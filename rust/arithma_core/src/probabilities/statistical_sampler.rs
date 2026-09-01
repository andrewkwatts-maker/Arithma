//====== Arithma/rust/arithma_core/src/probabilities/statistical_sampler.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Statistical sampler
//!
//! Generic sampling driver — feeds any [`ArithmaDistribution`] through
//! inverse-CDF (Smirnov) transform sampling: draw `u ~ Uniform(0, 1)` and
//! return `Q(u)`, which is distributed exactly as the target.
//!
//! ## Why a hand-rolled PRNG
//!
//! `arithma_core` deliberately carries no `rand` dependency, and `std` has no
//! random number generator, so [`ArithmaXorShift64Star`] is implemented here.
//! It is seeded **explicitly** and never from the clock or the OS, which means
//! every draw in this crate is bit-for-bit reproducible across runs, machines
//! and platforms — a property the test suite relies on and which matters for
//! reproducible numerical work generally.
//!
//! xorshift64* (Marsaglia 2003, with Vigna's 64-bit multiplier finaliser) was
//! chosen because it is ~5 lines, has a full `2^64 − 1` period, and passes
//! BigCrush. It is **not** cryptographically secure and must not be used for
//! keys, nonces or anything security-bearing.

use crate::probabilities::{ArithmaDistribution, ArithmaQuantileFunction};

/// Default seed for [`ArithmaStatisticalSampler::sample`].
///
/// The value is `⌊2⁶⁴/φ⌋`, the golden-ratio odd constant — an arbitrary but
/// well-mixed starting state. Because it is fixed, `sample` is a pure
/// function: the same distribution and `n` always yield the same vector. Call
/// [`ArithmaStatisticalSampler::sample_with_seed`] for an independent stream.
pub const DEFAULT_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// Vigna's 64-bit finalising multiplier for xorshift64*.
const XORSHIFT_MULTIPLIER: u64 = 0x2545_F491_4F6C_DD1D;

/// `2^-53`, the scale that turns 53 random bits into a unit-interval double.
const TWO_POW_NEG_53: f64 = 1.0 / 9_007_199_254_740_992.0;

/// Marsaglia's xorshift64 generator with Vigna's multiplicative finaliser.
///
/// Period `2^64 − 1`. Deterministic given its seed. Not cryptographically
/// secure.
#[derive(Debug, Clone)]
pub struct ArithmaXorShift64Star {
    state: u64,
}

impl ArithmaXorShift64Star {
    /// Seed the generator. Zero is the generator's single fixed point, so it
    /// is silently replaced by [`DEFAULT_SEED`].
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { DEFAULT_SEED } else { seed },
        }
    }

    /// Next 64 raw bits.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(XORSHIFT_MULTIPLIER)
    }

    /// Next double on the **open** interval `(0, 1)`.
    ///
    /// The top 53 bits give an integer in `[0, 2^53)`; adding `½` before
    /// scaling shifts the lattice off both endpoints, so neither `0.0` nor
    /// `1.0` is ever produced. That matters because
    /// [`ArithmaQuantileFunction::inverse_cdf`] rejects `p ≤ 0` and `p ≥ 1`.
    pub fn next_open_unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) * TWO_POW_NEG_53
    }
}

/// Static sampler.
pub struct ArithmaStatisticalSampler;

impl ArithmaStatisticalSampler {
    /// Draw `n` samples from `dist` using the fixed [`DEFAULT_SEED`].
    ///
    /// Deterministic: repeated calls with the same arguments return the same
    /// values. Use [`Self::sample_with_seed`] when independent streams are
    /// needed. `n == 0` yields an empty vector rather than an error.
    ///
    /// # Errors
    ///
    /// Propagates any error from the distribution's CDF or from the quantile
    /// bisection.
    pub fn sample(dist: &dyn ArithmaDistribution, n: usize) -> Result<Vec<f64>, String> {
        Self::sample_with_seed(dist, n, DEFAULT_SEED)
    }

    /// Draw `n` samples from `dist` with an explicit PRNG seed.
    ///
    /// Uses inverse-transform sampling: `xᵢ = Q(uᵢ)` for `uᵢ` uniform on
    /// `(0, 1)`. This works for continuous and discrete distributions alike —
    /// for a discrete one the quantile function returns the support point (up
    /// to the bisection tolerance), so the draws are the correct atoms.
    pub fn sample_with_seed(
        dist: &dyn ArithmaDistribution,
        n: usize,
        seed: u64,
    ) -> Result<Vec<f64>, String> {
        let mut rng = ArithmaXorShift64Star::new(seed);
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let u = rng.next_open_unit();
            out.push(ArithmaQuantileFunction::inverse_cdf(dist, u)?);
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Backward-compatibility aliases for the pre-rename `Arithmos*` names.
// Retained for one release; downstream (eml-math, eml-spectral, metaphysica,
// periodica) should migrate to the `Arithma*` names above.
// ---------------------------------------------------------------------------
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaStatisticalSampler`")]
#[allow(unused)]
pub use self::ArithmaStatisticalSampler as ArithmosStatisticalSampler;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probabilities::{ArithmaBernoulli, ArithmaNormal, ArithmaStatisticalMoment};

    #[test]
    fn rng_stays_inside_the_open_unit_interval() {
        let mut rng = ArithmaXorShift64Star::new(1);
        for _ in 0..100_000 {
            let u = rng.next_open_unit();
            assert!(u > 0.0 && u < 1.0, "u = {u} escaped (0, 1)");
        }
        // Zero seed is remapped, not a fixed point.
        let mut zero = ArithmaXorShift64Star::new(0);
        assert_ne!(zero.next_u64(), 0);
        assert_ne!(zero.next_u64(), 0);
    }

    #[test]
    fn rng_is_roughly_uniform() {
        // 10 equal bins over (0, 1); each should hold ~10% of 100k draws.
        let mut rng = ArithmaXorShift64Star::new(12_345);
        let mut bins = [0_u32; 10];
        for _ in 0..100_000 {
            let idx = (rng.next_open_unit() * 10.0) as usize;
            bins[idx.min(9)] += 1;
        }
        for (i, count) in bins.iter().enumerate() {
            assert!(
                (9_000..=11_000).contains(count),
                "bin {i} held {count}, expected ~10000"
            );
        }
    }

    #[test]
    fn sampling_is_reproducible() {
        let n = ArithmaNormal::standard();
        let a = ArithmaStatisticalSampler::sample(&n, 16).unwrap();
        let b = ArithmaStatisticalSampler::sample(&n, 16).unwrap();
        assert_eq!(a, b, "the default seed must make sampling deterministic");

        let c = ArithmaStatisticalSampler::sample_with_seed(&n, 16, DEFAULT_SEED).unwrap();
        assert_eq!(a, c, "sample() must equal sample_with_seed(DEFAULT_SEED)");

        // A different seed gives a different stream.
        let d = ArithmaStatisticalSampler::sample_with_seed(&n, 16, 7).unwrap();
        assert_ne!(a, d);
    }

    #[test]
    fn normal_samples_recover_their_moments() {
        let n = ArithmaNormal::new(5.0, 2.0);
        let draws = ArithmaStatisticalSampler::sample(&n, 4_000).unwrap();
        assert_eq!(draws.len(), 4_000);
        assert!(draws.iter().all(|v| v.is_finite()));

        // Standard error of the mean is 2/sqrt(4000) ~ 0.032; allow ~5 SE.
        let mean = ArithmaStatisticalMoment::mean(&draws).unwrap();
        assert!((mean - 5.0).abs() < 0.16, "sample mean {mean}, want ~5");

        // Population variance is 4.
        let var = ArithmaStatisticalMoment::variance(&draws).unwrap();
        assert!((var - 4.0).abs() < 0.4, "sample variance {var}, want ~4");

        // Roughly half the draws sit below the mean.
        let below = draws.iter().filter(|v| **v < 5.0).count();
        assert!(
            (1_800..=2_200).contains(&below),
            "{below} draws below the mean, want ~2000"
        );
    }

    #[test]
    fn discrete_samples_land_on_the_support() {
        let b = ArithmaBernoulli::new(0.3);
        let draws = ArithmaStatisticalSampler::sample(&b, 4_000).unwrap();
        assert!(
            draws.iter().all(|v| *v == 0.0 || *v == 1.0),
            "Bernoulli draws must be exactly 0 or 1"
        );
        let successes = draws.iter().filter(|v| **v == 1.0).count();
        // SE of the count is sqrt(4000*0.3*0.7) ~ 29; allow ~5 SE.
        assert!(
            (1_055..=1_345).contains(&successes),
            "{successes} successes out of 4000, want ~1200"
        );
    }

    #[test]
    fn zero_samples_and_invalid_distributions() {
        let n = ArithmaNormal::standard();
        // Edge case: n = 0 is an empty draw, not an error.
        assert_eq!(
            ArithmaStatisticalSampler::sample(&n, 0).unwrap(),
            Vec::<f64>::new()
        );

        // Errors from the underlying CDF propagate instead of panicking.
        let degenerate = ArithmaNormal::new(0.0, 0.0);
        assert!(ArithmaStatisticalSampler::sample(&degenerate, 4).is_err());
    }
}
