//====== Arithma/rust/arithma_core/src/probabilities/binomial.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Binomial distribution
//!
//! `Binomial(n, p)` — number of successes in `n` independent trials each with
//! success probability `p`.

use crate::probabilities::{ln_gamma, ArithmaDistribution};

/// Upper bound on the number of PMF terms [`ArithmaBinomial::cdf`] will sum.
///
/// `n` is a `u64`, so a naive `0..=n` summation is unbounded in practice. The
/// CDF refuses rather than spinning; callers with `n` this large want a
/// regularised-incomplete-beta implementation, not a term-by-term sum.
const MAX_CDF_TERMS: u64 = 1_000_000;

/// Binomial coefficient `C(n, k)` computed multiplicatively.
///
/// Uses the ascending recurrence `C(n, i) = C(n, i-1) · (n-i+1) / i` with the
/// symmetry `C(n, k) = C(n, n-k)` applied first to minimise the iteration
/// count. Factorials are never formed, so `C(52, 5)` does not go anywhere near
/// `52!`.
///
/// **Exactness domain.** Every partial result is itself an integer binomial
/// coefficient, so the result is *bit-exact* as long as each intermediate
/// product `C(n, i-1) · (n-i+1)` stays ≤ `2^53 = 9_007_199_254_740_992`. That
/// holds comfortably for all `n ≤ 50` (worst case `C(50, 25) ≈ 1.26e14`, whose
/// largest intermediate is ≈ `6.3e15`). Beyond that the value remains accurate
/// to a few ULP but is no longer guaranteed exact, and it overflows to
/// infinity somewhere past `n ≈ 1030`.
///
/// Returns `0.0` when `k > n`.
pub fn binomial_coefficient(n: u64, k: u64) -> f64 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    // The loop ran `min(k, n-k)` times over `u64` arguments, so
    // `C(u64::MAX, u64::MAX/2)` would spin for ~9e18 iterations. The cap is
    // not a compromise: whenever `min(k, n-k) > 1024` the true value is
    // already past `f64::MAX`, so infinity is the correct answer.
    //
    // Proof: with `m = min(k, n-k)` we have `n >= 2m`, and
    // `C(n, m) >= (n/m)^m >= 2^m`. At `m = 1025` that is `2^1025`, and
    // `f64::MAX` is just under `2^1024`.
    if k > MAX_EXACT_BINOMIAL_TERMS {
        return f64::INFINITY;
    }
    debug_assert!(k <= MAX_EXACT_BINOMIAL_TERMS, "term cap not enforced");
    let mut result = 1.0_f64;
    for i in 1..=k {
        result = result * ((n - k + i) as f64) / (i as f64);
    }
    debug_assert!(!result.is_nan(), "coefficient came out NaN");
    result
}

/// Largest `min(k, n-k)` for which [`binomial_coefficient`] iterates.
///
/// Beyond this the exact value exceeds `f64::MAX`, so the function returns
/// infinity without looping. See the proof in its body.
pub const MAX_EXACT_BINOMIAL_TERMS: u64 = 1024;

/// Largest `min(k, n-k)` for which [`ln_binomial_coefficient`] sums terms
/// directly before switching to the log-gamma closed form.
///
/// Term summation is the more accurate of the two, so it is preferred while
/// it is cheap; the cap only exists so the loop is bounded.
const MAX_LN_BINOMIAL_TERMS: u64 = 4096;

/// `ln C(n, k)`, accumulated term by term.
///
/// Fallback for the regime where [`binomial_coefficient`] overflows `f64`.
/// Slower and only ~1e-13 relative, but it keeps the PMF finite for very large
/// `n` instead of returning `NaN` from `inf * 0.0`.
fn ln_binomial_coefficient(n: u64, k: u64) -> f64 {
    let k = k.min(n - k);
    // Term summation is more accurate, but it is `min(k, n-k)` iterations --
    // unbounded over `u64` arguments. Past the cap, fall back to the log-gamma
    // identity, which is O(1):
    //
    //     ln C(n, k) = lnGamma(n+1) - lnGamma(k+1) - lnGamma(n-k+1)
    if k > MAX_LN_BINOMIAL_TERMS {
        let out = ln_gamma((n as f64) + 1.0)
            - ln_gamma((k as f64) + 1.0)
            - ln_gamma(((n - k) as f64) + 1.0);
        debug_assert!(!out.is_nan(), "log-gamma path produced NaN");
        return out;
    }
    debug_assert!(k <= MAX_LN_BINOMIAL_TERMS, "term cap not enforced");
    let mut acc = 0.0_f64;
    for i in 1..=k {
        acc += ((n - k + i) as f64).ln() - (i as f64).ln();
    }
    acc
}

/// Binomial distribution.
#[derive(Debug, Clone, Copy)]
pub struct ArithmaBinomial {
    pub n: u64,
    pub p: f64,
}

impl ArithmaBinomial {
    /// `Binomial(n, p)`.
    pub fn new(n: u64, p: f64) -> Self {
        Self { n, p }
    }

    /// `p` must be a probability. The trait forbids panicking, so every entry
    /// point validates first.
    fn validate(&self) -> Result<(), String> {
        if !self.p.is_finite() || !(0.0..=1.0).contains(&self.p) {
            return Err(format!(
                "ArithmaBinomial: p must lie in [0, 1], got {}",
                self.p
            ));
        }
        Ok(())
    }

    /// PMF at an integral `k`, assuming `p` has already been validated.
    fn pmf_at(&self, k: u64) -> f64 {
        if k > self.n {
            return 0.0;
        }
        let q = 1.0 - self.p;
        // Degenerate parameters: 0^0 is 1 by IEEE `pow`, but spelling these out
        // keeps the answers exactly 0.0 / 1.0 rather than merely very close.
        if self.p == 0.0 {
            return if k == 0 { 1.0 } else { 0.0 };
        }
        if self.p == 1.0 {
            return if k == self.n { 1.0 } else { 0.0 };
        }

        let coeff = binomial_coefficient(self.n, k);
        if coeff.is_finite() {
            // `coeff` is finite and both power factors are ≤ 1, so this cannot
            // overflow; multiplying the (large) coefficient first also keeps
            // the (small) powers from underflowing prematurely.
            // `powf` rather than `powi`: libm's `pow` is accurate to well under
            // an ULP for integral exponents, whereas `powi`'s repeated squaring
            // accumulates error, and `n` does not fit an `i32` anyway.
            coeff * self.p.powf(k as f64) * q.powf((self.n - k) as f64)
        } else {
            (ln_binomial_coefficient(self.n, k)
                + (k as f64) * self.p.ln()
                + ((self.n - k) as f64) * q.ln())
            .exp()
        }
    }
}

impl ArithmaDistribution for ArithmaBinomial {
    /// Probability mass function `P(X = k) = C(n, k) pᵏ (1-p)ⁿ⁻ᵏ`.
    ///
    /// The support is `{0, 1, …, n}`. Non-integral `x`, negative `x` and
    /// `x > n` all have mass `0`. For `Binomial(5, 0.5)`, `pdf(2)` is exactly
    /// `0.3125`.
    ///
    /// The coefficient comes from [`binomial_coefficient`]; see its docs for
    /// the range over which it is bit-exact.
    fn pdf(&self, x: f64) -> Result<f64, String> {
        self.validate()?;
        if x.is_nan() {
            return Err("ArithmaBinomial::pdf: x must not be NaN".to_string());
        }
        if !x.is_finite() || x < 0.0 || x.fract() != 0.0 || x > self.n as f64 {
            return Ok(0.0);
        }
        Ok(self.pmf_at(x as u64))
    }

    /// `F(x) = Σ_{i=0}^{⌊x⌋} P(X = i)`.
    ///
    /// The loop runs at most `min(⌊x⌋, n) + 1` times and is refused outright
    /// when `n > MAX_CDF_TERMS` (1e6), so it always terminates. For
    /// `Binomial(5, 0.5)`, `cdf(2)` is `0.5`.
    fn cdf(&self, x: f64) -> Result<f64, String> {
        self.validate()?;
        if x.is_nan() {
            return Err("ArithmaBinomial::cdf: x must not be NaN".to_string());
        }
        if x < 0.0 {
            return Ok(0.0);
        }
        if x >= self.n as f64 {
            return Ok(1.0);
        }
        if self.n > MAX_CDF_TERMS {
            return Err(format!(
                "ArithmaBinomial::cdf: n = {} exceeds the {MAX_CDF_TERMS}-term summation limit",
                self.n
            ));
        }
        // x is finite, >= 0 and < n here, so the cast is in range.
        let k = (x.floor() as u64).min(self.n);
        let mut acc = 0.0_f64;
        for i in 0..=k {
            acc += self.pmf_at(i);
        }
        Ok(acc.clamp(0.0, 1.0))
    }
    fn mean(&self) -> Result<f64, String> {
        self.validate()?;
        Ok((self.n as f64) * self.p)
    }
    fn variance(&self) -> Result<f64, String> {
        self.validate()?;
        Ok((self.n as f64) * self.p * (1.0 - self.p))
    }
}

// ---------------------------------------------------------------------------
// Backward-compatibility aliases for the pre-rename `Arithmos*` names.
// Retained for one release; downstream (eml-math, eml-spectral, metaphysica,
// periodica) should migrate to the `Arithma*` names above.
// ---------------------------------------------------------------------------
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaBinomial`")]
#[allow(unused)]
pub use self::ArithmaBinomial as ArithmosBinomial;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binomial_mean_is_np() {
        let b = ArithmaBinomial::new(10, 0.5);
        assert!((b.mean().unwrap() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn binomial_coefficients_are_exact() {
        assert_eq!(binomial_coefficient(0, 0), 1.0);
        assert_eq!(binomial_coefficient(5, 0), 1.0);
        assert_eq!(binomial_coefficient(5, 2), 10.0);
        assert_eq!(binomial_coefficient(5, 5), 1.0);
        assert_eq!(binomial_coefficient(10, 5), 252.0);
        // Poker hands.
        assert_eq!(binomial_coefficient(52, 5), 2_598_960.0);
        // Still exact at the documented n <= 50 bound.
        assert_eq!(binomial_coefficient(50, 25), 126_410_606_437_752.0);
        // k > n has no combinations.
        assert_eq!(binomial_coefficient(3, 4), 0.0);
    }

    #[test]
    fn fair_five_coin_pmf_is_exact() {
        let b = ArithmaBinomial::new(5, 0.5);
        // Every value is a dyadic rational, so equality is legitimate here.
        assert_eq!(b.pdf(0.0).unwrap(), 0.031_25);
        assert_eq!(b.pdf(1.0).unwrap(), 0.156_25);
        assert_eq!(b.pdf(2.0).unwrap(), 0.312_5);
        assert_eq!(b.pdf(3.0).unwrap(), 0.312_5);
        assert_eq!(b.pdf(5.0).unwrap(), 0.031_25);
        // Off support.
        assert_eq!(b.pdf(2.5).unwrap(), 0.0);
        assert_eq!(b.pdf(6.0).unwrap(), 0.0);
        assert_eq!(b.pdf(-1.0).unwrap(), 0.0);
        // Masses sum to 1.
        let total: f64 = (0..=5).map(|k| b.pdf(f64::from(k)).unwrap()).sum();
        assert!((total - 1.0).abs() < 1e-15, "total mass {total}");
    }

    #[test]
    fn asymmetric_pmf_and_cdf_match_known_values() {
        let b = ArithmaBinomial::new(10, 0.3);
        // C(10,3) * 0.3^3 * 0.7^7
        assert!(
            (b.pdf(3.0).unwrap() - 0.266_827_932_000_000_04).abs() < 1e-14,
            "pdf(3) = {}",
            b.pdf(3.0).unwrap()
        );
        assert!((b.pdf(0.0).unwrap() - 0.028_247_524_9).abs() < 1e-14);
        // F(3) = 0.6496107184
        assert!(
            (b.cdf(3.0).unwrap() - 0.649_610_718_4).abs() < 1e-14,
            "cdf(3) = {}",
            b.cdf(3.0).unwrap()
        );
        // The CDF floors its argument.
        assert_eq!(b.cdf(3.9).unwrap(), b.cdf(3.0).unwrap());
        assert!((b.cdf(10.0).unwrap() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn fair_five_coin_cdf_is_exact() {
        let b = ArithmaBinomial::new(5, 0.5);
        assert_eq!(b.cdf(-0.5).unwrap(), 0.0);
        assert_eq!(b.cdf(0.0).unwrap(), 0.031_25);
        assert_eq!(b.cdf(2.0).unwrap(), 0.5);
        assert_eq!(b.cdf(5.0).unwrap(), 1.0);
        assert_eq!(b.cdf(100.0).unwrap(), 1.0);
    }

    #[test]
    fn degenerate_n_and_p_edge_cases() {
        // n = 0: the point mass at 0.
        let none = ArithmaBinomial::new(0, 0.5);
        assert_eq!(none.pdf(0.0).unwrap(), 1.0);
        assert_eq!(none.pdf(1.0).unwrap(), 0.0);
        assert_eq!(none.cdf(0.0).unwrap(), 1.0);
        assert_eq!(none.mean().unwrap(), 0.0);

        // p = 0: never succeeds.
        let never = ArithmaBinomial::new(7, 0.0);
        assert_eq!(never.pdf(0.0).unwrap(), 1.0);
        assert_eq!(never.pdf(1.0).unwrap(), 0.0);
        assert_eq!(never.cdf(0.0).unwrap(), 1.0);

        // p = 1: always succeeds.
        let always = ArithmaBinomial::new(7, 1.0);
        assert_eq!(always.pdf(7.0).unwrap(), 1.0);
        assert_eq!(always.pdf(6.0).unwrap(), 0.0);
        assert_eq!(always.cdf(6.0).unwrap(), 0.0);
        assert_eq!(always.cdf(7.0).unwrap(), 1.0);
    }

    #[test]
    fn large_n_stays_finite_via_the_log_fallback() {
        // C(4000, 2000) overflows f64, so this exercises the log path.
        let b = ArithmaBinomial::new(4000, 0.5);
        let peak = b.pdf(2000.0).unwrap();
        assert!(peak.is_finite() && peak > 0.0, "peak = {peak}");
        // Normal approximation: 1/sqrt(2*pi*n*p*q) with n*p*q = 1000.
        let expected = 1.0 / (2.0 * std::f64::consts::PI * 1000.0).sqrt();
        assert!(
            (peak - expected).abs() / expected < 1e-4,
            "peak {peak} vs normal approx {expected}"
        );
    }

    #[test]
    fn invalid_inputs_are_rejected_not_panicked() {
        assert!(ArithmaBinomial::new(5, 1.5).pdf(2.0).is_err());
        assert!(ArithmaBinomial::new(5, -0.1).cdf(2.0).is_err());
        assert!(ArithmaBinomial::new(5, f64::NAN).pdf(2.0).is_err());
        assert!(ArithmaBinomial::new(5, 0.5).pdf(f64::NAN).is_err());
        // Summation limit.
        assert!(ArithmaBinomial::new(2_000_000, 0.5).cdf(10.0).is_err());
    }
    // ─── regressions ───────────────────────────────────────────────────────

    #[test]
    fn binomial_coefficient_terminates_on_adversarial_input() {
        // This used to loop `min(k, n-k)` times -- about 9e18 iterations, so
        // the call never returned. It must now answer immediately.
        let huge = binomial_coefficient(u64::MAX, u64::MAX / 2);
        assert!(
            huge.is_infinite(),
            "C(u64::MAX, u64::MAX/2) is far past f64::MAX, so infinity is correct"
        );
        assert!(binomial_coefficient(u64::MAX, 0).is_finite());
    }

    #[test]
    fn the_term_cap_sits_exactly_where_the_value_leaves_f64() {
        // The cap is a mathematical boundary, not a guess: with
        // m = min(k, n-k) and n >= 2m, C(n, m) >= 2^m, and f64::MAX < 2^1024.
        // So every refused case genuinely is infinite, and the last accepted
        // case is genuinely computed.
        let at_cap = binomial_coefficient(2 * MAX_EXACT_BINOMIAL_TERMS, MAX_EXACT_BINOMIAL_TERMS);
        assert!(at_cap.is_infinite(), "C(2048, 1024) overflows f64 honestly");
        // A case just inside the cap that is genuinely finite still computes.
        assert_eq!(binomial_coefficient(1024, 1), 1024.0);
        assert_eq!(binomial_coefficient(50, 25), 126410606437752.0);
    }

    #[test]
    fn small_coefficients_are_still_exact() {
        assert_eq!(binomial_coefficient(5, 2), 10.0);
        assert_eq!(binomial_coefficient(10, 5), 252.0);
        assert_eq!(binomial_coefficient(0, 0), 1.0);
        assert_eq!(binomial_coefficient(3, 4), 0.0, "k > n has no combinations");
    }

    #[test]
    fn the_log_gamma_fallback_agrees_with_term_summation() {
        // Both paths must give the same answer where they overlap, or the
        // pmf would jump discontinuously as n crosses the cap.
        for (n, k) in [(20_000_u64, 5_000_u64), (100_000, 9_000)] {
            let by_terms = {
                let m = k.min(n - k);
                let mut acc = 0.0_f64;
                for i in 1..=m {
                    acc += ((n - m + i) as f64).ln() - (i as f64).ln();
                }
                acc
            };
            let by_gamma = ln_gamma((n as f64) + 1.0)
                - ln_gamma((k as f64) + 1.0)
                - ln_gamma(((n - k) as f64) + 1.0);
            let rel = (by_terms - by_gamma).abs() / by_terms.abs();
            assert!(
                rel < 1e-12,
                "ln C({n}, {k}): terms {by_terms} vs gamma {by_gamma}, rel {rel:e}"
            );
        }
    }

    #[test]
    fn mean_and_variance_refuse_an_invalid_distribution() {
        // These returned Ok on a distribution whose pdf errors.
        let bad = ArithmaBinomial::new(10, f64::NAN);
        assert!(bad.pdf(1.0).is_err());
        assert!(
            bad.mean().is_err(),
            "mean must not report a broken distribution as healthy"
        );
        assert!(bad.variance().is_err());
        let good = ArithmaBinomial::new(10, 0.5);
        assert_eq!(good.mean().expect("valid"), 5.0);
        assert_eq!(good.variance().expect("valid"), 2.5);
    }
}
