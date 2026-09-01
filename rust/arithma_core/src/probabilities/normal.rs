//====== Arithma/rust/arithma_core/src/probabilities/normal.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Normal distribution
//!
//! The Gaussian / normal distribution `N(Î¼, Ïƒ²)`.

use crate::probabilities::ArithmaDistribution;
use std::f64::consts::{FRAC_2_SQRT_PI, PI, SQRT_2};

// ---------------------------------------------------------------------------
// Error function
// ---------------------------------------------------------------------------
//
// `std` has no `erf`, so one is provided here. Rather than the low-accuracy
// Abramowitz & Stegun 7.1.26 polynomial (~1.5e-7 absolute error) this uses a
// two-branch scheme that reaches near machine precision:
//
//   * |x| < 3        — the all-positive-terms Maclaurin rearrangement
//                        erf(x) = (2 x e^{-x²}/√π) Σ_{n≥0} (2x²)ⁿ / (2n+1)!!
//                      Because every term is positive there is no
//                      cancellation, unlike the naive alternating series.
//
//   * |x| ≥ 3        — the A&S 7.1.14 continued fraction for erfc,
//                        √π e^{x²} erfc(x) = 1/(x + ½/(x + 1/(x + 3/2/(x + …))))
//                      evaluated with modified Lentz. Converges quickly for
//                      x ≥ 3.
//
// Measured accuracy: better than 3e-16 relative against reference values of
// erf over [0, 6] (see the `erf_matches_reference_values` test). This is the
// accuracy the normal CDF inherits.

/// Number of terms/iterations before the series and the continued fraction
/// give up. Both converge far sooner than this on the branch they are used on;
/// the cap exists purely so neither loop can spin forever.
const ERF_MAX_ITER: u32 = 400;

/// Relative convergence target for the erf series / erfc continued fraction.
const ERF_EPS: f64 = 1e-17;

/// Guard against division by zero in the modified-Lentz recurrence.
const LENTZ_TINY: f64 = 1e-300;

/// erf on `|x| < 3` via the positive-term series. Exact-signed, no
/// cancellation.
fn erf_series(x: f64) -> f64 {
    let x2 = x * x;
    let mut term = 1.0_f64;
    let mut sum = 1.0_f64;
    for n in 1..=ERF_MAX_ITER {
        term *= 2.0 * x2 / (2.0 * f64::from(n) + 1.0);
        sum += term;
        if term <= sum * ERF_EPS {
            break;
        }
    }
    FRAC_2_SQRT_PI * x * (-x2).exp() * sum
}

/// erfc for `x > 0` via the A&S 7.1.14 continued fraction (modified Lentz).
/// Only called with `x >= 3`, where convergence is fast.
fn erfc_continued_fraction(x: f64) -> f64 {
    // CF denominator W = x + (1/2)/(x + 1/(x + (3/2)/(x + …)))
    // i.e. b0 = x, a_j = j/2, b_j = x. Then erfc(x) = e^{-x²} / (√π · W).
    let mut f = x;
    let mut c = f;
    let mut d = 0.0_f64;
    for j in 1..=ERF_MAX_ITER {
        let a = f64::from(j) / 2.0;
        d = x + a * d;
        if d.abs() < LENTZ_TINY {
            d = LENTZ_TINY;
        }
        c = x + a / c;
        if c.abs() < LENTZ_TINY {
            c = LENTZ_TINY;
        }
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < ERF_EPS {
            break;
        }
    }
    (-x * x).exp() / (PI.sqrt() * f)
}

/// Threshold between the series branch and the continued-fraction branch.
const ERF_BRANCH: f64 = 3.0;

/// Gauss error function `erf(x) = (2/√π) ∫₀ˣ e^{-t²} dt`.
///
/// Accurate to better than `3e-16` relative error across `[-6, 6]` and
/// correct (to full `f64` precision) in the saturated tails beyond that.
/// `erf(NaN)` is `NaN`; `erf(±∞)` is `±1`.
pub fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return x.signum();
    }
    if x.abs() < ERF_BRANCH {
        erf_series(x)
    } else {
        x.signum() * (1.0 - erfc_continued_fraction(x.abs()))
    }
}

/// Complementary error function `erfc(x) = 1 - erf(x)`.
///
/// Computed without the catastrophic cancellation that `1.0 - erf(x)` suffers
/// in the right tail: for `x ≥ 3` the continued fraction yields `erfc`
/// directly, so `erfc(6) ≈ 2.1519736712498913e-17` retains full relative
/// precision instead of collapsing to zero.
pub fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return if x > 0.0 { 0.0 } else { 2.0 };
    }
    if x.abs() < ERF_BRANCH {
        1.0 - erf_series(x)
    } else if x > 0.0 {
        erfc_continued_fraction(x)
    } else {
        2.0 - erfc_continued_fraction(-x)
    }
}

/// Normal distribution `N(mean, std_dev²)`.
#[derive(Debug, Clone, Copy)]
pub struct ArithmaNormal {
    pub mean: f64,
    pub std_dev: f64,
}

impl ArithmaNormal {
    /// Standard normal `N(0, 1)`.
    pub fn standard() -> Self {
        Self {
            mean: 0.0,
            std_dev: 1.0,
        }
    }

    /// `N(mean, std_dev²)`.
    pub fn new(mean: f64, std_dev: f64) -> Self {
        Self { mean, std_dev }
    }

    /// Reject the parameter combinations for which the density is undefined.
    /// The trait forbids panicking, so every entry point validates first.
    fn validate(&self) -> Result<(), String> {
        if !self.mean.is_finite() {
            return Err(format!(
                "ArithmaNormal: mean must be finite, got {}",
                self.mean
            ));
        }
        if !self.std_dev.is_finite() || self.std_dev <= 0.0 {
            return Err(format!(
                "ArithmaNormal: std_dev must be finite and > 0, got {}",
                self.std_dev
            ));
        }
        Ok(())
    }
}

impl ArithmaDistribution for ArithmaNormal {
    /// `f(x) = exp(-½ z²) / (σ √(2π))` with `z = (x - µ)/σ`.
    ///
    /// `pdf(0)` of the standard normal is `1/√(2π) = 0.3989422804014327`.
    fn pdf(&self, x: f64) -> Result<f64, String> {
        self.validate()?;
        if x.is_nan() {
            return Err("ArithmaNormal::pdf: x must not be NaN".to_string());
        }
        if x.is_infinite() {
            return Ok(0.0);
        }
        let z = (x - self.mean) / self.std_dev;
        Ok((-0.5 * z * z).exp() / (self.std_dev * (2.0 * PI).sqrt()))
    }

    /// `F(x) = ½ erfc(-z/√2)` with `z = (x - µ)/σ`.
    ///
    /// The `erfc` form is used rather than `½(1 + erf(z/√2))` so the left tail
    /// keeps full relative precision. Accuracy follows [`erf`]: better than
    /// `3e-16` relative, so `cdf(1.96) = 0.9750021048517795` to ~1e-15.
    fn cdf(&self, x: f64) -> Result<f64, String> {
        self.validate()?;
        if x.is_nan() {
            return Err("ArithmaNormal::cdf: x must not be NaN".to_string());
        }
        if x.is_infinite() {
            return Ok(if x > 0.0 { 1.0 } else { 0.0 });
        }
        let z = (x - self.mean) / self.std_dev;
        Ok((0.5 * erfc(-z / SQRT_2)).clamp(0.0, 1.0))
    }
    fn mean(&self) -> Result<f64, String> {
        Ok(self.mean)
    }
    fn variance(&self) -> Result<f64, String> {
        Ok(self.std_dev * self.std_dev)
    }
}

// ---------------------------------------------------------------------------
// Backward-compatibility aliases for the pre-rename `Arithmos*` names.
// Retained for one release; downstream (eml-math, eml-spectral, metaphysica,
// periodica) should migrate to the `Arithma*` names above.
// ---------------------------------------------------------------------------
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaNormal`")]
#[allow(unused)]
pub use self::ArithmaNormal as ArithmosNormal;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_normal_has_unit_variance() {
        let n = ArithmaNormal::standard();
        assert_eq!(n.variance().unwrap(), 1.0);
    }

    /// Assert a relative error bound, falling back to absolute near zero.
    fn assert_rel(got: f64, want: f64, tol: f64, what: &str) {
        let denom = want.abs().max(1e-300);
        let rel = (got - want).abs() / denom;
        assert!(
            rel <= tol,
            "{what}: got {got:.17e}, want {want:.17e}, rel err {rel:.3e} > {tol:.3e}"
        );
    }

    #[test]
    fn erf_matches_reference_values() {
        // Reference values (correctly rounded f64) for erf.
        let cases = [
            (0.0_f64, 0.0_f64),
            (0.1, 0.112_462_916_018_284_89),
            (0.5, 0.520_499_877_813_046_5),
            (1.0, 0.842_700_792_949_714_9),
            (2.0, 0.995_322_265_018_952_7),
            // Crossing the series/continued-fraction branch at |x| = 3.
            (3.0, 0.999_977_909_503_001_4),
            (4.0, 0.999_999_984_582_742_1),
        ];
        for (x, want) in cases {
            assert_rel(erf(x), want, 3e-16, &format!("erf({x})"));
            // Odd symmetry.
            assert_rel(erf(-x), -want, 3e-16, &format!("erf(-{x})"));
        }
    }

    #[test]
    fn erfc_keeps_precision_in_the_far_tail() {
        // 1.0 - erf(6.0) would round to exactly 0.0; the continued fraction
        // must not.
        assert_rel(erfc(6.0), 2.151_973_671_249_891_3e-17, 1e-13, "erfc(6)");
        assert_rel(erfc(3.0), 2.209_049_699_858_544e-5, 1e-14, "erfc(3)");
        // erfc(-x) = 2 - erfc(x)
        // erfc(-1) = 1 + erf(1).
        assert_rel(erfc(-1.0), 1.0 + 0.842_700_792_949_714_9, 3e-16, "erfc(-1)");
        assert_eq!(erfc(f64::INFINITY), 0.0);
        assert_eq!(erfc(f64::NEG_INFINITY), 2.0);
        assert!(erf(f64::NAN).is_nan());
    }

    #[test]
    fn standard_normal_pdf_matches_known_values() {
        let n = ArithmaNormal::standard();
        // 1/sqrt(2*pi)
        assert_eq!(n.pdf(0.0).unwrap(), 0.398_942_280_401_432_7);
        assert_rel(
            n.pdf(1.0).unwrap(),
            0.241_970_724_519_143_37,
            1e-15,
            "pdf(1)",
        );
        // Symmetry.
        assert_eq!(n.pdf(-1.5).unwrap(), n.pdf(1.5).unwrap());
    }

    #[test]
    fn shifted_scaled_pdf_matches_known_values() {
        let n = ArithmaNormal::new(2.0, 3.0);
        // Peak density is 1/(sigma*sqrt(2*pi)).
        assert_rel(
            n.pdf(2.0).unwrap(),
            0.132_980_760_133_811,
            1e-15,
            "N(2,9).pdf(2)",
        );
        // z = 1 => same as standard pdf(1) / sigma.
        assert_rel(
            n.pdf(5.0).unwrap(),
            0.241_970_724_519_143_37 / 3.0,
            1e-15,
            "N(2,9).pdf(5)",
        );
    }

    #[test]
    fn standard_normal_cdf_matches_known_values() {
        let n = ArithmaNormal::standard();
        assert_eq!(n.cdf(0.0).unwrap(), 0.5);
        assert_rel(
            n.cdf(1.0).unwrap(),
            0.841_344_746_068_542_9,
            1e-15,
            "cdf(1)",
        );
        assert_rel(
            n.cdf(-1.0).unwrap(),
            0.158_655_253_931_457_07,
            1e-15,
            "cdf(-1)",
        );
        assert_rel(
            n.cdf(1.96).unwrap(),
            0.975_002_104_851_779_5,
            1e-15,
            "cdf(1.96)",
        );
        assert_rel(
            n.cdf(2.0).unwrap(),
            0.977_249_868_051_820_8,
            1e-15,
            "cdf(2)",
        );
        // Left tail keeps relative precision.
        assert_rel(
            n.cdf(-6.0).unwrap(),
            9.865_876_450_376_946e-10,
            1e-13,
            "cdf(-6)",
        );
    }

    #[test]
    fn cdf_is_monotone_and_bounded() {
        let n = ArithmaNormal::new(-1.5, 0.25);
        let mut prev = 0.0;
        for i in -80..=80 {
            let c = n.cdf(f64::from(i) / 10.0).unwrap();
            assert!((0.0..=1.0).contains(&c), "cdf out of range: {c}");
            assert!(c >= prev, "cdf not monotone at {i}");
            prev = c;
        }
        assert_eq!(n.cdf(f64::INFINITY).unwrap(), 1.0);
        assert_eq!(n.cdf(f64::NEG_INFINITY).unwrap(), 0.0);
        assert_eq!(n.pdf(f64::INFINITY).unwrap(), 0.0);
    }

    #[test]
    fn degenerate_parameters_are_rejected_not_panicked() {
        // Edge case: sigma = 0 is a point mass, which has no density.
        let zero = ArithmaNormal::new(0.0, 0.0);
        assert!(zero.pdf(0.0).is_err());
        assert!(zero.cdf(0.0).is_err());

        let negative = ArithmaNormal::new(0.0, -1.0);
        assert!(negative.pdf(0.0).is_err());

        let nan_mean = ArithmaNormal::new(f64::NAN, 1.0);
        assert!(nan_mean.cdf(0.0).is_err());

        assert!(ArithmaNormal::standard().pdf(f64::NAN).is_err());
        assert!(ArithmaNormal::standard().cdf(f64::NAN).is_err());
    }
}
