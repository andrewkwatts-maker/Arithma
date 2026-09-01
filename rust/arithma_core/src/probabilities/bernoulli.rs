//====== Arithma/rust/arithma_core/src/probabilities/bernoulli.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Bernoulli distribution
//!
//! `Bernoulli(p)` — single trial success / failure.

use crate::probabilities::ArithmaDistribution;

/// Bernoulli distribution.
#[derive(Debug, Clone, Copy)]
pub struct ArithmaBernoulli {
    pub p: f64,
}

impl ArithmaBernoulli {
    /// `Bernoulli(p)`.
    pub fn new(p: f64) -> Self {
        Self { p }
    }

    /// `p` must be a probability. The trait forbids panicking, so every entry
    /// point validates first.
    fn validate(&self) -> Result<(), String> {
        if !self.p.is_finite() || !(0.0..=1.0).contains(&self.p) {
            return Err(format!(
                "ArithmaBernoulli: p must lie in [0, 1], got {}",
                self.p
            ));
        }
        Ok(())
    }
}

impl ArithmaDistribution for ArithmaBernoulli {
    /// Probability mass function: `P(X = 1) = p`, `P(X = 0) = 1 - p`.
    ///
    /// The support is the two-point set `{0, 1}`; any other `x` — including
    /// non-integers such as `0.5` — has mass `0`. `x` is compared exactly
    /// against `0.0` and `1.0`, so a caller holding a value that merely
    /// rounds to an integer must round it before calling.
    fn pdf(&self, x: f64) -> Result<f64, String> {
        self.validate()?;
        if x.is_nan() {
            return Err("ArithmaBernoulli::pdf: x must not be NaN".to_string());
        }
        if x == 1.0 {
            Ok(self.p)
        } else if x == 0.0 {
            Ok(1.0 - self.p)
        } else {
            Ok(0.0)
        }
    }

    /// `F(x) = P(X ≤ x)`: `0` below `0`, `1 - p` on `[0, 1)`, `1` from `1` up.
    fn cdf(&self, x: f64) -> Result<f64, String> {
        self.validate()?;
        if x.is_nan() {
            return Err("ArithmaBernoulli::cdf: x must not be NaN".to_string());
        }
        if x < 0.0 {
            Ok(0.0)
        } else if x < 1.0 {
            Ok(1.0 - self.p)
        } else {
            Ok(1.0)
        }
    }
    fn mean(&self) -> Result<f64, String> {
        Ok(self.p)
    }
    fn variance(&self) -> Result<f64, String> {
        Ok(self.p * (1.0 - self.p))
    }
}

// ---------------------------------------------------------------------------
// Backward-compatibility aliases for the pre-rename `Arithmos*` names.
// Retained for one release; downstream (eml-math, eml-spectral, metaphysica,
// periodica) should migrate to the `Arithma*` names above.
// ---------------------------------------------------------------------------
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaBernoulli`")]
#[allow(unused)]
pub use self::ArithmaBernoulli as ArithmosBernoulli;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fair_coin_has_quarter_variance() {
        let b = ArithmaBernoulli::new(0.5);
        assert!((b.variance().unwrap() - 0.25).abs() < 1e-12);
    }

    #[test]
    fn pmf_matches_known_values() {
        let b = ArithmaBernoulli::new(0.3);
        assert_eq!(b.pdf(1.0).unwrap(), 0.3);
        assert!((b.pdf(0.0).unwrap() - 0.7).abs() < 1e-15);
        // Off-support points carry no mass.
        assert_eq!(b.pdf(0.5).unwrap(), 0.0);
        assert_eq!(b.pdf(2.0).unwrap(), 0.0);
        assert_eq!(b.pdf(-1.0).unwrap(), 0.0);
        // Total mass is exactly 1.
        assert_eq!(b.pdf(0.0).unwrap() + b.pdf(1.0).unwrap(), 1.0);
    }

    #[test]
    fn cdf_is_the_two_step_staircase() {
        let b = ArithmaBernoulli::new(0.3);
        assert_eq!(b.cdf(-0.001).unwrap(), 0.0);
        assert!((b.cdf(0.0).unwrap() - 0.7).abs() < 1e-15);
        assert!((b.cdf(0.999).unwrap() - 0.7).abs() < 1e-15);
        assert_eq!(b.cdf(1.0).unwrap(), 1.0);
        assert_eq!(b.cdf(10.0).unwrap(), 1.0);
    }

    #[test]
    fn degenerate_p_edge_cases() {
        // p = 0: all mass on failure.
        let never = ArithmaBernoulli::new(0.0);
        assert_eq!(never.pdf(0.0).unwrap(), 1.0);
        assert_eq!(never.pdf(1.0).unwrap(), 0.0);
        assert_eq!(never.cdf(0.0).unwrap(), 1.0);
        assert_eq!(never.variance().unwrap(), 0.0);

        // p = 1: all mass on success.
        let always = ArithmaBernoulli::new(1.0);
        assert_eq!(always.pdf(1.0).unwrap(), 1.0);
        assert_eq!(always.pdf(0.0).unwrap(), 0.0);
        assert_eq!(always.cdf(0.0).unwrap(), 0.0);
        assert_eq!(always.cdf(1.0).unwrap(), 1.0);
    }

    #[test]
    fn out_of_range_p_is_rejected_not_panicked() {
        assert!(ArithmaBernoulli::new(1.5).pdf(1.0).is_err());
        assert!(ArithmaBernoulli::new(-0.1).cdf(0.0).is_err());
        assert!(ArithmaBernoulli::new(f64::NAN).pdf(0.0).is_err());
        assert!(ArithmaBernoulli::new(0.5).pdf(f64::NAN).is_err());
    }
}
