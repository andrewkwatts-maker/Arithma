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
}

impl ArithmaDistribution for ArithmaBernoulli {
    fn pdf(&self, _x: f64) -> Result<f64, String> {
        unimplemented!("ArithmaBernoulli::pdf — populated in Wave 3")
    }
    fn cdf(&self, _x: f64) -> Result<f64, String> {
        unimplemented!("ArithmaBernoulli::cdf — populated in Wave 3")
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
}
