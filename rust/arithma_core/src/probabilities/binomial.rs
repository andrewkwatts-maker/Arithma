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

use crate::probabilities::ArithmaDistribution;

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
}

impl ArithmaDistribution for ArithmaBinomial {
    fn pdf(&self, _x: f64) -> Result<f64, String> {
        unimplemented!("ArithmaBinomial::pdf — populated in Wave 3")
    }
    fn cdf(&self, _x: f64) -> Result<f64, String> {
        unimplemented!("ArithmaBinomial::cdf — populated in Wave 3")
    }
    fn mean(&self) -> Result<f64, String> {
        Ok((self.n as f64) * self.p)
    }
    fn variance(&self) -> Result<f64, String> {
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
}
