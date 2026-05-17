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
//! `Binomial(n, p)` â€” number of successes in `n` independent trials each with
//! success probability `p`.

use crate::probabilities::ArithmosDistribution;

/// Binomial distribution.
#[derive(Debug, Clone, Copy)]
pub struct ArithmosBinomial {
    pub n: u64,
    pub p: f64,
}

impl ArithmosBinomial {
    /// `Binomial(n, p)`.
    pub fn new(n: u64, p: f64) -> Self {
        Self { n, p }
    }
}

impl ArithmosDistribution for ArithmosBinomial {
    fn pdf(&self, _x: f64) -> Result<f64, String> {
        unimplemented!("ArithmosBinomial::pdf â€” populated in Wave 3")
    }
    fn cdf(&self, _x: f64) -> Result<f64, String> {
        unimplemented!("ArithmosBinomial::cdf â€” populated in Wave 3")
    }
    fn mean(&self) -> Result<f64, String> {
        Ok((self.n as f64) * self.p)
    }
    fn variance(&self) -> Result<f64, String> {
        Ok((self.n as f64) * self.p * (1.0 - self.p))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binomial_mean_is_np() {
        let b = ArithmosBinomial::new(10, 0.5);
        assert!((b.mean().unwrap() - 5.0).abs() < 1e-12);
    }
}

