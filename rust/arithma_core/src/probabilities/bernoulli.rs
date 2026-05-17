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
//! `Bernoulli(p)` â€” single trial success / failure.

use crate::probabilities::ArithmosDistribution;

/// Bernoulli distribution.
#[derive(Debug, Clone, Copy)]
pub struct ArithmosBernoulli {
    pub p: f64,
}

impl ArithmosBernoulli {
    /// `Bernoulli(p)`.
    pub fn new(p: f64) -> Self {
        Self { p }
    }
}

impl ArithmosDistribution for ArithmosBernoulli {
    fn pdf(&self, _x: f64) -> Result<f64, String> {
        unimplemented!("ArithmosBernoulli::pdf â€” populated in Wave 3")
    }
    fn cdf(&self, _x: f64) -> Result<f64, String> {
        unimplemented!("ArithmosBernoulli::cdf â€” populated in Wave 3")
    }
    fn mean(&self) -> Result<f64, String> {
        Ok(self.p)
    }
    fn variance(&self) -> Result<f64, String> {
        Ok(self.p * (1.0 - self.p))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fair_coin_has_quarter_variance() {
        let b = ArithmosBernoulli::new(0.5);
        assert!((b.variance().unwrap() - 0.25).abs() < 1e-12);
    }
}

