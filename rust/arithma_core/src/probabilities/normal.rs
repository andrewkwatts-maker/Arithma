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
}

impl ArithmaDistribution for ArithmaNormal {
    fn pdf(&self, _x: f64) -> Result<f64, String> {
        unimplemented!("ArithmaNormal::pdf — populated in Wave 3")
    }
    fn cdf(&self, _x: f64) -> Result<f64, String> {
        unimplemented!("ArithmaNormal::cdf — populated in Wave 3")
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
}
