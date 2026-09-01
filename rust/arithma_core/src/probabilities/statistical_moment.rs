//====== Arithma/rust/arithma_core/src/probabilities/statistical_moment.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Statistical moment
//!
//! Moments of distributions and samples — variance, skewness, kurtosis.

/// Static helper for moment calculations on f64 datasets.
pub struct ArithmaStatisticalMoment;

impl ArithmaStatisticalMoment {
    /// Sample mean.
    pub fn mean(_data: &[f64]) -> Result<f64, String> {
        unimplemented!("ArithmaStatisticalMoment::mean — populated in Wave 3")
    }

    /// Sample variance (Bessel-corrected).
    pub fn variance(_data: &[f64]) -> Result<f64, String> {
        unimplemented!("ArithmaStatisticalMoment::variance — populated in Wave 3")
    }

    /// Sample skewness (third standardised moment).
    pub fn skewness(_data: &[f64]) -> Result<f64, String> {
        unimplemented!("ArithmaStatisticalMoment::skewness — populated in Wave 3")
    }

    /// Sample kurtosis (fourth standardised moment, excess form).
    pub fn kurtosis(_data: &[f64]) -> Result<f64, String> {
        unimplemented!("ArithmaStatisticalMoment::kurtosis — populated in Wave 3")
    }
}

// ---------------------------------------------------------------------------
// Backward-compatibility aliases for the pre-rename `Arithmos*` names.
// Retained for one release; downstream (eml-math, eml-spectral, metaphysica,
// periodica) should migrate to the `Arithma*` names above.
// ---------------------------------------------------------------------------
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaStatisticalMoment`")]
#[allow(unused)]
pub use self::ArithmaStatisticalMoment as ArithmosStatisticalMoment;

#[cfg(test)]
mod tests {
    #[test]
    fn module_compiles() {
        assert!(true);
    }
}
