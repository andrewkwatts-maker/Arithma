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
//! inverse-CDF or rejection sampling.

use crate::probabilities::ArithmaDistribution;

/// Static sampler.
pub struct ArithmaStatisticalSampler;

impl ArithmaStatisticalSampler {
    /// Draw `n` samples from the supplied distribution. Wave-2 stub.
    pub fn sample(_dist: &dyn ArithmaDistribution, _n: usize) -> Result<Vec<f64>, String> {
        unimplemented!("ArithmaStatisticalSampler::sample — populated in Wave 3")
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
    #[test]
    fn module_compiles() {
        assert!(true);
    }
}
