//====== Arithma/rust/arithma_core/src/probabilities/quantile_function.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Quantile function
//!
//! Inverse-CDF lookups. Wave-2 stub.

use crate::probabilities::ArithmaDistribution;

/// Helper for evaluating inverse CDFs.
pub struct ArithmaQuantileFunction;

impl ArithmaQuantileFunction {
    /// Inverse CDF `Q(p) = inf{ x : F(x) ≥ p }`. Wave-2 stub.
    pub fn inverse_cdf(_dist: &dyn ArithmaDistribution, _p: f64) -> Result<f64, String> {
        unimplemented!("ArithmaQuantileFunction::inverse_cdf — populated in Wave 3")
    }
}

// ---------------------------------------------------------------------------
// Backward-compatibility aliases for the pre-rename `Arithmos*` names.
// Retained for one release; downstream (eml-math, eml-spectral, metaphysica,
// periodica) should migrate to the `Arithma*` names above.
// ---------------------------------------------------------------------------
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaQuantileFunction`")]
#[allow(unused)]
pub use self::ArithmaQuantileFunction as ArithmosQuantileFunction;

#[cfg(test)]
mod tests {
    #[test]
    fn module_compiles() {
        assert!(true);
    }
}
