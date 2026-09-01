//====== Arithma/rust/arithma_core/src/probabilities/confidence_interval.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Confidence interval
//!
//! Construction of confidence intervals around point estimates. Wave-2 stub.

/// Confidence interval `[lower, upper]` at a given confidence level.
#[derive(Debug, Clone, Copy)]
pub struct ArithmaConfidenceInterval {
    pub lower: f64,
    pub upper: f64,
    /// Confidence level in (0, 1).
    pub level: f64,
}

impl ArithmaConfidenceInterval {
    /// Construct a fresh interval.
    pub fn new(lower: f64, upper: f64, level: f64) -> Self {
        Self {
            lower,
            upper,
            level,
        }
    }

    /// Width of the interval `upper - lower`.
    pub fn width(&self) -> f64 {
        self.upper - self.lower
    }
}

// ---------------------------------------------------------------------------
// Backward-compatibility aliases for the pre-rename `Arithmos*` names.
// Retained for one release; downstream (eml-math, eml-spectral, metaphysica,
// periodica) should migrate to the `Arithma*` names above.
// ---------------------------------------------------------------------------
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaConfidenceInterval`")]
#[allow(unused)]
pub use self::ArithmaConfidenceInterval as ArithmosConfidenceInterval;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn width_is_correct() {
        let ci = ArithmaConfidenceInterval::new(0.0, 1.0, 0.95);
        assert!((ci.width() - 1.0).abs() < f64::EPSILON);
    }
}
