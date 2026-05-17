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
pub struct ArithmosConfidenceInterval {
    pub lower: f64,
    pub upper: f64,
    /// Confidence level in (0, 1).
    pub level: f64,
}

impl ArithmosConfidenceInterval {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn width_is_correct() {
        let ci = ArithmosConfidenceInterval::new(0.0, 1.0, 0.95);
        assert!((ci.width() - 1.0).abs() < f64::EPSILON);
    }
}

