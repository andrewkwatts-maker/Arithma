//====== Arithma/rust/arithma_core/src/numerical/interval_analysis.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Interval analysis
//!
//! Interval arithmetic for guaranteed bounds. Used by the root finder to prune
//! search regions and by the simplifier to detect provably-empty branches.

use crate::expression::ArithmaExpression;

/// Closed interval `[lo, hi]` over the reals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArithmaInterval {
    pub lo: f64,
    pub hi: f64,
}

impl ArithmaInterval {
    /// Construct an interval. `lo` must be ≤ `hi`.
    pub fn new(lo: f64, hi: f64) -> Self {
        Self { lo, hi }
    }

    /// The whole real line.
    pub fn whole() -> Self {
        Self {
            lo: f64::NEG_INFINITY,
            hi: f64::INFINITY,
        }
    }

    /// The empty interval. Represented with `lo > hi`.
    pub fn empty() -> Self {
        Self {
            lo: f64::INFINITY,
            hi: f64::NEG_INFINITY,
        }
    }

    /// Width `hi - lo`.
    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }

    /// True iff this interval is empty.
    pub fn is_empty(&self) -> bool {
        self.lo > self.hi
    }

    /// True iff `point` is contained in `[lo, hi]`.
    pub fn contains(&self, point: f64) -> bool {
        !self.is_empty() && point >= self.lo && point <= self.hi
    }
}

/// Compute the interval-arithmetic image of `expr` when `var` ranges over
/// `interval`. Wave-2 stub.
pub fn evaluate_interval(
    _expr: &ArithmaExpression,
    _var: &str,
    _interval: ArithmaInterval,
) -> Result<ArithmaInterval, String> {
    unimplemented!("evaluate_interval — populated in Wave 3")
}

// ---------------------------------------------------------------------------
// Backward-compatibility aliases for the pre-rename `Arithmos*` names.
// Retained for one release; downstream (eml-math, eml-spectral, metaphysica,
// periodica) should migrate to the `Arithma*` names above.
// ---------------------------------------------------------------------------
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaInterval`")]
#[allow(unused)]
pub use self::ArithmaInterval as ArithmosInterval;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_interval_contains_zero_five() {
        let i = ArithmaInterval::new(0.0, 1.0);
        assert!(i.contains(0.5));
    }

    #[test]
    fn empty_interval_contains_nothing() {
        let i = ArithmaInterval::empty();
        assert!(i.is_empty());
    }
}
