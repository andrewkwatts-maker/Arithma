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

use crate::expression::ArithmosExpression;

/// Closed interval `[lo, hi]` over the reals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArithmosInterval {
    pub lo: f64,
    pub hi: f64,
}

impl ArithmosInterval {
    /// Construct an interval. `lo` must be â‰¤ `hi`.
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
    _expr: &ArithmosExpression,
    _var: &str,
    _interval: ArithmosInterval,
) -> Result<ArithmosInterval, String> {
    unimplemented!("evaluate_interval â€” populated in Wave 3")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_interval_contains_zero_five() {
        let i = ArithmosInterval::new(0.0, 1.0);
        assert!(i.contains(0.5));
    }

    #[test]
    fn empty_interval_contains_nothing() {
        let i = ArithmosInterval::empty();
        assert!(i.is_empty());
    }
}

