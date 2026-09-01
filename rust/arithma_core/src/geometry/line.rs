//====== Arithma/rust/arithma_core/src/geometry/line.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Line
//!
//! Parametric line `P(t) = origin + t · direction`. Both vectors are symbolic.

use crate::expression::ArithmaExpression;
use crate::geometry::vector::ArithmaVector;

/// Parametric line: `P(t) = origin + t * direction`.
#[derive(Debug, Clone)]
pub struct ArithmaLine {
    /// Origin point.
    pub origin: ArithmaVector,
    /// Direction vector. Need not be unit-length — the parametric form handles
    /// scaling.
    pub direction: ArithmaVector,
}

impl ArithmaLine {
    /// Construct a line from origin and direction.
    pub fn new(origin: ArithmaVector, direction: ArithmaVector) -> Self {
        Self { origin, direction }
    }

    /// Evaluate the line at parameter `t`, returning `origin + t * direction`.
    ///
    /// `t` is cloned into each of the three components, so a symbolic `t`
    /// survives into every coordinate of the result.
    pub fn at(&self, t: ArithmaExpression) -> ArithmaVector {
        self.origin.add_vec(&self.direction.scale(&t))
    }
}

// ---------------------------------------------------------------------------
// Backward-compatibility aliases for the pre-rename `Arithmos*` names.
// Retained for one release; downstream (eml-math, eml-spectral, metaphysica,
// periodica) should migrate to the `Arithma*` names above.
// ---------------------------------------------------------------------------
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaLine`")]
#[allow(unused)]
pub use self::ArithmaLine as ArithmosLine;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::{ArithmaBindings, Evaluable};

    fn v(x: i64, y: i64, z: i64) -> ArithmaVector {
        ArithmaVector::new(
            ArithmaExpression::from_i64(x),
            ArithmaExpression::from_i64(y),
            ArithmaExpression::from_i64(z),
        )
    }

    fn num(e: &ArithmaExpression) -> f64 {
        e.evaluate(&ArithmaBindings::new())
            .expect("expression should evaluate numerically")
    }

    #[test]
    fn new_line_round_trip() {
        let line = ArithmaLine::new(ArithmaVector::zero(), ArithmaVector::zero());
        // Just exercise construction.
        let _ = line.origin;
    }

    #[test]
    fn at_zero_returns_the_origin() {
        let line = ArithmaLine::new(v(1, 2, 3), v(4, 5, 6));
        let p = line.at(ArithmaExpression::zero());
        assert!((num(&p.x) - 1.0).abs() < 1e-12);
        assert!((num(&p.y) - 2.0).abs() < 1e-12);
        assert!((num(&p.z) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn at_one_returns_origin_plus_direction() {
        let line = ArithmaLine::new(v(1, 2, 3), v(4, 5, 6));
        let p = line.at(ArithmaExpression::from_i64(1));
        assert!((num(&p.x) - 5.0).abs() < 1e-12);
        assert!((num(&p.y) - 7.0).abs() < 1e-12);
        assert!((num(&p.z) - 9.0).abs() < 1e-12);
    }

    #[test]
    fn at_negative_and_fractional_parameters() {
        let line = ArithmaLine::new(v(0, 0, 0), v(2, 4, 8));
        let back = line.at(ArithmaExpression::from_i64(-3));
        assert!((num(&back.x) + 6.0).abs() < 1e-12);
        assert!((num(&back.z) + 24.0).abs() < 1e-12);

        let half = line.at(ArithmaExpression::from_f64(0.5));
        assert!((num(&half.x) - 1.0).abs() < 1e-12);
        assert!((num(&half.y) - 2.0).abs() < 1e-12);
        assert!((num(&half.z) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn at_keeps_a_symbolic_parameter() {
        let line = ArithmaLine::new(v(1, 0, 0), v(0, 3, 0));
        let p = line.at(ArithmaExpression::var("t"));
        // Unbound `t` cannot collapse to a number.
        assert!(p.y.evaluate(&ArithmaBindings::new()).is_err());
        let mut bindings = ArithmaBindings::new();
        bindings.insert("t".to_string(), 2.0);
        assert!((p.y.evaluate(&bindings).expect("bound") - 6.0).abs() < 1e-12);
        assert!((p.x.evaluate(&bindings).expect("bound") - 1.0).abs() < 1e-12);
    }
}
