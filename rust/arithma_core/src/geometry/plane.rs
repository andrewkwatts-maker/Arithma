//====== Arithma/rust/arithma_core/src/geometry/plane.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Plane
//!
//! Infinite plane defined by a normal vector and a scalar offset (`n · p = d`).

use crate::expression::ArithmaExpression;
use crate::geometry::vector::ArithmaVector;

/// Plane in normal-and-offset form: `normal · p = offset`.
#[derive(Debug, Clone)]
pub struct ArithmaPlane {
    /// Plane normal (need not be unit-length).
    pub normal: ArithmaVector,
    /// Scalar offset along the normal.
    pub offset: ArithmaExpression,
}

impl ArithmaPlane {
    /// Construct a plane from normal and offset.
    pub fn new(normal: ArithmaVector, offset: ArithmaExpression) -> Self {
        Self { normal, offset }
    }

    /// Signed distance from `point` to this plane.
    ///
    /// Computes `(normal · point - offset) / |normal|`. Because the normal is
    /// not required to be unit-length, the division by `sqrt(normal · normal)`
    /// is always emitted; a caller holding a unit normal can simplify it away.
    ///
    /// The sign follows the normal: positive on the side the normal points
    /// toward, negative on the other side, zero on the plane.
    ///
    /// Degenerate case: a zero normal yields a division by zero, which the
    /// [`crate::expression::Evaluable`] implementation reports as `Err` rather
    /// than a numeric value. The expression is built structurally regardless so
    /// symbolic normals are never rejected at construction time.
    pub fn signed_distance(&self, point: &ArithmaVector) -> ArithmaExpression {
        let numerator = ArithmaExpression::sub(self.normal.dot(point), self.offset.clone());
        let denominator = ArithmaExpression::sqrt(self.normal.magnitude_squared());
        ArithmaExpression::div(numerator, denominator)
    }
}

// ---------------------------------------------------------------------------
// Backward-compatibility aliases for the pre-rename `Arithmos*` names.
// Retained for one release; downstream (eml-math, eml-spectral, metaphysica,
// periodica) should migrate to the `Arithma*` names above.
// ---------------------------------------------------------------------------
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaPlane`")]
#[allow(unused)]
pub use self::ArithmaPlane as ArithmosPlane;

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
    fn construct_plane() {
        let p = ArithmaPlane::new(ArithmaVector::zero(), ArithmaExpression::zero());
        let _ = p.normal;
    }

    #[test]
    fn distance_from_xy_plane_is_the_z_coordinate() {
        // z = 0 plane: normal (0,0,1), offset 0.
        let plane = ArithmaPlane::new(v(0, 0, 1), ArithmaExpression::zero());
        assert!((num(&plane.signed_distance(&v(9, -4, 5))) - 5.0).abs() < 1e-12);
        assert!((num(&plane.signed_distance(&v(0, 0, -2))) + 2.0).abs() < 1e-12);
        assert!(num(&plane.signed_distance(&v(3, 3, 0))).abs() < 1e-12);
    }

    #[test]
    fn offset_shifts_the_plane_along_the_normal() {
        // x = 4 plane.
        let plane = ArithmaPlane::new(v(1, 0, 0), ArithmaExpression::from_i64(4));
        assert!((num(&plane.signed_distance(&v(10, 0, 0))) - 6.0).abs() < 1e-12);
        assert!((num(&plane.signed_distance(&v(1, 0, 0))) + 3.0).abs() < 1e-12);
    }

    #[test]
    fn non_unit_normal_is_normalised() {
        // Same geometric plane as z = 0 but with a normal of length 5.
        let plane = ArithmaPlane::new(v(0, 0, 5), ArithmaExpression::zero());
        assert!((num(&plane.signed_distance(&v(0, 0, 3))) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn diagonal_plane_distance() {
        // x + y + z = 3, i.e. the plane through (1,1,1) with normal (1,1,1).
        let plane = ArithmaPlane::new(v(1, 1, 1), ArithmaExpression::from_i64(3));
        assert!(num(&plane.signed_distance(&v(1, 1, 1))).abs() < 1e-12);
        // Origin: (0 - 3) / sqrt(3) = -sqrt(3)
        let d = num(&plane.signed_distance(&v(0, 0, 0)));
        assert!((d + 3.0f64.sqrt()).abs() < 1e-9, "got {d}");
    }

    #[test]
    fn zero_normal_is_not_numerically_evaluable() {
        let plane = ArithmaPlane::new(ArithmaVector::zero(), ArithmaExpression::zero());
        let d = plane.signed_distance(&v(1, 2, 3));
        assert!(d.evaluate(&ArithmaBindings::new()).is_err());
    }

    #[test]
    fn signed_distance_keeps_symbolic_points() {
        let plane = ArithmaPlane::new(v(0, 0, 1), ArithmaExpression::zero());
        let point = ArithmaVector::new(
            ArithmaExpression::zero(),
            ArithmaExpression::zero(),
            ArithmaExpression::var("h"),
        );
        let d = plane.signed_distance(&point);
        assert!(d.evaluate(&ArithmaBindings::new()).is_err());
        let mut bindings = ArithmaBindings::new();
        bindings.insert("h".to_string(), 7.5);
        assert!((d.evaluate(&bindings).expect("bound") - 7.5).abs() < 1e-12);
    }
}
