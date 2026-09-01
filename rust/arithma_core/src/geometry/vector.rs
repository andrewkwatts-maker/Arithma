//====== Arithma/rust/arithma_core/src/geometry/vector.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Vector
//!
//! `ArithmaVector` — a 3-vector whose components are `ArithmaExpression`
//! values, allowing coordinates to remain symbolic throughout the geometry
//! pipeline.

use crate::expression::ArithmaExpression;

/// A symbolic 3-vector. Components are full expressions so geometry calculations
/// stay symbolic until a numeric value is required.
#[derive(Debug, Clone)]
pub struct ArithmaVector {
    pub x: ArithmaExpression,
    pub y: ArithmaExpression,
    pub z: ArithmaExpression,
}

impl ArithmaVector {
    /// Construct from three components.
    pub fn new(x: ArithmaExpression, y: ArithmaExpression, z: ArithmaExpression) -> Self {
        Self { x, y, z }
    }

    /// The zero vector.
    pub fn zero() -> Self {
        Self {
            x: ArithmaExpression::zero(),
            y: ArithmaExpression::zero(),
            z: ArithmaExpression::zero(),
        }
    }

    /// Dot product `self · other`.
    ///
    /// Built symbolically as `x·x' + y·y' + z·z'`; no numeric collapse is
    /// attempted, so the result stays exact for symbolic components.
    pub fn dot(&self, other: &Self) -> ArithmaExpression {
        let xx = ArithmaExpression::mul(self.x.clone(), other.x.clone());
        let yy = ArithmaExpression::mul(self.y.clone(), other.y.clone());
        let zz = ArithmaExpression::mul(self.z.clone(), other.z.clone());
        ArithmaExpression::add(ArithmaExpression::add(xx, yy), zz)
    }

    /// Cross product `self × other`.
    ///
    /// Right-handed convention: `x̂ × ŷ = ẑ`. Each component is built as a
    /// symbolic difference of products.
    pub fn cross(&self, other: &Self) -> Self {
        let x = ArithmaExpression::sub(
            ArithmaExpression::mul(self.y.clone(), other.z.clone()),
            ArithmaExpression::mul(self.z.clone(), other.y.clone()),
        );
        let y = ArithmaExpression::sub(
            ArithmaExpression::mul(self.z.clone(), other.x.clone()),
            ArithmaExpression::mul(self.x.clone(), other.z.clone()),
        );
        let z = ArithmaExpression::sub(
            ArithmaExpression::mul(self.x.clone(), other.y.clone()),
            ArithmaExpression::mul(self.y.clone(), other.x.clone()),
        );
        Self { x, y, z }
    }

    /// Component-wise sum `self + other`.
    pub(crate) fn add_vec(&self, other: &Self) -> Self {
        Self {
            x: ArithmaExpression::add(self.x.clone(), other.x.clone()),
            y: ArithmaExpression::add(self.y.clone(), other.y.clone()),
            z: ArithmaExpression::add(self.z.clone(), other.z.clone()),
        }
    }

    /// Component-wise difference `self - other`.
    pub(crate) fn sub_vec(&self, other: &Self) -> Self {
        Self {
            x: ArithmaExpression::sub(self.x.clone(), other.x.clone()),
            y: ArithmaExpression::sub(self.y.clone(), other.y.clone()),
            z: ArithmaExpression::sub(self.z.clone(), other.z.clone()),
        }
    }

    /// Scale every component by the scalar expression `s`.
    pub(crate) fn scale(&self, s: &ArithmaExpression) -> Self {
        Self {
            x: ArithmaExpression::mul(self.x.clone(), s.clone()),
            y: ArithmaExpression::mul(self.y.clone(), s.clone()),
            z: ArithmaExpression::mul(self.z.clone(), s.clone()),
        }
    }

    /// Squared magnitude `self · self`.
    pub fn magnitude_squared(&self) -> ArithmaExpression {
        self.dot(self)
    }
}

// ---------------------------------------------------------------------------
// Backward-compatibility aliases for the pre-rename `Arithmos*` names.
// Retained for one release; downstream (eml-math, eml-spectral, metaphysica,
// periodica) should migrate to the `Arithma*` names above.
// ---------------------------------------------------------------------------
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaVector`")]
#[allow(unused)]
pub use self::ArithmaVector as ArithmosVector;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::{ArithmaBindings, Evaluable};

    /// Build a vector from three integers.
    fn v(x: i64, y: i64, z: i64) -> ArithmaVector {
        ArithmaVector::new(
            ArithmaExpression::from_i64(x),
            ArithmaExpression::from_i64(y),
            ArithmaExpression::from_i64(z),
        )
    }

    /// Evaluate an expression against an empty binding map.
    fn num(e: &ArithmaExpression) -> f64 {
        e.evaluate(&ArithmaBindings::new())
            .expect("expression should evaluate numerically")
    }

    #[test]
    fn zero_vector_has_zero_components() {
        let z = ArithmaVector::zero();
        assert!(matches!(z.x, ArithmaExpression::Number(_)));
        assert!(matches!(z.y, ArithmaExpression::Number(_)));
        assert!(matches!(z.z, ArithmaExpression::Number(_)));
    }

    #[test]
    fn dot_of_123_and_456_is_32() {
        let d = v(1, 2, 3).dot(&v(4, 5, 6));
        assert!((num(&d) - 32.0).abs() < 1e-12, "got {}", num(&d));
    }

    #[test]
    fn dot_of_orthogonal_axes_is_zero() {
        let d = v(1, 0, 0).dot(&v(0, 1, 0));
        assert!(num(&d).abs() < 1e-12);
    }

    #[test]
    fn dot_is_symmetric_and_handles_negatives() {
        let a = v(2, -3, 4);
        let b = v(-1, 5, 6);
        // 2*-1 + -3*5 + 4*6 = -2 - 15 + 24 = 7
        assert!((num(&a.dot(&b)) - 7.0).abs() < 1e-12);
        assert!((num(&b.dot(&a)) - 7.0).abs() < 1e-12);
    }

    #[test]
    fn cross_x_and_y_is_z() {
        let c = v(1, 0, 0).cross(&v(0, 1, 0));
        assert!(num(&c.x).abs() < 1e-12);
        assert!(num(&c.y).abs() < 1e-12);
        assert!((num(&c.z) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cross_is_anticommutative() {
        let c = v(0, 1, 0).cross(&v(1, 0, 0));
        assert!((num(&c.z) + 1.0).abs() < 1e-12);
    }

    #[test]
    fn cross_of_general_vectors() {
        // [1,2,3] x [4,5,6] = [-3, 6, -3]
        let c = v(1, 2, 3).cross(&v(4, 5, 6));
        assert!((num(&c.x) + 3.0).abs() < 1e-12);
        assert!((num(&c.y) - 6.0).abs() < 1e-12);
        assert!((num(&c.z) + 3.0).abs() < 1e-12);
    }

    #[test]
    fn cross_of_parallel_vectors_is_zero() {
        let c = v(1, 2, 3).cross(&v(2, 4, 6));
        assert!(num(&c.x).abs() < 1e-12);
        assert!(num(&c.y).abs() < 1e-12);
        assert!(num(&c.z).abs() < 1e-12);
    }

    #[test]
    fn cross_result_is_orthogonal_to_both_inputs() {
        let a = v(1, 2, 3);
        let b = v(4, 5, 6);
        let c = a.cross(&b);
        assert!(num(&c.dot(&a)).abs() < 1e-9);
        assert!(num(&c.dot(&b)).abs() < 1e-9);
    }

    #[test]
    fn magnitude_squared_of_345_is_25() {
        // magnitude_squared() routes through dot(), so this also proves the
        // pre-existing panic is gone.
        let m = v(3, 4, 0).magnitude_squared();
        assert!((num(&m) - 25.0).abs() < 1e-12);
    }

    #[test]
    fn dot_stays_symbolic_with_free_variables() {
        let a = ArithmaVector::new(
            ArithmaExpression::var("a"),
            ArithmaExpression::zero(),
            ArithmaExpression::zero(),
        );
        let b = v(3, 0, 0);
        let d = a.dot(&b);
        let mut bindings = ArithmaBindings::new();
        bindings.insert("a".to_string(), 7.0);
        let got = d.evaluate(&bindings).expect("binds cleanly");
        assert!((got - 21.0).abs() < 1e-12, "got {got}");
        // Unbound, it must refuse rather than silently produce a number.
        assert!(d.evaluate(&ArithmaBindings::new()).is_err());
    }

    #[test]
    fn scale_and_component_ops() {
        let a = v(1, 2, 3);
        let b = v(4, 5, 6);
        let sum = a.add_vec(&b);
        assert!((num(&sum.x) - 5.0).abs() < 1e-12);
        assert!((num(&sum.z) - 9.0).abs() < 1e-12);
        let diff = b.sub_vec(&a);
        assert!((num(&diff.y) - 3.0).abs() < 1e-12);
        let scaled = a.scale(&ArithmaExpression::from_i64(2));
        assert!((num(&scaled.z) - 6.0).abs() < 1e-12);
    }
}
