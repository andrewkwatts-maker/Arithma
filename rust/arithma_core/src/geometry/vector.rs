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

    /// Dot product `self · other`. Wave-2 stub.
    pub fn dot(&self, _other: &Self) -> ArithmaExpression {
        unimplemented!("ArithmaVector::dot — populated in Wave 3")
    }

    /// Cross product `self × other`. Wave-2 stub.
    pub fn cross(&self, _other: &Self) -> Self {
        unimplemented!("ArithmaVector::cross — populated in Wave 3")
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

    #[test]
    fn zero_vector_has_zero_components() {
        let v = ArithmaVector::zero();
        assert!(matches!(v.x, ArithmaExpression::Number(_)));
        assert!(matches!(v.y, ArithmaExpression::Number(_)));
        assert!(matches!(v.z, ArithmaExpression::Number(_)));
    }
}
