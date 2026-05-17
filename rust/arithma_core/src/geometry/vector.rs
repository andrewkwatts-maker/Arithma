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
//! `ArithmosVector` â€” a 3-vector whose components are `ArithmosExpression`
//! values, allowing coordinates to remain symbolic throughout the geometry
//! pipeline.

use crate::expression::ArithmosExpression;

/// A symbolic 3-vector. Components are full expressions so geometry calculations
/// stay symbolic until a numeric value is required.
#[derive(Debug, Clone)]
pub struct ArithmosVector {
    pub x: ArithmosExpression,
    pub y: ArithmosExpression,
    pub z: ArithmosExpression,
}

impl ArithmosVector {
    /// Construct from three components.
    pub fn new(x: ArithmosExpression, y: ArithmosExpression, z: ArithmosExpression) -> Self {
        Self { x, y, z }
    }

    /// The zero vector.
    pub fn zero() -> Self {
        Self {
            x: ArithmosExpression::zero(),
            y: ArithmosExpression::zero(),
            z: ArithmosExpression::zero(),
        }
    }

    /// Dot product `self Â· other`. Wave-2 stub.
    pub fn dot(&self, _other: &Self) -> ArithmosExpression {
        unimplemented!("ArithmosVector::dot â€” populated in Wave 3")
    }

    /// Cross product `self Ã— other`. Wave-2 stub.
    pub fn cross(&self, _other: &Self) -> Self {
        unimplemented!("ArithmosVector::cross â€” populated in Wave 3")
    }

    /// Squared magnitude `self Â· self`.
    pub fn magnitude_squared(&self) -> ArithmosExpression {
        self.dot(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_vector_has_zero_components() {
        let v = ArithmosVector::zero();
        assert!(matches!(v.x, ArithmosExpression::Number(_)));
        assert!(matches!(v.y, ArithmosExpression::Number(_)));
        assert!(matches!(v.z, ArithmosExpression::Number(_)));
    }
}

