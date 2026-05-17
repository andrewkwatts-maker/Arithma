//====== Arithma/rust/arithma_core/src/geometry/sphere.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Sphere
//!
//! Sphere defined by symbolic centre and radius.

use crate::expression::ArithmosExpression;
use crate::geometry::vector::ArithmosVector;

/// Sphere: `|p - centre| = radius`.
#[derive(Debug, Clone)]
pub struct ArithmosSphere {
    pub centre: ArithmosVector,
    pub radius: ArithmosExpression,
}

impl ArithmosSphere {
    /// Construct a sphere.
    pub fn new(centre: ArithmosVector, radius: ArithmosExpression) -> Self {
        Self { centre, radius }
    }

    /// Returns the squared radius `radius * radius`.
    pub fn radius_squared(&self) -> ArithmosExpression {
        ArithmosExpression::mul(self.radius.clone(), self.radius.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_sphere_construction() {
        let s = ArithmosSphere::new(ArithmosVector::zero(), ArithmosExpression::zero());
        let _ = s.centre;
    }
}

