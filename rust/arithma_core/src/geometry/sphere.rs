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

use crate::expression::ArithmaExpression;
use crate::geometry::vector::ArithmaVector;

/// Sphere: `|p - centre| = radius`.
#[derive(Debug, Clone)]
pub struct ArithmaSphere {
    pub centre: ArithmaVector,
    pub radius: ArithmaExpression,
}

impl ArithmaSphere {
    /// Construct a sphere.
    pub fn new(centre: ArithmaVector, radius: ArithmaExpression) -> Self {
        Self { centre, radius }
    }

    /// Returns the squared radius `radius * radius`.
    pub fn radius_squared(&self) -> ArithmaExpression {
        ArithmaExpression::mul(self.radius.clone(), self.radius.clone())
    }
}

// ---------------------------------------------------------------------------
// Backward-compatibility aliases for the pre-rename `Arithmos*` names.
// Retained for one release; downstream (eml-math, eml-spectral, metaphysica,
// periodica) should migrate to the `Arithma*` names above.
// ---------------------------------------------------------------------------
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaSphere`")]
#[allow(unused)]
pub use self::ArithmaSphere as ArithmosSphere;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_sphere_construction() {
        let s = ArithmaSphere::new(ArithmaVector::zero(), ArithmaExpression::zero());
        let _ = s.centre;
    }
}
