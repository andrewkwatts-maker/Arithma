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

    /// Signed distance from `point` to this plane. Wave-2 stub.
    pub fn signed_distance(&self, _point: &ArithmaVector) -> ArithmaExpression {
        unimplemented!("ArithmaPlane::signed_distance — populated in Wave 3")
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

    #[test]
    fn construct_plane() {
        let p = ArithmaPlane::new(ArithmaVector::zero(), ArithmaExpression::zero());
        let _ = p.normal;
    }
}
