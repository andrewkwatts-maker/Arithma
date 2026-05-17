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
//! Infinite plane defined by a normal vector and a scalar offset (`n Â· p = d`).

use crate::expression::ArithmosExpression;
use crate::geometry::vector::ArithmosVector;

/// Plane in normal-and-offset form: `normal Â· p = offset`.
#[derive(Debug, Clone)]
pub struct ArithmosPlane {
    /// Plane normal (need not be unit-length).
    pub normal: ArithmosVector,
    /// Scalar offset along the normal.
    pub offset: ArithmosExpression,
}

impl ArithmosPlane {
    /// Construct a plane from normal and offset.
    pub fn new(normal: ArithmosVector, offset: ArithmosExpression) -> Self {
        Self { normal, offset }
    }

    /// Signed distance from `point` to this plane. Wave-2 stub.
    pub fn signed_distance(&self, _point: &ArithmosVector) -> ArithmosExpression {
        unimplemented!("ArithmosPlane::signed_distance â€” populated in Wave 3")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_plane() {
        let p = ArithmosPlane::new(ArithmosVector::zero(), ArithmosExpression::zero());
        let _ = p.normal;
    }
}

