//====== Arithma/rust/arithma_core/src/geometry/intersection.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Intersection
//!
//! Intersection routines between geometry primitives. All results carry
//! symbolic parameters so they survive through the simplifier and the Fourier
//! pipeline.

use crate::expression::ArithmaExpression;
use crate::geometry::line::ArithmaLine;
use crate::geometry::plane::ArithmaPlane;
use crate::geometry::sphere::ArithmaSphere;
use crate::geometry::vector::ArithmaVector;

/// Outcome of an intersection test. Variants cover both empty and one-or-more-
/// hit cases.
#[derive(Debug, Clone)]
pub enum ArithmaIntersectionResult {
    /// No intersection.
    None,
    /// Single hit at the given point.
    Point(ArithmaVector),
    /// Two hits — typical for ray/sphere.
    TwoPoints(ArithmaVector, ArithmaVector),
    /// Continuous overlap (e.g. line lies in plane).
    Continuous,
}

/// Static collection of intersection routines.
pub struct ArithmaIntersection;

impl ArithmaIntersection {
    /// Line vs plane intersection. Wave-2 stub.
    pub fn line_plane(_line: &ArithmaLine, _plane: &ArithmaPlane) -> ArithmaIntersectionResult {
        unimplemented!("ArithmaIntersection::line_plane — populated in Wave 3")
    }

    /// Line vs sphere intersection. Wave-2 stub.
    pub fn line_sphere(_line: &ArithmaLine, _sphere: &ArithmaSphere) -> ArithmaIntersectionResult {
        unimplemented!("ArithmaIntersection::line_sphere — populated in Wave 3")
    }

    /// Plane vs plane intersection (returns a line). Wave-2 stub.
    pub fn plane_plane(_a: &ArithmaPlane, _b: &ArithmaPlane) -> Option<ArithmaLine> {
        unimplemented!("ArithmaIntersection::plane_plane — populated in Wave 3")
    }

    /// Closest-point parameter `t` on a line for a target point.
    pub fn closest_point_param(_line: &ArithmaLine, _point: &ArithmaVector) -> ArithmaExpression {
        unimplemented!("ArithmaIntersection::closest_point_param — populated in Wave 3")
    }
}

// ---------------------------------------------------------------------------
// Backward-compatibility aliases for the pre-rename `Arithmos*` names.
// Retained for one release; downstream (eml-math, eml-spectral, metaphysica,
// periodica) should migrate to the `Arithma*` names above.
// ---------------------------------------------------------------------------
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaIntersection`")]
#[allow(unused)]
pub use self::ArithmaIntersection as ArithmosIntersection;
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaIntersectionResult`")]
#[allow(unused)]
pub use self::ArithmaIntersectionResult as ArithmosIntersectionResult;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intersection_none_variant_constructs() {
        let r = ArithmaIntersectionResult::None;
        assert!(matches!(r, ArithmaIntersectionResult::None));
    }
}
