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

use crate::expression::ArithmosExpression;
use crate::geometry::line::ArithmosLine;
use crate::geometry::plane::ArithmosPlane;
use crate::geometry::sphere::ArithmosSphere;
use crate::geometry::vector::ArithmosVector;

/// Outcome of an intersection test. Variants cover both empty and one-or-more-
/// hit cases.
#[derive(Debug, Clone)]
pub enum ArithmosIntersectionResult {
    /// No intersection.
    None,
    /// Single hit at the given point.
    Point(ArithmosVector),
    /// Two hits â€” typical for ray/sphere.
    TwoPoints(ArithmosVector, ArithmosVector),
    /// Continuous overlap (e.g. line lies in plane).
    Continuous,
}

/// Static collection of intersection routines.
pub struct ArithmosIntersection;

impl ArithmosIntersection {
    /// Line vs plane intersection. Wave-2 stub.
    pub fn line_plane(_line: &ArithmosLine, _plane: &ArithmosPlane) -> ArithmosIntersectionResult {
        unimplemented!("ArithmosIntersection::line_plane â€” populated in Wave 3")
    }

    /// Line vs sphere intersection. Wave-2 stub.
    pub fn line_sphere(_line: &ArithmosLine, _sphere: &ArithmosSphere) -> ArithmosIntersectionResult {
        unimplemented!("ArithmosIntersection::line_sphere â€” populated in Wave 3")
    }

    /// Plane vs plane intersection (returns a line). Wave-2 stub.
    pub fn plane_plane(_a: &ArithmosPlane, _b: &ArithmosPlane) -> Option<ArithmosLine> {
        unimplemented!("ArithmosIntersection::plane_plane â€” populated in Wave 3")
    }

    /// Closest-point parameter `t` on a line for a target point.
    pub fn closest_point_param(_line: &ArithmosLine, _point: &ArithmosVector) -> ArithmosExpression {
        unimplemented!("ArithmosIntersection::closest_point_param â€” populated in Wave 3")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intersection_none_variant_constructs() {
        let r = ArithmosIntersectionResult::None;
        assert!(matches!(r, ArithmosIntersectionResult::None));
    }
}

