//====== Arithma/rust/arithma_core/src/geometry/mod.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Geometry
//!
//! 3D vector / line / plane / sphere primitives plus intersection routines.
//! Coordinates are stored as `ArithmaExpression` so geometry can carry
//! symbolic data through the simplifier and Fourier-bake pipeline.
//!
//! ## Submodules
//!
//! - [`vector`] — `ArithmaVector` 3-vector.
//! - [`line`] — `ArithmaLine` parametric line.
//! - [`plane`] — `ArithmaPlane` infinite plane.
//! - [`sphere`] — `ArithmaSphere` sphere.
//! - [`intersection`] — closed-form intersection routines.

pub mod intersection;
pub mod line;
pub mod plane;
pub mod sphere;
pub mod vector;

pub use intersection::{ArithmaIntersection, ArithmaIntersectionResult};
pub use line::ArithmaLine;
pub use plane::ArithmaPlane;
pub use sphere::ArithmaSphere;
pub use vector::ArithmaVector;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_re_exports_resolve() {
        let _: Option<ArithmaVector> = None;
        let _: Option<ArithmaLine> = None;
        let _: Option<ArithmaPlane> = None;
        let _: Option<ArithmaSphere> = None;
    }
}
