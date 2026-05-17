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
//! Coordinates are stored as `ArithmosExpression` so geometry can carry
//! symbolic data through the simplifier and Fourier-bake pipeline.
//!
//! ## Submodules
//!
//! - [`vector`] â€” `ArithmosVector` 3-vector.
//! - [`line`] â€” `ArithmosLine` parametric line.
//! - [`plane`] â€” `ArithmosPlane` infinite plane.
//! - [`sphere`] â€” `ArithmosSphere` sphere.
//! - [`intersection`] â€” closed-form intersection routines.

pub mod intersection;
pub mod line;
pub mod plane;
pub mod sphere;
pub mod vector;

pub use intersection::{ArithmosIntersection, ArithmosIntersectionResult};
pub use line::ArithmosLine;
pub use plane::ArithmosPlane;
pub use sphere::ArithmosSphere;
pub use vector::ArithmosVector;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_re_exports_resolve() {
        let _: Option<ArithmosVector> = None;
        let _: Option<ArithmosLine> = None;
        let _: Option<ArithmosPlane> = None;
        let _: Option<ArithmosSphere> = None;
    }
}

