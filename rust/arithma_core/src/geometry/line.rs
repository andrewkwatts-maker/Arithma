//====== Arithma/rust/arithma_core/src/geometry/line.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Line
//!
//! Parametric line `P(t) = origin + t · direction`. Both vectors are symbolic.

use crate::expression::ArithmaExpression;
use crate::geometry::vector::ArithmaVector;

/// Parametric line: `P(t) = origin + t * direction`.
#[derive(Debug, Clone)]
pub struct ArithmaLine {
    /// Origin point.
    pub origin: ArithmaVector,
    /// Direction vector. Need not be unit-length — the parametric form handles
    /// scaling.
    pub direction: ArithmaVector,
}

impl ArithmaLine {
    /// Construct a line from origin and direction.
    pub fn new(origin: ArithmaVector, direction: ArithmaVector) -> Self {
        Self { origin, direction }
    }

    /// Evaluate the line at parameter `t`. Wave-2 stub.
    pub fn at(&self, _t: ArithmaExpression) -> ArithmaVector {
        unimplemented!("ArithmaLine::at — populated in Wave 3")
    }
}

// ---------------------------------------------------------------------------
// Backward-compatibility aliases for the pre-rename `Arithmos*` names.
// Retained for one release; downstream (eml-math, eml-spectral, metaphysica,
// periodica) should migrate to the `Arithma*` names above.
// ---------------------------------------------------------------------------
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaLine`")]
#[allow(unused)]
pub use self::ArithmaLine as ArithmosLine;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_line_round_trip() {
        let line = ArithmaLine::new(ArithmaVector::zero(), ArithmaVector::zero());
        // Just exercise construction.
        let _ = line.origin;
    }
}
