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
//! Parametric line `P(t) = origin + t Â· direction`. Both vectors are symbolic.

use crate::expression::ArithmosExpression;
use crate::geometry::vector::ArithmosVector;

/// Parametric line: `P(t) = origin + t * direction`.
#[derive(Debug, Clone)]
pub struct ArithmosLine {
    /// Origin point.
    pub origin: ArithmosVector,
    /// Direction vector. Need not be unit-length â€” the parametric form handles
    /// scaling.
    pub direction: ArithmosVector,
}

impl ArithmosLine {
    /// Construct a line from origin and direction.
    pub fn new(origin: ArithmosVector, direction: ArithmosVector) -> Self {
        Self { origin, direction }
    }

    /// Evaluate the line at parameter `t`. Wave-2 stub.
    pub fn at(&self, _t: ArithmosExpression) -> ArithmosVector {
        unimplemented!("ArithmosLine::at â€” populated in Wave 3")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_line_round_trip() {
        let line = ArithmosLine::new(ArithmosVector::zero(), ArithmosVector::zero());
        // Just exercise construction.
        let _ = line.origin;
    }
}

