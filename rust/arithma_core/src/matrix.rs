//====== Arithma/rust/arithma_core/src/matrix.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! Symbolic matrix algebra. Wave-2 placeholder; full impl migrates from
//! pt-arithmos `pt_matrix.rs` in Wave 3.

use crate::expression::ArithmaExpression;

/// Symbolic matrix of `ArithmaExpression` cells.
#[derive(Debug, Clone)]
pub struct ArithmaMatrix {
    pub rows: usize,
    pub cols: usize,
    pub cells: Vec<ArithmaExpression>,
}

impl ArithmaMatrix {
    /// Construct a matrix from a flat row-major cell list. Panics if the
    /// length does not match `rows * cols`.
    pub fn new(rows: usize, cols: usize, cells: Vec<ArithmaExpression>) -> Self {
        assert_eq!(
            cells.len(),
            rows * cols,
            "cell count must equal rows * cols"
        );
        Self { rows, cols, cells }
    }

    /// Total cell count.
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    /// Whether the matrix is empty.
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Backward-compatibility aliases for the pre-rename `Arithmos*` names.
// Retained for one release; downstream (eml-math, eml-spectral, metaphysica,
// periodica) should migrate to the `Arithma*` names above.
// ---------------------------------------------------------------------------
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaMatrix`")]
#[allow(unused)]
pub use self::ArithmaMatrix as ArithmosMatrix;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_with_correct_count_works() {
        let m = ArithmaMatrix::new(0, 0, vec![]);
        assert!(m.is_empty());
    }

    #[test]
    #[should_panic]
    fn new_with_wrong_count_panics() {
        let _ = ArithmaMatrix::new(2, 2, vec![]);
    }
}
