//====== Arithma/rust/arithma_core/src/tensor.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! N-dimensional symbolic tensor. Wave-2 placeholder.

use crate::expression::ArithmosExpression;

/// A general N-dimensional tensor of `ArithmosExpression` cells.
#[derive(Debug, Clone)]
pub struct ArithmosTensor {
    pub shape: Vec<usize>,
    pub cells: Vec<ArithmosExpression>,
}

impl ArithmosTensor {
    /// Total cell count = âˆ(shape).
    pub fn cell_count(shape: &[usize]) -> usize {
        shape.iter().product()
    }

    /// Construct a tensor; panics if `cells.len()` â‰  âˆ(shape).
    pub fn new(shape: Vec<usize>, cells: Vec<ArithmosExpression>) -> Self {
        assert_eq!(cells.len(), Self::cell_count(&shape), "cell count must equal âˆ(shape)");
        Self { shape, cells }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_count_zero_dimensional_is_one() {
        assert_eq!(ArithmosTensor::cell_count(&[]), 1);
    }

    #[test]
    fn cell_count_three_d() {
        assert_eq!(ArithmosTensor::cell_count(&[2, 3, 4]), 24);
    }
}

