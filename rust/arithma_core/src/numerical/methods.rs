//====== Arithma/rust/arithma_core/src/numerical/methods.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Numerical methods dispatcher
//!
//! Generic entry point for choosing a numerical solver. Routes to the
//! specialised implementations in [`super::root_finding`].

use crate::expression::ArithmosExpression;

/// Choice of numerical method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArithmosNumericalMethod {
    Bisection,
    NewtonRaphson,
    Secant,
    Brent,
}

/// Solve `expr = 0` for `var` using the specified method. Wave-2 stub.
pub fn solve_with_method(
    _expr: &ArithmosExpression,
    _var: &str,
    _method: ArithmosNumericalMethod,
    _initial: f64,
) -> Result<f64, String> {
    unimplemented!("solve_with_method â€” populated in Wave 3")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn methods_are_distinct() {
        assert_ne!(
            ArithmosNumericalMethod::Bisection,
            ArithmosNumericalMethod::NewtonRaphson
        );
    }
}

