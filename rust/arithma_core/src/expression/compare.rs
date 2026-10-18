//====== Arithma/rust/arithma_core/src/expression/compare.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Comparison builders
//!
//! Arithma has no dedicated comparison operator, so `>=` and `<` are built as
//! Sign/Abs-based Heaviside steps over existing expressions and stay symbolic
//! until evaluation.

use super::ArithmaExpression;
use crate::function::ArithmaFunction;

/// `1` when `x` evaluates to a non-negative number (exact zero included), `0` otherwise.
pub fn step_ge_zero(x: ArithmaExpression) -> ArithmaExpression {
    let sign = ArithmaExpression::func(ArithmaFunction::Sign, vec![x]);
    let abs_sign = ArithmaExpression::func(ArithmaFunction::Abs, vec![sign.clone()]);
    ArithmaExpression::div(
        ArithmaExpression::add(sign, abs_sign),
        ArithmaExpression::from_i64(2),
    )
}

/// `1` when `a >= b`, `0` otherwise.
pub fn ge(a: ArithmaExpression, b: ArithmaExpression) -> ArithmaExpression {
    step_ge_zero(ArithmaExpression::sub(a, b))
}

/// `1` when `a < b`, `0` otherwise.
pub fn lt(a: ArithmaExpression, b: ArithmaExpression) -> ArithmaExpression {
    ArithmaExpression::sub(ArithmaExpression::from_i64(1), ge(a, b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::{ArithmaBindings, Evaluable};

    fn eval(expr: ArithmaExpression) -> f64 {
        expr.evaluate(&ArithmaBindings::new()).unwrap()
    }

    #[test]
    fn step_ge_zero_negative_is_zero() {
        assert_eq!(eval(step_ge_zero(ArithmaExpression::from_i64(-3))), 0.0);
    }

    #[test]
    fn step_ge_zero_zero_is_one() {
        assert_eq!(eval(step_ge_zero(ArithmaExpression::from_i64(0))), 1.0);
    }

    #[test]
    fn step_ge_zero_exact_rational_zero_is_one() {
        let rational_zero = ArithmaExpression::div(
            ArithmaExpression::from_i64(0),
            ArithmaExpression::from_i64(3),
        );
        assert_eq!(eval(step_ge_zero(rational_zero)), 1.0);
    }

    #[test]
    fn step_ge_zero_positive_is_one() {
        assert_eq!(eval(step_ge_zero(ArithmaExpression::from_i64(5))), 1.0);
    }

    #[test]
    fn ge_compares_two_expressions() {
        assert_eq!(
            eval(ge(ArithmaExpression::from_i64(2), ArithmaExpression::from_i64(2))),
            1.0
        );
        assert_eq!(
            eval(ge(ArithmaExpression::from_i64(1), ArithmaExpression::from_i64(2))),
            0.0
        );
        assert_eq!(
            eval(ge(ArithmaExpression::from_i64(3), ArithmaExpression::from_i64(2))),
            1.0
        );
    }

    #[test]
    fn lt_compares_two_expressions() {
        assert_eq!(
            eval(lt(ArithmaExpression::from_i64(1), ArithmaExpression::from_i64(2))),
            1.0
        );
        assert_eq!(
            eval(lt(ArithmaExpression::from_i64(2), ArithmaExpression::from_i64(2))),
            0.0
        );
        assert_eq!(
            eval(lt(ArithmaExpression::from_i64(3), ArithmaExpression::from_i64(2))),
            0.0
        );
    }
}
