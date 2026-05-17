//====== Arithma/rust/arithma_core/src/function.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Function
//!
//! `ArithmosFunction` â€” the operator catalogue carried inside
//! `ArithmosExpression::Function(op, args)`. Variants mirror `PTFunction` so the
//! Wave-3 migration is mechanical.
//!
//! Variants are grouped:
//!
//! - **Basic arithmetic** â€” Add, Subtract, Multiply, Divide, Power, Negate.
//! - **Trigonometric** â€” Sin, Cos, Tan, Cot, Sec, Csc.
//! - **Inverse trigonometric** â€” Asin, Acos, Atan, Atan2.
//! - **Hyperbolic** â€” Sinh, Cosh, Tanh, Asinh, Acosh, Atanh.
//! - **Exponential / logarithmic** â€” Exp, Ln, Log, Log10, Log2, LogBase, Pow.
//! - **Roots** â€” Sqrt, Cbrt, Root.
//! - **Special functions** â€” Gamma, Beta, Erf, Factorial.
//! - **Rounding** â€” Abs, Sign, Floor, Ceil, Round.
//! - **Complex** â€” Real, Imag, Conjugate, Arg.
//! - **Calculus operators** â€” Derivative, PartialDerivative, Integral,
//!   DefiniteIntegral, plus vector-calculus Laplacian/Gradient/Divergence/Curl.
//! - **Numerical methods** â€” FindRoots, NewtonRaphson, FindCriticalPoints,
//!   Optimize.
//! - **Limit / series** â€” Limit, Summation, Product.
//! - **Statistical** â€” Median/Mode/Mean/Sum/Variance/StandardDeviation/Min/Max/
//!   Range/Quartiles/InterquartileRange/Percentile/Z-score/CorrelationCoefficient/
//!   LinearRegression.
//! - **Geometry** â€” Area, Volume, Perimeter, SurfaceArea.

use serde::{Deserialize, Serialize};

use crate::expression::ArithmosExpression;
use crate::integer::ArithmosInteger;

/// Direction marker for one-sided limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArithmosLimitDirection {
    /// Approach from the left (x â†’ aâ»).
    Left,
    /// Approach from the right (x â†’ aâº).
    Right,
    /// Two-sided.
    Both,
}

/// The operator catalogue.
///
/// Note: `PartialEq`/`Eq`/`Hash` are intentionally omitted because variants
/// carry `Box<ArithmosExpression>` and `ArithmosInteger`, which themselves
/// transitively contain `f64` fields and cannot satisfy `Eq`/`Hash` without
/// a hand-rolled impl that defines a canonical comparison for floats. That
/// impl will land alongside the equation-ID hashing work in pt-phantasia
/// (see plan Â§C "Equation-ID texture mechanism"). For now, comparing two
/// `ArithmosFunction`s structurally is the consumer's responsibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArithmosFunction {
    // Basic arithmetic
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    Negate,

    // Trigonometric
    Sin,
    Cos,
    Tan,
    Cot,
    Sec,
    Csc,

    // Inverse trigonometric
    Asin,
    Acos,
    Atan,
    Atan2,

    // Hyperbolic
    Sinh,
    Cosh,
    Tanh,
    Asinh,
    Acosh,
    Atanh,

    // Exponential / logarithmic
    Exp,
    Ln,
    Log,
    Log10,
    Log2,
    LogBase(ArithmosInteger),
    Pow(ArithmosInteger),

    // Roots
    Sqrt,
    Cbrt,
    Root(u64),

    // Special functions
    Gamma,
    Beta,
    Erf,
    Factorial,

    // Rounding / abs
    Abs,
    Sign,
    Floor,
    Ceil,
    Round,

    // Complex
    Real,
    Imag,
    Conjugate,
    Arg,

    // Calculus operators
    Derivative {
        var: String,
    },
    PartialDerivative {
        var: String,
    },
    Integral {
        var: String,
    },
    DefiniteIntegral {
        var: String,
        lower_bound: Box<ArithmosExpression>,
        upper_bound: Box<ArithmosExpression>,
    },

    // Vector calculus
    LaplacianOperator {
        vars: Vec<String>,
    },
    GradientOperator {
        vars: Vec<String>,
    },
    DivergenceOperator {
        vars: Vec<String>,
    },
    CurlOperator {
        vars: Vec<String>,
    },

    // Numerical methods
    FindRoots {
        var: String,
        lower_bound: Box<ArithmosExpression>,
        upper_bound: Box<ArithmosExpression>,
    },
    NewtonRaphson {
        var: String,
        initial_guess: Box<ArithmosExpression>,
    },
    FindCriticalPoints {
        var: String,
        lower_bound: Box<ArithmosExpression>,
        upper_bound: Box<ArithmosExpression>,
    },
    Optimize {
        var: String,
        lower_bound: Box<ArithmosExpression>,
        upper_bound: Box<ArithmosExpression>,
        maximize: bool,
    },

    // Limit / series
    Limit {
        var: String,
        approach: Box<ArithmosExpression>,
        direction: ArithmosLimitDirection,
    },
    Summation {
        var: String,
        lower_bound: Box<ArithmosExpression>,
        upper_bound: Box<ArithmosExpression>,
    },
    Product {
        var: String,
        lower_bound: Box<ArithmosExpression>,
        upper_bound: Box<ArithmosExpression>,
    },

    // Statistical
    Median,
    Mode,
    Quartiles,
    InterquartileRange,
    Percentile {
        percentile: ArithmosInteger,
    },
    Mean,
    Sum,
    Variance,
    StandardDeviation,
    Min,
    Max,
    Range,
    CorrelationCoefficient,
    LinearRegression,
    StandardScore,

    // Geometry
    Area,
    Volume,
    Perimeter,
    SurfaceArea,
}

impl ArithmosFunction {
    /// Return the expected number of arguments. Wave-2 stub knows the obvious
    /// cases; the real arity table lands in Wave 3.
    pub fn arity(&self) -> usize {
        match self {
            // Binary
            Self::Add
            | Self::Subtract
            | Self::Multiply
            | Self::Divide
            | Self::Power
            | Self::Beta
            | Self::Atan2
            | Self::CorrelationCoefficient
            | Self::LinearRegression => 2,

            // Ternary
            Self::StandardScore => 3,

            // Vector calculus depends on `vars.len()`
            Self::DivergenceOperator { vars } | Self::CurlOperator { vars } => vars.len(),

            // Default unary for everything else (calculus, transcendentals, â€¦)
            _ => 1,
        }
    }

    /// Best-effort exact evaluation when all arguments are themselves constants.
    /// Returns `None` if no exact form is known. Stub for Wave 2.
    pub fn evaluate_exact(&self, _args: &[ArithmosExpression]) -> Option<ArithmosExpression> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_arithmetic_is_arity_two() {
        assert_eq!(ArithmosFunction::Add.arity(), 2);
        assert_eq!(ArithmosFunction::Multiply.arity(), 2);
        assert_eq!(ArithmosFunction::Power.arity(), 2);
    }

    #[test]
    fn unary_transcendentals_are_arity_one() {
        assert_eq!(ArithmosFunction::Sin.arity(), 1);
        assert_eq!(ArithmosFunction::Exp.arity(), 1);
        assert_eq!(ArithmosFunction::Sqrt.arity(), 1);
    }

    #[test]
    fn standard_score_is_arity_three() {
        assert_eq!(ArithmosFunction::StandardScore.arity(), 3);
    }
}

