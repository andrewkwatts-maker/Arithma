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
//! `ArithmaFunction` — the operator catalogue carried inside
//! `ArithmaExpression::Function(op, args)`. Variants mirror `PTFunction` so the
//! Wave-3 migration is mechanical.
//!
//! Variants are grouped:
//!
//! - **Basic arithmetic** — Add, Subtract, Multiply, Divide, Power, Negate.
//! - **Trigonometric** — Sin, Cos, Tan, Cot, Sec, Csc.
//! - **Inverse trigonometric** — Asin, Acos, Atan, Atan2.
//! - **Hyperbolic** — Sinh, Cosh, Tanh, Asinh, Acosh, Atanh.
//! - **Exponential / logarithmic** — Exp, Ln, Log, Log10, Log2, LogBase, Pow.
//! - **Roots** — Sqrt, Cbrt, Root.
//! - **Special functions** — Gamma, Beta, Erf, Factorial.
//! - **Rounding** — Abs, Sign, Floor, Ceil, Round.
//! - **Complex** — Real, Imag, Conjugate, Arg.
//! - **Calculus operators** — Derivative, PartialDerivative, Integral,
//!   DefiniteIntegral, plus vector-calculus Laplacian/Gradient/Divergence/Curl.
//! - **Numerical methods** — FindRoots, NewtonRaphson, FindCriticalPoints,
//!   Optimize.
//! - **Limit / series** — Limit, Summation, Product.
//! - **Statistical** — Median/Mode/Mean/Sum/Variance/StandardDeviation/Min/Max/
//!   Range/Quartiles/InterquartileRange/Percentile/Z-score/CorrelationCoefficient/
//!   LinearRegression.
//! - **Geometry** — Area, Volume, Perimeter, SurfaceArea.

use serde::{Deserialize, Serialize};

use crate::expression::ArithmaExpression;
use crate::integer::ArithmaInteger;

/// Direction marker for one-sided limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArithmaLimitDirection {
    /// Approach from the left (x → a⁻).
    Left,
    /// Approach from the right (x → a⁺).
    Right,
    /// Two-sided.
    Both,
}

/// The operator catalogue.
///
/// Note: `PartialEq`/`Eq`/`Hash` are intentionally omitted because variants
/// carry `Box<ArithmaExpression>` and `ArithmaInteger`, which themselves
/// transitively contain `f64` fields and cannot satisfy `Eq`/`Hash` without
/// a hand-rolled impl that defines a canonical comparison for floats. That
/// impl will land alongside the equation-ID hashing work in pt-phantasia
/// (see plan §C "Equation-ID texture mechanism"). For now, comparing two
/// `ArithmaFunction`s structurally is the consumer's responsibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArithmaFunction {
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
    LogBase(ArithmaInteger),
    Pow(ArithmaInteger),

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
        lower_bound: Box<ArithmaExpression>,
        upper_bound: Box<ArithmaExpression>,
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
        lower_bound: Box<ArithmaExpression>,
        upper_bound: Box<ArithmaExpression>,
    },
    NewtonRaphson {
        var: String,
        initial_guess: Box<ArithmaExpression>,
    },
    FindCriticalPoints {
        var: String,
        lower_bound: Box<ArithmaExpression>,
        upper_bound: Box<ArithmaExpression>,
    },
    Optimize {
        var: String,
        lower_bound: Box<ArithmaExpression>,
        upper_bound: Box<ArithmaExpression>,
        maximize: bool,
    },

    // Limit / series
    Limit {
        var: String,
        approach: Box<ArithmaExpression>,
        direction: ArithmaLimitDirection,
    },
    Summation {
        var: String,
        lower_bound: Box<ArithmaExpression>,
        upper_bound: Box<ArithmaExpression>,
    },
    Product {
        var: String,
        lower_bound: Box<ArithmaExpression>,
        upper_bound: Box<ArithmaExpression>,
    },

    // Statistical
    Median,
    Mode,
    Quartiles,
    InterquartileRange,
    Percentile {
        percentile: ArithmaInteger,
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

impl ArithmaFunction {
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

            // Default unary for everything else (calculus, transcendentals, …)
            _ => 1,
        }
    }

    /// Best-effort exact evaluation when all arguments are themselves constants.
    /// Returns `None` if no exact form is known. Stub for Wave 2.
    pub fn evaluate_exact(&self, _args: &[ArithmaExpression]) -> Option<ArithmaExpression> {
        None
    }
}

// ---------------------------------------------------------------------------
// Backward-compatibility aliases for the pre-rename `Arithmos*` names.
// Retained for one release; downstream (eml-math, eml-spectral, metaphysica,
// periodica) should migrate to the `Arithma*` names above.
// ---------------------------------------------------------------------------
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaFunction`")]
#[allow(unused)]
pub use self::ArithmaFunction as ArithmosFunction;
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaLimitDirection`")]
#[allow(unused)]
pub use self::ArithmaLimitDirection as ArithmosLimitDirection;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_arithmetic_is_arity_two() {
        assert_eq!(ArithmaFunction::Add.arity(), 2);
        assert_eq!(ArithmaFunction::Multiply.arity(), 2);
        assert_eq!(ArithmaFunction::Power.arity(), 2);
    }

    #[test]
    fn unary_transcendentals_are_arity_one() {
        assert_eq!(ArithmaFunction::Sin.arity(), 1);
        assert_eq!(ArithmaFunction::Exp.arity(), 1);
        assert_eq!(ArithmaFunction::Sqrt.arity(), 1);
    }

    #[test]
    fn standard_score_is_arity_three() {
        assert_eq!(ArithmaFunction::StandardScore.arity(), 3);
    }
}
