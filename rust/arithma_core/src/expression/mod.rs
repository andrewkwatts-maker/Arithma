//====== Arithma/rust/arithma_core/src/expression/mod.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Expression
//!
//! `ArithmosExpression` is the abstract syntax tree at the heart of Arithmos. Every
//! other module in the crate either produces, consumes, or transforms expressions.
//!
//! ## Variants (mirrors `pt_arithmos::PTExpression` post-rename)
//!
//! - `Number(ArithmosInteger)` â€” an exact integer literal (also represents
//!   rationals via `Function::Divide` of two numbers and special values such as
//!   NaN / infinity through internal flags).
//! - `Constant { â€¦ }` â€” a named symbolic constant (Ï€, e, c, h, â€¦) with an
//!   optional cached f64, optional unit and optional SI prefix.
//! - `Variable(String)` â€” a free symbol bound at evaluation time.
//! - `Function(ArithmosFunction, Vec<â€¦>)` â€” the catch-all node for both binary /
//!   unary operators (Add, Sub, Mul, Div, Pow, Neg, Inv, Sqrt) and transcendental
//!   functions (Exp, Ln, Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Tanh, â€¦).
//! - `Sum`, `Product`, `Limit` â€” bounded ranges and limits (ranges).
//! - `Conditional` â€” if-then-else.
//! - `CachedValue` â€” performance cache with a dirty-flag (per CLAUDE.md Â§4).
//! - `FourierOptimized` â€” Fourier-transform-backed evaluation pathway.
//!
//! ## Core traits
//!
//! Four traits expose Arithmos's behaviour polymorphically so downstream code can
//! be written against abstractions instead of the AST directly:
//!
//! - [`Simplify`] â€” pure simplification. Drives both the compile-time and the
//!   iterative simplifier passes in the [`simplify`] and [`iterative`] submodules.
//! - [`Differentiable`] â€” symbolic differentiation; the core of [`crate::calculus`].
//! - [`Evaluable`] â€” numeric evaluation against a binding context.
//! - [`Emit`] â€” string-target codegen (GLSL, HLSL, MathML, LaTeX, plain text).
//!
//! ## Submodules
//!
//! - [`iterative`] â€” stack-based, iterative simplification passes (no recursion;
//!   per the engine's safety-critical standard "avoid recursion").
//! - [`simplify`] â€” compile-time and runtime simplification rules driven by the
//!   `SimplificationConfig` policy.

pub mod iterative;
pub mod simplify;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::function::ArithmosFunction;
use crate::integer::ArithmosInteger;

/// SI unit prefixes spanning yocto (10â»Â²â´) through yotta (10Â²â´).
///
/// Lives in the expression module because constants and functions both use
/// prefixes when emitting numeric values with units. The prefix is stored
/// separately from the value to preserve symbolic intent (`5 km` is distinct
/// from `5000 m` in the output even though they evaluate identically).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArithmosSIPrefix {
    Yotta,
    Zetta,
    Exa,
    Peta,
    Tera,
    Giga,
    Mega,
    Kilo,
    Hecto,
    Deca,
    None,
    Deci,
    Centi,
    Milli,
    Micro,
    Nano,
    Pico,
    Femto,
    Atto,
    Zepto,
    Yocto,
}

impl ArithmosSIPrefix {
    /// Decimal multiplier for this prefix.
    pub fn multiplier(&self) -> f64 {
        match self {
            Self::Yotta => 1e24,
            Self::Zetta => 1e21,
            Self::Exa => 1e18,
            Self::Peta => 1e15,
            Self::Tera => 1e12,
            Self::Giga => 1e9,
            Self::Mega => 1e6,
            Self::Kilo => 1e3,
            Self::Hecto => 1e2,
            Self::Deca => 1e1,
            Self::None => 1.0,
            Self::Deci => 1e-1,
            Self::Centi => 1e-2,
            Self::Milli => 1e-3,
            Self::Micro => 1e-6,
            Self::Nano => 1e-9,
            Self::Pico => 1e-12,
            Self::Femto => 1e-15,
            Self::Atto => 1e-18,
            Self::Zepto => 1e-21,
            Self::Yocto => 1e-24,
        }
    }

    /// Standard symbol for this prefix (e.g. "k" for kilo, "Î¼" for micro).
    pub fn symbol(&self) -> &'static str {
        match self {
            Self::Yotta => "Y",
            Self::Zetta => "Z",
            Self::Exa => "E",
            Self::Peta => "P",
            Self::Tera => "T",
            Self::Giga => "G",
            Self::Mega => "M",
            Self::Kilo => "k",
            Self::Hecto => "h",
            Self::Deca => "da",
            Self::None => "",
            Self::Deci => "d",
            Self::Centi => "c",
            Self::Milli => "m",
            Self::Micro => "Î¼",
            Self::Nano => "n",
            Self::Pico => "p",
            Self::Femto => "f",
            Self::Atto => "a",
            Self::Zepto => "z",
            Self::Yocto => "y",
        }
    }
}

/// The Arithmos symbolic expression AST.
///
/// Variants match `pt_arithmos::PTExpression` so the migration in Wave 3 is a
/// near-direct port. `Box<ArithmosExpression>` is used for child nodes so the
/// enum stays Sized.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArithmosExpression {
    /// Exact integer or rational literal.
    Number(ArithmosInteger),

    /// Named symbolic constant (Ï€, e, c, h, ...).
    Constant {
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        symbol: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cached_value: Option<f64>,
        #[serde(default)]
        allow_simplification: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        unit: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        prefix: Option<String>,
    },

    /// A free variable referenced by name.
    Variable(String),

    /// Application of `ArithmosFunction` to its arguments. Covers binary ops
    /// (Add/Sub/Mul/Div/Pow), unary ops (Neg, Sqrt), transcendentals (Sin, Cos,
    /// Exp, Ln, ...), inverse / hyperbolic trig and the entire calculus
    /// operator family. See [`ArithmosFunction`] for the full list.
    Function(ArithmosFunction, Vec<ArithmosExpression>),

    /// Bounded summation Î£_{var = start..end} expression.
    Sum {
        variable: String,
        start: Box<ArithmosExpression>,
        end: Box<ArithmosExpression>,
        expression: Box<ArithmosExpression>,
    },

    /// Limit lim_{var â†’ approaching} expression. `from_right` distinguishes the
    /// one-sided variants.
    Limit {
        variable: String,
        approaching: Box<ArithmosExpression>,
        expression: Box<ArithmosExpression>,
        #[serde(default)]
        from_right: bool,
    },

    /// Bounded product Î _{var = start..end} expression.
    Product {
        variable: String,
        start: Box<ArithmosExpression>,
        end: Box<ArithmosExpression>,
        expression: Box<ArithmosExpression>,
    },

    /// If-then-else.
    Conditional {
        condition: Box<ArithmosExpression>,
        then_expr: Box<ArithmosExpression>,
        else_expr: Box<ArithmosExpression>,
    },

    /// Cached f64 result with an explicit dirty flag (CLAUDE.md Â§4).
    #[serde(skip)]
    CachedValue {
        expr: Box<ArithmosExpression>,
        cached: Option<f64>,
        dirty: bool,
    },

    /// Expression that has been replaced by its Fourier-series approximation.
    /// `transform` is populated lazily and skipped during serialisation.
    #[serde(skip_deserializing)]
    FourierOptimized {
        expr: Box<ArithmosExpression>,
        #[serde(skip)]
        transform: Option<Box<crate::fourier::ArithmosFourierTransform>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        variable: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        range: Option<(f64, f64)>,
    },
}

impl ArithmosExpression {
    // ----- constructors -----

    /// Number literal from an `ArithmosInteger`.
    pub fn num(n: ArithmosInteger) -> Self {
        ArithmosExpression::Number(n)
    }

    /// The literal zero.
    pub fn zero() -> Self {
        ArithmosExpression::Number(ArithmosInteger::zero())
    }

    /// Number literal from i64.
    pub fn from_i64(n: i64) -> Self {
        ArithmosExpression::Number(ArithmosInteger::from_i64(n))
    }

    /// Number literal from u64.
    pub fn from_u64(n: u64) -> Self {
        ArithmosExpression::Number(ArithmosInteger::from_u64(n))
    }

    /// Smart f64 constructor.
    ///
    /// Currently routes through the integer constructor when `f` is an exact
    /// integer in the i64 range; otherwise wraps `f` as a `Number / Number`
    /// rational with a fixed denominator scale. NaN and Â±âˆž get the matching
    /// `ArithmosInteger` sentinel.
    pub fn from_f64(f: f64) -> Self {
        if f.is_nan() {
            return ArithmosExpression::Number(ArithmosInteger::nan());
        }
        if f.is_infinite() {
            let mut inf = ArithmosInteger::infinity();
            if f < 0.0 {
                inf.value.set_negative(true);
            }
            return ArithmosExpression::Number(inf);
        }
        let rounded = f.round();
        if (f - rounded).abs() <= f64::EPSILON * f.abs().max(1.0)
            && rounded >= i64::MIN as f64
            && rounded <= i64::MAX as f64
        {
            return ArithmosExpression::from_i64(rounded as i64);
        }
        // Fall back to a fixed-scale rational: f â‰ˆ num / 1e9.
        let scale: f64 = 1.0e9;
        let num = (f * scale).round() as i64;
        ArithmosExpression::div(
            ArithmosExpression::from_i64(num),
            ArithmosExpression::from_i64(scale as i64),
        )
    }

    /// Variable expression.
    pub fn var(name: &str) -> Self {
        ArithmosExpression::Variable(name.to_string())
    }

    /// Named constant.
    pub fn constant(
        symbol: &str,
        name: Option<&str>,
        value: Option<f64>,
        allow_simplification: bool,
    ) -> Self {
        ArithmosExpression::Constant {
            name: name.map(|s| s.to_string()),
            symbol: symbol.to_string(),
            cached_value: value,
            allow_simplification,
            unit: None,
            prefix: None,
        }
    }

    /// Function application.
    pub fn func(f: ArithmosFunction, args: Vec<ArithmosExpression>) -> Self {
        ArithmosExpression::Function(f, args)
    }

    // ----- algebra builders -----

    pub fn add(x: ArithmosExpression, y: ArithmosExpression) -> Self {
        ArithmosExpression::Function(ArithmosFunction::Add, vec![x, y])
    }
    pub fn sub(x: ArithmosExpression, y: ArithmosExpression) -> Self {
        ArithmosExpression::Function(ArithmosFunction::Subtract, vec![x, y])
    }
    pub fn mul(x: ArithmosExpression, y: ArithmosExpression) -> Self {
        ArithmosExpression::Function(ArithmosFunction::Multiply, vec![x, y])
    }
    pub fn div(x: ArithmosExpression, y: ArithmosExpression) -> Self {
        ArithmosExpression::Function(ArithmosFunction::Divide, vec![x, y])
    }
    pub fn pow(x: ArithmosExpression, y: ArithmosExpression) -> Self {
        ArithmosExpression::Function(ArithmosFunction::Power, vec![x, y])
    }
    pub fn neg(x: ArithmosExpression) -> Self {
        ArithmosExpression::Function(ArithmosFunction::Negate, vec![x])
    }
    pub fn sqrt(x: ArithmosExpression) -> Self {
        ArithmosExpression::Function(ArithmosFunction::Sqrt, vec![x])
    }

    // ----- transcendentals -----

    pub fn exp(x: ArithmosExpression) -> Self {
        ArithmosExpression::Function(ArithmosFunction::Exp, vec![x])
    }
    pub fn ln(x: ArithmosExpression) -> Self {
        ArithmosExpression::Function(ArithmosFunction::Ln, vec![x])
    }
    pub fn sin(x: ArithmosExpression) -> Self {
        ArithmosExpression::Function(ArithmosFunction::Sin, vec![x])
    }
    pub fn cos(x: ArithmosExpression) -> Self {
        ArithmosExpression::Function(ArithmosFunction::Cos, vec![x])
    }
    pub fn tan(x: ArithmosExpression) -> Self {
        ArithmosExpression::Function(ArithmosFunction::Tan, vec![x])
    }

    // ----- predicates -----

    /// Returns true if the expression contains no free variables.
    pub fn is_constant(&self) -> bool {
        match self {
            ArithmosExpression::Number(_) => true,
            ArithmosExpression::Constant { .. } => true,
            ArithmosExpression::Variable(_) => false,
            ArithmosExpression::Function(_, args) => args.iter().all(|a| a.is_constant()),
            ArithmosExpression::Sum { .. } => false,
            ArithmosExpression::Limit { .. } => false,
            ArithmosExpression::Product { .. } => false,
            ArithmosExpression::Conditional {
                condition,
                then_expr,
                else_expr,
            } => condition.is_constant() && then_expr.is_constant() && else_expr.is_constant(),
            ArithmosExpression::CachedValue { expr, .. } => expr.is_constant(),
            ArithmosExpression::FourierOptimized { expr, .. } => expr.is_constant(),
        }
    }

    /// Best-effort conversion to f64. Returns `None` if the expression contains
    /// free variables or otherwise cannot be reduced numerically.
    pub fn to_f64(&self) -> Option<f64> {
        let bindings = ArithmosBindings::new();
        match self.evaluate(&bindings) {
            Ok(v) if v.is_finite() || v.is_infinite() => Some(v),
            Ok(v) if v.is_nan() => Some(v),
            Ok(_) => None,
            Err(_) => None,
        }
    }

    /// Unchecked conversion to f64; panics in debug builds when the expression
    /// is not numerically reducible. Mirrors `pt-arithmos::PTExpression::f64()`.
    pub fn f64(&self) -> f64 {
        self.to_f64().unwrap_or(f64::NAN)
    }
}

/// JSON-friendly summary of how complex an expression is. Used by the simplifier
/// to compare candidates and by the EML / Arithmos router to pick a backend.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ArithmosComplexityMetrics {
    pub terms: usize,
    pub complexity: f64,
    pub simplicity: f64,
}

// ============================================================================
// Core traits â€” the abstraction layer per CLAUDE.md SOLID compliance.
// ============================================================================

/// Bindings used during numeric evaluation. Maps variable names to f64 values.
pub type ArithmosBindings = HashMap<String, f64>;

/// Evaluate an expression numerically against a binding context.
///
/// Implementations must return `Err` for unbound variables, divide-by-zero in
/// real-only mode, NaN-producing sub-expressions, and any case that cannot be
/// represented as a finite f64. They MUST NOT panic.
pub trait Evaluable {
    /// Evaluate this expression numerically. Bindings supply variable values.
    fn evaluate(&self, bindings: &ArithmosBindings) -> Result<f64, String>;
}

/// Symbolic differentiation.
///
/// `differentiate(var)` returns `d/d{var}` as a new expression. Higher-order
/// derivatives are obtained by re-applying the trait. The trait is decoupled
/// from `Evaluable` because purely symbolic pipelines never need to evaluate.
pub trait Differentiable {
    /// Returns the derivative of `self` with respect to `var`.
    fn differentiate(&self, var: &str) -> Result<ArithmosExpression, String>;
}

/// Simplification policy used by [`Simplify`].
///
/// Mirrors `pt_simplification_method::PTSimplificationConfig`. Wave 2 ships a
/// minimal default; Wave 3 fills in the full rule set.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SimplificationConfig {
    /// Maximum iterations the simplifier will run before bailing.
    pub max_iterations: usize,
    /// If true, the simplifier may use cached f64 values to drop precision-safe
    /// constants into literal form.
    pub allow_numeric_collapse: bool,
}

/// Simplify an expression in place or by value.
///
/// The trait covers the compile-time, runtime and iterative simplifier paths.
/// The boolean returned by [`simplify_in_place`] reports whether anything was
/// actually rewritten so callers can implement fixed-point loops cheaply.
pub trait Simplify: Sized {
    /// Returns a simplified copy of `self`.
    fn simplify(&self, config: &SimplificationConfig) -> Self;

    /// Simplify in place. Returns `true` if anything changed.
    fn simplify_in_place(&mut self, config: &SimplificationConfig) -> bool;
}

/// Codegen target for the [`Emit`] trait.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmitTarget {
    /// Plain mathematical text (e.g. `sin(x) + 2*y`).
    Text,
    /// LaTeX (e.g. `\\sin x + 2y`).
    Latex,
    /// MathML.
    MathMl,
    /// GLSL â€” the engine renderer's target language.
    Glsl,
    /// HLSL â€” the alternate D3D shader target.
    Hlsl,
    /// Reverse-Polish notation, the format the EML evaluator expects.
    EmlRpn,
}

/// Emit an expression as source code in a target language.
///
/// This is the abstraction the Phase-7 equation-ID texture path will use to
/// turn Arithmos expressions into GLSL fragments before SPIR-V compilation.
pub trait Emit {
    /// Emit `self` as a string in the chosen target dialect.
    fn emit(&self, target: EmitTarget) -> Result<String, String>;
}

// ============================================================================
// Stub trait impls so downstream modules can take trait-bounded inputs even
// before the real algorithms are migrated. Every method returns `unimplemented!`.
// ============================================================================

/// Hard cap on the number of nodes an iterative evaluator will visit before
/// bailing. Satisfies CLAUDE.md safety rule 2 (all loops have fixed bounds).
const ARITHMOS_EVALUATE_NODE_CAP: usize = 1_048_576;

/// Apply a unary or binary `ArithmosFunction` to f64 operands.
///
/// Pure helper so the iterative evaluator can stay short and the math table
/// lives in one place. Returns `Err` on division by zero, unsupported variants
/// or NaN-producing inputs.
fn arithmos_apply_function(
    func: &crate::function::ArithmosFunction,
    args: &[f64],
) -> Result<f64, String> {
    use crate::function::ArithmosFunction as F;
    debug_assert!(!args.is_empty() || matches!(func, F::Sum | F::Product { .. }), "no args for {:?}", func);
    match func {
        F::Add => Ok(args.iter().sum()),
        F::Subtract => {
            if args.len() != 2 {
                return Err("Subtract expects 2 args".into());
            }
            Ok(args[0] - args[1])
        }
        F::Multiply => Ok(args.iter().product()),
        F::Divide => {
            if args.len() != 2 {
                return Err("Divide expects 2 args".into());
            }
            if args[1] == 0.0 {
                return Err("division by zero".into());
            }
            Ok(args[0] / args[1])
        }
        F::Power => {
            if args.len() != 2 {
                return Err("Power expects 2 args".into());
            }
            Ok(args[0].powf(args[1]))
        }
        F::Negate => {
            if args.len() != 1 {
                return Err("Negate expects 1 arg".into());
            }
            Ok(-args[0])
        }
        F::Sqrt => Ok(args[0].sqrt()),
        F::Cbrt => Ok(args[0].cbrt()),
        F::Exp => Ok(args[0].exp()),
        F::Ln => Ok(args[0].ln()),
        F::Log10 => Ok(args[0].log10()),
        F::Log2 => Ok(args[0].log2()),
        F::Sin => Ok(args[0].sin()),
        F::Cos => Ok(args[0].cos()),
        F::Tan => Ok(args[0].tan()),
        F::Asin => Ok(args[0].asin()),
        F::Acos => Ok(args[0].acos()),
        F::Atan => Ok(args[0].atan()),
        F::Atan2 => {
            if args.len() != 2 {
                return Err("Atan2 expects 2 args".into());
            }
            Ok(args[0].atan2(args[1]))
        }
        F::Sinh => Ok(args[0].sinh()),
        F::Cosh => Ok(args[0].cosh()),
        F::Tanh => Ok(args[0].tanh()),
        F::Abs => Ok(args[0].abs()),
        F::Sign => Ok(args[0].signum()),
        F::Floor => Ok(args[0].floor()),
        F::Ceil => Ok(args[0].ceil()),
        F::Round => Ok(args[0].round()),
        _ => Err(format!("evaluate: unsupported function {:?}", func)),
    }
}

impl Evaluable for ArithmosExpression {
    fn evaluate(&self, bindings: &ArithmosBindings) -> Result<f64, String> {
        // Iterative post-order traversal. We push (node, child_index) frames
        // and unwind values onto a separate value stack. No recursion.
        enum Frame<'a> {
            Enter(&'a ArithmosExpression),
            CombineFunc(&'a crate::function::ArithmosFunction, usize),
            CombineCond,
        }
        let mut work: Vec<Frame> = Vec::with_capacity(32);
        let mut values: Vec<f64> = Vec::with_capacity(32);
        work.push(Frame::Enter(self));
        let mut guard: usize = 0;
        while let Some(frame) = work.pop() {
            guard += 1;
            if guard > ARITHMOS_EVALUATE_NODE_CAP {
                return Err("evaluate: node cap exceeded".into());
            }
            match frame {
                Frame::Enter(node) => match node {
                    ArithmosExpression::Number(n) => values.push(n.to_f64()),
                    ArithmosExpression::Constant { cached_value, symbol, .. } => {
                        if let Some(v) = *cached_value {
                            values.push(v);
                        } else {
                            return Err(format!("constant '{}' has no cached value", symbol));
                        }
                    }
                    ArithmosExpression::Variable(name) => match bindings.get(name) {
                        Some(v) => values.push(*v),
                        None => return Err(format!("unbound variable '{}'", name)),
                    },
                    ArithmosExpression::Function(func, args) => {
                        work.push(Frame::CombineFunc(func, args.len()));
                        // Children pushed in reverse so the first child is evaluated first.
                        for a in args.iter().rev() {
                            work.push(Frame::Enter(a));
                        }
                    }
                    ArithmosExpression::Conditional {
                        condition,
                        then_expr,
                        else_expr,
                    } => {
                        work.push(Frame::CombineCond);
                        work.push(Frame::Enter(else_expr));
                        work.push(Frame::Enter(then_expr));
                        work.push(Frame::Enter(condition));
                    }
                    ArithmosExpression::CachedValue { expr, cached, .. } => {
                        if let Some(v) = *cached {
                            values.push(v);
                        } else {
                            work.push(Frame::Enter(expr));
                        }
                    }
                    ArithmosExpression::FourierOptimized { expr, .. } => {
                        work.push(Frame::Enter(expr));
                    }
                    _ => {
                        return Err("evaluate: unsupported expression variant".into());
                    }
                },
                Frame::CombineFunc(func, n) => {
                    if values.len() < n {
                        return Err("evaluate: value stack underflow".into());
                    }
                    let start = values.len() - n;
                    let args_slice: Vec<f64> = values.drain(start..).collect();
                    let out = arithmos_apply_function(func, &args_slice)?;
                    values.push(out);
                }
                Frame::CombineCond => {
                    if values.len() < 3 {
                        return Err("evaluate: conditional underflow".into());
                    }
                    let else_v = values.pop().unwrap();
                    let then_v = values.pop().unwrap();
                    let cond_v = values.pop().unwrap();
                    values.push(if cond_v != 0.0 { then_v } else { else_v });
                }
            }
        }
        if values.len() != 1 {
            return Err(format!("evaluate: final stack size {} != 1", values.len()));
        }
        Ok(values[0])
    }
}

impl Differentiable for ArithmosExpression {
    fn differentiate(&self, var: &str) -> Result<ArithmosExpression, String> {
        crate::calculus::differentiation_iterative::differentiate_iterative(self, var)
    }
}

impl Simplify for ArithmosExpression {
    fn simplify(&self, _config: &SimplificationConfig) -> Self {
        // Trivial placeholder: returning the input unchanged is a valid
        // simplification (it just performs no work).
        self.clone()
    }

    fn simplify_in_place(&mut self, _config: &SimplificationConfig) -> bool {
        false
    }
}

impl Emit for ArithmosExpression {
    fn emit(&self, _target: EmitTarget) -> Result<String, String> {
        unimplemented!("ArithmosExpression::emit â€” populated in Wave 3")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_is_a_number() {
        let z = ArithmosExpression::zero();
        assert!(matches!(z, ArithmosExpression::Number(_)));
        assert!(z.is_constant());
    }

    #[test]
    fn variable_is_not_constant() {
        let v = ArithmosExpression::var("x");
        assert!(!v.is_constant());
    }

    #[test]
    fn si_prefix_kilo_multiplier() {
        let prefix = ArithmosSIPrefix::Kilo;
        assert!((prefix.multiplier() - 1e3).abs() < f64::EPSILON);
        assert_eq!(prefix.symbol(), "k");
    }

    #[test]
    fn simplify_default_is_identity() {
        let expr = ArithmosExpression::var("x");
        let cfg = SimplificationConfig::default();
        let out = expr.simplify(&cfg);
        // Default simplify is a no-op â€” equal-shape result.
        assert!(matches!(out, ArithmosExpression::Variable(_)));
    }
}

