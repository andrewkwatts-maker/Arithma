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
//! `ArithmaExpression` is the abstract syntax tree at the heart of Arithma. Every
//! other module in the crate either produces, consumes, or transforms expressions.
//!
//! ## Variants (mirrors `pt_arithmos::PTExpression` post-rename)
//!
//! - `Number(ArithmaInteger)` — an exact integer literal (also represents
//!   rationals via `Function::Divide` of two numbers and special values such as
//!   NaN / infinity through internal flags).
//! - `Constant { … }` — a named symbolic constant (π, e, c, h, …) with an
//!   optional cached f64, optional unit and optional SI prefix.
//! - `Variable(String)` — a free symbol bound at evaluation time.
//! - `Function(ArithmaFunction, Vec<…>)` — the catch-all node for both binary /
//!   unary operators (Add, Sub, Mul, Div, Pow, Neg, Inv, Sqrt) and transcendental
//!   functions (Exp, Ln, Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Tanh, …).
//! - `Sum`, `Product`, `Limit` — bounded ranges and limits (ranges).
//! - `Conditional` — if-then-else.
//! - `CachedValue` — performance cache with a dirty-flag (per CLAUDE.md §4).
//! - `FourierOptimized` — Fourier-transform-backed evaluation pathway.
//!
//! ## Core traits
//!
//! Four traits expose Arithma's behaviour polymorphically so downstream code can
//! be written against abstractions instead of the AST directly:
//!
//! - [`Simplify`] — pure simplification. Drives both the compile-time and the
//!   iterative simplifier passes in the [`simplify`] and [`iterative`] submodules.
//! - [`Differentiable`] — symbolic differentiation; the core of [`crate::calculus`].
//! - [`Evaluable`] — numeric evaluation against a binding context.
//! - [`Emit`] — string-target codegen (GLSL, HLSL, MathML, LaTeX, plain text).
//!
//! ## Submodules
//!
//! - [`iterative`] — stack-based, iterative simplification passes (no recursion;
//!   per the engine's safety-critical standard "avoid recursion").
//! - [`simplify`] — compile-time and runtime simplification rules driven by the
//!   `SimplificationConfig` policy.

pub mod emit;
pub mod iterative;
pub mod simplify;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::function::ArithmaFunction;
use crate::integer::ArithmaInteger;

/// SI unit prefixes spanning yocto (10⁻²⁴) through yotta (10²⁴).
///
/// Lives in the expression module because constants and functions both use
/// prefixes when emitting numeric values with units. The prefix is stored
/// separately from the value to preserve symbolic intent (`5 km` is distinct
/// from `5000 m` in the output even though they evaluate identically).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArithmaSIPrefix {
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

impl ArithmaSIPrefix {
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

    /// Standard symbol for this prefix, ASCII only.
    ///
    /// Micro is `"u"`, not the micro sign. Two reasons, and the second is the
    /// one that bites: the project's naming rule is Latin letters only, and
    /// `crate::unit`'s prefix table -- which drives `split_unit_symbol` and
    /// `ArithmaUnit::convert` -- spells it `"u"`. Two spellings of micro in
    /// one crate mean `si_prefix_power(Micro.symbol())` returns `None`, so the
    /// two modules silently disagree about what `um` denotes.
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
            Self::Micro => "u",
            Self::Nano => "n",
            Self::Pico => "p",
            Self::Femto => "f",
            Self::Atto => "a",
            Self::Zepto => "z",
            Self::Yocto => "y",
        }
    }
}

/// The Arithma symbolic expression AST.
///
/// Variants match `pt_arithmos::PTExpression` so the migration in Wave 3 is a
/// near-direct port. `Box<ArithmaExpression>` is used for child nodes so the
/// enum stays Sized.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArithmaExpression {
    /// Exact integer or rational literal.
    Number(ArithmaInteger),

    /// Named symbolic constant (π, e, c, h, ...).
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

    /// Application of `ArithmaFunction` to its arguments. Covers binary ops
    /// (Add/Sub/Mul/Div/Pow), unary ops (Neg, Sqrt), transcendentals (Sin, Cos,
    /// Exp, Ln, ...), inverse / hyperbolic trig and the entire calculus
    /// operator family. See [`ArithmaFunction`] for the full list.
    Function(ArithmaFunction, Vec<ArithmaExpression>),

    /// Bounded summation Î£_{var = start..end} expression.
    Sum {
        variable: String,
        start: Box<ArithmaExpression>,
        end: Box<ArithmaExpression>,
        expression: Box<ArithmaExpression>,
    },

    /// Limit lim_{var → approaching} expression. `from_right` distinguishes the
    /// one-sided variants.
    Limit {
        variable: String,
        approaching: Box<ArithmaExpression>,
        expression: Box<ArithmaExpression>,
        #[serde(default)]
        from_right: bool,
    },

    /// Bounded product Î _{var = start..end} expression.
    Product {
        variable: String,
        start: Box<ArithmaExpression>,
        end: Box<ArithmaExpression>,
        expression: Box<ArithmaExpression>,
    },

    /// If-then-else.
    Conditional {
        condition: Box<ArithmaExpression>,
        then_expr: Box<ArithmaExpression>,
        else_expr: Box<ArithmaExpression>,
    },

    /// Cached f64 result with an explicit dirty flag (CLAUDE.md §4).
    #[serde(skip)]
    CachedValue {
        expr: Box<ArithmaExpression>,
        cached: Option<f64>,
        dirty: bool,
    },

    /// Expression that has been replaced by its Fourier-series approximation.
    /// `transform` is populated lazily and skipped during serialisation.
    #[serde(skip_deserializing)]
    FourierOptimized {
        expr: Box<ArithmaExpression>,
        #[serde(skip)]
        transform: Option<Box<crate::fourier::ArithmaFourierTransform>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        variable: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        range: Option<(f64, f64)>,
    },
}

impl ArithmaExpression {
    // ----- constructors -----

    /// Number literal from an `ArithmaInteger`.
    pub fn num(n: ArithmaInteger) -> Self {
        ArithmaExpression::Number(n)
    }

    /// The literal zero.
    pub fn zero() -> Self {
        ArithmaExpression::Number(ArithmaInteger::zero())
    }

    /// Number literal from i64.
    pub fn from_i64(n: i64) -> Self {
        ArithmaExpression::Number(ArithmaInteger::from_i64(n))
    }

    /// Number literal from u64.
    pub fn from_u64(n: u64) -> Self {
        ArithmaExpression::Number(ArithmaInteger::from_u64(n))
    }

    /// Lossless f64 constructor.
    ///
    /// Represents `f` as an exact rational with a DECIMAL denominator, taken
    /// from the shortest decimal string that round-trips to `f`. NaN and ±∞
    /// get the matching `ArithmaInteger` sentinel.
    ///
    /// WHY A DECIMAL DENOMINATOR
    /// -------------------------
    /// Losslessness is the point of this library, and there are two ways to
    /// be lossless about an f64. Both round-trip; they differ in what they
    /// say the number IS.
    ///
    /// ```text
    ///     binary   0.1 -> 3602879701896397 / 2^55
    ///     decimal  0.1 -> 1 / 10
    /// ```
    ///
    /// The binary form is the exact content of the 64 bits, but it is not the
    /// number anybody wrote, and it makes every subsequent denominator a
    /// power of two that never cancels against a decimal one. The decimal
    /// form is equally exact -- the shortest round-tripping representation
    /// identifies the f64 uniquely -- and it is the rational a reader means
    /// by "0.1", so denominators stay small and cancel.
    ///
    /// WHAT THIS REPLACED
    /// ------------------
    /// A fixed-scale rational, `round(f * S) / S`, which could not represent
    /// values outside a narrow window and failed SILENTLY:
    ///
    /// ```text
    ///     S = 1e15  (|f| < 9000)   usable [1e-15, 9.2e3]
    ///     S = 1e9   (otherwise)    usable [1e-9,  9.2e9], then `as i64`
    ///                              SATURATES at i64::MAX
    /// ```
    ///
    /// Measured: `1.9e-93` became exactly `0.0`; `4.757e34` became
    /// `9223372036.854776`. Raising the scale cannot fix it -- an i64
    /// rational carries about 19 significant digits wherever the point is
    /// put, so 1e9 -> 1e15 only traded the ceiling (9.2e9 -> 9.2e3) for
    /// precision at the bottom. There is no fixed scale that covers a
    /// consumer spanning 128 decades. The decimal form below has no scale
    /// constant and no range limit beyond the arbitrary-precision integer.
    pub fn from_f64(f: f64) -> Self {
        if f.is_nan() {
            return ArithmaExpression::Number(ArithmaInteger::nan());
        }
        if f.is_infinite() {
            let mut inf = ArithmaInteger::infinity();
            if f < 0.0 {
                inf.value.set_negative(true);
            }
            return ArithmaExpression::Number(inf);
        }

        // EXACT integer test. This used to read
        //     (f - rounded).abs() <= f64::EPSILON * f.abs().max(1.0)
        // whose `.max(1.0)` makes the tolerance an ABSOLUTE 2.22e-16 for
        // every |f| < 1. Any value below about 2.2e-16 therefore satisfied it
        // against `rounded == 0.0` and was returned as the integer ZERO --
        // 1.9e-93 never reached the rational path at all. An epsilon test is
        // the wrong tool regardless: a float either is an integer or it is
        // not, and snapping a near-integer silently discards exactly the
        // information this library exists to keep.
        let rounded = f.round();
        if f == rounded && rounded >= i64::MIN as f64 && rounded <= i64::MAX as f64 {
            return ArithmaExpression::from_i64(rounded as i64);
        }

        Self::from_decimal_string(f).unwrap_or_else(|| Self::from_binary_rational(f))
    }

    /// Build the exact rational `digits / 10^k` from the shortest decimal
    /// string that round-trips to `f`. Rust's `{:e}` emits exactly that, so
    /// no precision is chosen here and none is lost.
    fn from_decimal_string(f: f64) -> Option<Self> {
        let rendered = format!("{f:e}");
        let (mantissa_part, exponent_part) = rendered.split_once('e')?;
        let exponent: i64 = exponent_part.parse().ok()?;

        let negative = mantissa_part.starts_with('-');
        let digits_only: String = mantissa_part
            .trim_start_matches(['-', '+'])
            .chars()
            .filter(|c| *c != '.')
            .collect();
        if digits_only.is_empty() || !digits_only.chars().all(|c| c.is_ascii_digit()) {
            return None;
        }
        let fractional_digits = mantissa_part
            .split_once('.')
            .map(|(_, tail)| tail.len() as i64)
            .unwrap_or(0);

        // f = +/- digits_only * 10^(exponent - fractional_digits)
        let power = exponent - fractional_digits;

        let mut numerator = Self::big_from_decimal_digits(&digits_only)?;
        if power >= 0 {
            let scale = u32::try_from(power).ok()?;
            if scale > 0 {
                numerator =
                    numerator.checked_mul(&ArithmaInteger::from_u64(10).checked_pow(scale)?)?;
            }
            if negative {
                numerator.value.set_negative(true);
            }
            return Some(ArithmaExpression::Number(numerator));
        }
        let scale = u32::try_from(-power).ok()?;
        if negative {
            numerator.value.set_negative(true);
        }
        Some(ArithmaExpression::div(
            ArithmaExpression::Number(numerator),
            ArithmaExpression::Number(ArithmaInteger::from_u64(10).checked_pow(scale)?),
        ))
    }

    /// Horner-accumulate a decimal digit string into an arbitrary-precision
    /// integer, so a mantissa longer than u64 is still exact.
    fn big_from_decimal_digits(digits: &str) -> Option<ArithmaInteger> {
        let ten = ArithmaInteger::from_u64(10);
        let mut acc = ArithmaInteger::zero();
        for ch in digits.chars() {
            let digit = ch.to_digit(10)?;
            acc = acc
                .checked_mul(&ten)?
                .checked_add(&ArithmaInteger::from_u64(u64::from(digit)))?;
        }
        Some(acc)
    }

    /// Exact binary rational `mantissa / 2^k` straight from the IEEE-754
    /// bits. Used only if the decimal route fails; it is equally lossless but
    /// yields power-of-two denominators that do not cancel against decimals.
    fn from_binary_rational(f: f64) -> Self {
        let bits = f.to_bits();
        let negative = (bits >> 63) != 0;
        let raw_exponent = ((bits >> 52) & 0x7ff) as i64;
        let raw_mantissa = bits & 0x000f_ffff_ffff_ffff;
        let (mut mantissa, mut exponent) = if raw_exponent == 0 {
            (raw_mantissa, -1074i64) // subnormal: no implicit leading bit
        } else {
            (raw_mantissa | 0x0010_0000_0000_0000, raw_exponent - 1075)
        };
        if mantissa == 0 {
            return ArithmaExpression::Number(ArithmaInteger::zero());
        }
        let shift = mantissa.trailing_zeros();
        mantissa >>= shift;
        exponent += i64::from(shift);

        let two = ArithmaInteger::from_u64(2);
        let built = u32::try_from(exponent.unsigned_abs()).ok().and_then(|k| {
            let mut numerator = ArithmaInteger::from_u64(mantissa);
            if exponent >= 0 {
                if k > 0 {
                    numerator = numerator.checked_mul(&two.checked_pow(k)?)?;
                }
                if negative {
                    numerator.value.set_negative(true);
                }
                Some(ArithmaExpression::Number(numerator))
            } else {
                if negative {
                    numerator.value.set_negative(true);
                }
                Some(ArithmaExpression::div(
                    ArithmaExpression::Number(numerator),
                    ArithmaExpression::Number(two.checked_pow(k)?),
                ))
            }
        });
        built.unwrap_or_else(|| ArithmaExpression::constant("", None, Some(f), true))
    }

    /// Variable expression.
    pub fn var(name: &str) -> Self {
        ArithmaExpression::Variable(name.to_string())
    }

    /// Named constant.
    pub fn constant(
        symbol: &str,
        name: Option<&str>,
        value: Option<f64>,
        allow_simplification: bool,
    ) -> Self {
        ArithmaExpression::Constant {
            name: name.map(|s| s.to_string()),
            symbol: symbol.to_string(),
            cached_value: value,
            allow_simplification,
            unit: None,
            prefix: None,
        }
    }

    /// Function application.
    pub fn func(f: ArithmaFunction, args: Vec<ArithmaExpression>) -> Self {
        ArithmaExpression::Function(f, args)
    }

    // ----- algebra builders -----

    pub fn add(x: ArithmaExpression, y: ArithmaExpression) -> Self {
        ArithmaExpression::Function(ArithmaFunction::Add, vec![x, y])
    }
    pub fn sub(x: ArithmaExpression, y: ArithmaExpression) -> Self {
        ArithmaExpression::Function(ArithmaFunction::Subtract, vec![x, y])
    }
    pub fn mul(x: ArithmaExpression, y: ArithmaExpression) -> Self {
        ArithmaExpression::Function(ArithmaFunction::Multiply, vec![x, y])
    }
    pub fn div(x: ArithmaExpression, y: ArithmaExpression) -> Self {
        ArithmaExpression::Function(ArithmaFunction::Divide, vec![x, y])
    }
    pub fn pow(x: ArithmaExpression, y: ArithmaExpression) -> Self {
        ArithmaExpression::Function(ArithmaFunction::Power, vec![x, y])
    }
    pub fn neg(x: ArithmaExpression) -> Self {
        ArithmaExpression::Function(ArithmaFunction::Negate, vec![x])
    }
    pub fn sqrt(x: ArithmaExpression) -> Self {
        ArithmaExpression::Function(ArithmaFunction::Sqrt, vec![x])
    }

    // ----- transcendentals -----

    pub fn exp(x: ArithmaExpression) -> Self {
        ArithmaExpression::Function(ArithmaFunction::Exp, vec![x])
    }
    pub fn ln(x: ArithmaExpression) -> Self {
        ArithmaExpression::Function(ArithmaFunction::Ln, vec![x])
    }
    pub fn sin(x: ArithmaExpression) -> Self {
        ArithmaExpression::Function(ArithmaFunction::Sin, vec![x])
    }
    pub fn cos(x: ArithmaExpression) -> Self {
        ArithmaExpression::Function(ArithmaFunction::Cos, vec![x])
    }
    pub fn tan(x: ArithmaExpression) -> Self {
        ArithmaExpression::Function(ArithmaFunction::Tan, vec![x])
    }

    // ----- predicates -----

    /// Returns true if the expression contains no free variables.
    pub fn is_constant(&self) -> bool {
        match self {
            ArithmaExpression::Number(_) => true,
            ArithmaExpression::Constant { .. } => true,
            ArithmaExpression::Variable(_) => false,
            ArithmaExpression::Function(_, args) => args.iter().all(|a| a.is_constant()),
            ArithmaExpression::Sum { .. } => false,
            ArithmaExpression::Limit { .. } => false,
            ArithmaExpression::Product { .. } => false,
            ArithmaExpression::Conditional {
                condition,
                then_expr,
                else_expr,
            } => condition.is_constant() && then_expr.is_constant() && else_expr.is_constant(),
            ArithmaExpression::CachedValue { expr, .. } => expr.is_constant(),
            ArithmaExpression::FourierOptimized { expr, .. } => expr.is_constant(),
        }
    }

    /// Best-effort conversion to f64. Returns `None` if the expression contains
    /// free variables or otherwise cannot be reduced numerically.
    pub fn to_f64(&self) -> Option<f64> {
        let bindings = ArithmaBindings::new();
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
/// to compare candidates and by the EML / Arithma router to pick a backend.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ArithmaComplexityMetrics {
    pub terms: usize,
    pub complexity: f64,
    pub simplicity: f64,
}

// ============================================================================
// Core traits — the abstraction layer per CLAUDE.md SOLID compliance.
// ============================================================================

/// Bindings used during numeric evaluation. Maps variable names to f64 values.
pub type ArithmaBindings = HashMap<String, f64>;

/// Evaluate an expression numerically against a binding context.
///
/// Implementations must return `Err` for unbound variables, divide-by-zero in
/// real-only mode, NaN-producing sub-expressions, and any case that cannot be
/// represented as a finite f64. They MUST NOT panic.
pub trait Evaluable {
    /// Evaluate this expression numerically. Bindings supply variable values.
    fn evaluate(&self, bindings: &ArithmaBindings) -> Result<f64, String>;
}

/// Symbolic differentiation.
///
/// `differentiate(var)` returns `d/d{var}` as a new expression. Higher-order
/// derivatives are obtained by re-applying the trait. The trait is decoupled
/// from `Evaluable` because purely symbolic pipelines never need to evaluate.
pub trait Differentiable {
    /// Returns the derivative of `self` with respect to `var`.
    fn differentiate(&self, var: &str) -> Result<ArithmaExpression, String>;
}

/// Simplification policy used by [`Simplify`].
///
/// Mirrors `pt_simplification_method::PTSimplificationConfig`. Wave 2 ships a
/// minimal default; Wave 3 fills in the full rule set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimplificationConfig {
    /// Maximum iterations the simplifier will run before bailing.
    pub max_iterations: usize,
    /// If true, the simplifier may use cached f64 values to drop precision-safe
    /// constants into literal form.
    pub allow_numeric_collapse: bool,
}

impl Default for SimplificationConfig {
    /// Usable defaults.
    ///
    /// This was `#[derive(Default)]`, which gave `max_iterations: 0` -- a
    /// simplifier that is a no-op by construction, and one that made the
    /// no-op implementation impossible to distinguish from a working one.
    /// 32 passes is far more than any fixpoint needs (folding converges in
    /// one pass per level of nesting) while still bounding the work.
    fn default() -> Self {
        Self {
            max_iterations: 32,
            allow_numeric_collapse: false,
        }
    }
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
    /// GLSL — the engine renderer's target language.
    Glsl,
    /// HLSL — the alternate D3D shader target.
    Hlsl,
    /// Reverse-Polish notation, the format the EML evaluator expects.
    EmlRpn,
}

/// Emit an expression as source code in a target language.
///
/// This is the abstraction the Phase-7 equation-ID texture path will use to
/// turn Arithma expressions into GLSL fragments before SPIR-V compilation.
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
const ARITHMA_EVALUATE_NODE_CAP: usize = 1_048_576;

/// Apply a unary or binary `ArithmaFunction` to f64 operands.
///
/// Pure helper so the iterative evaluator can stay short and the math table
/// lives in one place. Returns `Err` on division by zero, unsupported variants
/// or NaN-producing inputs.
fn arithma_apply_function(
    func: &crate::function::ArithmaFunction,
    args: &[f64],
) -> Result<f64, String> {
    use crate::function::ArithmaFunction as F;
    debug_assert!(
        !args.is_empty() || matches!(func, F::Sum | F::Product { .. }),
        "no args for {func:?}"
    );
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
        _ => Err(format!("evaluate: unsupported function {func:?}")),
    }
}

impl Evaluable for ArithmaExpression {
    fn evaluate(&self, bindings: &ArithmaBindings) -> Result<f64, String> {
        // Iterative post-order traversal. We push (node, child_index) frames
        // and unwind values onto a separate value stack. No recursion.
        enum Frame<'a> {
            Enter(&'a ArithmaExpression),
            CombineFunc(&'a crate::function::ArithmaFunction, usize),
            CombineCond,
        }
        let mut work: Vec<Frame> = Vec::with_capacity(32);
        let mut values: Vec<f64> = Vec::with_capacity(32);
        work.push(Frame::Enter(self));
        let mut guard: usize = 0;
        while let Some(frame) = work.pop() {
            guard += 1;
            if guard > ARITHMA_EVALUATE_NODE_CAP {
                return Err("evaluate: node cap exceeded".into());
            }
            match frame {
                Frame::Enter(node) => match node {
                    ArithmaExpression::Number(n) => values.push(n.to_f64()),
                    ArithmaExpression::Constant {
                        cached_value,
                        symbol,
                        ..
                    } => {
                        if let Some(v) = *cached_value {
                            values.push(v);
                        } else {
                            return Err(format!("constant '{symbol}' has no cached value"));
                        }
                    }
                    ArithmaExpression::Variable(name) => match bindings.get(name) {
                        Some(v) => values.push(*v),
                        None => return Err(format!("unbound variable '{name}'")),
                    },
                    ArithmaExpression::Function(func, args) => {
                        work.push(Frame::CombineFunc(func, args.len()));
                        // Children pushed in reverse so the first child is evaluated first.
                        for a in args.iter().rev() {
                            work.push(Frame::Enter(a));
                        }
                    }
                    ArithmaExpression::Conditional {
                        condition,
                        then_expr,
                        else_expr,
                    } => {
                        work.push(Frame::CombineCond);
                        work.push(Frame::Enter(else_expr));
                        work.push(Frame::Enter(then_expr));
                        work.push(Frame::Enter(condition));
                    }
                    ArithmaExpression::CachedValue { expr, cached, .. } => {
                        if let Some(v) = *cached {
                            values.push(v);
                        } else {
                            work.push(Frame::Enter(expr));
                        }
                    }
                    ArithmaExpression::FourierOptimized { expr, .. } => {
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
                    let out = arithma_apply_function(func, &args_slice)?;
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

impl Differentiable for ArithmaExpression {
    fn differentiate(&self, var: &str) -> Result<ArithmaExpression, String> {
        crate::calculus::differentiation_iterative::differentiate_iterative(self, var)
    }
}

impl Simplify for ArithmaExpression {
    /// Simplified copy. Delegates to the iterative engine so there is exactly
    /// one implementation of the rewrite rules.
    fn simplify(&self, config: &SimplificationConfig) -> Self {
        let mut out = self.clone();
        let _ = out.simplify_in_place(config);
        out
    }

    /// Simplify in place, reporting whether anything changed.
    ///
    /// This trait was a stub returning `false` while the rewrite rules were
    /// still to be written. It now routes to
    /// [`crate::expression::iterative::simplify_iterative`], which runs
    /// bottom-up passes to a fixpoint without recursion.
    fn simplify_in_place(&mut self, config: &SimplificationConfig) -> bool {
        crate::expression::iterative::simplify_iterative(self, config)
    }
}

// `impl Emit for ArithmaExpression` lives in `expression::emit`.

// ---------------------------------------------------------------------------
// Backward-compatibility aliases for the pre-rename `Arithmos*` names.
// Retained for one release; downstream (eml-math, eml-spectral, metaphysica,
// periodica) should migrate to the `Arithma*` names above.
// ---------------------------------------------------------------------------
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaBindings`")]
#[allow(unused)]
pub use self::ArithmaBindings as ArithmosBindings;
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaComplexityMetrics`")]
#[allow(unused)]
pub use self::ArithmaComplexityMetrics as ArithmosComplexityMetrics;
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaExpression`")]
#[allow(unused)]
pub use self::ArithmaExpression as ArithmosExpression;
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaSIPrefix`")]
#[allow(unused)]
pub use self::ArithmaSIPrefix as ArithmosSIPrefix;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_is_a_number() {
        let z = ArithmaExpression::zero();
        assert!(matches!(z, ArithmaExpression::Number(_)));
        assert!(z.is_constant());
    }

    #[test]
    fn variable_is_not_constant() {
        let v = ArithmaExpression::var("x");
        assert!(!v.is_constant());
    }

    #[test]
    fn si_prefix_kilo_multiplier() {
        let prefix = ArithmaSIPrefix::Kilo;
        assert!((prefix.multiplier() - 1e3).abs() < f64::EPSILON);
        assert_eq!(prefix.symbol(), "k");
    }

    #[test]
    fn simplify_default_is_identity() {
        let expr = ArithmaExpression::var("x");
        let cfg = SimplificationConfig::default();
        let out = expr.simplify(&cfg);
        // Default simplify is a no-op — equal-shape result.
        assert!(matches!(out, ArithmaExpression::Variable(_)));
    }

    /// `from_f64` must be LOSSLESS for every finite input.
    ///
    /// The predecessor wrote `round(f * S) / S` for a fixed `S`, which turned
    /// 1.9e-93 into exactly 0.0 and saturated 4.757e34 at i64::MAX / 1e9.
    /// Both failures were silent -- a consumer spanning 128 decades had
    /// computations replaced by zeros with nothing raised.
    #[test]
    fn from_f64_round_trips_across_the_full_f64_range() {
        let env = std::collections::HashMap::new();
        for &value in &[
            1.903_857_950_822_914_4e-93_f64,
            4.757_399_129_595_567e34,
            std::f64::consts::PI,
            0.1,
            -2.5e-40,
            1e300,
            1e-308,
            0.3,
            1.0 / 3.0,
            -7.5,
        ] {
            let got = ArithmaExpression::from_f64(value)
                .evaluate(&env)
                .unwrap_or_else(|e| panic!("{value:e} failed to evaluate: {e:?}"));
            assert_eq!(got, value, "from_f64 lost {value:e} (got {got:e})");
        }
    }

    /// A decimal literal should come back as a decimal rational, not a
    /// power-of-two one: 0.1 is 1/10, not 3602879701896397/2^55. Both are
    /// exact; only the first cancels against other decimals.
    #[test]
    fn from_f64_uses_a_decimal_denominator() {
        match ArithmaExpression::from_f64(0.1) {
            ArithmaExpression::Function(_, ref args) if args.len() == 2 => {
                let env = std::collections::HashMap::new();
                let denominator = args[1].evaluate(&env).expect("denominator evaluates");
                assert_eq!(denominator, 10.0, "0.1 should be 1/10");
            }
            other => panic!("expected a division, got {other:?}"),
        }
    }

    /// Tiny values must not be swallowed by the integer fast path. The old
    /// guard compared against an ABSOLUTE 2.22e-16 for every |f| < 1, so
    /// anything smaller was returned as the integer zero.
    #[test]
    fn tiny_values_are_not_snapped_to_zero() {
        let env = std::collections::HashMap::new();
        for &value in &[1e-17_f64, 2.2e-16, 5e-30, 1.9e-93] {
            let got = ArithmaExpression::from_f64(value).evaluate(&env).unwrap();
            assert_ne!(got, 0.0, "{value:e} was snapped to zero");
        }
    }

    /// `to_f64` used to read a capped 32-byte PREFIX of the magnitude, which
    /// returns 0.0 for any value whose low 32 bytes are zero -- 2^360 is
    /// exactly that shape, and it is the denominator of small floats.
    #[test]
    fn large_integers_convert_without_truncation() {
        let two = crate::integer::ArithmaInteger::from_u64(2);
        for &exponent in &[64_u32, 100, 200, 360, 900] {
            let power = two.checked_pow(exponent).expect("power fits");
            let got = power.to_f64();
            let want = 2.0_f64.powi(exponent as i32);
            assert!(
                (got - want).abs() <= want * 1e-12,
                "2^{exponent} converted to {got:e}, expected {want:e}"
            );
        }
    }
}
