//====== Arithma/rust/arithma_core/src/pyfacade.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! PyO3 facade for the `arithma` Python package.
//!
//! Gated behind the `python` feature so the engine workspace builds without
//! a Python interpreter on the path. When `pip install arithma` triggers a
//! maturin build, this module is the entry point that exposes the Rust core
//! to Python under the module name `arithma._arithma_core`.
//!
//! Wave-3 surface
//! --------------
//!
//! - `Expression` wraps `ArithmosExpression`. Constructors: `Expression.variable(name)`,
//!   `Expression.number(value)`, `Expression.constant(...)`. Builders for
//!   `add/sub/mul/div/pow/neg` plus the transcendental family `sin/cos/tan/exp/ln/sqrt`.
//!   Python dunder operators map onto the same builders. `evaluate(env)`,
//!   `to_latex()`, `children()`, `is_constant()` round out the surface.
//! - `Integer` wraps `ArithmosInteger`. Construction from arbitrary-precision
//!   decimal strings (`Integer.from_str(s)`) and from i64. `value()` returns a
//!   true Python int reconstructed from the little-endian byte vector so callers
//!   never lose precision through f64.
//! - `Variable` wraps `ArithmosVariable`. The `binding` keyword accepts a
//!   `float`, an `Expression`, or `None` so callers can express bound, symbolic,
//!   or free variables in one constructor.
//!
//! LaTeX rendering is performed by [`expression_to_latex`] in this file because
//! the `Emit` trait in the expression module is currently a stub. Pushing the
//! LaTeX walker down into the Rust core was considered but every other consumer
//! (calculus / equation solver / matrix / tensor) already has its own emission
//! pathway and does not call `Emit::emit`. Centralising LaTeX in the facade
//! avoids breaking the wider crate's still-incomplete `Emit` trait contract.

use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyInt, PyList, PyString};

use crate::expression::{ArithmosBindings, ArithmosExpression, Evaluable};
use crate::function::ArithmosFunction;
use crate::integer::{flag, ArithmosInteger, ArithmosInternalInteger};
use crate::variable::{ArithmosVariable, ArithmosVariableValue};

// ============================================================================
// Module-level functions (retained from Wave 2 scaffold).
// ============================================================================

/// Returns the underlying Rust crate version. The Python facade matches this
/// against `arithma.__version__` to detect maturin/wheel/python desyncs.
#[pyfunction]
fn version_rust() -> &'static str {
    crate::version()
}

/// Sentinel that the Python facade can probe to confirm the Rust backend
/// loaded successfully (analogous to `_HAS_RUST = True` in pure-Python).
#[pyfunction]
fn is_rust_backend() -> bool {
    true
}

// ============================================================================
// Helpers.
// ============================================================================

/// Convert a Python object into an `ArithmosExpression`. Accepts ints, floats,
/// and existing `Expression` instances. Anything else is a `TypeError`.
fn coerce_to_expression(obj: &Bound<'_, PyAny>) -> PyResult<ArithmosExpression> {
    if let Ok(py_expr) = obj.extract::<PyRef<Expression>>() {
        return Ok(py_expr.inner.clone());
    }
    if let Ok(int_val) = obj.extract::<i64>() {
        return Ok(ArithmosExpression::from_i64(int_val));
    }
    if let Ok(float_val) = obj.extract::<f64>() {
        return Ok(ArithmosExpression::from_f64(float_val));
    }
    Err(PyTypeError::new_err(
        "Expected Expression, int, or float operand",
    ))
}

// ============================================================================
// LaTeX walker.
// ============================================================================

/// Render an `ArithmosExpression` as a LaTeX string. Walks the tree iteratively
/// to keep the safety-critical "avoid recursion" rule (CLAUDE.md §4) intact for
/// pathological inputs. Wraps subexpressions in `\left(\right)` when operator
/// precedence demands it.
fn expression_to_latex(expr: &ArithmosExpression) -> String {
    // Precedence ranking — higher binds tighter. Used to decide when to add
    // parentheses around a child of a binary operator. Matches standard math
    // typesetting conventions.
    fn precedence(expr: &ArithmosExpression) -> u8 {
        match expr {
            ArithmosExpression::Number(_)
            | ArithmosExpression::Constant { .. }
            | ArithmosExpression::Variable(_) => 100,
            ArithmosExpression::Function(func, _) => match func {
                ArithmosFunction::Add | ArithmosFunction::Subtract => 10,
                ArithmosFunction::Multiply | ArithmosFunction::Divide => 20,
                ArithmosFunction::Negate => 25,
                ArithmosFunction::Power | ArithmosFunction::Pow(_) => 30,
                _ => 90, // unary functions like sin/cos/exp don't need outer parens
            },
            _ => 50,
        }
    }

    fn wrap(child: &ArithmosExpression, min_prec: u8) -> String {
        let s = expression_to_latex(child);
        if precedence(child) < min_prec {
            format!("\\left({}\\right)", s)
        } else {
            s
        }
    }

    fn fmt_number(n: &ArithmosInteger) -> String {
        if n.value.is_nan() {
            return "\\text{NaN}".to_string();
        }
        if n.value.is_infinity() {
            return if n.value.is_negative() {
                "-\\infty".to_string()
            } else {
                "\\infty".to_string()
            };
        }
        // Try to render exactly via the byte vector → decimal string conversion.
        // For ordinary i64-range values this is exact; for larger we fall back to
        // the f64 conversion (which only loses precision past 2^53 anyway and the
        // Integer wrapper exposes the full byte vector for callers who care).
        match arithmos_integer_to_decimal_string(n) {
            Some(s) => s,
            None => format!("{}", n.to_f64()),
        }
    }

    match expr {
        ArithmosExpression::Number(n) => fmt_number(n),
        ArithmosExpression::Constant { symbol, .. } => match symbol.as_str() {
            "pi" => "\\pi".to_string(),
            "e" => "e".to_string(),
            "tau" => "\\tau".to_string(),
            "phi" => "\\varphi".to_string(),
            other => other.to_string(),
        },
        ArithmosExpression::Variable(name) => name.clone(),
        ArithmosExpression::Function(func, args) => {
            match func {
                ArithmosFunction::Add => {
                    if args.len() == 2 {
                        format!("{} + {}", wrap(&args[0], 10), wrap(&args[1], 10))
                    } else {
                        args.iter()
                            .map(|a| wrap(a, 10))
                            .collect::<Vec<_>>()
                            .join(" + ")
                    }
                }
                ArithmosFunction::Subtract => {
                    if args.len() == 2 {
                        format!("{} - {}", wrap(&args[0], 10), wrap(&args[1], 11))
                    } else {
                        // best-effort
                        args.iter()
                            .map(|a| wrap(a, 10))
                            .collect::<Vec<_>>()
                            .join(" - ")
                    }
                }
                ArithmosFunction::Multiply => {
                    if args.len() == 2 {
                        format!("{} \\cdot {}", wrap(&args[0], 20), wrap(&args[1], 20))
                    } else {
                        args.iter()
                            .map(|a| wrap(a, 20))
                            .collect::<Vec<_>>()
                            .join(" \\cdot ")
                    }
                }
                ArithmosFunction::Divide => {
                    if args.len() == 2 {
                        format!(
                            "\\frac{{{}}}{{{}}}",
                            expression_to_latex(&args[0]),
                            expression_to_latex(&args[1])
                        )
                    } else {
                        "\\text{div?}".to_string()
                    }
                }
                ArithmosFunction::Power => {
                    if args.len() == 2 {
                        format!(
                            "{}^{{{}}}",
                            wrap(&args[0], 31),
                            expression_to_latex(&args[1])
                        )
                    } else {
                        "\\text{pow?}".to_string()
                    }
                }
                ArithmosFunction::Negate => {
                    if args.len() == 1 {
                        format!("-{}", wrap(&args[0], 25))
                    } else {
                        "\\text{neg?}".to_string()
                    }
                }
                ArithmosFunction::Sqrt => {
                    if args.len() == 1 {
                        format!("\\sqrt{{{}}}", expression_to_latex(&args[0]))
                    } else {
                        "\\text{sqrt?}".to_string()
                    }
                }
                ArithmosFunction::Cbrt => {
                    if args.len() == 1 {
                        format!("\\sqrt[3]{{{}}}", expression_to_latex(&args[0]))
                    } else {
                        "\\text{cbrt?}".to_string()
                    }
                }
                ArithmosFunction::Exp => {
                    if args.len() == 1 {
                        format!("e^{{{}}}", expression_to_latex(&args[0]))
                    } else {
                        "\\text{exp?}".to_string()
                    }
                }
                ArithmosFunction::Ln => unary_function("\\ln", args),
                ArithmosFunction::Log => unary_function("\\log", args),
                ArithmosFunction::Log10 => unary_function("\\log_{10}", args),
                ArithmosFunction::Log2 => unary_function("\\log_{2}", args),
                ArithmosFunction::Sin => unary_function("\\sin", args),
                ArithmosFunction::Cos => unary_function("\\cos", args),
                ArithmosFunction::Tan => unary_function("\\tan", args),
                ArithmosFunction::Cot => unary_function("\\cot", args),
                ArithmosFunction::Sec => unary_function("\\sec", args),
                ArithmosFunction::Csc => unary_function("\\csc", args),
                ArithmosFunction::Asin => unary_function("\\arcsin", args),
                ArithmosFunction::Acos => unary_function("\\arccos", args),
                ArithmosFunction::Atan => unary_function("\\arctan", args),
                ArithmosFunction::Sinh => unary_function("\\sinh", args),
                ArithmosFunction::Cosh => unary_function("\\cosh", args),
                ArithmosFunction::Tanh => unary_function("\\tanh", args),
                ArithmosFunction::Abs => {
                    if args.len() == 1 {
                        format!("\\left|{}\\right|", expression_to_latex(&args[0]))
                    } else {
                        "\\text{abs?}".to_string()
                    }
                }
                other => {
                    // Generic fallback: function-name(args).
                    let inside = args
                        .iter()
                        .map(expression_to_latex)
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("\\operatorname{{{:?}}}\\left({}\\right)", other, inside)
                }
            }
        }
        ArithmosExpression::Sum {
            variable,
            start,
            end,
            expression,
        } => format!(
            "\\sum_{{{}={}}}^{{{}}} {}",
            variable,
            expression_to_latex(start),
            expression_to_latex(end),
            expression_to_latex(expression)
        ),
        ArithmosExpression::Product {
            variable,
            start,
            end,
            expression,
        } => format!(
            "\\prod_{{{}={}}}^{{{}}} {}",
            variable,
            expression_to_latex(start),
            expression_to_latex(end),
            expression_to_latex(expression)
        ),
        ArithmosExpression::Limit {
            variable,
            approaching,
            expression,
            ..
        } => format!(
            "\\lim_{{{} \\to {}}} {}",
            variable,
            expression_to_latex(approaching),
            expression_to_latex(expression)
        ),
        ArithmosExpression::Conditional {
            condition,
            then_expr,
            else_expr,
        } => format!(
            "\\begin{{cases}} {} & \\text{{if }} {} \\\\ {} & \\text{{otherwise}} \\end{{cases}}",
            expression_to_latex(then_expr),
            expression_to_latex(condition),
            expression_to_latex(else_expr)
        ),
        ArithmosExpression::CachedValue { expr, .. } => expression_to_latex(expr),
        ArithmosExpression::FourierOptimized { expr, .. } => expression_to_latex(expr),
    }
}

fn unary_function(name: &str, args: &[ArithmosExpression]) -> String {
    if args.len() == 1 {
        format!("{}\\left({}\\right)", name, expression_to_latex(&args[0]))
    } else {
        format!("{}\\left(?\\right)", name)
    }
}

// ============================================================================
// Arbitrary-precision integer ↔ Python int.
// ============================================================================

/// Convert an `ArithmosInteger`'s little-endian byte vector into a base-10
/// string. Returns `None` for NaN / infinity (the caller handles those).
fn arithmos_integer_to_decimal_string(n: &ArithmosInteger) -> Option<String> {
    if n.value.is_nan() || n.value.is_infinity() {
        return None;
    }
    // Long division by 10 over the little-endian byte vector. Bounded loop:
    // the byte vector has finite length, and each iteration reduces the
    // running magnitude by at least one digit when non-zero.
    let mut bytes: Vec<u8> = n.value.value.clone();
    let mut digits: Vec<u8> = Vec::with_capacity(bytes.len() * 3);
    // Guard against pathological inputs (vec of length 100k would still be
    // ~300k digits which is reasonable, but cap loop iterations to a generous
    // value tied to the byte vector length).
    let max_iters = bytes.len().saturating_mul(8).saturating_add(64);
    let mut iter_count: usize = 0;
    while !bytes.iter().all(|&b| b == 0) {
        iter_count += 1;
        if iter_count > max_iters * 4 {
            return None;
        }
        // Divide bytes (interpreted as little-endian big integer) by 10.
        let mut remainder: u32 = 0;
        for byte in bytes.iter_mut().rev() {
            let cur = remainder * 256 + (*byte as u32);
            *byte = (cur / 10) as u8;
            remainder = cur % 10;
        }
        digits.push(remainder as u8);
        // Trim trailing zero bytes for cheaper subsequent iterations.
        while bytes.len() > 1 && *bytes.last().unwrap() == 0 {
            bytes.pop();
        }
    }
    let mut out = String::with_capacity(digits.len() + 1);
    if n.value.is_negative() {
        out.push('-');
    }
    if digits.is_empty() {
        out.push('0');
    } else {
        for d in digits.iter().rev() {
            out.push(char::from(b'0' + d));
        }
    }
    Some(out)
}

/// Construct an `ArithmosInternalInteger` from a decimal string. Handles a
/// leading sign and rejects empty / non-digit inputs.
fn arithmos_integer_from_decimal_string(s: &str) -> Result<ArithmosInteger, String> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return Err("empty integer string".to_string());
    }
    let (negative, digits_str) = match trimmed.as_bytes()[0] {
        b'-' => (true, &trimmed[1..]),
        b'+' => (false, &trimmed[1..]),
        _ => (false, trimmed),
    };
    if digits_str.is_empty() || !digits_str.bytes().all(|b| b.is_ascii_digit()) {
        return Err(format!("invalid integer literal: {:?}", s));
    }
    // Build little-endian byte vector via repeated *10 + digit on a byte vec.
    let mut bytes: Vec<u8> = vec![0];
    for ch in digits_str.bytes() {
        let digit = (ch - b'0') as u32;
        let mut carry: u32 = digit;
        for byte in bytes.iter_mut() {
            let cur = (*byte as u32) * 10 + carry;
            *byte = (cur & 0xFF) as u8;
            carry = cur >> 8;
        }
        while carry > 0 {
            bytes.push((carry & 0xFF) as u8);
            carry >>= 8;
        }
    }
    // Strip leading zero bytes (least significant is at index 0; trailing in vec).
    while bytes.len() > 1 && *bytes.last().unwrap() == 0 {
        bytes.pop();
    }
    let mut flags: u16 = 0;
    if negative {
        // Don't set negative flag for "−0".
        if !(bytes.len() == 1 && bytes[0] == 0) {
            flags |= flag::NEGATIVE;
        }
    }
    Ok(ArithmosInteger {
        value: ArithmosInternalInteger {
            flags,
            value: bytes,
        },
        unit: None,
    })
}

// ============================================================================
// Compact form (JSON-friendly list-of-lists serialisation).
// ============================================================================
//
// Schema (mirrors EML-Math's tree.to_compact pattern — tagged Python lists):
//
//   ["num",   "<decimal>"]                 - exact integer literal (or "NaN" / "Inf" / "-Inf")
//   ["var",   "<name>"]                    - free variable
//   ["const", "<symbol>", value_or_None]   - named constant with optional cached f64
//   ["fn",    "<op>", *child_compacts]     - function application (Add, Sin, Exp, …)
//   ["sum",     "<var>", start, end, body]
//   ["product", "<var>", start, end, body]
//   ["limit",   "<var>", approaching, body, from_right_bool]
//   ["if",      cond, then_branch, else_branch]
//
// The format is intentionally lossless for the variants emitted by
// `formulas.json`'s `arithma_compact` payload and round-trips back to a live
// `Expression` for client-side evaluation. Variants with no natural compact
// form (CachedValue, FourierOptimized, Function carrying inner `ArithmosInteger`
// such as Pow(n) or LogBase(n), the calculus / numerical-method operators, …)
// are flattened: CachedValue / FourierOptimized are serialised as their wrapped
// expression; unsupported operators raise `ValueError` from `from_compact`.

/// Map an `ArithmosFunction` to its compact-form tag string. Returns `None` for
/// operator variants we cannot losslessly express in the compact schema (e.g.
/// `Pow(ArithmosInteger)`, calculus / numerical-method nodes that carry inner
/// expressions or per-variant payload). Callers that hit a `None` fall back to
/// serialising the operator's `Debug` name and emit a generic `"fn"` node — but
/// `from_compact` will refuse to inflate those.
fn function_tag(func: &ArithmosFunction) -> Option<&'static str> {
    match func {
        ArithmosFunction::Add => Some("add"),
        ArithmosFunction::Subtract => Some("sub"),
        ArithmosFunction::Multiply => Some("mul"),
        ArithmosFunction::Divide => Some("div"),
        ArithmosFunction::Power => Some("pow"),
        ArithmosFunction::Negate => Some("neg"),
        ArithmosFunction::Sqrt => Some("sqrt"),
        ArithmosFunction::Cbrt => Some("cbrt"),
        ArithmosFunction::Exp => Some("exp"),
        ArithmosFunction::Ln => Some("ln"),
        ArithmosFunction::Log => Some("log"),
        ArithmosFunction::Log10 => Some("log10"),
        ArithmosFunction::Log2 => Some("log2"),
        ArithmosFunction::Sin => Some("sin"),
        ArithmosFunction::Cos => Some("cos"),
        ArithmosFunction::Tan => Some("tan"),
        ArithmosFunction::Cot => Some("cot"),
        ArithmosFunction::Sec => Some("sec"),
        ArithmosFunction::Csc => Some("csc"),
        ArithmosFunction::Asin => Some("asin"),
        ArithmosFunction::Acos => Some("acos"),
        ArithmosFunction::Atan => Some("atan"),
        ArithmosFunction::Sinh => Some("sinh"),
        ArithmosFunction::Cosh => Some("cosh"),
        ArithmosFunction::Tanh => Some("tanh"),
        ArithmosFunction::Abs => Some("abs"),
        ArithmosFunction::Sign => Some("sign"),
        ArithmosFunction::Floor => Some("floor"),
        ArithmosFunction::Ceil => Some("ceil"),
        ArithmosFunction::Round => Some("round"),
        ArithmosFunction::Gamma => Some("gamma"),
        ArithmosFunction::Erf => Some("erf"),
        ArithmosFunction::Factorial => Some("factorial"),
        _ => None,
    }
}

/// Inverse of [`function_tag`] — map a compact tag back to its `ArithmosFunction`
/// variant. Variants requiring inner payload (e.g. `Pow(n)`, calculus operators,
/// `LogBase(n)`) are intentionally unsupported.
fn function_from_tag(tag: &str) -> Option<ArithmosFunction> {
    Some(match tag {
        "add" => ArithmosFunction::Add,
        "sub" => ArithmosFunction::Subtract,
        "mul" => ArithmosFunction::Multiply,
        "div" => ArithmosFunction::Divide,
        "pow" => ArithmosFunction::Power,
        "neg" => ArithmosFunction::Negate,
        "sqrt" => ArithmosFunction::Sqrt,
        "cbrt" => ArithmosFunction::Cbrt,
        "exp" => ArithmosFunction::Exp,
        "ln" => ArithmosFunction::Ln,
        "log" => ArithmosFunction::Log,
        "log10" => ArithmosFunction::Log10,
        "log2" => ArithmosFunction::Log2,
        "sin" => ArithmosFunction::Sin,
        "cos" => ArithmosFunction::Cos,
        "tan" => ArithmosFunction::Tan,
        "cot" => ArithmosFunction::Cot,
        "sec" => ArithmosFunction::Sec,
        "csc" => ArithmosFunction::Csc,
        "asin" => ArithmosFunction::Asin,
        "acos" => ArithmosFunction::Acos,
        "atan" => ArithmosFunction::Atan,
        "sinh" => ArithmosFunction::Sinh,
        "cosh" => ArithmosFunction::Cosh,
        "tanh" => ArithmosFunction::Tanh,
        "abs" => ArithmosFunction::Abs,
        "sign" => ArithmosFunction::Sign,
        "floor" => ArithmosFunction::Floor,
        "ceil" => ArithmosFunction::Ceil,
        "round" => ArithmosFunction::Round,
        "gamma" => ArithmosFunction::Gamma,
        "erf" => ArithmosFunction::Erf,
        "factorial" => ArithmosFunction::Factorial,
        _ => return None,
    })
}

/// Serialise an `ArithmosInteger` literal to its compact string form. Handles
/// NaN / ±Infinity as sentinel strings so the JSON round-trip is unambiguous.
fn integer_to_compact_string(n: &ArithmosInteger) -> String {
    if n.value.is_nan() {
        return "NaN".to_string();
    }
    if n.value.is_infinity() {
        return if n.value.is_negative() {
            "-Inf".to_string()
        } else {
            "Inf".to_string()
        };
    }
    arithmos_integer_to_decimal_string(n).unwrap_or_else(|| format!("{}", n.to_f64()))
}

/// Inverse of [`integer_to_compact_string`].
fn integer_from_compact_string(s: &str) -> Result<ArithmosInteger, String> {
    match s {
        "NaN" => Ok(ArithmosInteger::nan()),
        "Inf" | "+Inf" => Ok(ArithmosInteger::infinity()),
        "-Inf" => {
            let mut inf = ArithmosInteger::infinity();
            inf.value.set_negative(true);
            Ok(inf)
        }
        other => arithmos_integer_from_decimal_string(other),
    }
}

/// Build the compact-form Python list for an `ArithmosExpression`.
fn expression_to_compact_py(py: Python<'_>, expr: &ArithmosExpression) -> PyResult<PyObject> {
    match expr {
        ArithmosExpression::Number(n) => {
            let lst = PyList::new_bound(
                py,
                &[
                    "num".into_py(py),
                    integer_to_compact_string(n).into_py(py),
                ],
            );
            Ok(lst.into())
        }
        ArithmosExpression::Variable(name) => {
            let lst = PyList::new_bound(
                py,
                &["var".into_py(py), name.clone().into_py(py)],
            );
            Ok(lst.into())
        }
        ArithmosExpression::Constant {
            symbol,
            cached_value,
            ..
        } => {
            let val_obj: PyObject = match cached_value {
                Some(v) => v.into_py(py),
                None => py.None(),
            };
            let lst = PyList::new_bound(
                py,
                &["const".into_py(py), symbol.clone().into_py(py), val_obj],
            );
            Ok(lst.into())
        }
        ArithmosExpression::Function(func, args) => {
            let tag = function_tag(func).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "Expression.to_compact: operator {:?} is not supported by the compact schema",
                    func
                ))
            })?;
            let items = PyList::empty_bound(py);
            items.append("fn")?;
            items.append(tag)?;
            for a in args {
                items.append(expression_to_compact_py(py, a)?)?;
            }
            Ok(items.into())
        }
        ArithmosExpression::Sum {
            variable,
            start,
            end,
            expression,
        } => {
            let items = PyList::empty_bound(py);
            items.append("sum")?;
            items.append(variable.clone())?;
            items.append(expression_to_compact_py(py, start)?)?;
            items.append(expression_to_compact_py(py, end)?)?;
            items.append(expression_to_compact_py(py, expression)?)?;
            Ok(items.into())
        }
        ArithmosExpression::Product {
            variable,
            start,
            end,
            expression,
        } => {
            let items = PyList::empty_bound(py);
            items.append("product")?;
            items.append(variable.clone())?;
            items.append(expression_to_compact_py(py, start)?)?;
            items.append(expression_to_compact_py(py, end)?)?;
            items.append(expression_to_compact_py(py, expression)?)?;
            Ok(items.into())
        }
        ArithmosExpression::Limit {
            variable,
            approaching,
            expression,
            from_right,
        } => {
            let items = PyList::empty_bound(py);
            items.append("limit")?;
            items.append(variable.clone())?;
            items.append(expression_to_compact_py(py, approaching)?)?;
            items.append(expression_to_compact_py(py, expression)?)?;
            items.append(*from_right)?;
            Ok(items.into())
        }
        ArithmosExpression::Conditional {
            condition,
            then_expr,
            else_expr,
        } => {
            let items = PyList::empty_bound(py);
            items.append("if")?;
            items.append(expression_to_compact_py(py, condition)?)?;
            items.append(expression_to_compact_py(py, then_expr)?)?;
            items.append(expression_to_compact_py(py, else_expr)?)?;
            Ok(items.into())
        }
        // Performance-cache wrappers serialise as their underlying expression —
        // the cache state is rebuilt on demand by the engine.
        ArithmosExpression::CachedValue { expr, .. }
        | ArithmosExpression::FourierOptimized { expr, .. } => {
            expression_to_compact_py(py, expr)
        }
    }
}

/// Inverse of [`expression_to_compact_py`]. Walks a Python list/tuple matching
/// the compact schema and reconstructs the corresponding `ArithmosExpression`.
fn expression_from_compact_py(blob: &Bound<'_, PyAny>) -> PyResult<ArithmosExpression> {
    let lst: Bound<'_, PyList> = blob.downcast::<PyList>().map(|l| l.clone()).or_else(|_| {
        // Allow tuples too — JSON round-trip from Python typically yields lists,
        // but tuples are equally natural to write at the call site.
        let as_seq: Vec<Bound<'_, PyAny>> = blob.extract().map_err(|_| {
            PyTypeError::new_err(
                "Expression.from_compact expects a list or tuple at every level",
            )
        })?;
        Ok::<Bound<'_, PyList>, PyErr>(PyList::new_bound(blob.py(), &as_seq))
    })?;
    if lst.is_empty() {
        return Err(PyValueError::new_err(
            "Expression.from_compact: empty compact node",
        ));
    }
    let tag: String = lst.get_item(0)?.extract().map_err(|_| {
        PyValueError::new_err(
            "Expression.from_compact: first element of every compact node must be a tag string",
        )
    })?;
    match tag.as_str() {
        "num" => {
            if lst.len() != 2 {
                return Err(PyValueError::new_err(
                    "Expression.from_compact: 'num' node expects 2 elements",
                ));
            }
            let s: String = lst.get_item(1)?.extract()?;
            let n = integer_from_compact_string(&s).map_err(PyValueError::new_err)?;
            Ok(ArithmosExpression::Number(n))
        }
        "var" => {
            if lst.len() != 2 {
                return Err(PyValueError::new_err(
                    "Expression.from_compact: 'var' node expects 2 elements",
                ));
            }
            let name: String = lst.get_item(1)?.extract()?;
            Ok(ArithmosExpression::Variable(name))
        }
        "const" => {
            if lst.len() != 3 {
                return Err(PyValueError::new_err(
                    "Expression.from_compact: 'const' node expects 3 elements",
                ));
            }
            let symbol: String = lst.get_item(1)?.extract()?;
            let val_item = lst.get_item(2)?;
            let cached: Option<f64> = if val_item.is_none() {
                None
            } else {
                Some(val_item.extract()?)
            };
            Ok(ArithmosExpression::constant(&symbol, None, cached, true))
        }
        "fn" => {
            if lst.len() < 2 {
                return Err(PyValueError::new_err(
                    "Expression.from_compact: 'fn' node expects at least an operator tag",
                ));
            }
            let op_tag: String = lst.get_item(1)?.extract()?;
            let func = function_from_tag(&op_tag).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "Expression.from_compact: unsupported operator tag {:?}",
                    op_tag
                ))
            })?;
            let mut args: Vec<ArithmosExpression> = Vec::with_capacity(lst.len() - 2);
            for i in 2..lst.len() {
                args.push(expression_from_compact_py(&lst.get_item(i)?)?);
            }
            Ok(ArithmosExpression::Function(func, args))
        }
        "sum" => {
            if lst.len() != 5 {
                return Err(PyValueError::new_err(
                    "Expression.from_compact: 'sum' node expects 5 elements",
                ));
            }
            let variable: String = lst.get_item(1)?.extract()?;
            let start = expression_from_compact_py(&lst.get_item(2)?)?;
            let end = expression_from_compact_py(&lst.get_item(3)?)?;
            let body = expression_from_compact_py(&lst.get_item(4)?)?;
            Ok(ArithmosExpression::Sum {
                variable,
                start: Box::new(start),
                end: Box::new(end),
                expression: Box::new(body),
            })
        }
        "product" => {
            if lst.len() != 5 {
                return Err(PyValueError::new_err(
                    "Expression.from_compact: 'product' node expects 5 elements",
                ));
            }
            let variable: String = lst.get_item(1)?.extract()?;
            let start = expression_from_compact_py(&lst.get_item(2)?)?;
            let end = expression_from_compact_py(&lst.get_item(3)?)?;
            let body = expression_from_compact_py(&lst.get_item(4)?)?;
            Ok(ArithmosExpression::Product {
                variable,
                start: Box::new(start),
                end: Box::new(end),
                expression: Box::new(body),
            })
        }
        "limit" => {
            if lst.len() != 5 {
                return Err(PyValueError::new_err(
                    "Expression.from_compact: 'limit' node expects 5 elements",
                ));
            }
            let variable: String = lst.get_item(1)?.extract()?;
            let approaching = expression_from_compact_py(&lst.get_item(2)?)?;
            let body = expression_from_compact_py(&lst.get_item(3)?)?;
            let from_right: bool = lst.get_item(4)?.extract()?;
            Ok(ArithmosExpression::Limit {
                variable,
                approaching: Box::new(approaching),
                expression: Box::new(body),
                from_right,
            })
        }
        "if" => {
            if lst.len() != 4 {
                return Err(PyValueError::new_err(
                    "Expression.from_compact: 'if' node expects 4 elements",
                ));
            }
            let condition = expression_from_compact_py(&lst.get_item(1)?)?;
            let then_expr = expression_from_compact_py(&lst.get_item(2)?)?;
            let else_expr = expression_from_compact_py(&lst.get_item(3)?)?;
            Ok(ArithmosExpression::Conditional {
                condition: Box::new(condition),
                then_expr: Box::new(then_expr),
                else_expr: Box::new(else_expr),
            })
        }
        other => Err(PyValueError::new_err(format!(
            "Expression.from_compact: unknown node tag {:?}",
            other
        ))),
    }
}

// ============================================================================
// `Expression` pyclass.
// ============================================================================

/// Symbolic expression — Python wrapper around `ArithmosExpression`.
#[pyclass(name = "Expression", module = "arithma")]
#[derive(Clone)]
pub struct Expression {
    pub(crate) inner: ArithmosExpression,
}

impl Expression {
    fn from_inner(inner: ArithmosExpression) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl Expression {
    /// Construct an `Expression` wrapping a variable name.
    #[staticmethod]
    fn variable(name: &str) -> Self {
        Self::from_inner(ArithmosExpression::var(name))
    }

    /// Construct an `Expression` wrapping a numeric literal. Accepts `int` or
    /// `float`. Booleans (an `int` subclass in Python) are rejected to keep
    /// downstream type-checks honest.
    #[staticmethod]
    fn number(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        if value.is_instance_of::<pyo3::types::PyBool>() {
            return Err(PyTypeError::new_err(
                "Expression.number does not accept bool; use 0 or 1 explicitly",
            ));
        }
        if let Ok(int_val) = value.extract::<i64>() {
            return Ok(Self::from_inner(ArithmosExpression::from_i64(int_val)));
        }
        if let Ok(float_val) = value.extract::<f64>() {
            return Ok(Self::from_inner(ArithmosExpression::from_f64(float_val)));
        }
        Err(PyTypeError::new_err(
            "Expression.number expects int or float",
        ))
    }

    /// Construct a named symbolic constant (e.g. `pi`, `e`).
    #[staticmethod]
    #[pyo3(signature = (symbol, value = None))]
    fn constant(symbol: &str, value: Option<f64>) -> Self {
        Self::from_inner(ArithmosExpression::constant(symbol, None, value, true))
    }

    // -------- algebra builders (functional form) --------

    fn add(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = coerce_to_expression(other)?;
        Ok(Self::from_inner(ArithmosExpression::add(
            self.inner.clone(),
            rhs,
        )))
    }

    fn sub(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = coerce_to_expression(other)?;
        Ok(Self::from_inner(ArithmosExpression::sub(
            self.inner.clone(),
            rhs,
        )))
    }

    fn mul(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = coerce_to_expression(other)?;
        Ok(Self::from_inner(ArithmosExpression::mul(
            self.inner.clone(),
            rhs,
        )))
    }

    fn div(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = coerce_to_expression(other)?;
        Ok(Self::from_inner(ArithmosExpression::div(
            self.inner.clone(),
            rhs,
        )))
    }

    fn pow_(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = coerce_to_expression(other)?;
        Ok(Self::from_inner(ArithmosExpression::pow(
            self.inner.clone(),
            rhs,
        )))
    }

    fn neg(&self) -> Self {
        Self::from_inner(ArithmosExpression::neg(self.inner.clone()))
    }

    // -------- transcendentals --------

    #[staticmethod]
    fn sin(arg: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner = coerce_to_expression(arg)?;
        Ok(Self::from_inner(ArithmosExpression::sin(inner)))
    }

    #[staticmethod]
    fn cos(arg: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner = coerce_to_expression(arg)?;
        Ok(Self::from_inner(ArithmosExpression::cos(inner)))
    }

    #[staticmethod]
    fn tan(arg: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner = coerce_to_expression(arg)?;
        Ok(Self::from_inner(ArithmosExpression::tan(inner)))
    }

    #[staticmethod]
    fn exp(arg: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner = coerce_to_expression(arg)?;
        Ok(Self::from_inner(ArithmosExpression::exp(inner)))
    }

    #[staticmethod]
    fn ln(arg: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner = coerce_to_expression(arg)?;
        Ok(Self::from_inner(ArithmosExpression::ln(inner)))
    }

    #[staticmethod]
    fn sqrt(arg: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner = coerce_to_expression(arg)?;
        Ok(Self::from_inner(ArithmosExpression::sqrt(inner)))
    }

    // -------- Python operator dunder methods --------

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.add(other)
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let lhs = coerce_to_expression(other)?;
        Ok(Self::from_inner(ArithmosExpression::add(
            lhs,
            self.inner.clone(),
        )))
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.sub(other)
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let lhs = coerce_to_expression(other)?;
        Ok(Self::from_inner(ArithmosExpression::sub(
            lhs,
            self.inner.clone(),
        )))
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.mul(other)
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let lhs = coerce_to_expression(other)?;
        Ok(Self::from_inner(ArithmosExpression::mul(
            lhs,
            self.inner.clone(),
        )))
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.div(other)
    }

    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let lhs = coerce_to_expression(other)?;
        Ok(Self::from_inner(ArithmosExpression::div(
            lhs,
            self.inner.clone(),
        )))
    }

    /// `__pow__` accepts the modulo argument Python passes (always `None` for
    /// `Expression ** n`) and ignores it.
    fn __pow__(
        &self,
        other: &Bound<'_, PyAny>,
        _modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        self.pow_(other)
    }

    fn __rpow__(
        &self,
        other: &Bound<'_, PyAny>,
        _modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let lhs = coerce_to_expression(other)?;
        Ok(Self::from_inner(ArithmosExpression::pow(
            lhs,
            self.inner.clone(),
        )))
    }

    fn __neg__(&self) -> Self {
        self.neg()
    }

    // -------- evaluation, rendering, walking --------

    /// Evaluate this expression against a `{name: float}` binding dict.
    /// Raises `KeyError` for unbound variables and `RuntimeError` for any
    /// other failure (division by zero, unsupported operator, ...).
    fn evaluate(&self, env: &Bound<'_, PyDict>) -> PyResult<f64> {
        let mut bindings: ArithmosBindings = ArithmosBindings::new();
        for (key, val) in env.iter() {
            let k: String = key.extract().map_err(|_| {
                PyTypeError::new_err("evaluate: bindings keys must be str")
            })?;
            let v: f64 = val.extract().map_err(|_| {
                PyTypeError::new_err(format!("evaluate: binding {:?} must be number", k))
            })?;
            bindings.insert(k, v);
        }
        match self.inner.evaluate(&bindings) {
            Ok(v) => Ok(v),
            Err(msg) => {
                if msg.starts_with("unbound variable") {
                    Err(PyKeyError::new_err(msg))
                } else {
                    Err(PyRuntimeError::new_err(msg))
                }
            }
        }
    }

    /// Render this expression as a LaTeX string.
    fn to_latex(&self) -> String {
        expression_to_latex(&self.inner)
    }

    /// Return the immediate child sub-expressions (depth = 1). Useful for
    /// dependency-graph walking.
    fn children(&self) -> Vec<Expression> {
        match &self.inner {
            ArithmosExpression::Function(_, args) => {
                args.iter().map(|a| Expression::from_inner(a.clone())).collect()
            }
            ArithmosExpression::Sum {
                start,
                end,
                expression,
                ..
            }
            | ArithmosExpression::Product {
                start,
                end,
                expression,
                ..
            } => vec![
                Expression::from_inner(*start.clone()),
                Expression::from_inner(*end.clone()),
                Expression::from_inner(*expression.clone()),
            ],
            ArithmosExpression::Limit {
                approaching,
                expression,
                ..
            } => vec![
                Expression::from_inner(*approaching.clone()),
                Expression::from_inner(*expression.clone()),
            ],
            ArithmosExpression::Conditional {
                condition,
                then_expr,
                else_expr,
            } => vec![
                Expression::from_inner(*condition.clone()),
                Expression::from_inner(*then_expr.clone()),
                Expression::from_inner(*else_expr.clone()),
            ],
            ArithmosExpression::CachedValue { expr, .. }
            | ArithmosExpression::FourierOptimized { expr, .. } => {
                vec![Expression::from_inner(*expr.clone())]
            }
            // Leaf nodes have no children.
            ArithmosExpression::Number(_)
            | ArithmosExpression::Constant { .. }
            | ArithmosExpression::Variable(_) => Vec::new(),
        }
    }

    /// True if the expression is fully numeric (no free variables).
    fn is_constant(&self) -> bool {
        self.inner.is_constant()
    }

    /// Serialise this expression to its compact JSON-friendly list form.
    ///
    /// The compact schema mirrors EML-Math's tree compaction — every node is a
    /// tagged Python list, recursively built so the result is directly
    /// `json.dump`-able:
    ///
    /// - ``["num", "<decimal>"]`` — exact integer literal (NaN / ±Inf use the
    ///   sentinels ``"NaN"`` / ``"Inf"`` / ``"-Inf"``).
    /// - ``["var", "<name>"]`` — free variable.
    /// - ``["const", "<symbol>", value_or_None]`` — named constant.
    /// - ``["fn", "<op>", *child_compacts]`` — operator application; ``op`` is
    ///   ``"add"`` / ``"sub"`` / ``"mul"`` / ``"div"`` / ``"pow"`` / ``"neg"`` /
    ///   ``"sin"`` / ``"cos"`` / ``"tan"`` / ``"exp"`` / ``"ln"`` / ``"sqrt"`` /
    ///   ``"log"`` / ``"log10"`` / ``"log2"`` / ``"abs"`` / ``"sign"`` / …
    /// - ``["sum", "<var>", start, end, body]`` — bounded summation.
    /// - ``["product", "<var>", start, end, body]`` — bounded product.
    /// - ``["limit", "<var>", approaching, body, from_right_bool]`` — limit.
    /// - ``["if", cond, then, else]`` — conditional.
    ///
    /// Round-trips losslessly through :meth:`from_compact` for every variant
    /// listed above. Operator variants carrying inner expressions (e.g.
    /// ``Pow(n)``, ``LogBase(n)``, calculus / numerical-method nodes) raise
    /// :class:`ValueError`. Performance-cache wrappers (``CachedValue`` /
    /// ``FourierOptimized``) serialise as their underlying expression — the
    /// cache layer is rebuilt on demand by the engine.
    fn to_compact(&self, py: Python<'_>) -> PyResult<PyObject> {
        expression_to_compact_py(py, &self.inner)
    }

    /// Inflate a compact-form list (see :meth:`to_compact`) back into a live
    /// :class:`Expression`. Accepts the list/tuple structures emitted by
    /// :meth:`to_compact`, including the JSON round-trip form
    /// (``json.loads(json.dumps(expr.to_compact()))``).
    #[staticmethod]
    fn from_compact(blob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner = expression_from_compact_py(blob)?;
        Ok(Self::from_inner(inner))
    }

    /// Discriminator string useful for dispatch from Python ("number", "variable",
    /// "constant", "function:Sin", ...).
    fn kind(&self) -> String {
        match &self.inner {
            ArithmosExpression::Number(_) => "number".into(),
            ArithmosExpression::Variable(_) => "variable".into(),
            ArithmosExpression::Constant { .. } => "constant".into(),
            ArithmosExpression::Function(f, _) => format!("function:{:?}", f),
            ArithmosExpression::Sum { .. } => "sum".into(),
            ArithmosExpression::Product { .. } => "product".into(),
            ArithmosExpression::Limit { .. } => "limit".into(),
            ArithmosExpression::Conditional { .. } => "conditional".into(),
            ArithmosExpression::CachedValue { .. } => "cached".into(),
            ArithmosExpression::FourierOptimized { .. } => "fourier".into(),
        }
    }

    fn __repr__(&self) -> String {
        format!("Expression({})", self.to_latex())
    }
}

// ============================================================================
// `Integer` pyclass.
// ============================================================================

/// Arbitrary-precision integer — Python wrapper around `ArithmosInteger`.
#[pyclass(name = "Integer", module = "arithma")]
#[derive(Clone)]
pub struct Integer {
    pub(crate) inner: ArithmosInteger,
}

#[pymethods]
impl Integer {
    /// Construct from a Python int (any precision).
    #[new]
    fn new(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Prefer the string round-trip so we get arbitrary precision from
        // Python ints rather than going through i64.
        let py_int: Bound<'_, PyInt> = value
            .downcast::<PyInt>()
            .map_err(|_| PyTypeError::new_err("Integer() expects int"))?
            .clone();
        let s: String = py_int.str()?.to_str()?.to_string();
        let inner = arithmos_integer_from_decimal_string(&s).map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    /// Construct from a decimal string. Accepts an optional leading sign.
    #[staticmethod]
    fn from_str(s: &Bound<'_, PyString>) -> PyResult<Self> {
        let s_str: &str = s.to_str()?;
        let inner = arithmos_integer_from_decimal_string(s_str).map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    /// Construct from i64.
    #[staticmethod]
    fn from_i64(value: i64) -> Self {
        Self {
            inner: ArithmosInteger::from_i64(value),
        }
    }

    /// Return the value as a Python int (preserves arbitrary precision).
    fn value<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if self.inner.value.is_nan() {
            return Err(PyValueError::new_err("cannot convert NaN to int"));
        }
        if self.inner.value.is_infinity() {
            return Err(PyValueError::new_err("cannot convert infinity to int"));
        }
        let dec = arithmos_integer_to_decimal_string(&self.inner)
            .ok_or_else(|| PyRuntimeError::new_err("integer is not finite"))?;
        // Parse via Python's int() so arbitrary precision survives.
        let builtins = py.import_bound("builtins")?;
        let int_cls = builtins.getattr("int")?;
        int_cls.call1((dec,))
    }

    /// Best-effort conversion to f64. Loses precision past 2^53.
    fn to_float(&self) -> f64 {
        self.inner.to_f64()
    }

    /// Decimal string form.
    fn to_string(&self) -> String {
        arithmos_integer_to_decimal_string(&self.inner)
            .unwrap_or_else(|| format!("{}", self.inner.to_f64()))
    }

    fn __repr__(&self) -> String {
        format!("Integer({})", self.to_string())
    }

    fn __str__(&self) -> String {
        self.to_string()
    }

    fn __int__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.value(py)
    }
}

// ============================================================================
// `Variable` pyclass.
// ============================================================================

/// Named variable — Python wrapper around `ArithmosVariable`.
///
/// `binding` keyword accepts:
/// - `None` → free / unbound variable.
/// - `float` / `int` → numeric binding.
/// - `Expression` → symbolic binding.
#[pyclass(name = "Variable", module = "arithma")]
#[derive(Clone)]
pub struct Variable {
    pub(crate) inner: ArithmosVariable,
}

#[pymethods]
impl Variable {
    #[new]
    #[pyo3(signature = (name, binding = None))]
    fn new(name: &str, binding: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let mut v = ArithmosVariable::new(name);
        if let Some(b) = binding {
            if b.is_none() {
                // Explicit None → leave unbound.
            } else if let Ok(expr) = b.extract::<PyRef<Expression>>() {
                v = v.with_symbolic(expr.inner.clone());
            } else if let Ok(f) = b.extract::<f64>() {
                v = v.with_float(f);
            } else {
                return Err(PyTypeError::new_err(
                    "Variable binding must be None, float, int, or Expression",
                ));
            }
        }
        Ok(Self { inner: v })
    }

    /// Variable name.
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// True if the variable is unbound.
    fn is_unbound(&self) -> bool {
        self.inner.is_unbound()
    }

    /// Return the current binding as a Python object: `None`, a float, or an
    /// `Expression`.
    fn binding<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner.value {
            ArithmosVariableValue::Unbound => Ok(py.None().into_bound(py)),
            ArithmosVariableValue::Float(f) => Ok(f.into_py(py).into_bound(py)),
            ArithmosVariableValue::Symbolic(expr) => {
                let py_expr = Expression::from_inner(*expr.clone());
                Ok(Py::new(py, py_expr)?.into_bound(py).into_any())
            }
        }
    }

    /// Set the binding (same semantics as the constructor's `binding` keyword).
    #[pyo3(signature = (binding=None))]
    fn set_binding(&mut self, binding: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let mut v = ArithmosVariable::new(self.inner.name.clone());
        // Preserve unit/description.
        v.unit = self.inner.unit.clone();
        v.description = self.inner.description.clone();
        if let Some(b) = binding {
            if b.is_none() {
                // leave unbound
            } else if let Ok(expr) = b.extract::<PyRef<Expression>>() {
                v = v.with_symbolic(expr.inner.clone());
            } else if let Ok(f) = b.extract::<f64>() {
                v = v.with_float(f);
            } else {
                return Err(PyTypeError::new_err(
                    "Variable binding must be None, float, int, or Expression",
                ));
            }
        }
        self.inner = v;
        Ok(())
    }

    /// Lift the variable to an `Expression` (uses its name as a free symbol).
    fn to_expression(&self) -> Expression {
        Expression::from_inner(ArithmosExpression::var(&self.inner.name))
    }

    fn __repr__(&self) -> String {
        match &self.inner.value {
            ArithmosVariableValue::Unbound => format!("Variable({}, binding=None)", self.inner.name),
            ArithmosVariableValue::Float(f) => {
                format!("Variable({}, binding={})", self.inner.name, f)
            }
            ArithmosVariableValue::Symbolic(_) => {
                format!("Variable({}, binding=<Expression>)", self.inner.name)
            }
        }
    }
}

// ============================================================================
// Module entry point.
// ============================================================================

/// `arithma._arithma_core` module entry point. Maturin invokes this through
/// the `[tool.maturin] module-name` setting in `pyproject.toml`.
#[pymodule]
fn _arithma_core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version_rust, m)?)?;
    m.add_function(wrap_pyfunction!(is_rust_backend, m)?)?;
    m.add_class::<Expression>()?;
    m.add_class::<Integer>()?;
    m.add_class::<Variable>()?;
    Ok(())
}
