//====== Arithma/rust/arithma_core/src/pyfacade/calculus.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! PyO3 facade for [`crate::calculus`] — symbolic differentiation and integration.
//!
//! Surface
//! -------
//!
//! | Python | Rust |
//! |---|---|
//! | `differentiate(expr, var)` | [`crate::calculus::differentiation::differentiate`] |
//! | `differentiate_iterative(expr, var)` | [`crate::calculus::differentiation_iterative::differentiate_iterative`] |
//! | `differentiate_n(expr, var, order)` | repeated `differentiate`, bounded by [`MAX_DERIVATIVE_ORDER`] |
//! | `gradient(expr, variables)` | one `differentiate` per name in a bounded sequence |
//! | `diff_sum(a, b, var)` | [`crate::calculus::differentiation::diff_sum`] |
//! | `diff_product(a, b, var)` | [`crate::calculus::differentiation::diff_product`] |
//! | `diff_chain(outer, inner, var)` | [`crate::calculus::differentiation::diff_chain`] |
//! | `integrate(expr, var)` | [`crate::calculus::integration::integrate`] |
//! | `integrate_definite(expr, var, lo, hi)` | [`crate::calculus::integration::integrate_definite`] |
//! | `integrate_numeric(expr, var, lo, hi)` | [`crate::calculus::integration::integrate_numeric`] |
//!
//! Notes on the wrapped API
//! ------------------------
//!
//! - `diff_sum` / `diff_product` / `diff_chain` return a bare
//!   `ArithmaExpression` rather than a `Result`: internally they swallow a
//!   failed sub-differentiation and substitute `0`, and `diff_chain` also
//!   returns `0` when `outer` is not one of the unary functions it recognises.
//!   The facade cannot turn that into an exception without re-deriving the
//!   result, so these wrappers validate their inputs up front, reject the one
//!   case they can detect (a non-function `outer`), and document the `0`
//!   fallback. Callers who need failure to be loud should use `differentiate`,
//!   which does return a `Result`.
//! - `integrate` refuses — rather than guesses — anything outside linearity,
//!   the power rule and the standard table; that refusal surfaces as
//!   `ValueError`.
//! - `Expression::from_inner` in [`crate::pyfacade::core`] is private and that
//!   file is not ours to edit, so this module builds the wrapper through a
//!   struct literal in [`to_py_expression`]. That is legal because
//!   `Expression::inner` is `pub(crate)` and this module is in the same crate;
//!   no change to `core.rs` was required.

// ── Lint policy ──────────────────────────────────────────────────────────────
// Every `#[pyfunction]` returning `PyResult<T>` expands to a `PyErr -> PyErr`
// conversion inside the generated trampoline, which `clippy::useless_conversion`
// flags at the *return type* of our source function. There is nothing to remove
// at the call site — the conversion is not ours — so the lint is silenced here
// rather than left to fire once per exported function. Scoped to this module,
// per lib.rs's policy of justifying every exception where it is taken.
#![allow(clippy::useless_conversion)]

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PySequence, PyString};

use crate::calculus::differentiation::{
    diff_chain as rust_diff_chain, diff_product as rust_diff_product, diff_sum as rust_diff_sum,
    differentiate as rust_differentiate,
};
use crate::calculus::differentiation_iterative::differentiate_iterative as rust_differentiate_iterative;
use crate::calculus::integration::{
    integrate as rust_integrate, integrate_definite as rust_integrate_definite,
    integrate_numeric as rust_integrate_numeric,
};
use crate::expression::ArithmaExpression;
use crate::pyfacade::core::Expression;
use crate::pyfacade::MAX_SEQUENCE_LEN;

// ============================================================================
// Bounds.
// ============================================================================

/// Upper bound on the byte length of a differentiation / integration variable
/// name. Names are identifiers, not payloads; anything longer is a bug or an
/// attack. The bound also fixes the iteration count of the character scan in
/// [`check_variable_name`].
const MAX_VARIABLE_NAME_LEN: usize = 256;

/// Upper bound on the `order` accepted by `differentiate_n`. Each order costs a
/// full traversal of a tree that grows quickly under repeated differentiation,
/// so the repeat loop gets a fixed bound like every other loop here.
const MAX_DERIVATIVE_ORDER: usize = 64;

/// Pre-allocation cap for result vectors. The sequence length is already
/// bounded by [`MAX_SEQUENCE_LEN`], but reserving a million slots up front for
/// a caller who then supplies a bad element would hand a hostile input a large
/// allocation for free.
const RESULT_PREALLOC_CAP: usize = 64;

// ============================================================================
// Pure validation helpers (unit-tested below without a Python interpreter).
// ============================================================================

/// Validate a variable name arriving from Python. `str` guarantees UTF-8 and
/// nothing else — emptiness, padding and absurd lengths all have to be checked
/// here, because the Rust layer only `debug_assert!`s them and a release build
/// would happily differentiate with respect to `""`.
fn check_variable_name(var: &str) -> Result<(), String> {
    if var.is_empty() {
        return Err("variable name must not be empty".to_string());
    }
    if var.len() > MAX_VARIABLE_NAME_LEN {
        return Err(format!(
            "variable name is {} bytes, exceeding the {MAX_VARIABLE_NAME_LEN}-byte limit",
            var.len()
        ));
    }
    // Bounded by the length check immediately above.
    if var.chars().any(char::is_whitespace) {
        return Err(format!("variable name {var:?} must not contain whitespace"));
    }
    Ok(())
}

/// Validate the repeat count for `differentiate_n`.
fn check_derivative_order(order: usize) -> Result<(), String> {
    if order > MAX_DERIVATIVE_ORDER {
        return Err(format!(
            "derivative order {order} exceeds the maximum of {MAX_DERIVATIVE_ORDER}"
        ));
    }
    Ok(())
}

/// Validate the limits handed to numeric quadrature. The Rust routine rejects
/// non-finite limits too, but catching it here lets the message name the
/// offending side.
fn check_finite_limits(lo: f64, hi: f64) -> Result<(), String> {
    if !lo.is_finite() {
        return Err(format!("lower limit must be finite, got {lo}"));
    }
    if !hi.is_finite() {
        return Err(format!("upper limit must be finite, got {hi}"));
    }
    Ok(())
}

/// Check a caller-supplied sequence length against [`MAX_SEQUENCE_LEN`] before
/// any iteration over it begins.
fn check_sequence_len(len: usize, what: &str) -> Result<(), String> {
    if len > MAX_SEQUENCE_LEN {
        return Err(format!(
            "{what}: sequence of {len} elements exceeds the maximum of {MAX_SEQUENCE_LEN}"
        ));
    }
    Ok(())
}

// ============================================================================
// Python <-> Rust conversion.
// ============================================================================

/// Wrap an `ArithmaExpression` in the `Expression` pyclass.
///
/// `Expression::from_inner` is private to `core.rs`, so we use the struct
/// literal: `Expression::inner` is `pub(crate)` and `pyfacade::calculus` lives
/// in the same crate.
fn to_py_expression(inner: ArithmaExpression) -> Expression {
    let wrapped = Expression { inner };
    debug_assert!(
        !matches!(&wrapped.inner, ArithmaExpression::Variable(n) if n.is_empty()),
        "to_py_expression: wrapped a variable with an empty name"
    );
    wrapped
}

/// Convert a Python object into an `ArithmaExpression`. Accepts `Expression`,
/// `int` and `float`; `bool` is rejected so `True` cannot silently become `1`.
fn coerce_expression(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<ArithmaExpression> {
    if obj.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(format!(
            "{what}: bool is not a valid operand; pass 0 or 1 explicitly"
        )));
    }
    if let Ok(py_expr) = obj.extract::<PyRef<Expression>>() {
        return Ok(py_expr.inner.clone());
    }
    if let Ok(int_val) = obj.extract::<i64>() {
        return Ok(ArithmaExpression::from_i64(int_val));
    }
    if let Ok(float_val) = obj.extract::<f64>() {
        return Ok(ArithmaExpression::from_f64(float_val));
    }
    Err(PyTypeError::new_err(format!(
        "{what}: expected Expression, int, or float"
    )))
}

/// Validate a variable name and surface a failure as `ValueError`.
fn variable_name_or_err(var: &str, what: &str) -> PyResult<()> {
    check_variable_name(var).map_err(|e| PyValueError::new_err(format!("{what}: {e}")))
}

// ============================================================================
// Differentiation.
// ============================================================================

/// Differentiate `expr` with respect to `var`, returning the symbolic
/// derivative.
///
/// `expr` may be an :class:`Expression`, an ``int`` or a ``float``. Raises
/// :class:`ValueError` for a malformed variable name, :class:`TypeError` for an
/// operand that cannot be coerced, and :class:`RuntimeError` if the
/// differentiator itself fails (for example when its node cap is hit).
#[pyfunction]
#[pyo3(signature = (expr, var))]
fn differentiate(expr: &Bound<'_, PyAny>, var: &str) -> PyResult<Expression> {
    variable_name_or_err(var, "differentiate")?;
    let inner = coerce_expression(expr, "differentiate")?;
    debug_assert!(!var.is_empty(), "differentiate: empty variable name");
    debug_assert!(
        var.len() <= MAX_VARIABLE_NAME_LEN,
        "differentiate: oversized variable name"
    );
    let out = rust_differentiate(&inner, var).map_err(PyRuntimeError::new_err)?;
    debug_assert!(
        !inner.is_constant() || out.is_constant(),
        "differentiate: derivative of a constant expression is not constant"
    );
    Ok(to_py_expression(out))
}

/// Differentiate using the explicit work-stack implementation.
///
/// Produces the same result as :func:`differentiate` (which routes here
/// internally) but is named so callers can pin the stack-safe traversal for
/// very deep trees.
#[pyfunction]
#[pyo3(signature = (expr, var))]
fn differentiate_iterative(expr: &Bound<'_, PyAny>, var: &str) -> PyResult<Expression> {
    variable_name_or_err(var, "differentiate_iterative")?;
    let inner = coerce_expression(expr, "differentiate_iterative")?;
    debug_assert!(
        !var.is_empty(),
        "differentiate_iterative: empty variable name"
    );
    debug_assert!(
        var.len() <= MAX_VARIABLE_NAME_LEN,
        "differentiate_iterative: oversized variable name"
    );
    let out = rust_differentiate_iterative(&inner, var).map_err(PyRuntimeError::new_err)?;
    debug_assert!(
        !inner.is_constant() || out.is_constant(),
        "differentiate_iterative: derivative of a constant expression is not constant"
    );
    Ok(to_py_expression(out))
}

/// Take the `order`-th derivative of `expr` with respect to `var`.
///
/// ``order=0`` returns the expression unchanged. ``order`` is capped at 64
/// because each step is a full traversal over a tree that grows quickly.
/// Raises :class:`ValueError` if the cap is exceeded and :class:`RuntimeError`
/// naming the failing step if any single differentiation fails.
#[pyfunction]
#[pyo3(signature = (expr, var, order = 1))]
fn differentiate_n(expr: &Bound<'_, PyAny>, var: &str, order: usize) -> PyResult<Expression> {
    variable_name_or_err(var, "differentiate_n")?;
    check_derivative_order(order)
        .map_err(|e| PyValueError::new_err(format!("differentiate_n: {e}")))?;
    let mut current = coerce_expression(expr, "differentiate_n")?;
    debug_assert!(!var.is_empty(), "differentiate_n: empty variable name");
    debug_assert!(
        order <= MAX_DERIVATIVE_ORDER,
        "differentiate_n: order passed validation but exceeds the cap"
    );
    // Fixed bound: `order` was checked against MAX_DERIVATIVE_ORDER above.
    for step in 0..order {
        current = rust_differentiate(&current, var).map_err(|e| {
            PyRuntimeError::new_err(format!(
                "differentiate_n: derivative {} of {order} failed: {e}",
                step + 1
            ))
        })?;
    }
    Ok(to_py_expression(current))
}

/// Partial derivatives of `expr` with respect to each name in `variables`.
///
/// `variables` must be a non-empty list/tuple of ``str``; a bare ``str`` is
/// rejected rather than iterated character by character. The sequence length is
/// checked against ``MAX_SEQUENCE_LEN`` before any iteration begins. Returns a
/// list of :class:`Expression` in the same order as the input.
#[pyfunction]
#[pyo3(signature = (expr, variables))]
fn gradient(expr: &Bound<'_, PyAny>, variables: &Bound<'_, PyAny>) -> PyResult<Vec<Expression>> {
    if variables.is_instance_of::<PyString>() {
        return Err(PyTypeError::new_err(
            "gradient: variables must be a sequence of str, not a single str",
        ));
    }
    let seq = variables
        .downcast::<PySequence>()
        .map_err(|_| PyTypeError::new_err("gradient: variables must be a list or tuple of str"))?;
    let len = seq.len()?;
    check_sequence_len(len, "gradient").map_err(PyValueError::new_err)?;
    if len == 0 {
        return Err(PyValueError::new_err(
            "gradient: variables must not be empty",
        ));
    }
    let inner = coerce_expression(expr, "gradient")?;
    debug_assert!(len > 0, "gradient: empty sequence reached the walk");
    debug_assert!(
        len <= MAX_SEQUENCE_LEN,
        "gradient: sequence passed validation but exceeds the cap"
    );
    let mut out: Vec<Expression> = Vec::with_capacity(len.min(RESULT_PREALLOC_CAP));
    // Fixed bound: `len` was checked against MAX_SEQUENCE_LEN above.
    for i in 0..len {
        let name: String = seq
            .get_item(i)?
            .extract()
            .map_err(|_| PyTypeError::new_err(format!("gradient: variables[{i}] must be str")))?;
        variable_name_or_err(&name, &format!("gradient: variables[{i}]"))?;
        let d = rust_differentiate(&inner, &name)
            .map_err(|e| PyRuntimeError::new_err(format!("gradient: d/d{name} failed: {e}")))?;
        out.push(to_py_expression(d));
    }
    debug_assert_eq!(out.len(), len, "gradient: produced the wrong arity");
    Ok(out)
}

/// Sum rule: ``d/dvar (a + b)`` assembled as ``da + db``.
///
/// Mirrors the Rust helper exactly, including its failure mode: a sub-term the
/// differentiator cannot handle contributes ``0`` instead of raising. Use
/// :func:`differentiate` on ``a + b`` if that needs to be an error.
#[pyfunction]
#[pyo3(signature = (a, b, var))]
fn diff_sum(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>, var: &str) -> PyResult<Expression> {
    variable_name_or_err(var, "diff_sum")?;
    let lhs = coerce_expression(a, "diff_sum (a)")?;
    let rhs = coerce_expression(b, "diff_sum (b)")?;
    debug_assert!(!var.is_empty(), "diff_sum: empty variable name");
    let out = rust_diff_sum(&lhs, &rhs, var);
    debug_assert!(
        !lhs.is_constant() || !rhs.is_constant() || out.is_constant(),
        "diff_sum: derivative of two constants is not constant"
    );
    Ok(to_py_expression(out))
}

/// Product rule: ``d/dvar (a * b)`` assembled as ``da*b + a*db``.
///
/// Shares the ``0``-on-failure behaviour documented on :func:`diff_sum`.
#[pyfunction]
#[pyo3(signature = (a, b, var))]
fn diff_product(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>, var: &str) -> PyResult<Expression> {
    variable_name_or_err(var, "diff_product")?;
    let lhs = coerce_expression(a, "diff_product (a)")?;
    let rhs = coerce_expression(b, "diff_product (b)")?;
    debug_assert!(!var.is_empty(), "diff_product: empty variable name");
    let out = rust_diff_product(&lhs, &rhs, var);
    debug_assert!(
        !lhs.is_constant() || !rhs.is_constant() || out.is_constant(),
        "diff_product: derivative of two constants is not constant"
    );
    Ok(to_py_expression(out))
}

/// Chain rule: rebuild `outer` around `inner`, then differentiate the composite.
///
/// `outer` must be an :class:`Expression` whose root is one of ``sin``,
/// ``cos``, ``tan``, ``exp``, ``ln``, ``sqrt`` or unary negation. The Rust
/// helper answers ``0`` for anything else, so this wrapper raises
/// :class:`TypeError` for the one case it can detect up front — a non-function
/// `outer`. A function root outside the recognised set still yields ``0``.
#[pyfunction]
#[pyo3(signature = (outer, inner, var))]
fn diff_chain(
    outer: &Bound<'_, PyAny>,
    inner: &Bound<'_, PyAny>,
    var: &str,
) -> PyResult<Expression> {
    variable_name_or_err(var, "diff_chain")?;
    let outer_expr = coerce_expression(outer, "diff_chain (outer)")?;
    let inner_expr = coerce_expression(inner, "diff_chain (inner)")?;
    if !matches!(outer_expr, ArithmaExpression::Function(_, _)) {
        return Err(PyTypeError::new_err(
            "diff_chain: outer must be a function application (sin, cos, tan, exp, ln, sqrt, neg)",
        ));
    }
    debug_assert!(!var.is_empty(), "diff_chain: empty variable name");
    debug_assert!(
        matches!(outer_expr, ArithmaExpression::Function(_, _)),
        "diff_chain: outer is not a function application"
    );
    Ok(to_py_expression(rust_diff_chain(
        &outer_expr,
        &inner_expr,
        var,
    )))
}

// ============================================================================
// Integration.
// ============================================================================

/// Indefinite integral of `expr` with respect to `var`, without the constant of
/// integration.
///
/// Covers linearity, constant multiples, the power rule, ``1/x`` and the
/// standard table (``exp``, ``sin``, ``cos``, ``sqrt``). Anything else raises
/// :class:`ValueError` — the engine refuses rather than returning a wrong
/// antiderivative.
#[pyfunction]
#[pyo3(signature = (expr, var))]
fn integrate(expr: &Bound<'_, PyAny>, var: &str) -> PyResult<Expression> {
    variable_name_or_err(var, "integrate")?;
    let inner = coerce_expression(expr, "integrate")?;
    debug_assert!(!var.is_empty(), "integrate: empty variable name");
    debug_assert!(
        var.len() <= MAX_VARIABLE_NAME_LEN,
        "integrate: oversized variable name"
    );
    let out = rust_integrate(&inner, var)
        .map_err(|e| PyValueError::new_err(format!("integrate: {e}")))?;
    Ok(to_py_expression(out))
}

/// Definite integral of `expr` over ``[lo, hi]``.
///
/// Evaluates the antiderivative at both limits. When no symbolic antiderivative
/// exists and both limits are numeric, falls back to Simpson quadrature and
/// returns the value as a literal. `lo` and `hi` accept :class:`Expression`,
/// ``int`` or ``float``. Raises :class:`ValueError` when neither path applies,
/// or when a limit reduces to a non-finite number.
#[pyfunction]
#[pyo3(signature = (expr, var, lo, hi))]
fn integrate_definite(
    expr: &Bound<'_, PyAny>,
    var: &str,
    lo: &Bound<'_, PyAny>,
    hi: &Bound<'_, PyAny>,
) -> PyResult<Expression> {
    variable_name_or_err(var, "integrate_definite")?;
    let inner = coerce_expression(expr, "integrate_definite")?;
    let lo_expr = coerce_expression(lo, "integrate_definite (lo)")?;
    let hi_expr = coerce_expression(hi, "integrate_definite (hi)")?;
    debug_assert!(!var.is_empty(), "integrate_definite: empty variable name");
    // A limit that reduces to a number must reduce to a *finite* number, or the
    // numeric fallback would integrate over an infinite interval. Fixed bound:
    // exactly two limits.
    for (label, limit) in [("lo", &lo_expr), ("hi", &hi_expr)] {
        if let Some(v) = limit.to_f64() {
            if !v.is_finite() {
                return Err(PyValueError::new_err(format!(
                    "integrate_definite: {label} limit must be finite, got {v}"
                )));
            }
        }
    }
    debug_assert!(
        lo_expr.to_f64().map(f64::is_finite).unwrap_or(true),
        "integrate_definite: non-finite lower limit survived validation"
    );
    let out = rust_integrate_definite(&inner, var, &lo_expr, &hi_expr)
        .map_err(|e| PyValueError::new_err(format!("integrate_definite: {e}")))?;
    Ok(to_py_expression(out))
}

/// Composite Simpson quadrature over ``[lo, hi]``, returning a ``float``.
///
/// Both limits must be finite. A sample point where the integrand fails to
/// evaluate, or is non-finite, raises :class:`ValueError` rather than being
/// skipped. Integrating backwards (``lo > hi``) negates the result, and
/// ``lo == hi`` returns ``0.0``.
#[pyfunction]
#[pyo3(signature = (expr, var, lo, hi))]
fn integrate_numeric(expr: &Bound<'_, PyAny>, var: &str, lo: f64, hi: f64) -> PyResult<f64> {
    variable_name_or_err(var, "integrate_numeric")?;
    check_finite_limits(lo, hi)
        .map_err(|e| PyValueError::new_err(format!("integrate_numeric: {e}")))?;
    let inner = coerce_expression(expr, "integrate_numeric")?;
    debug_assert!(!var.is_empty(), "integrate_numeric: empty variable name");
    debug_assert!(
        lo.is_finite() && hi.is_finite(),
        "integrate_numeric: limits passed validation but are not finite"
    );
    let value = rust_integrate_numeric(&inner, var, lo, hi)
        .map_err(|e| PyValueError::new_err(format!("integrate_numeric: {e}")))?;
    if !value.is_finite() {
        return Err(PyRuntimeError::new_err(format!(
            "integrate_numeric: quadrature produced a non-finite total ({value})"
        )));
    }
    Ok(value)
}

// ============================================================================
// Registration
// ============================================================================

/// Register this module's surface on the extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(differentiate, m)?)?;
    m.add_function(wrap_pyfunction!(differentiate_iterative, m)?)?;
    m.add_function(wrap_pyfunction!(differentiate_n, m)?)?;
    m.add_function(wrap_pyfunction!(gradient, m)?)?;
    m.add_function(wrap_pyfunction!(diff_sum, m)?)?;
    m.add_function(wrap_pyfunction!(diff_product, m)?)?;
    m.add_function(wrap_pyfunction!(diff_chain, m)?)?;
    m.add_function(wrap_pyfunction!(integrate, m)?)?;
    m.add_function(wrap_pyfunction!(integrate_definite, m)?)?;
    m.add_function(wrap_pyfunction!(integrate_numeric, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variable_name_rejects_empty() {
        assert!(check_variable_name("").is_err());
    }

    #[test]
    fn variable_name_accepts_ordinary_identifiers() {
        assert!(check_variable_name("x").is_ok());
        assert!(check_variable_name("theta_1").is_ok());
        // Non-ASCII identifiers are legitimate in this engine (pi, phi, ...).
        assert!(check_variable_name("\u{3b8}").is_ok());
    }

    #[test]
    fn variable_name_rejects_whitespace() {
        assert!(check_variable_name(" x").is_err());
        assert!(check_variable_name("x ").is_err());
        assert!(check_variable_name("a b").is_err());
        assert!(check_variable_name("\t").is_err());
    }

    #[test]
    fn variable_name_rejects_oversized() {
        let ok = "v".repeat(MAX_VARIABLE_NAME_LEN);
        assert!(check_variable_name(&ok).is_ok());
        let too_long = "v".repeat(MAX_VARIABLE_NAME_LEN + 1);
        assert!(check_variable_name(&too_long).is_err());
    }

    #[test]
    fn derivative_order_is_capped() {
        assert!(check_derivative_order(0).is_ok());
        assert!(check_derivative_order(1).is_ok());
        assert!(check_derivative_order(MAX_DERIVATIVE_ORDER).is_ok());
        assert!(check_derivative_order(MAX_DERIVATIVE_ORDER + 1).is_err());
    }

    #[test]
    fn limits_must_be_finite() {
        assert!(check_finite_limits(0.0, 1.0).is_ok());
        assert!(check_finite_limits(f64::NAN, 1.0).is_err());
        assert!(check_finite_limits(0.0, f64::INFINITY).is_err());
        assert!(check_finite_limits(f64::NEG_INFINITY, 0.0).is_err());
    }

    #[test]
    fn sequence_len_is_capped() {
        assert!(check_sequence_len(0, "t").is_ok());
        assert!(check_sequence_len(MAX_SEQUENCE_LEN, "t").is_ok());
        assert!(check_sequence_len(MAX_SEQUENCE_LEN + 1, "t").is_err());
    }

    #[test]
    fn sequence_len_error_names_the_caller() {
        let err = check_sequence_len(MAX_SEQUENCE_LEN + 1, "gradient").unwrap_err();
        assert!(err.contains("gradient"), "message was {err:?}");
    }

    #[test]
    fn to_py_expression_preserves_the_tree() {
        let e = ArithmaExpression::add(ArithmaExpression::var("x"), ArithmaExpression::from_i64(3));
        let wrapped = to_py_expression(e.clone());
        assert_eq!(format!("{:?}", wrapped.inner), format!("{e:?}"));
    }
}
