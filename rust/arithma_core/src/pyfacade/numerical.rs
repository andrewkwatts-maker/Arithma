//====== Arithma/rust/arithma_core/src/pyfacade/numerical.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! PyO3 facade for [`crate::numerical`] — root finding, critical points and
//! interval arithmetic.
//!
//! Surface
//! -------
//!
//! | Python | Rust |
//! |---|---|
//! | `find_root_bisection(expr, var, lo, hi, tol, max_iterations)` | [`crate::numerical::root_finding::find_root_bisection`] |
//! | `find_root_newton_raphson(expr, var, initial, tol, max_iterations)` | [`crate::numerical::root_finding::find_root_newton_raphson`] |
//! | `find_root_secant(expr, var, x0, x1, tol, max_iterations)` | [`crate::numerical::root_finding::find_root_secant`] |
//! | `solve_with_method(expr, var, initial, method)` | [`crate::numerical::methods::solve_with_method`] |
//! | `find_stationary_points(expr, var, lo, hi, ...)` | `ArithmaCriticalPoints::find_stationary_points` |
//! | `find_inflection_points(expr, var, lo, hi, ...)` | `ArithmaCriticalPoints::find_inflection_points` |
//! | `find_extrema(expr, var, lo, hi, ...)` | `ArithmaCriticalPoints::find_extrema` |
//! | `analyze_point(expr, var, point, ...)` | `ArithmaCriticalPoints::analyze_point` |
//! | `classify_point(expr, var, point, ...)` | `ArithmaCriticalPoints::classify_point` |
//! | `analyze_intervals(expr, var, lo, hi, ...)` | `ArithmaCriticalPoints::analyze_intervals` |
//! | `evaluate_interval(expr, var, interval)` | [`crate::numerical::interval_analysis::evaluate_interval`] |
//! | `RootResult`, `CriticalPoint`, `FunctionAnalysis`, `Interval` | result types |
//!
//! Notes on the wrapped API
//! ------------------------
//!
//! - **Non-convergence raises.** The Rust root finders return `Ok` with
//!   `converged: false` and their best estimate when the iteration budget runs
//!   out, on the reasoning that the caller usually wants the estimate anyway.
//!   That is the wrong default across an FFI boundary, where a returned float
//!   reads as an answer, so this facade turns it into `RuntimeError`. The
//!   `converged` attribute on a returned :class:`RootResult` is therefore
//!   always ``True``; it is kept because it is part of the underlying contract
//!   and reads honestly in a repr.
//! - **`solve_with_method` caps differ.** That dispatcher carries its own fixed
//!   budget (512 iterations, 96 bracket doublings) with no caller override, so
//!   `tol` and `max_iterations` are not offered there. Its Newton and secant
//!   branches error on an exhausted budget; its bisection and Brent branches
//!   return the current midpoint instead, which the module doc-comment claims
//!   is an error. Reaching that branch needs a bracket that 512 halvings cannot
//!   resolve to a relative `1e-14`, which is unreachable for a finite bracket,
//!   so the mismatch is recorded here rather than worked around.
//! - **The configuration structs are keyword arguments**, not pyclasses.
//!   `ArithmaRootFindingConfig` has two fields and
//!   `ArithmaCriticalPointsConfig` four; a separate object for that is
//!   ceremony. The defaults are asserted against the Rust `Default` impls in
//!   the tests below, so the two cannot drift apart unnoticed.
//! - **`Interval` diverges from `ArithmaInterval` on NaN.** `hull` widens to
//!   the whole real line when it sees a NaN, and `ArithmaInterval::new` accepts
//!   any pair. Silently handing back `(-inf, inf)` for a caller arithmetic slip
//!   is exactly the "looks healthy while it is not" failure this facade exists
//!   to prevent, so NaN is rejected at the boundary.
//! - `Expression::from_inner` in [`crate::pyfacade::core`] is private, so
//!   expressions are read through the `pub(crate)` `inner` field, as
//!   `calculus.rs` and `linalg.rs` do.

// ── Lint policy ──────────────────────────────────────────────────────────────
// `useless_conversion`: pyo3 0.22 expands `-> PyResult<T>` into a `PyErr::from`
// round-trip that clippy attributes to *our* signature span. Nothing at the
// call site can be removed. Scoped here rather than widened in lib.rs, matching
// `calculus.rs` / `geometry.rs` / `linalg.rs`.
#![allow(clippy::useless_conversion)]
// `too_many_arguments`: the critical-point entry points take an expression, a
// variable, a search range and the four `ArithmaCriticalPointsConfig` fields as
// keyword arguments. The arity *is* the configuration's field count, exposed
// deliberately so Python callers get named defaults instead of a builder
// object; it is not incidental complexity that a refactor would remove.
#![allow(clippy::too_many_arguments)]

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PySequence, PyString};

use crate::expression::ArithmaExpression;
use crate::numerical::critical_points::{
    ArithmaCriticalPoint, ArithmaCriticalPointKind, ArithmaCriticalPoints,
    ArithmaCriticalPointsConfig, ArithmaSearchRange,
};
use crate::numerical::interval_analysis::{
    evaluate_interval as rust_evaluate_interval, ArithmaInterval,
};
use crate::numerical::methods::{
    solve_with_method as rust_solve_with_method, ArithmaNumericalMethod,
};
use crate::numerical::root_finding::{
    find_root_bisection as rust_find_root_bisection,
    find_root_newton_raphson as rust_find_root_newton_raphson,
    find_root_secant as rust_find_root_secant, ArithmaRootFindingConfig, ArithmaRootFindingResult,
};
use crate::pyfacade::core::Expression;
use crate::pyfacade::MAX_SEQUENCE_LEN;

// ============================================================================
// Bounds.
// ============================================================================

/// Upper bound on the byte length of a variable name. Names are identifiers,
/// not payloads; the bound also fixes the iteration count of the whitespace
/// scan in [`check_variable_name`].
const MAX_VARIABLE_NAME_LEN: usize = 256;

/// Ceiling on any caller-supplied iteration budget. Every solver loop reached
/// from here is bounded by a budget that passed through [`check_iterations`],
/// so this is the fixed bound the safety standard asks for. A million
/// evaluations of a symbolic tree is already well past the point where a
/// different method, not a bigger budget, is the answer.
const MAX_ITERATIONS: usize = 1_000_000;

/// Pre-allocation cap for result vectors. Lengths are bounded already, but
/// reserving a million slots up front for input that then fails hands a hostile
/// caller a large allocation for free.
const RESULT_PREALLOC_CAP: usize = 64;

// ============================================================================
// Pure validation helpers (unit-tested below without a Python interpreter).
// ============================================================================

/// Validate a variable name arriving from Python. `str` guarantees UTF-8 and
/// nothing else; emptiness, padding and absurd lengths have to be checked here
/// because the Rust layer only `debug_assert!`s them and a release build would
/// happily solve with respect to `""`.
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

/// Validate a convergence tolerance. Zero is rejected rather than clamped: it
/// asks for an exact floating-point root, which for most expressions means
/// burning the whole iteration budget and then raising.
fn check_tolerance(tol: f64, what: &str) -> Result<(), String> {
    if !tol.is_finite() {
        return Err(format!("{what} must be finite, got {tol}"));
    }
    if tol <= 0.0 {
        return Err(format!("{what} must be greater than zero, got {tol}"));
    }
    Ok(())
}

/// Validate an iteration budget against [`MAX_ITERATIONS`].
fn check_iterations(count: usize, what: &str) -> Result<(), String> {
    if count == 0 {
        return Err(format!("{what} must be at least 1"));
    }
    if count > MAX_ITERATIONS {
        return Err(format!(
            "{what} of {count} exceeds the maximum of {MAX_ITERATIONS}"
        ));
    }
    Ok(())
}

/// Validate a single scalar the solvers will evaluate at.
fn check_finite(value: f64, what: &str) -> Result<(), String> {
    if !value.is_finite() {
        return Err(format!("{what} must be finite, got {value}"));
    }
    Ok(())
}

/// Validate a closed `[lo, hi]` bracket or search range.
///
/// Infinite endpoints are rejected even though `ArithmaSearchRange` tolerates
/// them, because the critical-point scan divides the width into fixed steps and
/// an infinite width gives a step of infinity or NaN.
fn check_bounds(lo: f64, hi: f64) -> Result<(), String> {
    check_finite(lo, "lower bound")?;
    check_finite(hi, "upper bound")?;
    if lo > hi {
        return Err(format!("lower bound {lo} must not exceed upper bound {hi}"));
    }
    Ok(())
}

/// Map a Python method name onto the dispatcher enum. Case-insensitive, and
/// `newton` is accepted alongside `newton_raphson` because both spellings are
/// in common use.
fn method_from_name(name: &str) -> Result<ArithmaNumericalMethod, String> {
    match name.to_ascii_lowercase().as_str() {
        "bisection" => Ok(ArithmaNumericalMethod::Bisection),
        "newton" | "newton_raphson" => Ok(ArithmaNumericalMethod::NewtonRaphson),
        "secant" => Ok(ArithmaNumericalMethod::Secant),
        "brent" => Ok(ArithmaNumericalMethod::Brent),
        other => Err(format!(
            "unknown method {other:?}; expected bisection, newton, newton_raphson, secant or brent"
        )),
    }
}

/// Stable Python-facing spelling of a classification.
fn kind_name(kind: ArithmaCriticalPointKind) -> &'static str {
    match kind {
        ArithmaCriticalPointKind::Maximum => "maximum",
        ArithmaCriticalPointKind::Minimum => "minimum",
        ArithmaCriticalPointKind::Saddle => "saddle",
        ArithmaCriticalPointKind::Inflection => "inflection",
        ArithmaCriticalPointKind::Inconclusive => "inconclusive",
    }
}

/// Build and validate a root-finding configuration.
fn root_config(tol: f64, max_iterations: usize) -> Result<ArithmaRootFindingConfig, String> {
    check_tolerance(tol, "tol")?;
    check_iterations(max_iterations, "max_iterations")?;
    Ok(ArithmaRootFindingConfig {
        tol,
        max_iterations,
    })
}

/// Build and validate a critical-point configuration.
fn critical_config(
    convergence_threshold: f64,
    second_derivative_threshold: f64,
    numerical_tolerance: f64,
    max_search_iterations: usize,
) -> Result<ArithmaCriticalPointsConfig, String> {
    check_tolerance(convergence_threshold, "convergence_threshold")?;
    check_tolerance(second_derivative_threshold, "second_derivative_threshold")?;
    check_tolerance(numerical_tolerance, "numerical_tolerance")?;
    check_iterations(max_search_iterations, "max_search_iterations")?;
    Ok(ArithmaCriticalPointsConfig {
        convergence_threshold,
        second_derivative_threshold,
        numerical_tolerance,
        max_search_iterations,
    })
}

/// Reason to reject a root-finding outcome, or `None` when it is sound.
///
/// A run that exhausts its budget carries the last iterate, which is a number
/// that looks like a root and is not one. Handing it back would make a failed
/// solve indistinguishable from a successful one on the Python side.
fn convergence_error(
    result: &ArithmaRootFindingResult,
    what: &str,
    max_iterations: usize,
) -> Option<String> {
    if !result.converged {
        return Some(format!(
            "{what}: no root within tolerance after {} of {max_iterations} iterations; \
             the best estimate was {}",
            result.iterations, result.root
        ));
    }
    if !result.root.is_finite() {
        return Some(format!(
            "{what}: converged on a non-finite root {}",
            result.root
        ));
    }
    if result.iterations > max_iterations {
        return Some(format!(
            "{what}: reported {} iterations against a budget of {max_iterations}",
            result.iterations
        ));
    }
    None
}

/// Name the failing entry point in a Rust error message.
///
/// `interval_analysis` and `methods` already open their messages with their own
/// name, while `root_finding` and `critical_points` do not. Prefixing
/// unconditionally would produce `evaluate_interval: evaluate_interval: ...`,
/// which reads like a bug in the wrapper.
fn prefix_message(what: &str, message: &str) -> String {
    if message.starts_with(what) {
        return message.to_string();
    }
    format!("{what}: {message}")
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

/// Shared front half of every critical-point entry point that searches a range.
fn prepare_range(
    expr: &Bound<'_, PyAny>,
    var: &str,
    lo: f64,
    hi: f64,
    config: ArithmaCriticalPointsConfig,
    what: &str,
) -> PyResult<(ArithmaExpression, ArithmaCriticalPoints, ArithmaSearchRange)> {
    variable_name_or_err(var, what)?;
    check_bounds(lo, hi).map_err(|e| PyValueError::new_err(format!("{what}: {e}")))?;
    let inner = coerce_expression(expr, what)?;
    let range = ArithmaSearchRange::new(lo, hi);
    debug_assert!(
        range.is_valid(),
        "{what}: range passed validation but is not valid"
    );
    debug_assert!(
        config.max_search_iterations > 0,
        "{what}: a zero search budget reached the analyser"
    );
    Ok((inner, ArithmaCriticalPoints::with_config(config), range))
}

/// Shared front half of the two entry points that classify one given point.
fn prepare_point(
    expr: &Bound<'_, PyAny>,
    var: &str,
    point: f64,
    config: ArithmaCriticalPointsConfig,
    what: &str,
) -> PyResult<(ArithmaExpression, ArithmaCriticalPoints)> {
    variable_name_or_err(var, what)?;
    check_finite(point, "point").map_err(|e| PyValueError::new_err(format!("{what}: {e}")))?;
    let inner = coerce_expression(expr, what)?;
    debug_assert!(
        point.is_finite(),
        "{what}: non-finite point reached the analyser"
    );
    debug_assert!(
        config.convergence_threshold > 0.0,
        "{what}: a non-positive stationarity threshold reached the analyser"
    );
    Ok((inner, ArithmaCriticalPoints::with_config(config)))
}

/// Convert a run of classified points, rejecting any the analyser could not
/// place on the real line.
fn to_py_points(points: &[ArithmaCriticalPoint], what: &str) -> PyResult<Vec<CriticalPoint>> {
    check_sequence_len(points.len(), what).map_err(PyRuntimeError::new_err)?;
    let mut out: Vec<CriticalPoint> = Vec::with_capacity(points.len().min(RESULT_PREALLOC_CAP));
    // Fixed bound: the length was checked against MAX_SEQUENCE_LEN above.
    for point in points.iter().take(MAX_SEQUENCE_LEN) {
        out.push(CriticalPoint::from_rust(point, what)?);
    }
    debug_assert_eq!(
        out.len(),
        points.len(),
        "{what}: dropped a point in conversion"
    );
    Ok(out)
}

/// Wrap a Rust interval, rejecting NaN bounds as an internal failure. Interval
/// arithmetic is meant to produce an enclosure or an explicit error; a NaN
/// endpoint is neither, and is not something a caller can act on.
fn to_py_interval(inner: ArithmaInterval, what: &str) -> PyResult<Interval> {
    if inner.lo.is_nan() || inner.hi.is_nan() {
        return Err(PyRuntimeError::new_err(format!(
            "{what}: produced an interval with a NaN bound [{}, {}]",
            inner.lo, inner.hi
        )));
    }
    Ok(Interval { inner })
}

/// Accept either an :class:`Interval` or a plain ``(lo, hi)`` tuple.
fn coerce_interval(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<ArithmaInterval> {
    if let Ok(iv) = obj.extract::<PyRef<Interval>>() {
        return Ok(iv.inner);
    }
    if let Ok((lo, hi)) = obj.extract::<(f64, f64)>() {
        check_bounds(lo, hi).map_err(|e| PyValueError::new_err(format!("{what}: {e}")))?;
        return Ok(ArithmaInterval::new(lo, hi));
    }
    Err(PyTypeError::new_err(format!(
        "{what}: expected an Interval or a (lo, hi) tuple of floats"
    )))
}

// ============================================================================
// `RootResult` pyclass.
// ============================================================================

/// Outcome of a root-finding run.
///
/// ``converged`` is always ``True`` on an object you can hold: a run that fails
/// to reach tolerance raises :class:`RuntimeError` instead of returning. The
/// attribute is retained because it is part of the underlying Rust contract.
#[pyclass(name = "RootResult", module = "arithma")]
#[derive(Clone)]
pub struct RootResult {
    root: f64,
    iterations: usize,
    converged: bool,
}

#[pymethods]
impl RootResult {
    /// The root estimate.
    #[getter]
    fn root(&self) -> f64 {
        debug_assert!(
            self.root.is_finite(),
            "a non-finite root escaped validation"
        );
        debug_assert!(self.converged, "an unconverged result escaped validation");
        self.root
    }

    /// Iterations consumed. Zero means an endpoint or seed was already a root.
    #[getter]
    fn iterations(&self) -> usize {
        debug_assert!(
            self.root.is_finite(),
            "a non-finite root escaped validation"
        );
        debug_assert!(self.converged, "an unconverged result escaped validation");
        self.iterations
    }

    /// Always ``True``; see the class docstring.
    #[getter]
    fn converged(&self) -> bool {
        debug_assert!(
            self.root.is_finite(),
            "a non-finite root escaped validation"
        );
        debug_assert!(self.converged, "an unconverged result escaped validation");
        self.converged
    }

    /// The root, so a result can be used directly in arithmetic.
    fn __float__(&self) -> f64 {
        debug_assert!(
            self.root.is_finite(),
            "a non-finite root escaped validation"
        );
        debug_assert!(self.converged, "an unconverged result escaped validation");
        self.root
    }

    fn __repr__(&self) -> String {
        debug_assert!(
            self.root.is_finite(),
            "a non-finite root escaped validation"
        );
        debug_assert!(self.converged, "an unconverged result escaped validation");
        format!(
            "RootResult(root={}, iterations={}, converged={})",
            self.root,
            self.iterations,
            if self.converged { "True" } else { "False" }
        )
    }
}

// ============================================================================
// `CriticalPoint` pyclass.
// ============================================================================

/// One classified point on the curve.
///
/// ``kind`` is one of ``"maximum"``, ``"minimum"``, ``"saddle"``,
/// ``"inflection"`` or ``"inconclusive"``. The derivative samples are ``None``
/// where the analyser could not evaluate that order at this point.
#[pyclass(name = "CriticalPoint", module = "arithma")]
#[derive(Clone)]
pub struct CriticalPoint {
    x: f64,
    y: f64,
    kind: String,
    first_derivative: Option<f64>,
    second_derivative: Option<f64>,
    third_derivative: Option<f64>,
}

impl CriticalPoint {
    /// Convert one Rust record, rejecting a point that is not on the real line.
    fn from_rust(point: &ArithmaCriticalPoint, what: &str) -> PyResult<Self> {
        if !point.x.is_finite() || !point.y.is_finite() {
            return Err(PyRuntimeError::new_err(format!(
                "{what}: analyser produced a non-finite point ({}, {})",
                point.x, point.y
            )));
        }
        Ok(Self {
            x: point.x,
            y: point.y,
            kind: kind_name(point.kind).to_string(),
            first_derivative: point.first_derivative,
            second_derivative: point.second_derivative,
            third_derivative: point.third_derivative,
        })
    }
}

#[pymethods]
impl CriticalPoint {
    /// Location along the variable axis.
    #[getter]
    fn x(&self) -> f64 {
        debug_assert!(
            self.x.is_finite(),
            "a non-finite location escaped validation"
        );
        debug_assert!(!self.kind.is_empty(), "kind is always populated");
        self.x
    }

    /// Function value at [`CriticalPoint::x`].
    #[getter]
    fn y(&self) -> f64 {
        debug_assert!(self.y.is_finite(), "a non-finite value escaped validation");
        debug_assert!(!self.kind.is_empty(), "kind is always populated");
        self.y
    }

    /// Classification: ``"maximum"``, ``"minimum"``, ``"saddle"``,
    /// ``"inflection"`` or ``"inconclusive"``.
    #[getter]
    fn kind(&self) -> String {
        debug_assert!(!self.kind.is_empty(), "kind is always populated");
        debug_assert!(
            self.x.is_finite(),
            "a non-finite location escaped validation"
        );
        self.kind.clone()
    }

    /// Sampled first derivative, or ``None`` if it could not be evaluated.
    #[getter]
    fn first_derivative(&self) -> Option<f64> {
        debug_assert!(!self.kind.is_empty(), "kind is always populated");
        debug_assert!(
            self.x.is_finite(),
            "a non-finite location escaped validation"
        );
        self.first_derivative
    }

    /// Sampled second derivative; its sign drives the classification.
    #[getter]
    fn second_derivative(&self) -> Option<f64> {
        debug_assert!(!self.kind.is_empty(), "kind is always populated");
        debug_assert!(
            self.x.is_finite(),
            "a non-finite location escaped validation"
        );
        self.second_derivative
    }

    /// Sampled third derivative; confirms inflections and saddles.
    #[getter]
    fn third_derivative(&self) -> Option<f64> {
        debug_assert!(!self.kind.is_empty(), "kind is always populated");
        debug_assert!(
            self.x.is_finite(),
            "a non-finite location escaped validation"
        );
        self.third_derivative
    }

    fn __repr__(&self) -> String {
        debug_assert!(!self.kind.is_empty(), "kind is always populated");
        debug_assert!(
            self.x.is_finite(),
            "a non-finite location escaped validation"
        );
        format!(
            "CriticalPoint(x={}, y={}, kind={:?})",
            self.x, self.y, self.kind
        )
    }
}

// ============================================================================
// `FunctionAnalysis` pyclass.
// ============================================================================

/// Combined report over a search range.
///
/// The monotonic and concavity decompositions are lists of
/// ``(lo, hi, flag)`` tuples: the flag is ``True`` for an increasing
/// sub-interval and for a concave-up one respectively. A sub-interval whose
/// midpoint could not be evaluated is absent rather than guessed, so the
/// decompositions do not necessarily tile the whole range.
#[pyclass(name = "FunctionAnalysis", module = "arithma")]
#[derive(Clone)]
pub struct FunctionAnalysis {
    stationary_points: Vec<CriticalPoint>,
    inflection_points: Vec<CriticalPoint>,
    lo: f64,
    hi: f64,
    monotonic_intervals: Vec<(f64, f64, bool)>,
    concavity_intervals: Vec<(f64, f64, bool)>,
}

#[pymethods]
impl FunctionAnalysis {
    /// Classified stationary points, in increasing order of ``x``.
    #[getter]
    fn stationary_points(&self) -> Vec<CriticalPoint> {
        debug_assert!(self.lo <= self.hi, "the analysed range is inverted");
        debug_assert!(self.lo.is_finite(), "the analysed range is not finite");
        self.stationary_points.clone()
    }

    /// Confirmed inflection points, in increasing order of ``x``.
    #[getter]
    fn inflection_points(&self) -> Vec<CriticalPoint> {
        debug_assert!(self.lo <= self.hi, "the analysed range is inverted");
        debug_assert!(self.lo.is_finite(), "the analysed range is not finite");
        self.inflection_points.clone()
    }

    /// The searched range as ``(lo, hi)``.
    #[getter]
    fn range(&self) -> (f64, f64) {
        debug_assert!(self.lo <= self.hi, "the analysed range is inverted");
        debug_assert!(self.hi.is_finite(), "the analysed range is not finite");
        (self.lo, self.hi)
    }

    /// ``(lo, hi, increasing)`` per monotonic sub-interval.
    #[getter]
    fn monotonic_intervals(&self) -> Vec<(f64, f64, bool)> {
        debug_assert!(self.lo <= self.hi, "the analysed range is inverted");
        debug_assert!(self.hi.is_finite(), "the analysed range is not finite");
        self.monotonic_intervals.clone()
    }

    /// ``(lo, hi, concave_up)`` per concavity sub-interval.
    #[getter]
    fn concavity_intervals(&self) -> Vec<(f64, f64, bool)> {
        debug_assert!(self.lo <= self.hi, "the analysed range is inverted");
        debug_assert!(self.hi.is_finite(), "the analysed range is not finite");
        self.concavity_intervals.clone()
    }

    fn __repr__(&self) -> String {
        debug_assert!(self.lo <= self.hi, "the analysed range is inverted");
        debug_assert!(self.hi.is_finite(), "the analysed range is not finite");
        format!(
            "FunctionAnalysis(range=({}, {}), stationary={}, inflection={})",
            self.lo,
            self.hi,
            self.stationary_points.len(),
            self.inflection_points.len()
        )
    }
}

// ============================================================================
// `Interval` pyclass.
// ============================================================================

/// Closed interval ``[lo, hi]`` over the reals.
///
/// NaN bounds are rejected at construction. The empty interval is the sentinel
/// ``lo > hi`` produced by :meth:`Interval.empty` or by an :meth:`intersect`
/// that does not overlap; :meth:`is_empty` is the way to test for it.
#[pyclass(name = "Interval", module = "arithma")]
#[derive(Clone)]
pub struct Interval {
    inner: ArithmaInterval,
}

#[pymethods]
impl Interval {
    /// Construct ``[lo, hi]``. Raises :class:`ValueError` for a NaN bound or an
    /// inverted pair; use :meth:`Interval.empty` for the empty interval.
    #[new]
    #[pyo3(signature = (lo, hi))]
    fn new(lo: f64, hi: f64) -> PyResult<Self> {
        if lo.is_nan() || hi.is_nan() {
            return Err(PyValueError::new_err(format!(
                "Interval: bounds must not be NaN, got [{lo}, {hi}]"
            )));
        }
        if lo > hi {
            return Err(PyValueError::new_err(format!(
                "Interval: lo {lo} exceeds hi {hi}; use Interval.empty() for the empty interval"
            )));
        }
        debug_assert!(
            !lo.is_nan() && !hi.is_nan(),
            "a NaN bound escaped validation"
        );
        debug_assert!(lo <= hi, "an inverted bound escaped validation");
        Ok(Self {
            inner: ArithmaInterval::new(lo, hi),
        })
    }

    /// The whole real line, ``(-inf, inf)``.
    #[staticmethod]
    fn whole() -> Self {
        let inner = ArithmaInterval::whole();
        debug_assert!(!inner.is_empty(), "the whole line is not empty");
        debug_assert!(inner.contains(0.0), "the whole line contains zero");
        Self { inner }
    }

    /// The empty interval.
    #[staticmethod]
    fn empty() -> Self {
        let inner = ArithmaInterval::empty();
        debug_assert!(inner.is_empty(), "the empty interval must report empty");
        debug_assert!(!inner.contains(0.0), "the empty interval contains nothing");
        Self { inner }
    }

    /// The degenerate interval ``[value, value]``.
    #[staticmethod]
    #[pyo3(signature = (value))]
    fn point(value: f64) -> PyResult<Self> {
        check_finite(value, "value")
            .map_err(|e| PyValueError::new_err(format!("Interval.point: {e}")))?;
        let inner = ArithmaInterval::point(value);
        debug_assert!(inner.is_point(), "a point interval must report as one");
        debug_assert!(inner.contains(value), "a point interval contains its value");
        Ok(Self { inner })
    }

    /// Smallest interval containing every value in `values`.
    ///
    /// The sequence must be non-empty and NaN-free. The Rust helper answers
    /// with the whole real line in both cases; that is a silent widening a
    /// caller cannot detect, so both raise :class:`ValueError` here.
    #[staticmethod]
    #[pyo3(signature = (values))]
    fn hull(values: &Bound<'_, PyAny>) -> PyResult<Self> {
        let collected = collect_floats(values, "Interval.hull")?;
        debug_assert!(!collected.is_empty(), "an empty hull escaped validation");
        debug_assert!(
            collected.iter().take(MAX_SEQUENCE_LEN).all(|v| !v.is_nan()),
            "a NaN escaped validation"
        );
        let inner = ArithmaInterval::hull(&collected);
        Ok(Self { inner })
    }

    /// Lower bound.
    #[getter]
    fn lo(&self) -> f64 {
        debug_assert!(!self.inner.lo.is_nan(), "a NaN bound escaped validation");
        debug_assert!(!self.inner.hi.is_nan(), "a NaN bound escaped validation");
        self.inner.lo
    }

    /// Upper bound.
    #[getter]
    fn hi(&self) -> f64 {
        debug_assert!(!self.inner.lo.is_nan(), "a NaN bound escaped validation");
        debug_assert!(!self.inner.hi.is_nan(), "a NaN bound escaped validation");
        self.inner.hi
    }

    /// ``hi - lo``. Negative for the empty interval.
    #[getter]
    fn width(&self) -> f64 {
        debug_assert!(!self.inner.lo.is_nan(), "a NaN bound escaped validation");
        debug_assert!(!self.inner.hi.is_nan(), "a NaN bound escaped validation");
        self.inner.width()
    }

    /// True when this interval holds no real value.
    fn is_empty(&self) -> bool {
        debug_assert!(!self.inner.lo.is_nan(), "a NaN bound escaped validation");
        debug_assert!(!self.inner.hi.is_nan(), "a NaN bound escaped validation");
        self.inner.is_empty()
    }

    /// True when this interval is a single finite value.
    fn is_point(&self) -> bool {
        debug_assert!(!self.inner.lo.is_nan(), "a NaN bound escaped validation");
        debug_assert!(!self.inner.hi.is_nan(), "a NaN bound escaped validation");
        self.inner.is_point()
    }

    /// True when `point` lies in ``[lo, hi]``.
    #[pyo3(signature = (point))]
    fn contains(&self, point: f64) -> bool {
        debug_assert!(!self.inner.lo.is_nan(), "a NaN bound escaped validation");
        debug_assert!(!self.inner.hi.is_nan(), "a NaN bound escaped validation");
        self.inner.contains(point)
    }

    /// Smallest interval containing both operands.
    #[pyo3(signature = (other))]
    fn union(&self, other: PyRef<'_, Interval>) -> Self {
        debug_assert!(!self.inner.lo.is_nan(), "a NaN bound escaped validation");
        debug_assert!(!other.inner.hi.is_nan(), "a NaN bound escaped validation");
        Self {
            inner: self.inner.union(other.inner),
        }
    }

    /// Overlap of the two operands; may be empty.
    #[pyo3(signature = (other))]
    fn intersect(&self, other: PyRef<'_, Interval>) -> Self {
        debug_assert!(!self.inner.lo.is_nan(), "a NaN bound escaped validation");
        debug_assert!(!other.inner.hi.is_nan(), "a NaN bound escaped validation");
        Self {
            inner: self.inner.intersect(other.inner),
        }
    }

    fn __contains__(&self, point: f64) -> bool {
        debug_assert!(!self.inner.lo.is_nan(), "a NaN bound escaped validation");
        debug_assert!(!self.inner.hi.is_nan(), "a NaN bound escaped validation");
        self.inner.contains(point)
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        debug_assert!(!self.inner.lo.is_nan(), "a NaN bound escaped validation");
        debug_assert!(!self.inner.hi.is_nan(), "a NaN bound escaped validation");
        match other.extract::<PyRef<Interval>>() {
            Ok(rhs) => self.inner == rhs.inner,
            // Comparison against an unrelated type is false, not an error.
            Err(_) => false,
        }
    }

    fn __repr__(&self) -> String {
        debug_assert!(!self.inner.lo.is_nan(), "a NaN bound escaped validation");
        debug_assert!(!self.inner.hi.is_nan(), "a NaN bound escaped validation");
        if self.inner.is_empty() {
            return "Interval.empty()".to_string();
        }
        format!("Interval({}, {})", self.inner.lo, self.inner.hi)
    }
}

/// Read a Python sequence of floats, rejecting `str`, an empty sequence and
/// NaN. Bounded by [`MAX_SEQUENCE_LEN`] before any iteration starts.
fn collect_floats(values: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<f64>> {
    if values.is_instance_of::<PyString>() {
        return Err(PyTypeError::new_err(format!(
            "{what}: expected a sequence of floats, not a str"
        )));
    }
    let seq = values
        .downcast::<PySequence>()
        .map_err(|_| PyTypeError::new_err(format!("{what}: expected a list or tuple of floats")))?;
    let len = seq.len()?;
    check_sequence_len(len, what).map_err(PyValueError::new_err)?;
    if len == 0 {
        return Err(PyValueError::new_err(format!(
            "{what}: the sequence must not be empty"
        )));
    }
    let mut out: Vec<f64> = Vec::with_capacity(len.min(RESULT_PREALLOC_CAP));
    // Fixed bound: `len` was checked against MAX_SEQUENCE_LEN above.
    for i in 0..len {
        let value: f64 = seq
            .get_item(i)?
            .extract()
            .map_err(|_| PyTypeError::new_err(format!("{what}: values[{i}] must be a float")))?;
        if value.is_nan() {
            return Err(PyValueError::new_err(format!(
                "{what}: values[{i}] is NaN, which would silently widen the result"
            )));
        }
        out.push(value);
    }
    debug_assert_eq!(out.len(), len, "{what}: dropped a value in conversion");
    Ok(out)
}

// ============================================================================
// Root finding.
// ============================================================================

/// Bisect ``[lo, hi]`` for a root of `expr` in `var`.
///
/// Requires a sign change across the bracket — that is what guarantees a root —
/// and raises :class:`RuntimeError` if there is none, rather than returning a
/// confident midpoint. Raises :class:`RuntimeError` if `max_iterations` is
/// exhausted before ``|f(x)| <= tol``.
///
/// The bracket must be given in order. The Rust helper silently swaps a
/// reversed pair; here it is a :class:`ValueError`, because at this boundary a
/// reversed bracket is far more likely to be two arguments in the wrong order
/// than a deliberate choice.
#[pyfunction]
#[pyo3(signature = (expr, var, lo, hi, tol = 1e-12, max_iterations = 1024))]
fn find_root_bisection(
    expr: &Bound<'_, PyAny>,
    var: &str,
    lo: f64,
    hi: f64,
    tol: f64,
    max_iterations: usize,
) -> PyResult<RootResult> {
    let what = "find_root_bisection";
    variable_name_or_err(var, what)?;
    check_bounds(lo, hi).map_err(|e| PyValueError::new_err(format!("{what}: {e}")))?;
    let config = root_config(tol, max_iterations)
        .map_err(|e| PyValueError::new_err(format!("{what}: {e}")))?;
    let inner = coerce_expression(expr, what)?;
    debug_assert!(
        config.tol > 0.0,
        "{what}: a non-positive tolerance escaped validation"
    );
    debug_assert!(
        lo.is_finite() && hi.is_finite(),
        "{what}: a non-finite bracket escaped validation"
    );

    let result = rust_find_root_bisection(&inner, var, lo, hi, &config)
        .map_err(|e| PyRuntimeError::new_err(prefix_message(what, &e)))?;
    finish_root(result, what, max_iterations)
}

/// Newton-Raphson from `initial`, using the symbolic derivative of `expr`.
///
/// Raises :class:`RuntimeError` when the derivative vanishes, when the iterate
/// runs away to a non-finite value, and when `max_iterations` is exhausted
/// before ``|f(x)| <= tol``. Newton can diverge from a poor seed; fall back to
/// :func:`find_root_bisection` with a known bracket when it does.
#[pyfunction]
#[pyo3(signature = (expr, var, initial, tol = 1e-12, max_iterations = 1024))]
fn find_root_newton_raphson(
    expr: &Bound<'_, PyAny>,
    var: &str,
    initial: f64,
    tol: f64,
    max_iterations: usize,
) -> PyResult<RootResult> {
    let what = "find_root_newton_raphson";
    variable_name_or_err(var, what)?;
    check_finite(initial, "initial").map_err(|e| PyValueError::new_err(format!("{what}: {e}")))?;
    let config = root_config(tol, max_iterations)
        .map_err(|e| PyValueError::new_err(format!("{what}: {e}")))?;
    let inner = coerce_expression(expr, what)?;
    debug_assert!(
        config.tol > 0.0,
        "{what}: a non-positive tolerance escaped validation"
    );
    debug_assert!(
        initial.is_finite(),
        "{what}: a non-finite seed escaped validation"
    );

    let result = rust_find_root_newton_raphson(&inner, var, initial, &config)
        .map_err(|e| PyRuntimeError::new_err(prefix_message(what, &e)))?;
    finish_root(result, what, max_iterations)
}

/// Secant method seeded with two distinct points, needing no derivative.
///
/// Useful when `expr` contains an operator the differentiator cannot handle.
/// Raises :class:`ValueError` for identical seeds and :class:`RuntimeError`
/// when the secant line goes horizontal, when the iterate diverges, or when
/// `max_iterations` is exhausted before ``|f(x)| <= tol``.
#[pyfunction]
#[pyo3(signature = (expr, var, x0, x1, tol = 1e-12, max_iterations = 1024))]
fn find_root_secant(
    expr: &Bound<'_, PyAny>,
    var: &str,
    x0: f64,
    x1: f64,
    tol: f64,
    max_iterations: usize,
) -> PyResult<RootResult> {
    let what = "find_root_secant";
    variable_name_or_err(var, what)?;
    check_finite(x0, "x0").map_err(|e| PyValueError::new_err(format!("{what}: {e}")))?;
    check_finite(x1, "x1").map_err(|e| PyValueError::new_err(format!("{what}: {e}")))?;
    if x0 == x1 {
        return Err(PyValueError::new_err(format!(
            "{what}: x0 and x1 must be distinct, both were {x0}"
        )));
    }
    let config = root_config(tol, max_iterations)
        .map_err(|e| PyValueError::new_err(format!("{what}: {e}")))?;
    let inner = coerce_expression(expr, what)?;
    debug_assert!(x0 != x1, "{what}: identical seeds escaped validation");
    debug_assert!(
        config.max_iterations > 0,
        "{what}: a zero budget escaped validation"
    );

    let result = rust_find_root_secant(&inner, var, x0, x1, &config)
        .map_err(|e| PyRuntimeError::new_err(prefix_message(what, &e)))?;
    finish_root(result, what, max_iterations)
}

/// Turn a Rust outcome into a `RootResult`, raising if it is not a real root.
fn finish_root(
    result: ArithmaRootFindingResult,
    what: &str,
    max_iterations: usize,
) -> PyResult<RootResult> {
    if let Some(message) = convergence_error(&result, what, max_iterations) {
        return Err(PyRuntimeError::new_err(message));
    }
    debug_assert!(
        result.converged,
        "{what}: an unconverged result passed the check"
    );
    debug_assert!(
        result.root.is_finite(),
        "{what}: a non-finite root passed the check"
    );
    Ok(RootResult {
        root: result.root,
        iterations: result.iterations,
        converged: result.converged,
    })
}

/// Solve ``expr = 0`` for `var` from a single seed, choosing the method.
///
/// `method` is ``"bisection"``, ``"newton"`` (or ``"newton_raphson"``),
/// ``"secant"`` or ``"brent"``, case-insensitive. The two bracketing methods
/// grow a sign-changing bracket outward from `initial` first, so the seed need
/// not bracket a root, but a function with no sign change near it raises.
///
/// This dispatcher carries its own fixed budget and uses a *numeric* derivative
/// for Newton, so it takes no tolerance or iteration arguments; use
/// :func:`find_root_newton_raphson` when you want the symbolic derivative and
/// an explicit budget. Returns a bare ``float`` because the dispatcher does not
/// report an iteration count.
#[pyfunction]
#[pyo3(signature = (expr, var, initial, method = "brent"))]
fn solve_with_method(
    expr: &Bound<'_, PyAny>,
    var: &str,
    initial: f64,
    method: &str,
) -> PyResult<f64> {
    let what = "solve_with_method";
    variable_name_or_err(var, what)?;
    check_finite(initial, "initial").map_err(|e| PyValueError::new_err(format!("{what}: {e}")))?;
    let chosen =
        method_from_name(method).map_err(|e| PyValueError::new_err(format!("{what}: {e}")))?;
    let inner = coerce_expression(expr, what)?;
    debug_assert!(
        initial.is_finite(),
        "{what}: a non-finite seed escaped validation"
    );
    debug_assert!(
        !var.is_empty(),
        "{what}: an empty variable name escaped validation"
    );

    let root = rust_solve_with_method(&inner, var, chosen, initial)
        .map_err(|e| PyRuntimeError::new_err(prefix_message(what, &e)))?;
    if !root.is_finite() {
        return Err(PyRuntimeError::new_err(format!(
            "{what}: {method} returned a non-finite root {root}"
        )));
    }
    Ok(root)
}

// ============================================================================
// Critical points.
// ============================================================================

/// Stationary points of `expr` over ``[lo, hi]``, each classified.
///
/// Roots of ``f'`` are located by scanning the range for sign changes and
/// bisecting each bracket, so the scan resolution is
/// ``(hi - lo) / max_search_iterations``: two stationary points closer together
/// than that may be missed, and a densely oscillating function wants a larger
/// budget. A function whose derivative is identically zero over the range has
/// no *isolated* stationary points and yields an empty list.
#[pyfunction]
#[pyo3(signature = (
    expr, var, lo, hi,
    convergence_threshold = 1e-10,
    second_derivative_threshold = 1e-8,
    numerical_tolerance = 1e-12,
    max_search_iterations = 100
))]
fn find_stationary_points(
    expr: &Bound<'_, PyAny>,
    var: &str,
    lo: f64,
    hi: f64,
    convergence_threshold: f64,
    second_derivative_threshold: f64,
    numerical_tolerance: f64,
    max_search_iterations: usize,
) -> PyResult<Vec<CriticalPoint>> {
    let what = "find_stationary_points";
    let config = critical_config(
        convergence_threshold,
        second_derivative_threshold,
        numerical_tolerance,
        max_search_iterations,
    )
    .map_err(|e| PyValueError::new_err(format!("{what}: {e}")))?;
    let (inner, analyser, range) = prepare_range(expr, var, lo, hi, config, what)?;
    let points = analyser
        .find_stationary_points(&inner, var, range)
        .map_err(|e| PyRuntimeError::new_err(prefix_message(what, &e)))?;
    to_py_points(&points, what)
}

/// Inflection points of `expr` over ``[lo, hi]``.
///
/// A root of ``f''`` is reported only where the concavity actually flips —
/// confirmed by ``f''' != 0``, or failing that by the sign of ``f''`` either
/// side. That second check is what stops ``x**4`` reporting an inflection at
/// the origin.
#[pyfunction]
#[pyo3(signature = (
    expr, var, lo, hi,
    convergence_threshold = 1e-10,
    second_derivative_threshold = 1e-8,
    numerical_tolerance = 1e-12,
    max_search_iterations = 100
))]
fn find_inflection_points(
    expr: &Bound<'_, PyAny>,
    var: &str,
    lo: f64,
    hi: f64,
    convergence_threshold: f64,
    second_derivative_threshold: f64,
    numerical_tolerance: f64,
    max_search_iterations: usize,
) -> PyResult<Vec<CriticalPoint>> {
    let what = "find_inflection_points";
    let config = critical_config(
        convergence_threshold,
        second_derivative_threshold,
        numerical_tolerance,
        max_search_iterations,
    )
    .map_err(|e| PyValueError::new_err(format!("{what}: {e}")))?;
    let (inner, analyser, range) = prepare_range(expr, var, lo, hi, config, what)?;
    let points = analyser
        .find_inflection_points(&inner, var, range)
        .map_err(|e| PyRuntimeError::new_err(prefix_message(what, &e)))?;
    to_py_points(&points, what)
}

/// Local maxima and minima of `expr` over ``[lo, hi]``, as ``(maxima, minima)``.
///
/// Saddles and points the derivative tests cannot settle appear in neither
/// list; call :func:`find_stationary_points` to see them.
#[pyfunction]
#[pyo3(signature = (
    expr, var, lo, hi,
    convergence_threshold = 1e-10,
    second_derivative_threshold = 1e-8,
    numerical_tolerance = 1e-12,
    max_search_iterations = 100
))]
fn find_extrema(
    expr: &Bound<'_, PyAny>,
    var: &str,
    lo: f64,
    hi: f64,
    convergence_threshold: f64,
    second_derivative_threshold: f64,
    numerical_tolerance: f64,
    max_search_iterations: usize,
) -> PyResult<(Vec<CriticalPoint>, Vec<CriticalPoint>)> {
    let what = "find_extrema";
    let config = critical_config(
        convergence_threshold,
        second_derivative_threshold,
        numerical_tolerance,
        max_search_iterations,
    )
    .map_err(|e| PyValueError::new_err(format!("{what}: {e}")))?;
    let (inner, analyser, range) = prepare_range(expr, var, lo, hi, config, what)?;
    let (maxima, minima) = analyser
        .find_extrema(&inner, var, range)
        .map_err(|e| PyRuntimeError::new_err(prefix_message(what, &e)))?;
    Ok((to_py_points(&maxima, what)?, to_py_points(&minima, what)?))
}

/// Classify one specific `point` of `expr`, returning the full record.
///
/// The point need not be stationary: a non-stationary point is reported as an
/// inflection when ``f''`` vanishes there and the concavity flips, and as
/// ``"inconclusive"`` otherwise.
#[pyfunction]
#[pyo3(signature = (
    expr, var, point,
    convergence_threshold = 1e-10,
    second_derivative_threshold = 1e-8,
    numerical_tolerance = 1e-12,
    max_search_iterations = 100
))]
fn analyze_point(
    expr: &Bound<'_, PyAny>,
    var: &str,
    point: f64,
    convergence_threshold: f64,
    second_derivative_threshold: f64,
    numerical_tolerance: f64,
    max_search_iterations: usize,
) -> PyResult<CriticalPoint> {
    let what = "analyze_point";
    let config = critical_config(
        convergence_threshold,
        second_derivative_threshold,
        numerical_tolerance,
        max_search_iterations,
    )
    .map_err(|e| PyValueError::new_err(format!("{what}: {e}")))?;
    let (inner, analyser) = prepare_point(expr, var, point, config, what)?;
    let classified = analyser
        .analyze_point(&inner, var, point)
        .map_err(|e| PyRuntimeError::new_err(prefix_message(what, &e)))?;
    CriticalPoint::from_rust(&classified, what)
}

/// Classification of one specific `point`, as a string.
///
/// Shorthand for the ``kind`` of :func:`analyze_point`.
#[pyfunction]
#[pyo3(signature = (
    expr, var, point,
    convergence_threshold = 1e-10,
    second_derivative_threshold = 1e-8,
    numerical_tolerance = 1e-12,
    max_search_iterations = 100
))]
fn classify_point(
    expr: &Bound<'_, PyAny>,
    var: &str,
    point: f64,
    convergence_threshold: f64,
    second_derivative_threshold: f64,
    numerical_tolerance: f64,
    max_search_iterations: usize,
) -> PyResult<String> {
    let what = "classify_point";
    let config = critical_config(
        convergence_threshold,
        second_derivative_threshold,
        numerical_tolerance,
        max_search_iterations,
    )
    .map_err(|e| PyValueError::new_err(format!("{what}: {e}")))?;
    let (inner, analyser) = prepare_point(expr, var, point, config, what)?;
    let kind = analyser
        .classify_point(&inner, var, point)
        .map_err(|e| PyRuntimeError::new_err(prefix_message(what, &e)))?;
    let name = kind_name(kind);
    debug_assert!(!name.is_empty(), "{what}: produced an empty classification");
    debug_assert!(
        point.is_finite(),
        "{what}: a non-finite point reached the result"
    );
    Ok(name.to_string())
}

/// Full report over ``[lo, hi]``: stationary points, inflection points and the
/// monotonic and concavity decompositions.
///
/// Costs one stationary-point search plus one inflection search over the same
/// range, so prefer it to calling both separately when you want each.
#[pyfunction]
#[pyo3(signature = (
    expr, var, lo, hi,
    convergence_threshold = 1e-10,
    second_derivative_threshold = 1e-8,
    numerical_tolerance = 1e-12,
    max_search_iterations = 100
))]
fn analyze_intervals(
    expr: &Bound<'_, PyAny>,
    var: &str,
    lo: f64,
    hi: f64,
    convergence_threshold: f64,
    second_derivative_threshold: f64,
    numerical_tolerance: f64,
    max_search_iterations: usize,
) -> PyResult<FunctionAnalysis> {
    let what = "analyze_intervals";
    let config = critical_config(
        convergence_threshold,
        second_derivative_threshold,
        numerical_tolerance,
        max_search_iterations,
    )
    .map_err(|e| PyValueError::new_err(format!("{what}: {e}")))?;
    let (inner, analyser, range) = prepare_range(expr, var, lo, hi, config, what)?;
    let analysis = analyser
        .analyze_intervals(&inner, var, range)
        .map_err(|e| PyRuntimeError::new_err(prefix_message(what, &e)))?;
    check_sequence_len(analysis.monotonic_intervals.len(), what)
        .map_err(PyRuntimeError::new_err)?;
    check_sequence_len(analysis.concavity_intervals.len(), what)
        .map_err(PyRuntimeError::new_err)?;
    Ok(FunctionAnalysis {
        stationary_points: to_py_points(&analysis.stationary_points, what)?,
        inflection_points: to_py_points(&analysis.inflection_points, what)?,
        lo: analysis.range.lo,
        hi: analysis.range.hi,
        // Bounded: both lengths were checked against MAX_SEQUENCE_LEN above.
        monotonic_intervals: analysis
            .monotonic_intervals
            .iter()
            .take(MAX_SEQUENCE_LEN)
            .map(|iv| (iv.lo, iv.hi, iv.increasing))
            .collect(),
        concavity_intervals: analysis
            .concavity_intervals
            .iter()
            .take(MAX_SEQUENCE_LEN)
            .map(|iv| (iv.lo, iv.hi, iv.concave_up))
            .collect(),
    })
}

// ============================================================================
// Interval analysis.
// ============================================================================

/// Guaranteed enclosure of ``expr(interval)`` by interval arithmetic.
///
/// `interval` is an :class:`Interval` or a ``(lo, hi)`` tuple. The result
/// encloses the true image but is not generally tight: each occurrence of `var`
/// is bounded independently, so ``x - x`` over ``[0, 1]`` gives ``[-1, 1]``.
/// Partial functions are evaluated over the intersection with their real
/// domain, so ``sqrt`` over ``[-1, 4]`` gives ``[0, 2]`` and an input entirely
/// outside the domain gives the empty interval.
///
/// Raises :class:`RuntimeError` for a free variable other than `var`, an
/// operator the module does not model, or a node-cap overrun — never a
/// silently wrong bound.
#[pyfunction]
#[pyo3(signature = (expr, var, interval))]
fn evaluate_interval(
    expr: &Bound<'_, PyAny>,
    var: &str,
    interval: &Bound<'_, PyAny>,
) -> PyResult<Interval> {
    let what = "evaluate_interval";
    variable_name_or_err(var, what)?;
    let bounds = coerce_interval(interval, what)?;
    let inner = coerce_expression(expr, what)?;
    debug_assert!(
        !bounds.lo.is_nan(),
        "{what}: a NaN bound escaped validation"
    );
    debug_assert!(
        !bounds.hi.is_nan(),
        "{what}: a NaN bound escaped validation"
    );

    let image = rust_evaluate_interval(&inner, var, bounds)
        .map_err(|e| PyRuntimeError::new_err(prefix_message(what, &e)))?;
    to_py_interval(image, what)
}

// ============================================================================
// Registration.
// ============================================================================

/// Add the numerical surface to the `_arithma_core` module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RootResult>()?;
    m.add_class::<CriticalPoint>()?;
    m.add_class::<FunctionAnalysis>()?;
    m.add_class::<Interval>()?;
    m.add_function(wrap_pyfunction!(find_root_bisection, m)?)?;
    m.add_function(wrap_pyfunction!(find_root_newton_raphson, m)?)?;
    m.add_function(wrap_pyfunction!(find_root_secant, m)?)?;
    m.add_function(wrap_pyfunction!(solve_with_method, m)?)?;
    m.add_function(wrap_pyfunction!(find_stationary_points, m)?)?;
    m.add_function(wrap_pyfunction!(find_inflection_points, m)?)?;
    m.add_function(wrap_pyfunction!(find_extrema, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_point, m)?)?;
    m.add_function(wrap_pyfunction!(classify_point, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_intervals, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_interval, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── variable names ────────────────────────────────────────────────────

    #[test]
    fn rejects_an_empty_variable_name() {
        assert!(check_variable_name("").is_err());
    }

    #[test]
    fn accepts_ordinary_identifiers_as_variable_names() {
        assert!(check_variable_name("x").is_ok());
        assert!(check_variable_name("theta_1").is_ok());
        // Non-ASCII identifiers are legitimate in this engine (pi, phi, ...).
        assert!(check_variable_name("\u{3b8}").is_ok());
    }

    #[test]
    fn rejects_a_variable_name_containing_whitespace() {
        assert!(check_variable_name(" x").is_err());
        assert!(check_variable_name("x ").is_err());
        assert!(check_variable_name("a b").is_err());
    }

    #[test]
    fn rejects_a_variable_name_past_the_length_limit() {
        assert!(check_variable_name(&"v".repeat(MAX_VARIABLE_NAME_LEN)).is_ok());
        assert!(check_variable_name(&"v".repeat(MAX_VARIABLE_NAME_LEN + 1)).is_err());
    }

    // ── tolerances and budgets ────────────────────────────────────────────

    #[test]
    fn rejects_a_zero_tolerance() {
        assert!(check_tolerance(0.0, "tol").is_err());
    }

    #[test]
    fn rejects_a_negative_or_non_finite_tolerance() {
        assert!(check_tolerance(-1e-9, "tol").is_err());
        assert!(check_tolerance(f64::NAN, "tol").is_err());
        assert!(check_tolerance(f64::INFINITY, "tol").is_err());
    }

    #[test]
    fn accepts_a_small_positive_tolerance() {
        assert!(check_tolerance(1e-15, "tol").is_ok());
    }

    #[test]
    fn rejects_a_zero_iteration_budget() {
        assert!(check_iterations(0, "max_iterations").is_err());
    }

    #[test]
    fn rejects_an_iteration_budget_past_the_ceiling() {
        assert!(check_iterations(MAX_ITERATIONS, "max_iterations").is_ok());
        assert!(check_iterations(MAX_ITERATIONS + 1, "max_iterations").is_err());
    }

    // ── scalars and ranges ────────────────────────────────────────────────

    #[test]
    fn rejects_a_non_finite_scalar() {
        assert!(check_finite(f64::NAN, "initial").is_err());
        assert!(check_finite(f64::NEG_INFINITY, "initial").is_err());
        assert!(check_finite(0.0, "initial").is_ok());
    }

    #[test]
    fn rejects_an_inverted_bracket() {
        assert!(check_bounds(1.0, 0.0).is_err());
        assert!(check_bounds(0.0, 1.0).is_ok());
    }

    #[test]
    fn rejects_an_unbounded_search_range() {
        // The scan divides the width into fixed steps, so an infinite width
        // would give a step of infinity.
        assert!(check_bounds(f64::NEG_INFINITY, f64::INFINITY).is_err());
    }

    #[test]
    fn accepts_a_degenerate_range() {
        assert!(check_bounds(2.5, 2.5).is_ok());
    }

    // ── method names ──────────────────────────────────────────────────────

    #[test]
    fn maps_every_documented_method_name() {
        assert_eq!(
            method_from_name("bisection").unwrap(),
            ArithmaNumericalMethod::Bisection
        );
        assert_eq!(
            method_from_name("newton").unwrap(),
            ArithmaNumericalMethod::NewtonRaphson
        );
        assert_eq!(
            method_from_name("newton_raphson").unwrap(),
            ArithmaNumericalMethod::NewtonRaphson
        );
        assert_eq!(
            method_from_name("secant").unwrap(),
            ArithmaNumericalMethod::Secant
        );
        assert_eq!(
            method_from_name("brent").unwrap(),
            ArithmaNumericalMethod::Brent
        );
    }

    #[test]
    fn matches_a_method_name_case_insensitively() {
        assert_eq!(
            method_from_name("BRENT").unwrap(),
            ArithmaNumericalMethod::Brent
        );
    }

    #[test]
    fn rejects_an_unknown_method_name() {
        let err = method_from_name("halley").unwrap_err();
        assert!(
            err.contains("halley"),
            "message should name the input: {err}"
        );
    }

    // ── classification names ──────────────────────────────────────────────

    #[test]
    fn names_every_classification_distinctly() {
        let names = [
            kind_name(ArithmaCriticalPointKind::Maximum),
            kind_name(ArithmaCriticalPointKind::Minimum),
            kind_name(ArithmaCriticalPointKind::Saddle),
            kind_name(ArithmaCriticalPointKind::Inflection),
            kind_name(ArithmaCriticalPointKind::Inconclusive),
        ];
        for (i, a) in names.iter().enumerate() {
            assert!(!a.is_empty(), "classification {i} has no name");
            for b in names.iter().skip(i + 1) {
                assert_ne!(a, b, "two classifications share the name {a}");
            }
        }
    }

    // ── configuration ─────────────────────────────────────────────────────

    #[test]
    fn root_finding_keyword_defaults_match_the_rust_defaults() {
        // The literals in the `#[pyo3(signature = ...)]` attributes cannot
        // reference the Rust `Default` impl, so this is what stops the two
        // drifting apart.
        let rust = ArithmaRootFindingConfig::default();
        let facade = root_config(1e-12, 1024).unwrap();
        assert_eq!(facade.tol, rust.tol);
        assert_eq!(facade.max_iterations, rust.max_iterations);
    }

    #[test]
    fn critical_point_keyword_defaults_match_the_rust_defaults() {
        let rust = ArithmaCriticalPointsConfig::default();
        let facade = critical_config(1e-10, 1e-8, 1e-12, 100).unwrap();
        assert_eq!(facade, rust);
    }

    #[test]
    fn rejects_a_critical_point_config_with_a_bad_threshold() {
        assert!(critical_config(0.0, 1e-8, 1e-12, 100).is_err());
        assert!(critical_config(1e-10, -1.0, 1e-12, 100).is_err());
        assert!(critical_config(1e-10, 1e-8, f64::NAN, 100).is_err());
        assert!(critical_config(1e-10, 1e-8, 1e-12, 0).is_err());
    }

    // ── convergence gate ──────────────────────────────────────────────────

    #[test]
    fn accepts_a_converged_result() {
        let result = ArithmaRootFindingResult {
            root: 1.5,
            iterations: 12,
            converged: true,
        };
        assert!(convergence_error(&result, "test", 1024).is_none());
    }

    #[test]
    fn rejects_a_result_that_exhausted_its_budget() {
        let result = ArithmaRootFindingResult {
            root: 1.5,
            iterations: 1024,
            converged: false,
        };
        let message = convergence_error(&result, "test", 1024).expect("must be rejected");
        assert!(
            message.contains("1024"),
            "message should quote the budget: {message}"
        );
        assert!(
            message.contains("1.5"),
            "message should quote the best estimate: {message}"
        );
    }

    #[test]
    fn rejects_a_result_carrying_a_non_finite_root() {
        let result = ArithmaRootFindingResult {
            root: f64::NAN,
            iterations: 3,
            converged: true,
        };
        assert!(convergence_error(&result, "test", 1024).is_some());
    }

    #[test]
    fn rejects_a_result_that_overran_the_reported_budget() {
        // Defence against a solver that stops honouring its own cap.
        let result = ArithmaRootFindingResult {
            root: 1.0,
            iterations: 2048,
            converged: true,
        };
        assert!(convergence_error(&result, "test", 1024).is_some());
    }

    // ── error messages ────────────────────────────────────────────────────

    #[test]
    fn names_the_entry_point_in_an_unprefixed_message() {
        assert_eq!(
            prefix_message("find_root_bisection", "newton-raphson diverged"),
            "find_root_bisection: newton-raphson diverged"
        );
    }

    #[test]
    fn leaves_a_message_that_already_names_the_entry_point_alone() {
        // `interval_analysis` and `methods` self-prefix; doubling it up reads
        // like a bug in the wrapper.
        assert_eq!(
            prefix_message(
                "evaluate_interval",
                "evaluate_interval: unbound variable 'y'"
            ),
            "evaluate_interval: unbound variable 'y'"
        );
    }

    // ── sequence bounds ───────────────────────────────────────────────────

    #[test]
    fn rejects_a_sequence_longer_than_the_cap() {
        assert!(check_sequence_len(MAX_SEQUENCE_LEN, "hull").is_ok());
        assert!(check_sequence_len(MAX_SEQUENCE_LEN + 1, "hull").is_err());
    }
}
