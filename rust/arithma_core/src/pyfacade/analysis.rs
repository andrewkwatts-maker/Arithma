//====== Arithma/rust/arithma_core/src/pyfacade/analysis.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! PyO3 facade for the analysis domain: [`crate::equation_solver`] and
//! [`crate::fourier`].
//!
//! Surface
//! -------
//!
//! | Python | Rust |
//! |---|---|
//! | `solve(expr, var, strategy)` | [`crate::equation_solver::solve`] |
//! | `solve_equation(lhs, rhs, var, strategy)` | [`crate::equation_solver::solve_equation`] |
//! | `solve_system(equations, variables, strategy)` | [`crate::equation_solver::solve_system`] |
//! | `Solution` | [`crate::equation_solver::ArithmaSolution`] |
//! | `fourier_transform(expr, var, config)` | [`crate::fourier::fourier_transform`] |
//! | `FourierConfig` | [`crate::fourier::ArithmaFourierConfig`] |
//! | `FourierTransform` | [`crate::fourier::ArithmaFourierTransform`] |
//! | `fourier_window_weight(window, t)` | [`crate::fourier::ArithmaFourierWindow::weight`] |
//! | `SOLVER_STRATEGIES`, `FOURIER_WINDOWS` | the accepted enum spellings |
//!
//! Both wrapped modules are complete implementations, not scaffolding: the
//! solver runs a real closed-form pass plus a bisection scan, and the Fourier
//! pipeline runs real composite-Simpson quadrature.
//!
//! Enums as strings
//! ----------------
//!
//! [`ArithmaSolverStrategy`] and [`ArithmaFourierWindow`] cross the boundary as
//! lowercase `str`, not as wrapped enum types. A Python caller writing
//! `strategy="numeric"` needs no import and no second class to discover, and a
//! typo is answered by an error naming every accepted value. `geometry.rs` does
//! the same thing for its intersection discriminator. Case and the `_` / `-`
//! separators are normalised, so `"BlackmanHarris"` and `"blackman_harris"`
//! both land on the same variant.
//!
//! Hazards this wrapper absorbs
//! ----------------------------
//!
//! - `solve` with the `algebraic` strategy raises on an *identity* (`0 = 0`,
//!   satisfied by every value) instead of returning an empty list, because the
//!   Rust layer treats "every x" and "no x" as genuinely different answers.
//!   That error surfaces as `ValueError`; do not confuse it with `[]`, which
//!   means no solution exists.
//! - `solve_system` ignores its `strategy` argument — the Rust routine takes it
//!   and does not read it, because only the linear path is implemented. The
//!   parameter is still validated here so a typo is not silently accepted.
//! - `fourier_transform` cost is `sample_count × harmonics`. Both are capped by
//!   the core (`ARITHMA_FOURIER_MAX_SAMPLES`, `ARITHMA_FOURIER_MAX_HARMONICS`)
//!   and again by their product against an internal work budget.
//!   [`FourierConfig`] validates at construction so an excessive value raises
//!   `ValueError` at the point the caller wrote it, not deep inside a transform.
//! - `Expression::from_inner` is private to `core.rs` and that file is not ours
//!   to edit, so this module builds the wrapper through a struct literal.
//!   `Expression::inner` is `pub(crate)` and we are in the same crate; this is
//!   what `linalg.rs` and `calculus.rs` already do.

// ── Lint policy ──────────────────────────────────────────────────────────────
// pyo3 0.22 expands every `#[pyfunction]` / `#[pymethods]` member returning
// `PyResult<T>` into a trampoline containing a `PyErr -> PyErr` conversion, and
// clippy attributes that conversion to *our* return-type span. There is nothing
// to remove at the call site. Scoped to this file, matching `calculus.rs`,
// `geometry.rs` and `linalg.rs`.
#![allow(clippy::useless_conversion)]

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PySequence, PyString};

use crate::equation_solver::{
    solve as rust_solve, solve_equation as rust_solve_equation, solve_system as rust_solve_system,
    ArithmaSolution, ArithmaSolverStrategy,
};
use crate::expression::ArithmaExpression;
use crate::fourier::{
    fourier_transform as rust_fourier_transform, ArithmaFourierConfig, ArithmaFourierTransform,
    ArithmaFourierWindow, ARITHMA_FOURIER_MAX_HARMONICS, ARITHMA_FOURIER_MAX_SAMPLES,
};
use crate::pyfacade::core::Expression;
use crate::pyfacade::MAX_SEQUENCE_LEN;

// ============================================================================
// Bounds and accepted spellings.
// ============================================================================

/// Upper bound on the byte length of a variable name. Names are identifiers,
/// not payloads. The bound also fixes the iteration count of the whitespace
/// scan in [`check_variable_name`].
const MAX_VARIABLE_NAME_LEN: usize = 256;

/// Upper bound on the byte length of a strategy / window name. Fixes the
/// iteration count of the normalisation pass in [`normalise_enum_name`].
const MAX_ENUM_NAME_LEN: usize = 64;

/// Largest system `solve_system` will accept. Gaussian elimination is `O(n³)`
/// and coefficient recovery costs three expression evaluations per cell, so a
/// far tighter cap than [`MAX_SEQUENCE_LEN`] is the honest limit here.
const MAX_SYSTEM_SIZE: usize = 64;

/// Upper bound on the number of solution branches converted back to Python.
/// The numeric scan can legitimately return hundreds of roots for a periodic
/// function; this bounds the conversion loop without cutting into real results.
const MAX_SOLUTION_BRANCHES: usize = 8192;

/// Accepted spellings for [`ArithmaSolverStrategy`], in the order reported by
/// the error message and by the `SOLVER_STRATEGIES` module attribute.
const SOLVER_STRATEGY_NAMES: [&str; 4] = ["auto", "algebraic", "numeric", "hybrid"];

/// Accepted spellings for [`ArithmaFourierWindow`].
const FOURIER_WINDOW_NAMES: [&str; 6] = [
    "rectangular",
    "hann",
    "hamming",
    "blackman",
    "blackman_harris",
    "gaussian",
];

// ============================================================================
// Pure helpers (unit-tested below without a Python interpreter).
// ============================================================================

/// Fold a caller-supplied enum spelling to its canonical key: lowercase, with
/// `_`, `-` and spaces removed. Returns `Err` for an absurd length rather than
/// walking an unbounded string.
fn normalise_enum_name(name: &str) -> Result<String, String> {
    if name.is_empty() {
        return Err("name must not be empty".to_string());
    }
    if name.len() > MAX_ENUM_NAME_LEN {
        return Err(format!(
            "name is {} bytes, exceeding the {MAX_ENUM_NAME_LEN}-byte limit",
            name.len()
        ));
    }
    // Bounded by the length check immediately above.
    Ok(name
        .chars()
        .filter(|c| !matches!(c, '_' | '-' | ' '))
        .flat_map(char::to_lowercase)
        .collect())
}

/// Parse a solver strategy name. The error lists every accepted value, because
/// a Python caller has no enum type to consult.
fn parse_strategy(name: &str) -> Result<ArithmaSolverStrategy, String> {
    let key = normalise_enum_name(name).map_err(|e| format!("strategy: {e}"))?;
    match key.as_str() {
        "auto" => Ok(ArithmaSolverStrategy::Auto),
        "algebraic" => Ok(ArithmaSolverStrategy::Algebraic),
        "numeric" => Ok(ArithmaSolverStrategy::Numeric),
        "hybrid" => Ok(ArithmaSolverStrategy::Hybrid),
        _ => Err(format!(
            "unknown strategy {name:?}; valid values are: {}",
            SOLVER_STRATEGY_NAMES.join(", ")
        )),
    }
}

/// Canonical lowercase name of a strategy. Round-trips with [`parse_strategy`].
fn strategy_name(strategy: ArithmaSolverStrategy) -> &'static str {
    match strategy {
        ArithmaSolverStrategy::Auto => "auto",
        ArithmaSolverStrategy::Algebraic => "algebraic",
        ArithmaSolverStrategy::Numeric => "numeric",
        ArithmaSolverStrategy::Hybrid => "hybrid",
    }
}

/// Parse a window name. The error lists every accepted value.
fn parse_window(name: &str) -> Result<ArithmaFourierWindow, String> {
    let key = normalise_enum_name(name).map_err(|e| format!("window: {e}"))?;
    match key.as_str() {
        "rectangular" => Ok(ArithmaFourierWindow::Rectangular),
        "hann" => Ok(ArithmaFourierWindow::Hann),
        "hamming" => Ok(ArithmaFourierWindow::Hamming),
        "blackman" => Ok(ArithmaFourierWindow::Blackman),
        "blackmanharris" => Ok(ArithmaFourierWindow::BlackmanHarris),
        "gaussian" => Ok(ArithmaFourierWindow::Gaussian),
        _ => Err(format!(
            "unknown window {name:?}; valid values are: {}",
            FOURIER_WINDOW_NAMES.join(", ")
        )),
    }
}

/// Canonical lowercase name of a window. Round-trips with [`parse_window`].
fn window_name(window: ArithmaFourierWindow) -> &'static str {
    match window {
        ArithmaFourierWindow::Rectangular => "rectangular",
        ArithmaFourierWindow::Hann => "hann",
        ArithmaFourierWindow::Hamming => "hamming",
        ArithmaFourierWindow::Blackman => "blackman",
        ArithmaFourierWindow::BlackmanHarris => "blackman_harris",
        ArithmaFourierWindow::Gaussian => "gaussian",
    }
}

/// Validate a variable name arriving from Python. `str` guarantees UTF-8 and
/// nothing else; emptiness, padding and absurd lengths are checked here because
/// the Rust layer only `debug_assert!`s them and a release build would happily
/// solve for `""`.
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

/// Validate the length of a system-of-equations sequence before any iteration
/// over it begins.
fn check_system_size(len: usize, what: &str) -> Result<(), String> {
    if len > MAX_SEQUENCE_LEN {
        return Err(format!(
            "{what}: sequence of {len} elements exceeds the maximum of {MAX_SEQUENCE_LEN}"
        ));
    }
    if len > MAX_SYSTEM_SIZE {
        return Err(format!(
            "{what}: {len} entries exceed the {MAX_SYSTEM_SIZE}-equation limit"
        ));
    }
    if len == 0 {
        return Err(format!("{what}: must not be empty"));
    }
    Ok(())
}

/// Validate the branch count coming back from the solver before it is walked.
fn check_branch_count(len: usize) -> Result<(), String> {
    if len > MAX_SOLUTION_BRANCHES {
        return Err(format!(
            "solver returned {len} branches, exceeding the {MAX_SOLUTION_BRANCHES} conversion limit"
        ));
    }
    Ok(())
}

/// Validate a Fourier sample count against the core's own cap. Checked here so
/// the failure names the constructor argument rather than surfacing from inside
/// the transform.
fn check_sample_count(count: usize) -> Result<(), String> {
    if count < 2 {
        return Err(format!("sample_count must be at least 2, got {count}"));
    }
    if count > ARITHMA_FOURIER_MAX_SAMPLES {
        return Err(format!(
            "sample_count {count} exceeds the maximum of {ARITHMA_FOURIER_MAX_SAMPLES}"
        ));
    }
    Ok(())
}

/// Validate a harmonic count against the core's own cap.
fn check_harmonics(count: usize) -> Result<(), String> {
    if count == 0 {
        return Err("harmonics must be at least 1".to_string());
    }
    if count > ARITHMA_FOURIER_MAX_HARMONICS {
        return Err(format!(
            "harmonics {count} exceeds the maximum of {ARITHMA_FOURIER_MAX_HARMONICS}"
        ));
    }
    Ok(())
}

/// Validate the transform domain. A degenerate or reversed range would divide
/// the fundamental by zero or a negative length.
fn check_range(lo: f64, hi: f64) -> Result<(), String> {
    if !lo.is_finite() || !hi.is_finite() {
        return Err(format!("range ({lo}, {hi}) must be finite"));
    }
    if hi <= lo {
        return Err(format!("range ({lo}, {hi}) must satisfy lo < hi"));
    }
    Ok(())
}

/// Validate the RMS accuracy target. `0.0` is legal and means "keep every
/// coefficient"; NaN and negatives are not.
fn check_accuracy(accuracy: f64) -> Result<(), String> {
    if !accuracy.is_finite() {
        return Err(format!("accuracy must be finite, got {accuracy}"));
    }
    if accuracy < 0.0 {
        return Err(format!("accuracy must not be negative, got {accuracy}"));
    }
    Ok(())
}

// ============================================================================
// Python <-> Rust conversion.
// ============================================================================

/// Wrap an `ArithmaExpression` in the `Expression` pyclass. See the module note
/// on `Expression::from_inner` being private.
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

/// Borrow a Python object as a sequence. `str` is rejected up front: it passes
/// `PySequence_Check` and would otherwise decompose into single characters.
fn as_sequence<'py>(obj: &Bound<'py, PyAny>, what: &str) -> PyResult<Bound<'py, PySequence>> {
    if obj.is_instance_of::<PyString>() {
        return Err(PyTypeError::new_err(format!(
            "{what}: expected a list or tuple, not a str"
        )));
    }
    let seq = obj
        .downcast::<PySequence>()
        .map_err(|_| PyTypeError::new_err(format!("{what}: expected a list or tuple")))?;
    Ok(seq.clone())
}

/// Validate a variable name and surface a failure as `ValueError`.
fn variable_name_or_err(var: &str, what: &str) -> PyResult<()> {
    check_variable_name(var).map_err(|e| PyValueError::new_err(format!("{what}: {e}")))
}

/// Parse a strategy name and surface a failure as `ValueError`.
fn strategy_or_err(name: &str, what: &str) -> PyResult<ArithmaSolverStrategy> {
    parse_strategy(name).map_err(|e| PyValueError::new_err(format!("{what}: {e}")))
}

/// Convert solver output into the Python-facing branch list. The length is
/// checked before the walk so the loop bound is fixed.
fn wrap_solutions(found: Vec<ArithmaSolution>, what: &str) -> PyResult<Vec<Solution>> {
    check_branch_count(found.len()).map_err(|e| PyRuntimeError::new_err(format!("{what}: {e}")))?;
    debug_assert!(
        found.len() <= MAX_SOLUTION_BRANCHES,
        "wrap_solutions: branch count passed validation but exceeds the cap"
    );
    debug_assert!(
        found.iter().all(|s| s.is_real || s.cached.is_none()),
        "wrap_solutions: a non-real branch carried a cached real value"
    );
    // Fixed bound: `found.len()` was checked against MAX_SOLUTION_BRANCHES.
    Ok(found.into_iter().map(|inner| Solution { inner }).collect())
}

// ============================================================================
// `Solution` pyclass.
// ============================================================================

/// One root or solution branch — Python wrapper around `ArithmaSolution`.
///
/// ``expression`` is the symbolic form, ``cached`` the numeric value when the
/// solver has one (``None`` for a complex branch), and ``is_real`` says whether
/// the branch lies on the real line. All three are read-only.
#[pyclass(name = "Solution", module = "arithma")]
#[derive(Clone)]
pub struct Solution {
    inner: ArithmaSolution,
}

#[pymethods]
impl Solution {
    /// Symbolic form of this branch.
    #[getter]
    fn expression(&self) -> Expression {
        debug_assert!(
            self.inner.is_real || self.inner.cached.is_none(),
            "a non-real branch must not carry a cached real value"
        );
        debug_assert!(
            !matches!(&self.inner.expression, ArithmaExpression::Variable(n) if n.is_empty()),
            "solution expression holds a variable with an empty name"
        );
        to_py_expression(self.inner.expression.clone())
    }

    /// Cached numeric value, or ``None`` when the branch has no real value.
    #[getter]
    fn cached(&self) -> Option<f64> {
        debug_assert!(
            self.inner.is_real || self.inner.cached.is_none(),
            "a non-real branch must not carry a cached real value"
        );
        debug_assert!(
            !matches!(&self.inner.expression, ArithmaExpression::Variable(n) if n.is_empty()),
            "solution expression holds a variable with an empty name"
        );
        self.inner.cached
    }

    /// True when this branch is real-valued.
    #[getter]
    fn is_real(&self) -> bool {
        debug_assert!(
            self.inner.is_real || self.inner.cached.is_none(),
            "a non-real branch must not carry a cached real value"
        );
        debug_assert!(
            !matches!(&self.inner.expression, ArithmaExpression::Variable(n) if n.is_empty()),
            "solution expression holds a variable with an empty name"
        );
        self.inner.is_real
    }

    fn __repr__(&self) -> String {
        debug_assert!(
            self.inner.is_real || self.inner.cached.is_none(),
            "a non-real branch must not carry a cached real value"
        );
        debug_assert!(
            !matches!(&self.inner.expression, ArithmaExpression::Variable(n) if n.is_empty()),
            "solution expression holds a variable with an empty name"
        );
        match self.inner.cached {
            Some(v) => format!("Solution({v}, is_real={})", self.inner.is_real),
            None => format!("Solution(<symbolic>, is_real={})", self.inner.is_real),
        }
    }
}

// ============================================================================
// Solver functions.
// ============================================================================

/// Solve ``expr = 0`` for ``var``, returning every branch the solver finds.
///
/// ``strategy`` is one of ``"auto"``, ``"algebraic"``, ``"numeric"`` or
/// ``"hybrid"`` (case and ``_``/``-`` separators are normalised):
///
/// - ``"algebraic"`` — closed forms only, detected by sampling the expression
///   as a polynomial of degree ≤ 2. Raises :class:`ValueError` when no closed
///   form applies, rather than quietly returning nothing.
/// - ``"numeric"`` — scans ``[-100, 100]`` for sign changes and bisects each
///   bracket. Roots outside that span, or of even multiplicity, are not found.
/// - ``"auto"`` / ``"hybrid"`` — closed form first, numeric scan as fallback.
///
/// An *identity* (``0 = 0``, satisfied by every value) raises
/// :class:`ValueError`; an empty list means no solution exists. Those are
/// different answers and the engine refuses to conflate them.
#[pyfunction]
#[pyo3(signature = (expr, var, strategy = "auto"))]
fn solve(expr: &Bound<'_, PyAny>, var: &str, strategy: &str) -> PyResult<Vec<Solution>> {
    variable_name_or_err(var, "solve")?;
    let picked = strategy_or_err(strategy, "solve")?;
    let inner = coerce_expression(expr, "solve")?;
    debug_assert!(!var.is_empty(), "solve: empty variable name");
    debug_assert!(
        var.len() <= MAX_VARIABLE_NAME_LEN,
        "solve: oversized variable name"
    );
    // The canonical strategy name goes into the message: "no closed form
    // available" is only actionable once you know which path produced it.
    let found = rust_solve(&inner, var, picked).map_err(|e| {
        PyValueError::new_err(format!("solve ({} strategy): {e}", strategy_name(picked)))
    })?;
    wrap_solutions(found, "solve")
}

/// Solve ``lhs = rhs`` for ``var`` by rewriting to ``lhs - rhs = 0``.
///
/// Shares every rule documented on :func:`solve`, including the strategy names
/// and the identity-is-an-error behaviour.
#[pyfunction]
#[pyo3(signature = (lhs, rhs, var, strategy = "auto"))]
fn solve_equation(
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
    var: &str,
    strategy: &str,
) -> PyResult<Vec<Solution>> {
    variable_name_or_err(var, "solve_equation")?;
    let picked = strategy_or_err(strategy, "solve_equation")?;
    let lhs_expr = coerce_expression(lhs, "solve_equation (lhs)")?;
    let rhs_expr = coerce_expression(rhs, "solve_equation (rhs)")?;
    debug_assert!(!var.is_empty(), "solve_equation: empty variable name");
    debug_assert!(
        var.len() <= MAX_VARIABLE_NAME_LEN,
        "solve_equation: oversized variable name"
    );
    let found = rust_solve_equation(&lhs_expr, &rhs_expr, var, picked).map_err(|e| {
        PyValueError::new_err(format!(
            "solve_equation ({} strategy): {e}",
            strategy_name(picked)
        ))
    })?;
    wrap_solutions(found, "solve_equation")
}

/// Read a sequence of ``(lhs, rhs)`` pairs into Rust equations.
fn extract_equations(
    obj: &Bound<'_, PyAny>,
) -> PyResult<Vec<(ArithmaExpression, ArithmaExpression)>> {
    let seq = as_sequence(obj, "solve_system: equations")?;
    let len = seq.len()?;
    check_system_size(len, "solve_system: equations").map_err(PyValueError::new_err)?;
    debug_assert!(
        len > 0,
        "extract_equations: empty sequence reached the walk"
    );
    debug_assert!(
        len <= MAX_SYSTEM_SIZE,
        "extract_equations: length passed validation but exceeds the cap"
    );
    let mut out = Vec::with_capacity(len);
    // Fixed bound: `len` was checked against MAX_SYSTEM_SIZE above.
    for i in 0..len {
        let item = seq.get_item(i)?;
        let pair = as_sequence(&item, &format!("solve_system: equations[{i}]"))?;
        if pair.len()? != 2 {
            return Err(PyValueError::new_err(format!(
                "solve_system: equations[{i}] must be a (lhs, rhs) pair"
            )));
        }
        let lhs = coerce_expression(
            &pair.get_item(0)?,
            &format!("solve_system: equations[{i}][0]"),
        )?;
        let rhs = coerce_expression(
            &pair.get_item(1)?,
            &format!("solve_system: equations[{i}][1]"),
        )?;
        out.push((lhs, rhs));
    }
    debug_assert_eq!(out.len(), len, "extract_equations: dropped an equation");
    Ok(out)
}

/// Read a sequence of variable names, validating each one.
fn extract_var_names(obj: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let seq = as_sequence(obj, "solve_system: variables")?;
    let len = seq.len()?;
    check_system_size(len, "solve_system: variables").map_err(PyValueError::new_err)?;
    debug_assert!(
        len > 0,
        "extract_var_names: empty sequence reached the walk"
    );
    debug_assert!(
        len <= MAX_SYSTEM_SIZE,
        "extract_var_names: length passed validation but exceeds the cap"
    );
    let mut out: Vec<String> = Vec::with_capacity(len);
    // Fixed bound: `len` was checked against MAX_SYSTEM_SIZE above.
    for i in 0..len {
        let name: String = seq.get_item(i)?.extract().map_err(|_| {
            PyTypeError::new_err(format!("solve_system: variables[{i}] must be str"))
        })?;
        variable_name_or_err(&name, &format!("solve_system: variables[{i}]"))?;
        out.push(name);
    }
    debug_assert_eq!(out.len(), len, "extract_var_names: dropped a name");
    Ok(out)
}

/// Solve a **linear** system of equations for the listed variables.
///
/// ``equations`` is a sequence of ``(lhs, rhs)`` pairs; ``variables`` a
/// sequence of names, one per equation. Each equation is sampled to recover its
/// coefficient row and the matrix is solved by Gaussian elimination with
/// partial pivoting. A non-linear equation is rejected with
/// :class:`ValueError` rather than linearised silently, and a singular system
/// is reported rather than answered with one arbitrary solution.
///
/// Returns a list of solution lists — a linear system has at most one branch,
/// so the outer list holds exactly one entry, ordered to match ``variables``.
///
/// ``strategy`` is accepted and validated but has no effect: the Rust routine
/// implements only the linear path and ignores it.
#[pyfunction]
#[pyo3(signature = (equations, variables, strategy = "auto"))]
fn solve_system(
    equations: &Bound<'_, PyAny>,
    variables: &Bound<'_, PyAny>,
    strategy: &str,
) -> PyResult<Vec<Vec<Solution>>> {
    let picked = strategy_or_err(strategy, "solve_system")?;
    let eqs = extract_equations(equations)?;
    let names = extract_var_names(variables)?;
    if eqs.len() != names.len() {
        return Err(PyValueError::new_err(format!(
            "solve_system: needs one equation per variable: {} equations, {} variables",
            eqs.len(),
            names.len()
        )));
    }
    debug_assert_eq!(
        eqs.len(),
        names.len(),
        "solve_system: shape mismatch survived validation"
    );
    debug_assert!(!names.is_empty(), "solve_system: empty variable list");
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let branches = rust_solve_system(&eqs, &refs, picked)
        .map_err(|e| PyValueError::new_err(format!("solve_system: {e}")))?;
    check_branch_count(branches.len())
        .map_err(|e| PyRuntimeError::new_err(format!("solve_system: {e}")))?;
    let mut out: Vec<Vec<Solution>> = Vec::with_capacity(branches.len());
    // Fixed bound: `branches.len()` was checked against MAX_SOLUTION_BRANCHES.
    for branch in branches {
        out.push(wrap_solutions(branch, "solve_system")?);
    }
    Ok(out)
}

// ============================================================================
// `FourierConfig` pyclass.
// ============================================================================

/// Fourier pipeline configuration — Python wrapper around
/// `ArithmaFourierConfig`.
///
/// Every field is validated at construction, so an oversized ``sample_count``
/// or ``harmonics`` raises :class:`ValueError` where the caller wrote it rather
/// than from inside a transform. ``window`` is a lowercase name; see
/// ``FOURIER_WINDOWS`` for the accepted set. All getters are read-only —
/// build a new config rather than mutating one that a transform already holds.
#[pyclass(name = "FourierConfig", module = "arithma")]
#[derive(Clone)]
pub struct FourierConfig {
    inner: ArithmaFourierConfig,
}

#[pymethods]
impl FourierConfig {
    /// Defaults match the Rust core: 1024 samples over ``(-pi, pi)``, 32
    /// harmonics, an RMS accuracy target of ``1e-6`` and the Hann window.
    #[new]
    #[pyo3(signature = (
        sample_count = 1024,
        range = None,
        harmonics = 32,
        accuracy = 1e-6,
        window = "hann",
    ))]
    fn new(
        sample_count: usize,
        range: Option<(f64, f64)>,
        harmonics: usize,
        accuracy: f64,
        window: &str,
    ) -> PyResult<Self> {
        let (lo, hi) = range.unwrap_or((-std::f64::consts::PI, std::f64::consts::PI));
        check_sample_count(sample_count)
            .map_err(|e| PyValueError::new_err(format!("FourierConfig: {e}")))?;
        check_harmonics(harmonics)
            .map_err(|e| PyValueError::new_err(format!("FourierConfig: {e}")))?;
        check_range(lo, hi).map_err(|e| PyValueError::new_err(format!("FourierConfig: {e}")))?;
        check_accuracy(accuracy)
            .map_err(|e| PyValueError::new_err(format!("FourierConfig: {e}")))?;
        let picked = parse_window(window)
            .map_err(|e| PyValueError::new_err(format!("FourierConfig: {e}")))?;
        debug_assert!(
            (2..=ARITHMA_FOURIER_MAX_SAMPLES).contains(&sample_count),
            "FourierConfig: sample_count passed validation but is out of range"
        );
        debug_assert!(
            hi > lo,
            "FourierConfig: range passed validation but is degenerate"
        );
        Ok(Self {
            inner: ArithmaFourierConfig {
                sample_count,
                range: (lo, hi),
                harmonics,
                accuracy,
                window: picked,
            },
        })
    }

    /// Number of samples in the discrete transform.
    #[getter]
    fn sample_count(&self) -> usize {
        debug_assert!(
            self.inner.sample_count >= 2,
            "sample_count was validated at construction"
        );
        debug_assert!(
            self.inner.harmonics >= 1,
            "harmonics was validated at construction"
        );
        self.inner.sample_count
    }

    /// Domain over which the transform is computed, as ``(lo, hi)``.
    #[getter]
    fn range(&self) -> (f64, f64) {
        debug_assert!(
            self.inner.range.1 > self.inner.range.0,
            "range was validated at construction"
        );
        debug_assert!(
            self.inner.range.0.is_finite() && self.inner.range.1.is_finite(),
            "range endpoints were validated as finite at construction"
        );
        self.inner.range
    }

    /// Number of harmonics retained in the truncated series.
    #[getter]
    fn harmonics(&self) -> usize {
        debug_assert!(
            self.inner.harmonics >= 1,
            "harmonics was validated at construction"
        );
        debug_assert!(
            self.inner.harmonics <= ARITHMA_FOURIER_MAX_HARMONICS,
            "harmonics was capped at construction"
        );
        self.inner.harmonics
    }

    /// Target RMS reconstruction error. ``0.0`` keeps every coefficient.
    #[getter]
    fn accuracy(&self) -> f64 {
        debug_assert!(
            self.inner.accuracy.is_finite(),
            "accuracy was validated as finite at construction"
        );
        debug_assert!(
            self.inner.accuracy >= 0.0,
            "accuracy was validated as non-negative at construction"
        );
        self.inner.accuracy
    }

    /// Window name, one of ``FOURIER_WINDOWS``.
    #[getter]
    fn window(&self) -> &'static str {
        debug_assert!(
            self.inner.sample_count >= 2,
            "sample_count was validated at construction"
        );
        debug_assert!(
            parse_window(window_name(self.inner.window)).is_ok(),
            "window name must round-trip through the parser"
        );
        window_name(self.inner.window)
    }

    fn __repr__(&self) -> String {
        debug_assert!(
            self.inner.sample_count >= 2,
            "sample_count was validated at construction"
        );
        debug_assert!(
            self.inner.range.1 > self.inner.range.0,
            "range was validated at construction"
        );
        format!(
            "FourierConfig(sample_count={}, range=({}, {}), harmonics={}, accuracy={}, window={:?})",
            self.inner.sample_count,
            self.inner.range.0,
            self.inner.range.1,
            self.inner.harmonics,
            self.inner.accuracy,
            window_name(self.inner.window)
        )
    }
}

// ============================================================================
// `FourierTransform` pyclass.
// ============================================================================

/// Result of the Fourier pipeline — Python wrapper around
/// `ArithmaFourierTransform`.
///
/// Carries the cosine and sine coefficient arrays plus the DC offset, alongside
/// the originating :class:`FourierConfig` so re-evaluation is deterministic.
/// All getters are read-only; the coefficient getters return copies.
#[pyclass(name = "FourierTransform", module = "arithma")]
#[derive(Clone)]
pub struct FourierTransform {
    inner: ArithmaFourierTransform,
}

#[pymethods]
impl FourierTransform {
    /// All-zero transform for a config. Useful as a neutral element.
    #[staticmethod]
    fn empty(config: PyRef<'_, FourierConfig>) -> Self {
        debug_assert!(
            config.inner.harmonics >= 1,
            "empty: harmonics was validated at construction"
        );
        debug_assert!(
            config.inner.range.1 > config.inner.range.0,
            "empty: range was validated at construction"
        );
        Self {
            inner: ArithmaFourierTransform::empty(config.inner.clone()),
        }
    }

    /// The configuration this transform was computed with.
    #[getter]
    fn config(&self) -> FourierConfig {
        debug_assert_eq!(
            self.inner.cos_coeffs.len(),
            self.inner.sin_coeffs.len(),
            "coefficient arrays must be paired"
        );
        debug_assert_eq!(
            self.inner.cos_coeffs.len(),
            self.inner.config.harmonics,
            "coefficient count must match the configured harmonics"
        );
        FourierConfig {
            inner: self.inner.config.clone(),
        }
    }

    /// Cosine (real) coefficients, harmonic 1 first.
    #[getter]
    fn cos_coeffs(&self) -> Vec<f64> {
        debug_assert_eq!(
            self.inner.cos_coeffs.len(),
            self.inner.sin_coeffs.len(),
            "coefficient arrays must be paired"
        );
        debug_assert_eq!(
            self.inner.cos_coeffs.len(),
            self.inner.config.harmonics,
            "coefficient count must match the configured harmonics"
        );
        self.inner.cos_coeffs.clone()
    }

    /// Sine (imaginary) coefficients, harmonic 1 first.
    #[getter]
    fn sin_coeffs(&self) -> Vec<f64> {
        debug_assert_eq!(
            self.inner.cos_coeffs.len(),
            self.inner.sin_coeffs.len(),
            "coefficient arrays must be paired"
        );
        debug_assert_eq!(
            self.inner.sin_coeffs.len(),
            self.inner.config.harmonics,
            "coefficient count must match the configured harmonics"
        );
        self.inner.sin_coeffs.clone()
    }

    /// Constant DC offset.
    #[getter]
    fn dc(&self) -> f64 {
        debug_assert_eq!(
            self.inner.cos_coeffs.len(),
            self.inner.sin_coeffs.len(),
            "coefficient arrays must be paired"
        );
        debug_assert!(
            self.inner.config.sample_count >= 2,
            "config was validated at construction"
        );
        self.inner.dc
    }

    /// Reconstruct the value at ``x`` from the truncated series.
    ///
    /// Raises :class:`ValueError` for a non-finite ``x`` and
    /// :class:`RuntimeError` if the sum itself comes out non-finite, which
    /// would mean the coefficients have overflowed.
    fn evaluate(&self, x: f64) -> PyResult<f64> {
        if !x.is_finite() {
            return Err(PyValueError::new_err(format!(
                "evaluate: x must be finite, got {x}"
            )));
        }
        debug_assert!(
            x.is_finite(),
            "evaluate: x passed validation but is not finite"
        );
        debug_assert_eq!(
            self.inner.cos_coeffs.len(),
            self.inner.sin_coeffs.len(),
            "coefficient arrays must be paired"
        );
        let value = self.inner.evaluate(x);
        if !value.is_finite() {
            return Err(PyRuntimeError::new_err(format!(
                "evaluate: reconstruction at x = {x} is not finite ({value})"
            )));
        }
        Ok(value)
    }

    /// RMS error of this reconstruction against ``expr``, sampled uniformly
    /// across the configured range.
    ///
    /// Use it to check whether the ``accuracy`` target was actually met. Raises
    /// :class:`ValueError` if the expression fails to evaluate anywhere on the
    /// grid.
    fn rms_error(&self, expr: &Bound<'_, PyAny>, var: &str) -> PyResult<f64> {
        variable_name_or_err(var, "rms_error")?;
        let inner = coerce_expression(expr, "rms_error")?;
        debug_assert!(!var.is_empty(), "rms_error: empty variable name");
        debug_assert_eq!(
            self.inner.cos_coeffs.len(),
            self.inner.sin_coeffs.len(),
            "coefficient arrays must be paired"
        );
        let value = self
            .inner
            .rms_error(&inner, var)
            .map_err(|e| PyValueError::new_err(format!("rms_error: {e}")))?;
        if !value.is_finite() {
            return Err(PyRuntimeError::new_err(format!(
                "rms_error: error metric is not finite ({value})"
            )));
        }
        Ok(value)
    }

    /// Number of harmonic pairs held by this transform.
    fn __len__(&self) -> usize {
        debug_assert_eq!(
            self.inner.cos_coeffs.len(),
            self.inner.sin_coeffs.len(),
            "coefficient arrays must be paired"
        );
        debug_assert_eq!(
            self.inner.cos_coeffs.len(),
            self.inner.config.harmonics,
            "coefficient count must match the configured harmonics"
        );
        self.inner.cos_coeffs.len()
    }

    fn __repr__(&self) -> String {
        debug_assert_eq!(
            self.inner.cos_coeffs.len(),
            self.inner.sin_coeffs.len(),
            "coefficient arrays must be paired"
        );
        debug_assert!(
            self.inner.config.sample_count >= 2,
            "config was validated at construction"
        );
        format!(
            "FourierTransform(dc={}, harmonics={}, window={:?})",
            self.inner.dc,
            self.inner.cos_coeffs.len(),
            window_name(self.inner.config.window)
        )
    }
}

// ============================================================================
// Fourier module functions.
// ============================================================================

/// Compute the real Fourier series of ``expr`` with respect to ``var``.
///
/// ``config`` defaults to ``FourierConfig()``. Sampling is composite Simpson
/// quadrature over ``config.range``; the window is divided out by its coherent
/// gain, so a constant signal reproduces its own value as ``dc`` and a pure
/// sinusoid keeps unit amplitude under any window.
///
/// Cost is ``sample_count × harmonics``. Both are capped individually by
/// :class:`FourierConfig` and their product is capped again by the core, so an
/// excessive request raises :class:`ValueError` instead of running for minutes.
///
/// Raises :class:`ValueError` when the expression fails to evaluate, or is not
/// finite, at any sample point — a bad sample is never skipped.
#[pyfunction]
#[pyo3(signature = (expr, var, config = None))]
fn fourier_transform(
    expr: &Bound<'_, PyAny>,
    var: &str,
    config: Option<PyRef<'_, FourierConfig>>,
) -> PyResult<FourierTransform> {
    variable_name_or_err(var, "fourier_transform")?;
    let inner = coerce_expression(expr, "fourier_transform")?;
    let cfg = match &config {
        Some(c) => c.inner.clone(),
        None => ArithmaFourierConfig::default(),
    };
    check_sample_count(cfg.sample_count)
        .map_err(|e| PyValueError::new_err(format!("fourier_transform: {e}")))?;
    check_harmonics(cfg.harmonics)
        .map_err(|e| PyValueError::new_err(format!("fourier_transform: {e}")))?;
    debug_assert!(!var.is_empty(), "fourier_transform: empty variable name");
    debug_assert!(
        (2..=ARITHMA_FOURIER_MAX_SAMPLES).contains(&cfg.sample_count),
        "fourier_transform: sample_count passed validation but is out of range"
    );
    let harmonics = cfg.harmonics;
    let out = rust_fourier_transform(&inner, var, &cfg)
        .map_err(|e| PyValueError::new_err(format!("fourier_transform: {e}")))?;
    debug_assert_eq!(
        out.cos_coeffs.len(),
        harmonics,
        "fourier_transform: coefficient count does not match the request"
    );
    Ok(FourierTransform { inner: out })
}

/// Window weight at normalised position ``t``, where ``0`` is the low end of
/// the range and ``1`` the high end.
///
/// ``t`` is clamped into ``[0, 1]``, so a value outside that interval yields
/// the endpoint weight rather than a nonsensical one. ``t`` must still be
/// finite; NaN and infinities raise :class:`ValueError` rather than being
/// silently folded to ``0``.
#[pyfunction]
#[pyo3(signature = (window, t))]
fn fourier_window_weight(window: &str, t: f64) -> PyResult<f64> {
    let picked = parse_window(window)
        .map_err(|e| PyValueError::new_err(format!("fourier_window_weight: {e}")))?;
    if !t.is_finite() {
        return Err(PyValueError::new_err(format!(
            "fourier_window_weight: t must be finite, got {t}"
        )));
    }
    debug_assert!(
        t.is_finite(),
        "fourier_window_weight: t passed validation but is not finite"
    );
    let weight = picked.weight(t);
    debug_assert!(
        weight.is_finite(),
        "fourier_window_weight: window produced a non-finite weight"
    );
    Ok(weight)
}

// ============================================================================
// Registration
// ============================================================================

/// Register this module's surface on the extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Solution>()?;
    m.add_class::<FourierConfig>()?;
    m.add_class::<FourierTransform>()?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_function(wrap_pyfunction!(solve_equation, m)?)?;
    m.add_function(wrap_pyfunction!(solve_system, m)?)?;
    m.add_function(wrap_pyfunction!(fourier_transform, m)?)?;
    m.add_function(wrap_pyfunction!(fourier_window_weight, m)?)?;
    // Published so a caller can enumerate the accepted enum spellings instead
    // of learning them from an exception.
    m.add("SOLVER_STRATEGIES", SOLVER_STRATEGY_NAMES.to_vec())?;
    m.add("FOURIER_WINDOWS", FOURIER_WINDOW_NAMES.to_vec())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── enum name normalisation ───────────────────────────────────────────

    #[test]
    fn normalises_case_and_separators() {
        assert_eq!(
            normalise_enum_name("Blackman-Harris").unwrap(),
            "blackmanharris"
        );
        assert_eq!(
            normalise_enum_name("BLACKMAN_HARRIS").unwrap(),
            "blackmanharris"
        );
        assert_eq!(
            normalise_enum_name("blackman harris").unwrap(),
            "blackmanharris"
        );
    }

    #[test]
    fn rejects_an_empty_enum_name() {
        assert!(normalise_enum_name("").is_err());
    }

    #[test]
    fn rejects_an_absurdly_long_enum_name() {
        let too_long = "x".repeat(MAX_ENUM_NAME_LEN + 1);
        assert!(normalise_enum_name(&too_long).is_err());
        let at_limit = "x".repeat(MAX_ENUM_NAME_LEN);
        assert!(normalise_enum_name(&at_limit).is_ok());
    }

    // ── strategy parsing ──────────────────────────────────────────────────

    #[test]
    fn parses_every_documented_strategy_name() {
        // Fixed bound: the published name table.
        for name in SOLVER_STRATEGY_NAMES {
            let parsed = parse_strategy(name)
                .unwrap_or_else(|e| panic!("published strategy {name:?} did not parse: {e}"));
            assert_eq!(strategy_name(parsed), name, "{name} did not round-trip");
        }
    }

    #[test]
    fn accepts_a_strategy_name_in_mixed_case() {
        assert_eq!(
            parse_strategy("Numeric").unwrap(),
            ArithmaSolverStrategy::Numeric
        );
        assert_eq!(
            parse_strategy("HYBRID").unwrap(),
            ArithmaSolverStrategy::Hybrid
        );
    }

    #[test]
    fn rejects_an_unknown_strategy_name() {
        let err = parse_strategy("quantum").unwrap_err();
        assert!(err.contains("quantum"), "message was {err:?}");
        // The whole accepted set has to be in the message; a Python caller has
        // no enum type to consult.
        for name in SOLVER_STRATEGY_NAMES {
            assert!(err.contains(name), "message {err:?} omitted {name}");
        }
    }

    #[test]
    fn rejects_an_empty_strategy_name() {
        assert!(parse_strategy("").is_err());
    }

    #[test]
    fn the_default_strategy_name_is_auto() {
        assert_eq!(strategy_name(ArithmaSolverStrategy::default()), "auto");
    }

    // ── window parsing ────────────────────────────────────────────────────

    #[test]
    fn parses_every_documented_window_name() {
        // Fixed bound: the published name table.
        for name in FOURIER_WINDOW_NAMES {
            let parsed = parse_window(name)
                .unwrap_or_else(|e| panic!("published window {name:?} did not parse: {e}"));
            assert_eq!(window_name(parsed), name, "{name} did not round-trip");
        }
    }

    #[test]
    fn accepts_a_window_name_in_mixed_case_or_with_a_dash() {
        assert_eq!(parse_window("Hann").unwrap(), ArithmaFourierWindow::Hann);
        assert_eq!(
            parse_window("blackman-harris").unwrap(),
            ArithmaFourierWindow::BlackmanHarris
        );
        assert_eq!(
            parse_window("BlackmanHarris").unwrap(),
            ArithmaFourierWindow::BlackmanHarris
        );
    }

    #[test]
    fn rejects_an_unknown_window_name() {
        let err = parse_window("kaiser").unwrap_err();
        assert!(err.contains("kaiser"), "message was {err:?}");
        for name in FOURIER_WINDOW_NAMES {
            assert!(err.contains(name), "message {err:?} omitted {name}");
        }
    }

    #[test]
    fn rejects_an_empty_window_name() {
        assert!(parse_window("").is_err());
    }

    #[test]
    fn the_default_config_window_is_a_published_name() {
        let default = ArithmaFourierConfig::default();
        assert!(FOURIER_WINDOW_NAMES.contains(&window_name(default.window)));
    }

    // ── numeric validation ────────────────────────────────────────────────

    #[test]
    fn variable_name_rejects_empty_padded_and_oversized_input() {
        assert!(check_variable_name("").is_err());
        assert!(check_variable_name(" x").is_err());
        assert!(check_variable_name("a b").is_err());
        assert!(check_variable_name("x").is_ok());
        assert!(check_variable_name("theta_1").is_ok());
        assert!(check_variable_name(&"v".repeat(MAX_VARIABLE_NAME_LEN)).is_ok());
        assert!(check_variable_name(&"v".repeat(MAX_VARIABLE_NAME_LEN + 1)).is_err());
    }

    #[test]
    fn system_size_must_be_non_empty_and_within_the_cap() {
        assert!(check_system_size(0, "t").is_err());
        assert!(check_system_size(1, "t").is_ok());
        assert!(check_system_size(MAX_SYSTEM_SIZE, "t").is_ok());
        assert!(check_system_size(MAX_SYSTEM_SIZE + 1, "t").is_err());
    }

    #[test]
    fn system_size_error_names_the_caller() {
        let err = check_system_size(0, "solve_system: variables").unwrap_err();
        assert!(err.contains("solve_system"), "message was {err:?}");
    }

    #[test]
    fn branch_count_is_capped() {
        assert!(check_branch_count(0).is_ok());
        assert!(check_branch_count(MAX_SOLUTION_BRANCHES).is_ok());
        assert!(check_branch_count(MAX_SOLUTION_BRANCHES + 1).is_err());
    }

    #[test]
    fn sample_count_must_sit_between_two_and_the_core_cap() {
        assert!(check_sample_count(0).is_err());
        assert!(check_sample_count(1).is_err());
        assert!(check_sample_count(2).is_ok());
        assert!(check_sample_count(ARITHMA_FOURIER_MAX_SAMPLES).is_ok());
        assert!(check_sample_count(ARITHMA_FOURIER_MAX_SAMPLES + 1).is_err());
    }

    #[test]
    fn harmonics_must_be_at_least_one_and_within_the_core_cap() {
        assert!(check_harmonics(0).is_err());
        assert!(check_harmonics(1).is_ok());
        assert!(check_harmonics(ARITHMA_FOURIER_MAX_HARMONICS).is_ok());
        assert!(check_harmonics(ARITHMA_FOURIER_MAX_HARMONICS + 1).is_err());
    }

    #[test]
    fn range_must_be_finite_and_ordered() {
        assert!(check_range(-1.0, 1.0).is_ok());
        assert!(check_range(0.0, 0.0).is_err());
        assert!(check_range(1.0, -1.0).is_err());
        assert!(check_range(f64::NAN, 1.0).is_err());
        assert!(check_range(0.0, f64::INFINITY).is_err());
    }

    #[test]
    fn accuracy_accepts_zero_but_not_nan_or_a_negative() {
        assert!(check_accuracy(0.0).is_ok());
        assert!(check_accuracy(1e-6).is_ok());
        assert!(check_accuracy(-1e-6).is_err());
        assert!(check_accuracy(f64::NAN).is_err());
        assert!(check_accuracy(f64::INFINITY).is_err());
    }

    #[test]
    fn the_default_fourier_config_passes_our_own_validation() {
        // A default the facade would reject would be a trap for anyone calling
        // `fourier_transform` without a config.
        let d = ArithmaFourierConfig::default();
        assert!(check_sample_count(d.sample_count).is_ok());
        assert!(check_harmonics(d.harmonics).is_ok());
        assert!(check_range(d.range.0, d.range.1).is_ok());
        assert!(check_accuracy(d.accuracy).is_ok());
    }

    #[test]
    fn to_py_expression_preserves_the_tree() {
        let e = ArithmaExpression::add(ArithmaExpression::var("x"), ArithmaExpression::from_i64(3));
        let wrapped = to_py_expression(e.clone());
        assert_eq!(format!("{:?}", wrapped.inner), format!("{e:?}"));
    }
}
