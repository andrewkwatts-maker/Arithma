//====== Arithma/rust/arithma_core/src/numerical/root_finding.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Root finding
//!
//! Bisection / Newton-Raphson / secant root finders.
//!
//! All three share one contract:
//!
//! - The expression is evaluated numerically at each step by binding `var`
//!   through [`Evaluable`]. Any other free variable makes evaluation fail and
//!   the error propagates — these solve in one dimension.
//! - Every loop is bounded by `config.max_iterations` (CLAUDE.md safety rule 2).
//! - `converged` reports whether the tolerance was actually met. A run that
//!   exhausts its iteration budget returns `Ok` with `converged: false` and the
//!   best estimate so far, rather than an error: the caller usually wants the
//!   estimate *and* the knowledge that it is not tight.
//! - A non-finite evaluation (NaN/±∞) is an error, not a silent result.

use std::collections::HashMap;

use crate::expression::{ArithmaBindings, ArithmaExpression, Evaluable};

/// Evaluate `expr` with `var` bound to `x`, rejecting non-finite results.
fn eval_at(expr: &ArithmaExpression, var: &str, x: f64) -> Result<f64, String> {
    let mut bindings: ArithmaBindings = HashMap::with_capacity(1);
    bindings.insert(var.to_string(), x);
    let y = expr.evaluate(&bindings)?;
    if y.is_finite() {
        Ok(y)
    } else {
        Err(format!("f({x}) evaluated to a non-finite value: {y}"))
    }
}

/// Termination criterion configuration.
#[derive(Debug, Clone, Copy)]
pub struct ArithmaRootFindingConfig {
    /// Relative tolerance on `|f(x)|`.
    pub tol: f64,
    /// Hard cap on iterations (CLAUDE.md safety rule 2: bounded loops).
    pub max_iterations: usize,
}

impl Default for ArithmaRootFindingConfig {
    fn default() -> Self {
        Self {
            tol: 1e-12,
            max_iterations: 1024,
        }
    }
}

/// Outcome of a single root-finding run.
#[derive(Debug, Clone, Copy)]
pub struct ArithmaRootFindingResult {
    /// The root estimate.
    pub root: f64,
    /// Iterations consumed.
    pub iterations: usize,
    /// Whether the run converged within tolerance.
    pub converged: bool,
}

/// Bisection root finder over `[lo, hi]`.
///
/// Requires a sign change across the bracket — that is what guarantees a root
/// by the intermediate value theorem, and rejecting a non-bracketing interval
/// up front is better than returning a confident wrong answer. Converges
/// linearly but unconditionally, which is why the solver uses it as the
/// fallback when Newton diverges.
pub fn find_root_bisection(
    expr: &ArithmaExpression,
    var: &str,
    lo: f64,
    hi: f64,
    config: &ArithmaRootFindingConfig,
) -> Result<ArithmaRootFindingResult, String> {
    if !lo.is_finite() || !hi.is_finite() {
        return Err("bisection requires a finite bracket".to_string());
    }
    let (mut a, mut b) = if lo <= hi { (lo, hi) } else { (hi, lo) };

    let mut fa = eval_at(expr, var, a)?;
    let fb = eval_at(expr, var, b)?;

    // An endpoint may already be the root.
    if fa.abs() <= config.tol {
        return Ok(ArithmaRootFindingResult {
            root: a,
            iterations: 0,
            converged: true,
        });
    }
    if fb.abs() <= config.tol {
        return Ok(ArithmaRootFindingResult {
            root: b,
            iterations: 0,
            converged: true,
        });
    }
    if fa * fb > 0.0 {
        return Err(format!(
            "bisection needs a sign change over the bracket: f({a}) = {fa}, f({b}) = {fb}"
        ));
    }

    let mut mid = a;
    for iteration in 1..=config.max_iterations {
        mid = a + (b - a) * 0.5;
        let fm = eval_at(expr, var, mid)?;

        // Converged either on the residual or on a bracket narrower than the
        // representable gap — the latter stops us spinning on ulp-width ranges.
        if fm.abs() <= config.tol || (b - a).abs() <= f64::EPSILON * mid.abs().max(1.0) {
            return Ok(ArithmaRootFindingResult {
                root: mid,
                iterations: iteration,
                converged: true,
            });
        }

        if fa * fm < 0.0 {
            b = mid;
        } else {
            a = mid;
            fa = fm;
        }
    }

    Ok(ArithmaRootFindingResult {
        root: mid,
        iterations: config.max_iterations,
        converged: false,
    })
}

/// Newton-Raphson root finder from `initial`.
///
/// Differentiates `expr` symbolically once up front (via the iterative
/// differentiator) rather than approximating the derivative — that is the whole
/// advantage of having a symbolic core. Quadratic convergence near a simple
/// root; the caller should fall back to [`find_root_bisection`] when this
/// reports `converged: false`, since Newton can oscillate or run away.
pub fn find_root_newton_raphson(
    expr: &ArithmaExpression,
    var: &str,
    initial: f64,
    config: &ArithmaRootFindingConfig,
) -> Result<ArithmaRootFindingResult, String> {
    if !initial.is_finite() {
        return Err("newton-raphson requires a finite starting point".to_string());
    }
    let derivative = crate::calculus::differentiation::differentiate(expr, var)?;

    let mut x = initial;
    for iteration in 1..=config.max_iterations {
        let fx = eval_at(expr, var, x)?;
        if fx.abs() <= config.tol {
            return Ok(ArithmaRootFindingResult {
                root: x,
                iterations: iteration,
                converged: true,
            });
        }

        let dfx = eval_at(&derivative, var, x)?;
        if dfx == 0.0 {
            return Err(format!(
                "newton-raphson stalled: f'({x}) = 0, no tangent to follow"
            ));
        }

        let next = x - fx / dfx;
        if !next.is_finite() {
            return Err(format!("newton-raphson diverged at x = {x}"));
        }
        // A step that no longer moves x means we are at the precision floor.
        if next == x {
            return Ok(ArithmaRootFindingResult {
                root: next,
                iterations: iteration,
                converged: eval_at(expr, var, next)?.abs() <= config.tol,
            });
        }
        x = next;
    }

    Ok(ArithmaRootFindingResult {
        root: x,
        iterations: config.max_iterations,
        converged: false,
    })
}

/// Secant-method root finder seeded with `x0` and `x1`.
///
/// Newton's convergence without needing a derivative — useful when `expr`
/// contains an operator the differentiator does not handle. Requires two
/// distinct starting points.
pub fn find_root_secant(
    expr: &ArithmaExpression,
    var: &str,
    x0: f64,
    x1: f64,
    config: &ArithmaRootFindingConfig,
) -> Result<ArithmaRootFindingResult, String> {
    if !x0.is_finite() || !x1.is_finite() {
        return Err("secant requires finite starting points".to_string());
    }
    if x0 == x1 {
        return Err("secant requires two distinct starting points".to_string());
    }

    let mut a = x0;
    let mut b = x1;
    let mut fa = eval_at(expr, var, a)?;
    let mut fb = eval_at(expr, var, b)?;

    if fa.abs() <= config.tol {
        return Ok(ArithmaRootFindingResult {
            root: a,
            iterations: 0,
            converged: true,
        });
    }

    for iteration in 1..=config.max_iterations {
        if fb.abs() <= config.tol {
            return Ok(ArithmaRootFindingResult {
                root: b,
                iterations: iteration,
                converged: true,
            });
        }

        let denom = fb - fa;
        if denom == 0.0 {
            return Err(format!(
                "secant stalled: f({a}) == f({b}) == {fb}, the secant line is horizontal"
            ));
        }

        let next = b - fb * (b - a) / denom;
        if !next.is_finite() {
            return Err(format!("secant diverged at x = {b}"));
        }

        a = b;
        fa = fb;
        b = next;
        fb = eval_at(expr, var, b)?;

        if a == b {
            return Ok(ArithmaRootFindingResult {
                root: b,
                iterations: iteration,
                converged: fb.abs() <= config.tol,
            });
        }
    }

    Ok(ArithmaRootFindingResult {
        root: b,
        iterations: config.max_iterations,
        converged: fb.abs() <= config.tol,
    })
}

// ---------------------------------------------------------------------------
// Backward-compatibility aliases for the pre-rename `Arithmos*` names.
// Retained for one release; downstream (eml-math, eml-spectral, metaphysica,
// periodica) should migrate to the `Arithma*` names above.
// ---------------------------------------------------------------------------
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaRootFindingConfig`")]
#[allow(unused)]
pub use self::ArithmaRootFindingConfig as ArithmosRootFindingConfig;
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaRootFindingResult`")]
#[allow(unused)]
pub use self::ArithmaRootFindingResult as ArithmosRootFindingResult;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_sane() {
        let cfg = ArithmaRootFindingConfig::default();
        assert!(cfg.tol > 0.0);
        assert!(cfg.max_iterations > 0);
    }

    /// x^2 - 2, whose positive root is sqrt(2).
    fn x_squared_minus_two() -> ArithmaExpression {
        let x = ArithmaExpression::var("x");
        ArithmaExpression::sub(
            ArithmaExpression::mul(x.clone(), x),
            ArithmaExpression::from_i64(2),
        )
    }

    /// x^3 - x, with roots at -1, 0 and 1.
    fn cubic_minus_x() -> ArithmaExpression {
        let x = ArithmaExpression::var("x");
        ArithmaExpression::sub(
            ArithmaExpression::mul(ArithmaExpression::mul(x.clone(), x.clone()), x.clone()),
            x,
        )
    }

    const SQRT2: f64 = std::f64::consts::SQRT_2;

    // ── bisection ─────────────────────────────────────────────────────────

    #[test]
    fn bisection_finds_sqrt_two() {
        let cfg = ArithmaRootFindingConfig::default();
        let r = find_root_bisection(&x_squared_minus_two(), "x", 0.0, 2.0, &cfg).unwrap();
        assert!(r.converged, "should converge in {} iters", r.iterations);
        assert!((r.root - SQRT2).abs() < 1e-9, "root = {}", r.root);
    }

    #[test]
    fn bisection_rejects_a_bracket_without_a_sign_change() {
        let cfg = ArithmaRootFindingConfig::default();
        // f > 0 across [2, 3]: no root, and bisection must say so rather than
        // returning a confident midpoint.
        let err = find_root_bisection(&x_squared_minus_two(), "x", 2.0, 3.0, &cfg).unwrap_err();
        assert!(err.contains("sign change"), "unexpected error: {err}");
    }

    #[test]
    fn bisection_accepts_a_reversed_bracket() {
        let cfg = ArithmaRootFindingConfig::default();
        let r = find_root_bisection(&x_squared_minus_two(), "x", 2.0, 0.0, &cfg).unwrap();
        assert!((r.root - SQRT2).abs() < 1e-9);
    }

    #[test]
    fn bisection_returns_an_endpoint_that_is_already_a_root() {
        let cfg = ArithmaRootFindingConfig::default();
        let r = find_root_bisection(&cubic_minus_x(), "x", 0.0, 0.5, &cfg).unwrap();
        assert!(r.converged);
        assert!(r.root.abs() < 1e-12, "root = {}", r.root);
        assert_eq!(
            r.iterations, 0,
            "an exact endpoint should cost no iterations"
        );
    }

    #[test]
    fn bisection_reports_non_convergence_within_a_tiny_budget() {
        let cfg = ArithmaRootFindingConfig {
            tol: 1e-15,
            max_iterations: 3,
        };
        let r = find_root_bisection(&x_squared_minus_two(), "x", 0.0, 2.0, &cfg).unwrap();
        assert!(!r.converged, "3 iterations cannot reach 1e-15");
        assert_eq!(r.iterations, 3);
    }

    // ── newton-raphson ────────────────────────────────────────────────────

    #[test]
    fn newton_finds_sqrt_two_and_beats_bisection_on_iterations() {
        let cfg = ArithmaRootFindingConfig::default();
        let n = find_root_newton_raphson(&x_squared_minus_two(), "x", 1.0, &cfg).unwrap();
        let b = find_root_bisection(&x_squared_minus_two(), "x", 0.0, 2.0, &cfg).unwrap();
        assert!(n.converged);
        assert!((n.root - SQRT2).abs() < 1e-12, "root = {}", n.root);
        assert!(
            n.iterations < b.iterations,
            "newton {} vs bisection {}",
            n.iterations,
            b.iterations
        );
    }

    #[test]
    fn newton_errors_on_a_zero_derivative() {
        let cfg = ArithmaRootFindingConfig::default();
        // f'(0) = 0 for x^3 - x ... actually f'(0) = -1; use x^2 whose f'(0) = 0.
        let x = ArithmaExpression::var("x");
        let sq = ArithmaExpression::mul(x.clone(), x);
        let err = find_root_newton_raphson(&sq, "x", 0.0, &cfg);
        // f(0) = 0 is already within tolerance, so this converges immediately.
        assert!(err.unwrap().converged);

        // Starting where the derivative vanishes but the value does not:
        let shifted = ArithmaExpression::add(
            ArithmaExpression::mul(ArithmaExpression::var("x"), ArithmaExpression::var("x")),
            ArithmaExpression::from_i64(1),
        );
        let err = find_root_newton_raphson(&shifted, "x", 0.0, &cfg).unwrap_err();
        assert!(err.contains("stalled"), "unexpected error: {err}");
    }

    #[test]
    fn newton_rejects_a_non_finite_start() {
        let cfg = ArithmaRootFindingConfig::default();
        assert!(find_root_newton_raphson(&x_squared_minus_two(), "x", f64::NAN, &cfg).is_err());
    }

    // ── secant ────────────────────────────────────────────────────────────

    #[test]
    fn secant_finds_sqrt_two() {
        let cfg = ArithmaRootFindingConfig::default();
        let r = find_root_secant(&x_squared_minus_two(), "x", 1.0, 2.0, &cfg).unwrap();
        assert!(r.converged);
        assert!((r.root - SQRT2).abs() < 1e-9, "root = {}", r.root);
    }

    #[test]
    fn secant_rejects_identical_starting_points() {
        let cfg = ArithmaRootFindingConfig::default();
        let err = find_root_secant(&x_squared_minus_two(), "x", 1.0, 1.0, &cfg).unwrap_err();
        assert!(err.contains("distinct"), "unexpected error: {err}");
    }

    // ── agreement ─────────────────────────────────────────────────────────

    #[test]
    fn all_three_methods_agree_on_the_same_root() {
        let cfg = ArithmaRootFindingConfig::default();
        let f = cubic_minus_x();
        let b = find_root_bisection(&f, "x", 0.5, 1.5, &cfg).unwrap().root;
        let n = find_root_newton_raphson(&f, "x", 1.4, &cfg).unwrap().root;
        let s = find_root_secant(&f, "x", 0.6, 1.5, &cfg).unwrap().root;
        for (name, got) in [("bisection", b), ("newton", n), ("secant", s)] {
            assert!((got - 1.0).abs() < 1e-8, "{name} gave {got}, expected 1.0");
        }
    }

    #[test]
    fn unbound_second_variable_is_an_error_not_a_wrong_answer() {
        let cfg = ArithmaRootFindingConfig::default();
        let expr = ArithmaExpression::sub(ArithmaExpression::var("x"), ArithmaExpression::var("y"));
        assert!(find_root_bisection(&expr, "x", -1.0, 1.0, &cfg).is_err());
    }
}
