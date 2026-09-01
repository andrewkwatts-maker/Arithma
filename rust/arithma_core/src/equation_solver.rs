//====== Arithma/rust/arithma_core/src/equation_solver.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Equation solver
//!
//! Symbolic and numeric equation solving. Mirrors
//! `pt_arithmos::pt_equation_solver`. Wave 2 ships type signatures only; Wave 3
//! ports the real solver passes (linear, quadratic, polynomial root,
//! transcendental, system-of-equations).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::expression::{ArithmaBindings, ArithmaExpression, Evaluable};
use crate::numerical::root_finding::{find_root_bisection, ArithmaRootFindingConfig};

/// Strategy hint for the solver. Implementations may ignore it and use their
/// own heuristics, but this gives callers a way to express priorities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ArithmaSolverStrategy {
    /// Auto-detect (default).
    #[default]
    Auto,
    /// Force the algebraic / closed-form path.
    Algebraic,
    /// Force the numeric (root-finding) path.
    Numeric,
    /// Try algebraic, fall back to numeric.
    Hybrid,
}

/// One root or solution branch returned by the solver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArithmaSolution {
    /// Symbolic form of the solution (e.g. `(-b ± √(b²-4ac)) / (2a)`).
    pub expression: ArithmaExpression,
    /// Optional cached numeric value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached: Option<f64>,
    /// Whether this branch is real-valued.
    pub is_real: bool,
}

/// Default half-width of the bracket scanned by the numeric fallback.
const NUMERIC_SEARCH_SPAN: f64 = 100.0;
/// Sub-intervals used when scanning for sign changes numerically.
const NUMERIC_SCAN_STEPS: usize = 2000;
/// Tolerance for treating a sampled polynomial coefficient as zero.
const COEFF_EPSILON: f64 = 1e-12;

/// Solve `expr = 0` for `var`. Returns every branch the solver finds.
///
/// Strategy:
///
/// - [`ArithmaSolverStrategy::Algebraic`] — closed forms only (linear and
///   quadratic, detected by sampling the expression as a polynomial). Errors if
///   no closed form applies, rather than quietly returning nothing.
/// - [`ArithmaSolverStrategy::Numeric`] — scans `[-100, 100]` for sign changes
///   and bisects each bracket. Roots outside that span, or of even multiplicity
///   (which do not produce a sign change), are not found.
/// - [`ArithmaSolverStrategy::Auto`] / [`Hybrid`](ArithmaSolverStrategy::Hybrid)
///   — try the closed form, fall back to the numeric scan.
///
/// Solutions carry both the symbolic expression and a cached numeric value
/// where one is available.
pub fn solve(
    expr: &ArithmaExpression,
    var: &str,
    strategy: ArithmaSolverStrategy,
) -> Result<Vec<ArithmaSolution>, String> {
    match strategy {
        ArithmaSolverStrategy::Algebraic => solve_algebraic(expr, var)?
            .ok_or_else(|| "no closed form available for this expression".to_string()),
        ArithmaSolverStrategy::Numeric => solve_numeric(expr, var),
        ArithmaSolverStrategy::Auto | ArithmaSolverStrategy::Hybrid => {
            match solve_algebraic(expr, var)? {
                Some(found) => Ok(found),
                None => solve_numeric(expr, var),
            }
        }
    }
}

/// Solve `lhs = rhs` for `var` by rewriting to `lhs - rhs = 0`.
pub fn solve_equation(
    lhs: &ArithmaExpression,
    rhs: &ArithmaExpression,
    var: &str,
    strategy: ArithmaSolverStrategy,
) -> Result<Vec<ArithmaSolution>, String> {
    solve(
        &ArithmaExpression::sub(lhs.clone(), rhs.clone()),
        var,
        strategy,
    )
}

/// Solve a system of equations for the listed variables.
///
/// Handles the **linear** case: each equation is sampled to recover its
/// coefficient row, and the resulting matrix is solved by Gaussian elimination
/// with partial pivoting. A non-linear system is rejected with an error rather
/// than linearised silently — a wrong answer here would be very hard to spot.
///
/// The result is one solution list, in the same order as `vars`, wrapped in the
/// outer `Vec` (a linear system has at most one solution branch).
pub fn solve_system(
    equations: &[(ArithmaExpression, ArithmaExpression)],
    vars: &[&str],
    _strategy: ArithmaSolverStrategy,
) -> Result<Vec<Vec<ArithmaSolution>>, String> {
    if vars.is_empty() {
        return Err("solve_system needs at least one variable".to_string());
    }
    if equations.len() != vars.len() {
        return Err(format!(
            "solve_system needs one equation per variable: {} equations, {} variables",
            equations.len(),
            vars.len()
        ));
    }

    let n = vars.len();
    // Augmented matrix: row i is [a_i0 .. a_i(n-1) | b_i] for  A·x = b.
    let mut m = vec![vec![0.0_f64; n + 1]; n];
    for (i, (lhs, rhs)) in equations.iter().enumerate() {
        let f = ArithmaExpression::sub(lhs.clone(), rhs.clone());
        let base = eval_vars(&f, vars, &vec![0.0; n])?;
        for (j, _) in vars.iter().enumerate() {
            let mut probe = vec![0.0; n];
            probe[j] = 1.0;
            let at_one = eval_vars(&f, vars, &probe)?;
            let coeff = at_one - base;

            // Linearity check: f must be affine in this variable, so the step
            // from 1 to 2 has to match the step from 0 to 1.
            probe[j] = 2.0;
            let at_two = eval_vars(&f, vars, &probe)?;
            if ((at_two - at_one) - coeff).abs() > 1e-6 * coeff.abs().max(1.0) {
                return Err(format!(
                    "equation {i} is not linear in `{}` — solve_system handles linear systems only",
                    vars[j]
                ));
            }
            m[i][j] = coeff;
        }
        m[i][n] = -base;
    }

    let xs = gaussian_solve(&mut m, n)?;
    let solutions = xs
        .into_iter()
        .map(|v| ArithmaSolution {
            expression: ArithmaExpression::from_f64(v),
            cached: Some(v),
            is_real: true,
        })
        .collect();
    Ok(vec![solutions])
}

// ── internals ───────────────────────────────────────────────────────────────

/// Evaluate `expr` with one variable bound.
fn eval_at(expr: &ArithmaExpression, var: &str, x: f64) -> Result<f64, String> {
    let mut b: ArithmaBindings = HashMap::with_capacity(1);
    b.insert(var.to_string(), x);
    expr.evaluate(&b)
}

/// Evaluate `expr` with several variables bound positionally.
fn eval_vars(expr: &ArithmaExpression, vars: &[&str], values: &[f64]) -> Result<f64, String> {
    let mut b: ArithmaBindings = HashMap::with_capacity(vars.len());
    for (v, x) in vars.iter().zip(values) {
        b.insert((*v).to_string(), *x);
    }
    expr.evaluate(&b)
}

fn solution_from(value: f64) -> ArithmaSolution {
    ArithmaSolution {
        expression: ArithmaExpression::from_f64(value),
        cached: Some(value),
        is_real: true,
    }
}

/// Try to read `expr` as a polynomial of degree ≤ 2 in `var` and solve in
/// closed form. Returns `Ok(None)` when the expression is not such a polynomial.
///
/// Detection samples the expression at four points and checks that a quadratic
/// fit reproduces a fifth — cheap, and it cannot mistake a transcendental for a
/// polynomial without the verification step failing.
fn solve_algebraic(
    expr: &ArithmaExpression,
    var: &str,
) -> Result<Option<Vec<ArithmaSolution>>, String> {
    // f(0), f(1), f(-1) determine a, b, c for f = a x² + b x + c.
    let (f0, f1, fm1) = match (
        eval_at(expr, var, 0.0),
        eval_at(expr, var, 1.0),
        eval_at(expr, var, -1.0),
    ) {
        (Ok(a), Ok(b), Ok(c)) => (a, b, c),
        _ => return Ok(None),
    };
    if !(f0.is_finite() && f1.is_finite() && fm1.is_finite()) {
        return Ok(None);
    }

    let c = f0;
    let a = (f1 + fm1) / 2.0 - f0;
    let b = (f1 - fm1) / 2.0;

    // Verify the fit against an independent sample; a mismatch means the
    // expression is not quadratic and the closed forms do not apply.
    let probe = 2.5_f64;
    match eval_at(expr, var, probe) {
        Ok(actual) => {
            let predicted = a * probe * probe + b * probe + c;
            if (actual - predicted).abs() > 1e-6 * actual.abs().max(1.0) {
                return Ok(None);
            }
        }
        Err(_) => return Ok(None),
    }

    if a.abs() <= COEFF_EPSILON {
        if b.abs() <= COEFF_EPSILON {
            // Constant: either no solution, or every x is one.
            return if c.abs() <= COEFF_EPSILON {
                Err("identity: every value of the variable is a solution".to_string())
            } else {
                Ok(Some(Vec::new()))
            };
        }
        // Linear: b·x + c = 0
        return Ok(Some(vec![solution_from(-c / b)]));
    }

    // Quadratic: a·x² + b·x + c = 0
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        // Complex pair — reported as non-real branches with no cached f64.
        let re = -b / (2.0 * a);
        let im = (-disc).sqrt() / (2.0 * a);
        let mk = |sign: f64| ArithmaSolution {
            expression: ArithmaExpression::add(
                ArithmaExpression::from_f64(re),
                ArithmaExpression::mul(
                    ArithmaExpression::from_f64(sign * im),
                    ArithmaExpression::var("i"),
                ),
            ),
            cached: None,
            is_real: false,
        };
        return Ok(Some(vec![mk(1.0), mk(-1.0)]));
    }

    let sq = disc.sqrt();
    // The numerically stable pair: computing both roots from the textbook
    // formula loses precision in the one where -b and ±√disc nearly cancel.
    let q = -0.5 * (b + b.signum() * sq);
    let (r1, r2) = if q == 0.0 { (0.0, 0.0) } else { (q / a, c / q) };

    let mut roots = vec![r1, r2];
    roots.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    if (roots[0] - roots[1]).abs() <= COEFF_EPSILON * roots[0].abs().max(1.0) {
        roots.truncate(1); // repeated root
    }
    Ok(Some(roots.into_iter().map(solution_from).collect()))
}

/// Scan for sign changes over `[-NUMERIC_SEARCH_SPAN, NUMERIC_SEARCH_SPAN]` and
/// bisect each bracket.
fn solve_numeric(expr: &ArithmaExpression, var: &str) -> Result<Vec<ArithmaSolution>, String> {
    let cfg = ArithmaRootFindingConfig::default();
    let lo = -NUMERIC_SEARCH_SPAN;
    let dx = (2.0 * NUMERIC_SEARCH_SPAN) / NUMERIC_SCAN_STEPS as f64;

    let mut roots: Vec<f64> = Vec::new();
    let push = |roots: &mut Vec<f64>, x: f64| {
        if !roots.iter().any(|r: &f64| (r - x).abs() <= dx) {
            roots.push(x);
        }
    };

    let mut prev_x = lo;
    let mut prev_y = eval_at(expr, var, prev_x).ok();

    for i in 1..=NUMERIC_SCAN_STEPS {
        let x = lo + dx * i as f64;
        let y = eval_at(expr, var, x).ok().filter(|v| v.is_finite());

        if let (Some(py), Some(cy)) = (prev_y, y) {
            if cy.abs() <= cfg.tol {
                push(&mut roots, x);
            } else if py * cy < 0.0 {
                if let Ok(r) = find_root_bisection(expr, var, prev_x, x, &cfg) {
                    push(&mut roots, r.root);
                }
            }
        }
        prev_x = x;
        prev_y = y;
    }

    roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(roots.into_iter().map(solution_from).collect())
}

/// Gaussian elimination with partial pivoting on an `n × (n+1)` augmented
/// matrix. Returns the solution vector.
fn gaussian_solve(m: &mut [Vec<f64>], n: usize) -> Result<Vec<f64>, String> {
    for col in 0..n {
        // Partial pivot: the largest magnitude in this column keeps the
        // elimination numerically stable.
        let mut pivot = col;
        for r in (col + 1)..n {
            if m[r][col].abs() > m[pivot][col].abs() {
                pivot = r;
            }
        }
        if m[pivot][col].abs() <= COEFF_EPSILON {
            return Err("singular system: no unique solution".to_string());
        }
        m.swap(col, pivot);

        let p = m[col][col];
        for r in (col + 1)..n {
            let factor = m[r][col] / p;
            if factor == 0.0 {
                continue;
            }
            for c in col..=n {
                m[r][c] -= factor * m[col][c];
            }
        }
    }

    let mut xs = vec![0.0; n];
    for i in (0..n).rev() {
        let mut acc = m[i][n];
        for (j, xj) in xs.iter().enumerate().take(n).skip(i + 1) {
            acc -= m[i][j] * xj;
        }
        xs[i] = acc / m[i][i];
    }
    Ok(xs)
}

// ---------------------------------------------------------------------------
// Backward-compatibility aliases for the pre-rename `Arithmos*` names.
// Retained for one release; downstream (eml-math, eml-spectral, metaphysica,
// periodica) should migrate to the `Arithma*` names above.
// ---------------------------------------------------------------------------
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaSolution`")]
#[allow(unused)]
pub use self::ArithmaSolution as ArithmosSolution;
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaSolverStrategy`")]
#[allow(unused)]
pub use self::ArithmaSolverStrategy as ArithmosSolverStrategy;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_strategy_is_auto() {
        assert_eq!(
            ArithmaSolverStrategy::default(),
            ArithmaSolverStrategy::Auto
        );
    }

    fn x() -> ArithmaExpression {
        ArithmaExpression::var("x")
    }

    fn n(v: i64) -> ArithmaExpression {
        ArithmaExpression::from_i64(v)
    }

    /// Roots, sorted, from a solve result.
    fn roots(sols: &[ArithmaSolution]) -> Vec<f64> {
        let mut v: Vec<f64> = sols.iter().filter_map(|s| s.cached).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v
    }

    // ── linear ────────────────────────────────────────────────────────────

    #[test]
    fn solves_a_linear_equation() {
        // 2x - 6 = 0  ->  x = 3
        let e = ArithmaExpression::sub(ArithmaExpression::mul(n(2), x()), n(6));
        let s = solve(&e, "x", ArithmaSolverStrategy::Auto).unwrap();
        assert_eq!(s.len(), 1);
        assert!((s[0].cached.unwrap() - 3.0).abs() < 1e-9);
        assert!(s[0].is_real);
    }

    #[test]
    fn solve_equation_rewrites_lhs_minus_rhs() {
        // 3x = 12  ->  x = 4
        let lhs = ArithmaExpression::mul(n(3), x());
        let s = solve_equation(&lhs, &n(12), "x", ArithmaSolverStrategy::Auto).unwrap();
        assert_eq!(s.len(), 1);
        assert!((s[0].cached.unwrap() - 4.0).abs() < 1e-9);
    }

    // ── quadratic ─────────────────────────────────────────────────────────

    #[test]
    fn solves_a_quadratic_with_two_real_roots() {
        // x² - 5x + 6 = 0  ->  2, 3
        let e = ArithmaExpression::add(
            ArithmaExpression::sub(
                ArithmaExpression::mul(x(), x()),
                ArithmaExpression::mul(n(5), x()),
            ),
            n(6),
        );
        let r = roots(&solve(&e, "x", ArithmaSolverStrategy::Auto).unwrap());
        assert_eq!(r.len(), 2, "got {r:?}");
        assert!((r[0] - 2.0).abs() < 1e-9, "{r:?}");
        assert!((r[1] - 3.0).abs() < 1e-9, "{r:?}");
    }

    #[test]
    fn repeated_root_is_reported_once() {
        // x² - 2x + 1 = (x-1)²
        let e = ArithmaExpression::add(
            ArithmaExpression::sub(
                ArithmaExpression::mul(x(), x()),
                ArithmaExpression::mul(n(2), x()),
            ),
            n(1),
        );
        let r = roots(&solve(&e, "x", ArithmaSolverStrategy::Algebraic).unwrap());
        assert_eq!(r.len(), 1, "got {r:?}");
        assert!((r[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn complex_roots_are_flagged_not_real() {
        // x² + 1 = 0
        let e = ArithmaExpression::add(ArithmaExpression::mul(x(), x()), n(1));
        let s = solve(&e, "x", ArithmaSolverStrategy::Algebraic).unwrap();
        assert_eq!(s.len(), 2);
        assert!(s.iter().all(|b| !b.is_real));
        assert!(s.iter().all(|b| b.cached.is_none()));
    }

    #[test]
    fn quadratic_formula_stays_accurate_when_the_roots_are_far_apart() {
        // x² - 1e8·x + 1 = 0. The naive formula loses the small root entirely
        // to cancellation; the stable pairing keeps it.
        let a = 1.0_f64;
        let b = -1e8_f64;
        let c = 1.0_f64;
        let e = ArithmaExpression::add(
            ArithmaExpression::add(
                ArithmaExpression::mul(x(), x()),
                ArithmaExpression::mul(ArithmaExpression::from_f64(b), x()),
            ),
            n(1),
        );
        let r = roots(&solve(&e, "x", ArithmaSolverStrategy::Algebraic).unwrap());
        assert_eq!(r.len(), 2, "got {r:?}");
        let small = c / (-b); // ≈ 1e-8 by Vieta
        assert!(
            (r[0] - small).abs() < 1e-12,
            "small root {} lost to cancellation (expected ~{small}), a={a}",
            r[0]
        );
    }

    // ── strategy behaviour ────────────────────────────────────────────────

    #[test]
    fn algebraic_strategy_refuses_a_transcendental() {
        // sin(x) is not a polynomial — Algebraic must say so, not return [].
        let e = ArithmaExpression::sin(x());
        assert!(solve(&e, "x", ArithmaSolverStrategy::Algebraic).is_err());
    }

    #[test]
    fn numeric_strategy_finds_transcendental_roots() {
        // sin(x) over the scan span: roots at 0, ±π, ±2π, ...
        let e = ArithmaExpression::sin(x());
        let r = roots(&solve(&e, "x", ArithmaSolverStrategy::Numeric).unwrap());
        assert!(r.len() > 10, "expected many roots, got {}", r.len());
        assert!(r.iter().any(|v| v.abs() < 1e-6), "missing the root at 0");
        assert!(
            r.iter().any(|v| (v - std::f64::consts::PI).abs() < 1e-6),
            "missing the root at pi"
        );
    }

    #[test]
    fn auto_falls_back_from_algebraic_to_numeric() {
        let e = ArithmaExpression::sin(x());
        let s = solve(&e, "x", ArithmaSolverStrategy::Auto).unwrap();
        assert!(
            !s.is_empty(),
            "Auto should have fallen back to the numeric scan"
        );
    }

    #[test]
    fn no_solution_constant_returns_empty() {
        // 7 = 0 has no solution.
        let r = solve(&n(7), "x", ArithmaSolverStrategy::Algebraic).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn identity_is_an_error_not_an_empty_list() {
        // 0 = 0 is satisfied by every x; conflating that with "no solution"
        // would be a silent wrong answer.
        assert!(solve(&n(0), "x", ArithmaSolverStrategy::Algebraic).is_err());
    }

    // ── systems ───────────────────────────────────────────────────────────

    #[test]
    fn solves_a_2x2_linear_system() {
        // x + y = 5 ; x - y = 1   ->   x = 3, y = 2
        let xv = ArithmaExpression::var("x");
        let yv = ArithmaExpression::var("y");
        let eqs = vec![
            (ArithmaExpression::add(xv.clone(), yv.clone()), n(5)),
            (ArithmaExpression::sub(xv, yv), n(1)),
        ];
        let out = solve_system(&eqs, &["x", "y"], ArithmaSolverStrategy::Auto).unwrap();
        assert_eq!(out.len(), 1);
        let vals: Vec<f64> = out[0].iter().map(|s| s.cached.unwrap()).collect();
        assert!((vals[0] - 3.0).abs() < 1e-9, "x = {}", vals[0]);
        assert!((vals[1] - 2.0).abs() < 1e-9, "y = {}", vals[1]);
    }

    #[test]
    fn solves_a_3x3_linear_system() {
        // x+y+z=6 ; 2y+z=8 ; x-z=-2  ->  x=1, y=3, z=2  (check by substitution)
        let (xv, yv, zv) = (
            ArithmaExpression::var("x"),
            ArithmaExpression::var("y"),
            ArithmaExpression::var("z"),
        );
        let eqs = vec![
            (
                ArithmaExpression::add(ArithmaExpression::add(xv.clone(), yv.clone()), zv.clone()),
                n(6),
            ),
            (
                ArithmaExpression::add(ArithmaExpression::mul(n(2), yv.clone()), zv.clone()),
                n(8),
            ),
            (ArithmaExpression::sub(xv, zv), n(-1)),
        ];
        let out = solve_system(&eqs, &["x", "y", "z"], ArithmaSolverStrategy::Auto).unwrap();
        let v: Vec<f64> = out[0].iter().map(|s| s.cached.unwrap()).collect();
        assert!(
            (v[0] + v[1] + v[2] - 6.0).abs() < 1e-9,
            "eq1 not satisfied: {v:?}"
        );
        assert!(
            (2.0 * v[1] + v[2] - 8.0).abs() < 1e-9,
            "eq2 not satisfied: {v:?}"
        );
        assert!((v[0] - v[2] + 1.0).abs() < 1e-9, "eq3 not satisfied: {v:?}");
    }

    #[test]
    fn singular_system_is_rejected() {
        // x + y = 1 ; 2x + 2y = 2  — dependent rows, no unique solution.
        let xv = ArithmaExpression::var("x");
        let yv = ArithmaExpression::var("y");
        let eqs = vec![
            (ArithmaExpression::add(xv.clone(), yv.clone()), n(1)),
            (
                ArithmaExpression::add(
                    ArithmaExpression::mul(n(2), xv),
                    ArithmaExpression::mul(n(2), yv),
                ),
                n(2),
            ),
        ];
        let err = solve_system(&eqs, &["x", "y"], ArithmaSolverStrategy::Auto).unwrap_err();
        assert!(err.contains("singular"), "unexpected error: {err}");
    }

    #[test]
    fn nonlinear_system_is_rejected_rather_than_linearised() {
        // x² + y = 1 is not linear in x; silently linearising would give a
        // confidently wrong answer.
        let xv = ArithmaExpression::var("x");
        let yv = ArithmaExpression::var("y");
        let eqs = vec![
            (
                ArithmaExpression::add(ArithmaExpression::mul(xv.clone(), xv.clone()), yv.clone()),
                n(1),
            ),
            (ArithmaExpression::add(xv, yv), n(2)),
        ];
        let err = solve_system(&eqs, &["x", "y"], ArithmaSolverStrategy::Auto).unwrap_err();
        assert!(err.contains("not linear"), "unexpected error: {err}");
    }

    #[test]
    fn system_shape_is_validated() {
        let eqs = vec![(ArithmaExpression::var("x"), n(1))];
        assert!(solve_system(&eqs, &["x", "y"], ArithmaSolverStrategy::Auto).is_err());
        assert!(solve_system(&eqs, &[], ArithmaSolverStrategy::Auto).is_err());
    }
}
