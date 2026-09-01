//====== Arithma/rust/arithma_core/src/numerical/methods.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Numerical methods dispatcher
//!
//! Generic entry point for choosing a numerical solver. Routes to the
//! specialised implementations in [`super::root_finding`].

use crate::expression::{ArithmaBindings, ArithmaExpression, Evaluable};

/// Choice of numerical method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArithmaNumericalMethod {
    Bisection,
    NewtonRaphson,
    Secant,
    Brent,
}

/// Residual tolerance: a point with `|f(x)|` below this counts as a root.
const ARITHMA_SOLVE_RESIDUAL_TOL: f64 = 1e-12;
/// Step tolerance: a bracket or update narrower than this counts as converged.
const ARITHMA_SOLVE_STEP_TOL: f64 = 1e-14;
/// Iteration cap shared by every method (CLAUDE.md safety rule 2).
const ARITHMA_SOLVE_MAX_ITERATIONS: usize = 512;
/// Cap on the outward doubling used to find a sign-changing bracket.
const ARITHMA_SOLVE_MAX_EXPANSIONS: usize = 96;
/// Below this magnitude a derivative is treated as vanished.
const ARITHMA_SOLVE_MIN_DERIVATIVE: f64 = 1e-300;

/// Single-variable numeric view of an expression.
///
/// Holds one reusable bindings map so a solve does not allocate per evaluation.
struct Sampler<'a> {
    expr: &'a ArithmaExpression,
    var: &'a str,
    bindings: ArithmaBindings,
}

impl<'a> Sampler<'a> {
    fn new(expr: &'a ArithmaExpression, var: &'a str) -> Self {
        let mut bindings = ArithmaBindings::new();
        bindings.insert(var.to_string(), 0.0);
        Self {
            expr,
            var,
            bindings,
        }
    }

    /// `f(x)`, rejecting non-finite results so a NaN can never masquerade as a
    /// bracket endpoint.
    fn f(&mut self, x: f64) -> Result<f64, String> {
        match self.bindings.get_mut(self.var) {
            Some(slot) => *slot = x,
            None => return Err("solve_with_method: binding slot missing".into()),
        }
        let v = self.expr.evaluate(&self.bindings)?;
        if v.is_finite() {
            Ok(v)
        } else {
            Err(format!(
                "solve_with_method: expression is not finite at {} = {x}",
                self.var
            ))
        }
    }

    /// Central-difference derivative with a relative step.
    ///
    /// A numeric derivative keeps this module self-contained (no dependency on
    /// symbolic differentiation) and works for expressions the differentiator
    /// cannot yet handle. The `h² + ε/h` error balance below lands around
    /// `1e-10`, which is ample for driving a Newton step.
    fn df(&mut self, x: f64) -> Result<f64, String> {
        let h = 1e-6 * x.abs().max(1.0);
        let a = self.f(x + h)?;
        let b = self.f(x - h)?;
        Ok((a - b) / (2.0 * h))
    }
}

/// Grow a symmetric window outward from `start` until a sign change appears.
///
/// Returns the tightest `(lo, hi)` found with `f(lo)·f(hi) ≤ 0`, or the exact
/// root if one is hit along the way. Bounded by
/// [`ARITHMA_SOLVE_MAX_EXPANSIONS`] doublings; a side that stops evaluating
/// (domain error, overflow to non-finite) is frozen rather than aborting the
/// whole search.
fn bracket_around(sampler: &mut Sampler<'_>, start: f64) -> Result<(f64, f64), String> {
    let f_start = sampler.f(start)?;
    if f_start == 0.0 {
        return Ok((start, start));
    }
    let mut lo = start;
    let mut hi = start;
    let mut f_lo = f_start;
    let mut f_hi = f_start;
    let mut lo_alive = true;
    let mut hi_alive = true;
    let mut step = (start.abs() * 0.1).max(0.1);

    for _ in 0..ARITHMA_SOLVE_MAX_EXPANSIONS {
        if !lo_alive && !hi_alive {
            break;
        }
        if lo_alive {
            let next = lo - step;
            match sampler.f(next) {
                Ok(v) => {
                    if v == 0.0 {
                        return Ok((next, next));
                    }
                    if v * f_lo < 0.0 {
                        return Ok((next, lo));
                    }
                    lo = next;
                    f_lo = v;
                }
                Err(_) => lo_alive = false,
            }
        }
        if hi_alive {
            let next = hi + step;
            match sampler.f(next) {
                Ok(v) => {
                    if v == 0.0 {
                        return Ok((next, next));
                    }
                    if f_hi * v < 0.0 {
                        return Ok((hi, next));
                    }
                    hi = next;
                    f_hi = v;
                }
                Err(_) => hi_alive = false,
            }
        }
        step *= 1.6;
        if !step.is_finite() {
            break;
        }
    }
    Err(format!(
        "solve_with_method: no sign change found within {ARITHMA_SOLVE_MAX_EXPANSIONS} expansions around {start}"
    ))
}

/// Plain bisection on a bracket that already brackets a sign change.
fn bisect(sampler: &mut Sampler<'_>, mut lo: f64, mut hi: f64) -> Result<f64, String> {
    let mut f_lo = sampler.f(lo)?;
    if f_lo == 0.0 {
        return Ok(lo);
    }
    let f_hi = sampler.f(hi)?;
    if f_hi == 0.0 {
        return Ok(hi);
    }
    if f_lo * f_hi > 0.0 {
        return Err(format!(
            "solve_with_method: bisection needs a sign change over [{lo}, {hi}]"
        ));
    }
    for _ in 0..ARITHMA_SOLVE_MAX_ITERATIONS {
        let mid = 0.5 * (lo + hi);
        let f_mid = sampler.f(mid)?;
        if f_mid == 0.0 || (hi - lo).abs() <= ARITHMA_SOLVE_STEP_TOL * mid.abs().max(1.0) {
            return Ok(mid);
        }
        if f_lo * f_mid < 0.0 {
            hi = mid;
        } else {
            lo = mid;
            f_lo = f_mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

/// Brent's method (van Wijngaarden–Dekker–Brent): inverse quadratic
/// interpolation with a guaranteed bisection fallback.
fn brent(sampler: &mut Sampler<'_>, lo: f64, hi: f64) -> Result<f64, String> {
    let mut a = lo;
    let mut b = hi;
    let mut fa = sampler.f(a)?;
    let mut fb = sampler.f(b)?;
    if fa == 0.0 {
        return Ok(a);
    }
    if fb == 0.0 {
        return Ok(b);
    }
    if fa * fb > 0.0 {
        return Err(format!(
            "solve_with_method: Brent needs a sign change over [{lo}, {hi}]"
        ));
    }
    if fa.abs() < fb.abs() {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }
    let mut c = a;
    let mut fc = fa;
    let mut d = a;
    let mut used_bisection = true;

    for _ in 0..ARITHMA_SOLVE_MAX_ITERATIONS {
        if fb.abs() <= ARITHMA_SOLVE_RESIDUAL_TOL
            || (b - a).abs() <= ARITHMA_SOLVE_STEP_TOL * b.abs().max(1.0)
        {
            return Ok(b);
        }
        let tol = ARITHMA_SOLVE_STEP_TOL * b.abs().max(1.0);
        let mut s = if fa != fc && fb != fc {
            a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb))
        } else {
            b - fb * (b - a) / (fb - fa)
        };
        let outside = (s - (3.0 * a + b) / 4.0) * (s - b) >= 0.0;
        let stalled = if used_bisection {
            (s - b).abs() >= (b - c).abs() / 2.0 || (b - c).abs() < tol
        } else {
            (s - b).abs() >= (c - d).abs() / 2.0 || (c - d).abs() < tol
        };
        if !s.is_finite() || outside || stalled {
            s = 0.5 * (a + b);
            used_bisection = true;
        } else {
            used_bisection = false;
        }
        let fs = sampler.f(s)?;
        d = c;
        c = b;
        fc = fb;
        if fa * fs < 0.0 {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }
        if fa.abs() < fb.abs() {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut fa, &mut fb);
        }
    }
    Ok(b)
}

/// Solve `expr = 0` for `var` using the specified method.
///
/// # Methods
///
/// - [`ArithmaNumericalMethod::NewtonRaphson`] — `x ← x − f(x)/f′(x)` from
///   `initial`, with `f′` taken by central differences (relative step
///   `1e-6·max(|x|, 1)`). Errors out if the derivative vanishes.
/// - [`ArithmaNumericalMethod::Secant`] — two-point variant seeded with
///   `initial` and `initial + 1e-4·max(|initial|, 1)`; needs no derivative.
/// - [`ArithmaNumericalMethod::Bisection`] — `initial` is a *seed*, not a
///   bracket, so a sign-changing bracket is first grown outward from it by
///   doubling; then plain bisection.
/// - [`ArithmaNumericalMethod::Brent`] — same bracketing step, then Brent's
///   method (inverse quadratic interpolation with a bisection fallback).
///
/// Every loop is bounded: [`ARITHMA_SOLVE_MAX_ITERATIONS`] iterations and
/// [`ARITHMA_SOLVE_MAX_EXPANSIONS`] bracket doublings. Convergence is declared
/// when `|f(x)| ≤ 1e-12` or the step/bracket shrinks below a relative `1e-14`.
///
/// # Note for the root-finding migration
///
/// This implementation is deliberately self-contained because
/// [`super::root_finding`] was still stubbed when it landed. Once
/// `find_root_bisection` / `find_root_newton_raphson` / `find_root_secant` are
/// implemented, this dispatcher should become a thin router over them —
/// forwarding an [`super::root_finding::ArithmaRootFindingConfig`] and
/// unwrapping the returned `ArithmaRootFindingResult::root` — and Brent should
/// move down beside them. Keep the bracket-growing behaviour above: the public
/// signature only supplies a single `initial` value, whereas
/// `find_root_bisection` takes an explicit `(lo, hi)`.
///
/// # Errors
///
/// Returns `Err` when `expr` cannot be evaluated (or is non-finite) at a probe
/// point, when a Newton derivative vanishes, when no sign-changing bracket can
/// be grown for the bracketing methods, or when the iteration cap is reached
/// without the residual falling under tolerance.
pub fn solve_with_method(
    expr: &ArithmaExpression,
    var: &str,
    method: ArithmaNumericalMethod,
    initial: f64,
) -> Result<f64, String> {
    if !initial.is_finite() {
        return Err(format!(
            "solve_with_method: initial guess {initial} is not finite"
        ));
    }
    let mut sampler = Sampler::new(expr, var);

    match method {
        ArithmaNumericalMethod::NewtonRaphson => {
            let mut x = initial;
            for _ in 0..ARITHMA_SOLVE_MAX_ITERATIONS {
                let fx = sampler.f(x)?;
                if fx.abs() <= ARITHMA_SOLVE_RESIDUAL_TOL {
                    return Ok(x);
                }
                let dfx = sampler.df(x)?;
                if dfx.abs() < ARITHMA_SOLVE_MIN_DERIVATIVE {
                    return Err(format!(
                        "solve_with_method: Newton-Raphson derivative vanished at {var} = {x}"
                    ));
                }
                let next = x - fx / dfx;
                if !next.is_finite() {
                    return Err(format!(
                        "solve_with_method: Newton-Raphson diverged from {var} = {x}"
                    ));
                }
                if (next - x).abs() <= ARITHMA_SOLVE_STEP_TOL * next.abs().max(1.0) {
                    return Ok(next);
                }
                x = next;
            }
            Err(format!(
                "solve_with_method: Newton-Raphson did not converge in {ARITHMA_SOLVE_MAX_ITERATIONS} iterations"
            ))
        }
        ArithmaNumericalMethod::Secant => {
            let mut x0 = initial;
            let mut x1 = initial + 1e-4 * initial.abs().max(1.0);
            let mut f0 = sampler.f(x0)?;
            let mut f1 = sampler.f(x1)?;
            for _ in 0..ARITHMA_SOLVE_MAX_ITERATIONS {
                if f1.abs() <= ARITHMA_SOLVE_RESIDUAL_TOL {
                    return Ok(x1);
                }
                let denom = f1 - f0;
                if denom.abs() < ARITHMA_SOLVE_MIN_DERIVATIVE {
                    return Err(format!(
                        "solve_with_method: secant slope vanished near {var} = {x1}"
                    ));
                }
                let next = x1 - f1 * (x1 - x0) / denom;
                if !next.is_finite() {
                    return Err(format!(
                        "solve_with_method: secant diverged from {var} = {x1}"
                    ));
                }
                if (next - x1).abs() <= ARITHMA_SOLVE_STEP_TOL * next.abs().max(1.0) {
                    return Ok(next);
                }
                x0 = x1;
                f0 = f1;
                x1 = next;
                f1 = sampler.f(x1)?;
            }
            Err(format!(
                "solve_with_method: secant did not converge in {ARITHMA_SOLVE_MAX_ITERATIONS} iterations"
            ))
        }
        ArithmaNumericalMethod::Bisection => {
            let (lo, hi) = bracket_around(&mut sampler, initial)?;
            if lo == hi {
                return Ok(lo);
            }
            bisect(&mut sampler, lo, hi)
        }
        ArithmaNumericalMethod::Brent => {
            let (lo, hi) = bracket_around(&mut sampler, initial)?;
            if lo == hi {
                return Ok(lo);
            }
            brent(&mut sampler, lo, hi)
        }
    }
}

// ---------------------------------------------------------------------------
// Backward-compatibility aliases for the pre-rename `Arithmos*` names.
// Retained for one release; downstream (eml-math, eml-spectral, metaphysica,
// periodica) should migrate to the `Arithma*` names above.
// ---------------------------------------------------------------------------
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaNumericalMethod`")]
#[allow(unused)]
pub use self::ArithmaNumericalMethod as ArithmosNumericalMethod;

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_METHODS: [ArithmaNumericalMethod; 4] = [
        ArithmaNumericalMethod::Bisection,
        ArithmaNumericalMethod::NewtonRaphson,
        ArithmaNumericalMethod::Secant,
        ArithmaNumericalMethod::Brent,
    ];

    fn x() -> ArithmaExpression {
        ArithmaExpression::var("x")
    }

    /// `x² - 2`, whose positive root is √2.
    fn x_squared_minus_two() -> ArithmaExpression {
        ArithmaExpression::sub(
            ArithmaExpression::pow(x(), ArithmaExpression::from_i64(2)),
            ArithmaExpression::from_i64(2),
        )
    }

    #[test]
    fn methods_are_distinct() {
        assert_ne!(
            ArithmaNumericalMethod::Bisection,
            ArithmaNumericalMethod::NewtonRaphson
        );
    }

    #[test]
    fn every_method_finds_the_square_root_of_two() {
        let expr = x_squared_minus_two();
        for method in ALL_METHODS {
            let root = solve_with_method(&expr, "x", method, 1.0)
                .unwrap_or_else(|e| panic!("{method:?} failed: {e}"));
            assert!(
                (root - std::f64::consts::SQRT_2).abs() < 1e-9,
                "{method:?} returned {root}, expected {}",
                std::f64::consts::SQRT_2
            );
        }
    }

    #[test]
    fn every_method_finds_the_negative_root_from_a_negative_seed() {
        let expr = x_squared_minus_two();
        for method in ALL_METHODS {
            let root = solve_with_method(&expr, "x", method, -1.0)
                .unwrap_or_else(|e| panic!("{method:?} failed: {e}"));
            assert!(
                (root + std::f64::consts::SQRT_2).abs() < 1e-9,
                "{method:?} returned {root}"
            );
        }
    }

    #[test]
    fn every_method_finds_the_first_zero_of_cosine() {
        let expr = ArithmaExpression::cos(x());
        for method in ALL_METHODS {
            let root = solve_with_method(&expr, "x", method, 1.0)
                .unwrap_or_else(|e| panic!("{method:?} failed: {e}"));
            assert!(
                (root - std::f64::consts::FRAC_PI_2).abs() < 1e-9,
                "{method:?} returned {root}"
            );
        }
    }

    #[test]
    fn methods_agree_with_each_other_on_a_cubic() {
        // x³ - x - 2, single real root at ≈ 1.5213797068045676.
        let expr = ArithmaExpression::sub(
            ArithmaExpression::sub(
                ArithmaExpression::pow(x(), ArithmaExpression::from_i64(3)),
                x(),
            ),
            ArithmaExpression::from_i64(2),
        );
        let expected = 1.521_379_706_804_567_6;
        for method in ALL_METHODS {
            let root = solve_with_method(&expr, "x", method, 1.0)
                .unwrap_or_else(|e| panic!("{method:?} failed: {e}"));
            assert!(
                (root - expected).abs() < 1e-8,
                "{method:?} returned {root}, expected {expected}"
            );
        }
    }

    #[test]
    fn newton_converges_on_a_transcendental_root() {
        // exp(x) - 3 → ln 3.
        let expr =
            ArithmaExpression::sub(ArithmaExpression::exp(x()), ArithmaExpression::from_i64(3));
        let root =
            solve_with_method(&expr, "x", ArithmaNumericalMethod::NewtonRaphson, 0.0).unwrap();
        assert!((root - 3.0_f64.ln()).abs() < 1e-9, "root = {root}");
    }

    #[test]
    fn a_seed_that_is_already_a_root_is_returned_unchanged() {
        // x - 4, seeded exactly at 4.
        let expr = ArithmaExpression::sub(x(), ArithmaExpression::from_i64(4));
        for method in ALL_METHODS {
            let root = solve_with_method(&expr, "x", method, 4.0).unwrap();
            assert!((root - 4.0).abs() < 1e-12, "{method:?} returned {root}");
        }
    }

    #[test]
    fn bracketing_methods_reject_a_function_with_no_sign_change() {
        // x² + 1 never crosses zero.
        let expr = ArithmaExpression::add(
            ArithmaExpression::pow(x(), ArithmaExpression::from_i64(2)),
            ArithmaExpression::from_i64(1),
        );
        for method in [
            ArithmaNumericalMethod::Bisection,
            ArithmaNumericalMethod::Brent,
        ] {
            let err = solve_with_method(&expr, "x", method, 0.0).unwrap_err();
            assert!(
                err.contains("sign change"),
                "{method:?} gave unexpected error: {err}"
            );
        }
    }

    #[test]
    fn newton_reports_a_vanished_derivative_on_a_flat_function() {
        // The constant 7 has no root and zero slope everywhere.
        let expr = ArithmaExpression::from_i64(7);
        let err =
            solve_with_method(&expr, "x", ArithmaNumericalMethod::NewtonRaphson, 0.0).unwrap_err();
        assert!(err.contains("derivative"), "unexpected error: {err}");
    }

    #[test]
    fn a_non_finite_seed_is_rejected() {
        let expr = x_squared_minus_two();
        assert!(solve_with_method(&expr, "x", ArithmaNumericalMethod::Secant, f64::NAN).is_err());
        assert!(
            solve_with_method(&expr, "x", ArithmaNumericalMethod::Bisection, f64::INFINITY)
                .is_err()
        );
    }

    #[test]
    fn an_unbound_variable_propagates_as_an_error() {
        let expr = ArithmaExpression::var("y");
        for method in ALL_METHODS {
            let err = solve_with_method(&expr, "x", method, 1.0).unwrap_err();
            assert!(err.contains('y'), "{method:?} gave: {err}");
        }
    }

    #[test]
    fn residuals_are_actually_small_at_the_returned_roots() {
        use crate::expression::ArithmaBindings;
        let expr = x_squared_minus_two();
        for method in ALL_METHODS {
            let root = solve_with_method(&expr, "x", method, 3.0).unwrap();
            let mut bindings = ArithmaBindings::new();
            bindings.insert("x".to_string(), root);
            let residual = expr.evaluate(&bindings).unwrap();
            assert!(residual.abs() < 1e-9, "{method:?}: f({root}) = {residual}");
        }
    }
}
