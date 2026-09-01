//====== Arithma/rust/arithma_core/src/calculus/integration.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Integration
//!
//! Symbolic integration with a numeric fallback.
//!
//! The symbolic side implements the rules that are safe to apply structurally:
//! linearity, the power rule, and a table of standard antiderivatives. It does
//! **not** attempt substitution or integration by parts — those need a
//! heuristic search, and a wrong antiderivative is far worse than an honest
//! refusal, so anything unrecognised returns `Err`. Callers that only need a
//! number should use [`integrate_numeric`], which handles everything the
//! evaluator can sample.
//!
//! Indefinite results omit the constant of integration.

use std::collections::HashMap;

use crate::expression::{ArithmaBindings, ArithmaExpression, Evaluable};
use crate::function::ArithmaFunction;

/// Panels used by [`integrate_numeric`]. Even, as Simpson's rule requires.
const QUADRATURE_PANELS: usize = 1000;

/// Indefinite integral ∫ expr d{var}, without the constant of integration.
///
/// Handles: constants, the variable itself, sums and differences (linearity),
/// constant multiples, the power rule `∫xⁿ = xⁿ⁺¹/(n+1)` for `n ≠ -1`, `1/x`,
/// and the standard table entries `exp`, `sin`, `cos`, `√x`.
///
/// Returns `Err` for anything else rather than guessing.
pub fn integrate(expr: &ArithmaExpression, var: &str) -> Result<ArithmaExpression, String> {
    let x = || ArithmaExpression::var(var);

    // A subtree that does not mention `var` is constant with respect to it,
    // so ∫c dx = c·x.
    if !mentions(expr, var) {
        return Ok(ArithmaExpression::mul(expr.clone(), x()));
    }

    match expr {
        // ∫x dx = x²/2
        ArithmaExpression::Variable(name) if name == var => Ok(ArithmaExpression::div(
            ArithmaExpression::mul(x(), x()),
            ArithmaExpression::from_i64(2),
        )),

        ArithmaExpression::Function(f, args) => integrate_function(f, args, var),

        other => Err(format!(
            "no integration rule for this expression form: {other:?}"
        )),
    }
}

fn integrate_function(
    f: &ArithmaFunction,
    args: &[ArithmaExpression],
    var: &str,
) -> Result<ArithmaExpression, String> {
    let x = || ArithmaExpression::var(var);
    let int = |e: &ArithmaExpression| integrate(e, var);

    match f {
        // Linearity: ∫(f ± g) = ∫f ± ∫g
        ArithmaFunction::Add if args.len() == 2 => {
            Ok(ArithmaExpression::add(int(&args[0])?, int(&args[1])?))
        }
        ArithmaFunction::Subtract if args.len() == 2 => {
            Ok(ArithmaExpression::sub(int(&args[0])?, int(&args[1])?))
        }
        ArithmaFunction::Negate if args.len() == 1 => Ok(ArithmaExpression::neg(int(&args[0])?)),

        // Constant multiple: pull whichever side is free of `var` out front.
        ArithmaFunction::Multiply if args.len() == 2 => {
            let (a, b) = (&args[0], &args[1]);
            match (mentions(a, var), mentions(b, var)) {
                (false, true) => Ok(ArithmaExpression::mul(a.clone(), int(b)?)),
                (true, false) => Ok(ArithmaExpression::mul(b.clone(), int(a)?)),
                _ => Err(
                    "product of two var-dependent factors needs integration by parts, \
                     which is not implemented"
                        .to_string(),
                ),
            }
        }

        // Quotient by a constant: ∫(f/c) = (∫f)/c
        ArithmaFunction::Divide if args.len() == 2 => {
            let (num, den) = (&args[0], &args[1]);
            if !mentions(den, var) {
                return Ok(ArithmaExpression::div(int(num)?, den.clone()));
            }
            // ∫(1/x) dx = ln|x|
            if !mentions(num, var) && is_var(den, var) {
                return Ok(ArithmaExpression::mul(
                    num.clone(),
                    ArithmaExpression::ln(abs_of(den.clone())),
                ));
            }
            Err("no rule for this quotient; substitution is not implemented".to_string())
        }

        // Power rule. `Pow(n)` carries the exponent in the variant.
        ArithmaFunction::Pow(n) if args.len() == 1 && is_var(&args[0], var) => {
            power_rule(&x(), n.to_f64())
        }

        // Table entries. Each requires its argument to be exactly `var` —
        // a composed argument would need the chain rule in reverse.
        ArithmaFunction::Exp if args.len() == 1 && is_var(&args[0], var) => {
            Ok(ArithmaExpression::exp(x()))
        }
        ArithmaFunction::Sin if args.len() == 1 && is_var(&args[0], var) => {
            Ok(ArithmaExpression::neg(ArithmaExpression::cos(x())))
        }
        ArithmaFunction::Cos if args.len() == 1 && is_var(&args[0], var) => {
            Ok(ArithmaExpression::sin(x()))
        }
        // ∫√x dx = (2/3)·x^(3/2)
        ArithmaFunction::Sqrt if args.len() == 1 && is_var(&args[0], var) => {
            Ok(ArithmaExpression::div(
                ArithmaExpression::mul(
                    ArithmaExpression::from_i64(2),
                    ArithmaExpression::mul(x(), ArithmaExpression::sqrt(x())),
                ),
                ArithmaExpression::from_i64(3),
            ))
        }

        other => Err(format!("no integration rule for `{other:?}`")),
    }
}

/// `∫xⁿ dx = xⁿ⁺¹/(n+1)`, with the `n = -1` case handled as `ln|x|`.
fn power_rule(x: &ArithmaExpression, exp: f64) -> Result<ArithmaExpression, String> {
    if (exp + 1.0).abs() < 1e-12 {
        return Ok(ArithmaExpression::ln(abs_of(x.clone())));
    }
    let next = exp + 1.0;
    // Build x^(n+1) by repeated multiplication for small non-negative integer
    // powers (exact), else fall back to the generic power node.
    let numerator = if next.fract() == 0.0 && (0.0..=16.0).contains(&next) {
        let k = next as u32;
        let mut acc = ArithmaExpression::from_i64(1);
        for _ in 0..k {
            acc = ArithmaExpression::mul(acc, x.clone());
        }
        acc
    } else {
        ArithmaExpression::pow(x.clone(), ArithmaExpression::from_f64(next))
    };
    Ok(ArithmaExpression::div(
        numerator,
        ArithmaExpression::from_f64(next),
    ))
}

/// Definite integral ∫_{lo}^{hi} expr d{var}.
///
/// Evaluates the antiderivative at both limits (the fundamental theorem). If no
/// symbolic antiderivative is available but both limits are numeric, falls back
/// to [`integrate_numeric`] and returns the result as a literal.
pub fn integrate_definite(
    expr: &ArithmaExpression,
    var: &str,
    lo: &ArithmaExpression,
    hi: &ArithmaExpression,
) -> Result<ArithmaExpression, String> {
    match integrate(expr, var) {
        Ok(antiderivative) => Ok(ArithmaExpression::sub(
            substitute(&antiderivative, var, hi),
            substitute(&antiderivative, var, lo),
        )),
        Err(symbolic_err) => {
            // Numeric fallback needs concrete limits.
            let empty: ArithmaBindings = HashMap::new();
            match (lo.evaluate(&empty), hi.evaluate(&empty)) {
                (Ok(a), Ok(b)) => Ok(ArithmaExpression::from_f64(integrate_numeric(
                    expr, var, a, b,
                )?)),
                _ => Err(format!(
                    "{symbolic_err}; numeric fallback needs numeric limits"
                )),
            }
        }
    }
}

/// Composite Simpson's rule over `[lo, hi]`.
///
/// Error is O(h⁴), exact for polynomials up to cubic. Uses
/// [`QUADRATURE_PANELS`] panels; a sample that fails to evaluate (a pole or a
/// domain gap) makes the whole call an error rather than silently skewing the
/// total.
pub fn integrate_numeric(
    expr: &ArithmaExpression,
    var: &str,
    lo: f64,
    hi: f64,
) -> Result<f64, String> {
    if !lo.is_finite() || !hi.is_finite() {
        return Err("numeric integration requires finite limits".to_string());
    }
    if lo == hi {
        return Ok(0.0);
    }
    // Integrating backwards negates the result.
    let (a, b, sign) = if lo < hi {
        (lo, hi, 1.0)
    } else {
        (hi, lo, -1.0)
    };

    let n = QUADRATURE_PANELS; // even by construction
    let h = (b - a) / n as f64;

    let sample = |x: f64| -> Result<f64, String> {
        let mut bindings: ArithmaBindings = HashMap::with_capacity(1);
        bindings.insert(var.to_string(), x);
        let y = expr.evaluate(&bindings)?;
        if y.is_finite() {
            Ok(y)
        } else {
            Err(format!("integrand is non-finite at {var} = {x}"))
        }
    };

    let mut acc = sample(a)? + sample(b)?;
    for i in 1..n {
        let x = a + h * i as f64;
        let w = if i % 2 == 0 { 2.0 } else { 4.0 };
        acc += w * sample(x)?;
    }
    Ok(sign * acc * h / 3.0)
}

// ── helpers ─────────────────────────────────────────────────────────────────

/// Whether `expr` references `var` anywhere. Iterative, per the crate's
/// no-recursion convention for tree walks.
fn mentions(expr: &ArithmaExpression, var: &str) -> bool {
    let mut stack = vec![expr];
    while let Some(node) = stack.pop() {
        match node {
            ArithmaExpression::Variable(name) if name == var => return true,
            ArithmaExpression::Function(_, args) => stack.extend(args.iter()),
            ArithmaExpression::Sum { expression, .. }
            | ArithmaExpression::Product { expression, .. } => stack.push(expression),
            ArithmaExpression::Limit { expression, .. } => stack.push(expression),
            ArithmaExpression::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                stack.push(condition);
                stack.push(then_expr);
                stack.push(else_expr);
            }
            _ => {}
        }
    }
    false
}

fn is_var(expr: &ArithmaExpression, var: &str) -> bool {
    matches!(expr, ArithmaExpression::Variable(n) if n == var)
}

fn abs_of(e: ArithmaExpression) -> ArithmaExpression {
    ArithmaExpression::func(ArithmaFunction::Abs, vec![e])
}

/// Replace every occurrence of `var` with `value`.
fn substitute(expr: &ArithmaExpression, var: &str, value: &ArithmaExpression) -> ArithmaExpression {
    match expr {
        ArithmaExpression::Variable(name) if name == var => value.clone(),
        ArithmaExpression::Function(f, args) => ArithmaExpression::func(
            f.clone(),
            args.iter().map(|a| substitute(a, var, value)).collect(),
        ),
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn x() -> ArithmaExpression {
        ArithmaExpression::var("x")
    }
    fn n(v: i64) -> ArithmaExpression {
        ArithmaExpression::from_i64(v)
    }

    fn at(e: &ArithmaExpression, v: f64) -> f64 {
        let mut b: ArithmaBindings = HashMap::new();
        b.insert("x".to_string(), v);
        e.evaluate(&b).expect("should evaluate")
    }

    // ── indefinite ────────────────────────────────────────────────────────

    #[test]
    fn integrates_a_constant() {
        // ∫5 dx = 5x
        let r = integrate(&n(5), "x").unwrap();
        assert!((at(&r, 3.0) - 15.0).abs() < 1e-9);
    }

    #[test]
    fn integrates_the_variable() {
        // ∫x dx = x²/2 ; at x=4 -> 8
        let r = integrate(&x(), "x").unwrap();
        assert!((at(&r, 4.0) - 8.0).abs() < 1e-9);
    }

    #[test]
    fn linearity_over_a_sum() {
        // ∫(x + 3) dx = x²/2 + 3x ; at x=2 -> 2 + 6 = 8
        let e = ArithmaExpression::add(x(), n(3));
        let r = integrate(&e, "x").unwrap();
        assert!((at(&r, 2.0) - 8.0).abs() < 1e-9);
    }

    #[test]
    fn pulls_out_a_constant_factor_from_either_side() {
        // ∫3x dx = 3x²/2 ; at x=2 -> 6
        let left = integrate(&ArithmaExpression::mul(n(3), x()), "x").unwrap();
        let right = integrate(&ArithmaExpression::mul(x(), n(3)), "x").unwrap();
        assert!((at(&left, 2.0) - 6.0).abs() < 1e-9);
        assert!((at(&right, 2.0) - 6.0).abs() < 1e-9);
    }

    #[test]
    fn table_entries_are_correct() {
        // ∫sin = -cos ; at 0 -> -1
        let s = integrate(&ArithmaExpression::sin(x()), "x").unwrap();
        assert!((at(&s, 0.0) + 1.0).abs() < 1e-9);
        // ∫cos = sin ; at π/2 -> 1
        let c = integrate(&ArithmaExpression::cos(x()), "x").unwrap();
        assert!((at(&c, std::f64::consts::FRAC_PI_2) - 1.0).abs() < 1e-9);
        // ∫exp = exp ; at 1 -> e
        let e = integrate(&ArithmaExpression::exp(x()), "x").unwrap();
        assert!((at(&e, 1.0) - std::f64::consts::E).abs() < 1e-9);
    }

    #[test]
    fn reciprocal_integrates_to_log() {
        // ∫(1/x) dx = ln|x| ; at e -> 1
        let e = ArithmaExpression::div(n(1), x());
        let r = integrate(&e, "x").unwrap();
        assert!((at(&r, std::f64::consts::E) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn refuses_a_product_of_two_var_terms() {
        // x·sin(x) needs integration by parts — must refuse, not guess.
        let e = ArithmaExpression::mul(x(), ArithmaExpression::sin(x()));
        let err = integrate(&e, "x").unwrap_err();
        assert!(err.contains("by parts"), "unexpected error: {err}");
    }

    /// The property that catches a wrong antiderivative: differentiating the
    /// result must recover the integrand.
    #[test]
    fn differentiating_the_antiderivative_recovers_the_integrand() {
        let cases = vec![
            n(7),
            x(),
            ArithmaExpression::add(x(), n(3)),
            ArithmaExpression::mul(n(4), x()),
            ArithmaExpression::sin(x()),
            ArithmaExpression::cos(x()),
            ArithmaExpression::exp(x()),
        ];
        for f in cases {
            let anti = integrate(&f, "x").expect("should integrate");
            let back = crate::calculus::differentiation::differentiate(&anti, "x")
                .expect("should differentiate");
            for probe in [0.3_f64, 1.0, 2.5] {
                let want = at(&f, probe);
                let got = at(&back, probe);
                assert!(
                    (want - got).abs() < 1e-6,
                    "d/dx ∫f != f at x={probe}: expected {want}, got {got}"
                );
            }
        }
    }

    // ── definite ──────────────────────────────────────────────────────────

    #[test]
    fn definite_integral_of_x_over_zero_to_two() {
        // ∫₀² x dx = 2
        let r = integrate_definite(&x(), "x", &n(0), &n(2)).unwrap();
        let empty: ArithmaBindings = HashMap::new();
        assert!((r.evaluate(&empty).unwrap() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn definite_integral_falls_back_to_quadrature() {
        // x·sin(x) has no symbolic rule here, but numeric limits let the
        // fallback close it. ∫₀^π x·sin(x) dx = π.
        let e = ArithmaExpression::mul(x(), ArithmaExpression::sin(x()));
        let pi = ArithmaExpression::from_f64(std::f64::consts::PI);
        let r = integrate_definite(&e, "x", &n(0), &pi).unwrap();
        let empty: ArithmaBindings = HashMap::new();
        let v = r.evaluate(&empty).unwrap();
        assert!((v - std::f64::consts::PI).abs() < 1e-6, "got {v}");
    }

    // ── numeric ───────────────────────────────────────────────────────────

    #[test]
    fn simpson_is_exact_for_a_cubic() {
        // ∫₀¹ x³ dx = 1/4. Simpson integrates cubics exactly.
        let cube = ArithmaExpression::mul(ArithmaExpression::mul(x(), x()), x());
        let v = integrate_numeric(&cube, "x", 0.0, 1.0).unwrap();
        assert!((v - 0.25).abs() < 1e-12, "got {v}");
    }

    #[test]
    fn numeric_matches_a_known_transcendental_integral() {
        // ∫₀^π sin(x) dx = 2
        let v = integrate_numeric(&ArithmaExpression::sin(x()), "x", 0.0, std::f64::consts::PI)
            .unwrap();
        assert!((v - 2.0).abs() < 1e-9, "got {v}");
    }

    #[test]
    fn reversed_limits_negate_the_result() {
        let f = ArithmaExpression::sin(x());
        let fwd = integrate_numeric(&f, "x", 0.0, std::f64::consts::PI).unwrap();
        let rev = integrate_numeric(&f, "x", std::f64::consts::PI, 0.0).unwrap();
        assert!((fwd + rev).abs() < 1e-12, "{fwd} vs {rev}");
    }

    #[test]
    fn zero_width_interval_is_zero() {
        assert_eq!(integrate_numeric(&x(), "x", 2.0, 2.0).unwrap(), 0.0);
    }

    #[test]
    fn non_finite_limits_are_rejected() {
        assert!(integrate_numeric(&x(), "x", 0.0, f64::INFINITY).is_err());
    }

    #[test]
    fn a_pole_in_the_interval_is_an_error_not_a_wrong_number() {
        // 1/x over [-1, 1] passes through a pole; silently returning a finite
        // total would be the dangerous outcome.
        let e = ArithmaExpression::div(n(1), x());
        assert!(integrate_numeric(&e, "x", -1.0, 1.0).is_err());
    }
}
