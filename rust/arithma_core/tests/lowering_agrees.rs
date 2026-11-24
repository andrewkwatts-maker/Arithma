//! Lowering conformance (ARI-A1): the float and Fourier lowerings of a fixed
//! sample set are compared with `evaluate_precise` on dyadic points, and the
//! EML lowering is checked as emitted. The EML value check is ARI-A5.
//!
//! The float lowering pinned here is the CPU oracle that every Phantasia
//! GPU-twin parity test compares against.

use std::sync::OnceLock;

use arithma_core::expression::{ArithmaBindings, Emit, EmitTarget, Evaluable};
use arithma_core::fourier::{fourier_transform, ArithmaFourierWindow};
use arithma_core::integer::ArithmaInteger;
use arithma_core::numerical::{evaluate_precise, Rational};
use arithma_core::parse::parse_expression;
use arithma_core::{ArithmaExpression, ArithmaFourierConfig};

/// Pinned gate: max relative error (against `max(|ref|, 1)`) of the float
/// lowering. The ratchet may only lower it.
const FLOAT_GATE: f64 = 1e-13;
/// Pinned gate: max absolute error of the Fourier lowering on the periodic
/// set. The ratchet may only lower it.
const FOURIER_GATE: f64 = 1e-9;
/// Precision requested from `evaluate_precise`.
const PRECISE_BITS: u32 = 64;
/// Points are `k / 16`; the brief's range is `-40..=40`.
const POINT_K_ALL: (i64, i64) = (-40, 40);
/// `sqrt(2+x)` is real only for `x >= -2`.
const POINT_K_SQRT: (i64, i64) = (-32, 40);
const POINT_DENOMINATOR: i64 = 16;

/// The sample set: source text, whether it is periodic over `(-pi, pi)`, and
/// the inclusive `k` range on which the function is defined and the precise
/// evaluator supports it.
const SAMPLES: &[(&str, bool, (i64, i64))] = &[
    ("sin(x)", false, POINT_K_ALL),
    ("cos(3*x)", false, POINT_K_ALL),
    ("exp(x)", false, POINT_K_ALL),
    ("ln(1+x^2)", false, POINT_K_ALL),
    ("sqrt(2+x)", false, POINT_K_SQRT),
    ("x^3-2*x+1", false, POINT_K_ALL),
    ("(x^2+1)/(x^2+2)", false, POINT_K_ALL),
    ("exp(-x^2)*cos(5*x)", false, POINT_K_ALL),
    ("exp(cos(x))", true, POINT_K_ALL),
    ("1/(2+sin(x))", true, POINT_K_ALL),
];

/// One printed row of the conformance table.
struct Row {
    text: &'static str,
    float_max_rel: f64,
    fourier_max_abs: Option<f64>,
    eml_tokens: usize,
}

fn parse(text: &str) -> ArithmaExpression {
    parse_expression(text).unwrap_or_else(|e| panic!("parse {text}: {e}"))
}

/// The dyadic sample points as `(rational, f64)`, both exact.
fn points((k_lo, k_hi): (i64, i64)) -> Vec<(Rational, f64)> {
    (k_lo..=k_hi)
        .map(|k| {
            let r = Rational::new(
                ArithmaInteger::from_i64(k),
                ArithmaInteger::from_i64(POINT_DENOMINATOR),
            )
            .expect("non-zero denominator");
            (r, k as f64 / POINT_DENOMINATOR as f64)
        })
        .collect()
}

/// The precise reference value at a rational point, as the interval midpoint.
fn reference(expr: &ArithmaExpression, x: &Rational) -> Result<f64, String> {
    let iv = evaluate_precise(expr, &[("x", x.clone())], PRECISE_BITS)?;
    Ok(iv.mid_f64())
}

fn float_max_rel(expr: &ArithmaExpression, pts: &[(Rational, f64)]) -> Result<f64, String> {
    let mut worst = 0.0_f64;
    for (r, x) in pts {
        let want = reference(expr, r)?;
        let mut b = ArithmaBindings::new();
        b.insert("x".to_string(), *x);
        let got = expr.evaluate(&b)?;
        worst = worst.max((got - want).abs() / want.abs().max(1.0));
    }
    Ok(worst)
}

fn periodic_config() -> ArithmaFourierConfig {
    ArithmaFourierConfig {
        sample_count: 1024,
        range: (-std::f64::consts::PI, std::f64::consts::PI),
        harmonics: 64,
        accuracy: 0.0,
        window: ArithmaFourierWindow::Rectangular,
    }
}

fn fourier_max_abs(expr: &ArithmaExpression, pts: &[(Rational, f64)]) -> Result<f64, String> {
    let series = fourier_transform(expr, "x", &periodic_config())?;
    let mut worst = 0.0_f64;
    for (r, x) in pts {
        worst = worst.max((series.evaluate(*x) - reference(expr, r)?).abs());
    }
    Ok(worst)
}

fn eml_tokens(expr: &ArithmaExpression) -> Result<usize, String> {
    Ok(expr.emit(EmitTarget::EmlRpn)?.split_whitespace().count())
}

/// The rows, computed once and shared by every test in this target.
fn rows() -> &'static [Row] {
    static ROWS: OnceLock<Vec<Row>> = OnceLock::new();
    ROWS.get_or_init(|| build_rows().expect("rows"))
}

fn build_rows() -> Result<Vec<Row>, String> {
    SAMPLES
        .iter()
        .map(|(text, periodic, k_range)| {
            let pts = points(*k_range);
            let expr = parse(text);
            Ok(Row {
                text,
                float_max_rel: float_max_rel(&expr, &pts)?,
                fourier_max_abs: if *periodic {
                    Some(fourier_max_abs(&expr, &pts)?)
                } else {
                    None
                },
                eml_tokens: eml_tokens(&expr)?,
            })
        })
        .collect()
}

fn render(rows: &[Row]) -> String {
    let mut out = String::from("| function | float max rel | Fourier max abs | EML tokens | EML value |\n");
    out.push_str("|---|---|---|---|---|\n");
    for r in rows {
        let fourier = r
            .fourier_max_abs
            .map_or_else(|| "n/a: not periodic on range".to_string(), |v| format!("{v:.3e}"));
        out.push_str(&format!(
            "| `{}` | {:.3e} | {} | {} | A5 |\n",
            r.text, r.float_max_rel, fourier, r.eml_tokens
        ));
    }
    out
}

#[test]
fn float_lowering_agrees_with_precise() {
    println!("{}", render(rows()));
    for r in rows() {
        assert!(
            r.float_max_rel <= FLOAT_GATE,
            "{}: float max rel {:.3e} exceeds gate {FLOAT_GATE:e}",
            r.text,
            r.float_max_rel
        );
    }
}

#[test]
fn fourier_lowering_agrees_on_periodic_set() {
    let mut checked = 0;
    for r in rows() {
        if let Some(err) = r.fourier_max_abs {
            checked += 1;
            assert!(
                err <= FOURIER_GATE,
                "{}: Fourier max abs {err:.3e} exceeds gate {FOURIER_GATE:e}",
                r.text
            );
        }
    }
    assert_eq!(checked, SAMPLES.iter().filter(|(_, p, _)| *p).count());
}

#[test]
fn eml_lowering_emits_every_sample() {
    for (text, _, _) in SAMPLES {
        let tokens = eml_tokens(&parse(text)).unwrap_or_else(|e| panic!("{text}: {e}"));
        assert!(tokens > 0, "{text}: empty EML emission");
    }
}

#[test]
fn table_prints_every_function() {
    let table = render(rows());
    for (text, periodic, _) in SAMPLES {
        let row = table
            .lines()
            .find(|l| l.contains(&format!("`{text}`")))
            .unwrap_or_else(|| panic!("row missing for {text}"));
        assert_eq!(row.contains("n/a: not periodic on range"), !periodic, "{text}");
    }
}

/// `num / den` for small integers.
fn ratio(num: i64, den: i64) -> Rational {
    Rational::new(ArithmaInteger::from_i64(num), ArithmaInteger::from_i64(den)).unwrap()
}

/// `2^e` for `e` of either sign; built in `2^500` chunks, because `pow_u32`
/// squares its base past the 4,096-bit cap for exponents near 4000.
fn pow2_rational(e: i32) -> Rational {
    const CHUNK: u32 = 500;
    let two = Rational::from_i64(2).unwrap();
    let mut left = e.unsigned_abs();
    let mut p = Rational::from_i64(1).unwrap();
    while left > 0 {
        let step = left.min(CHUNK);
        p = p.mul(&two.pow_u32(step).unwrap()).unwrap();
        left -= step;
    }
    if e >= 0 {
        p
    } else {
        Rational::from_i64(1).unwrap().div(&p).unwrap()
    }
}

/// Cases for the ln domain test: the argument and its `f64::ln`, where `f64`
/// is finite (`2^±4000` is not, so those use `±4000 * LN_2`).
fn ln_domain_cases() -> Vec<(Rational, f64)> {
    vec![
        (ratio(1, 1000), (1.0_f64 / 1000.0).ln()),
        (ratio(1, 10), 0.1_f64.ln()),
        (ratio(1, 8), 0.125_f64.ln()),
        (ratio(6, 1), 6.0_f64.ln()),
        (ratio(100, 1), 100.0_f64.ln()),
        (ratio(1_000_000, 1), 1.0e6_f64.ln()),
        (pow2_rational(-4000), -4000.0 * std::f64::consts::LN_2),
        (pow2_rational(4000), 4000.0 * std::f64::consts::LN_2),
    ]
}

/// The `f64` midpoint is itself rounded twice (`num / den`), so its floor is
/// about `2^-52`; the exact enclosure width is gated at the requested `2^-60`.
const LN_MID_REL_GATE: f64 = 1.0 / (1u64 << 50) as f64;

#[test]
fn ln_covers_every_positive_rational() {
    let expr = parse("ln(x)");
    let width_gate = pow2_rational(-60);
    for (x, want) in ln_domain_cases() {
        let iv = evaluate_precise(&expr, &[("x", x.clone())], PRECISE_BITS)
            .unwrap_or_else(|e| panic!("ln({}): {e}", x.to_f64()));
        let got = iv.mid_f64();
        let rel = (got - want).abs() / want.abs();
        assert!(rel <= LN_MID_REL_GATE, "ln({}): rel {rel:.3e}", x.to_f64());
        let width = iv.hi.sub(&iv.lo).unwrap();
        assert!(
            width.cmp(&width_gate).unwrap() != std::cmp::Ordering::Greater,
            "ln({}): enclosure wider than 2^-60",
            x.to_f64()
        );
    }
}

#[test]
fn ln_reduction_cap_is_named() {
    let expr = parse("ln(x)");
    for bad in [0, -1] {
        let err = evaluate_precise(&expr, &[("x", ratio(bad, 1))], PRECISE_BITS)
            .expect_err("ln of a non-positive value must be refused");
        assert!(err.contains("non-positive"), "unnamed error: {err}");
    }
}

#[test]
fn unsupported_node_is_named_not_skipped() {
    let expr = parse("tan(x)");
    let x = Rational::new(ArithmaInteger::from_i64(1), ArithmaInteger::from_i64(2)).unwrap();
    let err = float_max_rel(&expr, &[(x, 0.5)]).expect_err("tan must be refused");
    assert!(err.to_lowercase().contains("tan"), "unnamed error: {err}");
}
