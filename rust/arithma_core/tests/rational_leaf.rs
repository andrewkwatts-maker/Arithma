//====== Arithma/rust/arithma_core/tests/rational_leaf.rs ======//
//! Acceptance tests for the exact rational leaf (MP-ARI-2, ARI-A4b):
//! `1/3 + 1/6` simplifies to the single leaf `1/2`, and every evaluator and
//! emitter accepts it.

use arithma_core::expression::iterative::simplify_iterative;
use arithma_core::expression::{
    ArithmaBindings, ArithmaExpression, Emit, EmitTarget, Evaluable, SimplificationConfig,
};
use arithma_core::function::ArithmaFunction;
use arithma_core::integer::ArithmaInteger;
use arithma_core::numerical::{evaluate_precise, ArithmaInterval, Rational};
use arithma_core::parse::parse_expression;

fn int(n: i64) -> ArithmaInteger {
    ArithmaInteger::from_i64(n)
}

fn ratio(n: i64, d: i64) -> Rational {
    Rational::new(int(n), int(d)).expect("non-zero denominator")
}

fn simplified(text: &str) -> ArithmaExpression {
    let mut expr = parse_expression(text).expect("parses");
    simplify_iterative(&mut expr, &SimplificationConfig::default());
    expr
}

fn half() -> ArithmaExpression {
    ArithmaExpression::Rational(ratio(1, 2))
}

#[test]
fn one_third_plus_one_sixth_is_one_half() {
    match simplified("1/3 + 1/6") {
        ArithmaExpression::Rational(r) => assert_eq!(r, ratio(1, 2)),
        other => panic!("expected the leaf 1/2, got {other:?}"),
    }
}

#[test]
fn integer_results_stay_numbers() {
    match simplified("1/2 + 1/2") {
        ArithmaExpression::Number(n) => assert_eq!(n.to_i64(), Some(1)),
        other => panic!("expected the integer 1, got {other:?}"),
    }
}

#[test]
fn division_by_zero_is_not_folded() {
    match simplified("1/0") {
        ArithmaExpression::Function(ArithmaFunction::Divide, args) => assert_eq!(args.len(), 2),
        other => panic!("expected an unfolded Divide, got {other:?}"),
    }
}

#[test]
fn rational_leaf_round_trips_serde() {
    let json = serde_json::to_string(&half()).expect("serialises");
    let back: ArithmaExpression = serde_json::from_str(&json).expect("deserialises");
    match back {
        ArithmaExpression::Rational(r) => assert_eq!(r, ratio(1, 2)),
        other => panic!("round trip changed the leaf: {other:?}"),
    }
    let refused = serde_json::from_value::<ArithmaExpression>(with_zero_denominator(&json));
    assert!(refused.is_err(), "a zero denominator must be refused");
}

/// The serialised leaf with its denominator replaced by the integer zero.
fn with_zero_denominator(json: &str) -> serde_json::Value {
    let mut value: serde_json::Value = serde_json::from_str(json).expect("valid json");
    let zero = serde_json::to_value(int(0)).expect("serialises zero");
    let slot = value
        .get_mut("Rational")
        .and_then(|r| r.get_mut("den"))
        .expect("the leaf serialises as {\"Rational\": {\"num\", \"den\"}}");
    *slot = zero;
    value
}

#[test]
fn rational_leaf_evaluates_on_every_path() {
    let leaf = half();
    let float = leaf.evaluate(&ArithmaBindings::new()).expect("float path");
    assert_eq!(float, 0.5);

    let precise = evaluate_precise(&leaf, &[], 64).expect("precise path");
    assert_eq!(precise.lo, ratio(1, 2));
    assert_eq!(precise.hi, ratio(1, 2));

    let interval = arithma_core::numerical::interval_analysis::evaluate_interval(
        &leaf,
        "x",
        ArithmaInterval::point(0.0),
    )
    .expect("interval path");
    assert_eq!((interval.lo, interval.hi), (0.5, 0.5));
}

#[test]
fn rational_leaf_emits_on_every_target() {
    let expected = [
        (EmitTarget::Text, "1/2"),
        (EmitTarget::Latex, "\\frac{1}{2}"),
        (EmitTarget::MathMl, "<mfrac><mn>1</mn><mn>2</mn></mfrac>"),
        (EmitTarget::Glsl, "0.5"),
        (EmitTarget::Hlsl, "0.5"),
        (EmitTarget::EmlRpn, "1 2 /"),
    ];
    for (target, text) in expected {
        assert_eq!(half().emit(target).expect("emits"), text, "{target:?}");
    }
}
