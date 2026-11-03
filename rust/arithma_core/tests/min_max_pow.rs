use arithma_core::expression::{ArithmaBindings, Emit, EmitTarget, Evaluable};
use arithma_core::parse::parse_expression;

fn binds(pairs: &[(&str, f64)]) -> ArithmaBindings {
    pairs.iter().map(|(k, v)| ((*k).to_string(), *v)).collect()
}

fn eval(text: &str, pairs: &[(&str, f64)]) -> f64 {
    parse_expression(text).unwrap().evaluate(&binds(pairs)).unwrap()
}

#[test]
fn min_and_max_evaluate_literals_and_symbols() {
    assert_eq!(eval("min(2, 5)", &[]), 2.0);
    assert_eq!(eval("max(2, 5)", &[]), 5.0);
    assert_eq!(eval("min(x, y)", &[("x", -1.0), ("y", 3.0)]), -1.0);
    assert_eq!(eval("max(x, y)", &[("x", -1.0), ("y", 3.0)]), 3.0);
}

#[test]
fn text_round_trips() {
    for src in ["min(x, y)", "max(x, 2)"] {
        let e = parse_expression(src).unwrap();
        let text = e.emit(EmitTarget::Text).unwrap();
        assert_eq!(text, src);
        assert_eq!(format!("{:?}", parse_expression(&text).unwrap()), format!("{e:?}"));
    }
    let p = parse_expression("pow(x, 3)").unwrap();
    let text = p.emit(EmitTarget::Text).unwrap();
    assert_eq!(format!("{:?}", parse_expression(&text).unwrap()), format!("{p:?}"));
}

#[test]
fn pow_matches_caret_over_a_sweep() {
    assert_eq!(
        format!("{:?}", parse_expression("pow(x, 3)").unwrap()),
        format!("{:?}", parse_expression("x ^ 3").unwrap())
    );
    for i in -10..=10 {
        let x = f64::from(i) * 0.37;
        assert_eq!(eval("pow(x, 3)", &[("x", x)]), eval("x ^ 3", &[("x", x)]));
    }
}

#[test]
fn glsl_cbrt_has_no_one_argument_pow() {
    let e = parse_expression("cbrt(x)").unwrap();
    for t in [EmitTarget::Glsl, EmitTarget::Hlsl] {
        let s = e.emit(t).unwrap();
        assert!(s.contains("pow(abs(x), 1.0/3.0)"), "{s}");
        assert!(s.contains("sign(x)"), "{s}");
    }
    assert_eq!(e.emit(EmitTarget::Text).unwrap(), "cbrt(x)");
}

#[test]
fn min_max_emit_as_calls() {
    let e = parse_expression("min(a, max(b, c))").unwrap();
    assert_eq!(e.emit(EmitTarget::Glsl).unwrap(), "min(a, max(b, c))");
}
