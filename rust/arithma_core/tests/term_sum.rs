use arithma_core::expression::term_sum::{TermSum, TermSumError};
use arithma_core::expression::{ArithmaBindings, ArithmaExpression, Evaluable};
use arithma_core::integer::ArithmaInteger;

fn binds(pairs: &[(&str, f64)]) -> ArithmaBindings {
    pairs.iter().map(|(k, v)| ((*k).to_string(), *v)).collect()
}

#[test]
fn polynomial_lists_coefficients_and_inputs() {
    let p = TermSum::polynomial("x", 3, "c").unwrap();
    assert_eq!(p.coefficients(), vec!["c0", "c1", "c2", "c3"]);
    assert_eq!(p.inputs(), vec!["x"]);
}

#[test]
fn repeated_or_shadowing_coefficient_fails() {
    let x = ArithmaExpression::var("x");
    let dup = TermSum::new(vec![("a".into(), x.clone()), ("a".into(), x.clone())]);
    assert_eq!(dup.unwrap_err(), TermSumError::RepeatedCoefficient("a".into()));
    let shadow = TermSum::new(vec![("a".into(), ArithmaExpression::var("a"))]);
    assert_eq!(shadow.unwrap_err(), TermSumError::ShadowedCoefficient("a".into()));
}

#[test]
fn unbound_coefficient_is_named() {
    let p = TermSum::polynomial("x", 2, "c").unwrap();
    let err = p
        .evaluate(&binds(&[("x", 2.0)]), &binds(&[("c0", 1.0), ("c2", 3.0)]))
        .unwrap_err();
    assert_eq!(err, TermSumError::UnboundCoefficient("c1".into()));
    assert!(err.to_string().contains("c1"));
}

#[test]
fn to_expression_matches_evaluate() {
    let p = TermSum::polynomial("x", 3, "c").unwrap();
    let coeffs = binds(&[("c0", 1.0), ("c1", -2.0), ("c2", 0.5), ("c3", 3.0)]);
    let expr = p.to_expression();
    for x in [-2.0, -0.5, 0.0, 1.5, 4.0] {
        let inputs = binds(&[("x", x)]);
        let direct = p.evaluate(&inputs, &coeffs).unwrap();
        let mut all = inputs.clone();
        all.extend(coeffs.clone());
        let via = expr.evaluate(&all).unwrap();
        let want = 1.0 - 2.0 * x + 0.5 * x * x + 3.0 * x * x * x;
        assert!((direct - via).abs() < 1e-9 && (direct - want).abs() < 1e-9);
    }
}

#[test]
fn free_symbols_of_nested_expression() {
    let v = ArithmaExpression::var;
    let one = ArithmaExpression::num(ArithmaInteger::from_i64(1));
    let sum = ArithmaExpression::Sum {
        variable: "i".into(),
        start: Box::new(one),
        end: Box::new(v("n")),
        expression: Box::new(ArithmaExpression::mul(v("i"), v("y"))),
    };
    let e = ArithmaExpression::add(ArithmaExpression::sin(v("x")), sum);
    let got: Vec<String> = e.free_symbols().into_iter().collect();
    assert_eq!(got, vec!["n", "x", "y"]);
}
