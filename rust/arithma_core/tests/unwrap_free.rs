//! Conditional and iterative-derivative paths that used to pop a value stack
//! with `unwrap`, plus a tripwire that keeps the library's unwrap/expect lint.

use arithma_core::calculus::differentiate_iterative;
use arithma_core::expression::{ArithmaBindings, Evaluable};
use arithma_core::numerical::interval_analysis::{evaluate_interval, ArithmaInterval};
use arithma_core::parse::parse_expression;
use arithma_core::ArithmaExpression;

fn parse(text: &str) -> ArithmaExpression {
    parse_expression(text).unwrap()
}

fn binds(pairs: &[(&str, f64)]) -> ArithmaBindings {
    pairs.iter().map(|(k, v)| ((*k).to_string(), *v)).collect()
}

fn conditional(cond: &str, then: &str, otherwise: &str) -> ArithmaExpression {
    ArithmaExpression::Conditional {
        condition: Box::new(parse(cond)),
        then_expr: Box::new(parse(then)),
        else_expr: Box::new(parse(otherwise)),
    }
}

#[test]
fn conditional_evaluates_both_branches() {
    let expr = conditional("c", "10", "20");
    assert_eq!(expr.evaluate(&binds(&[("c", 1.0)])).unwrap(), 10.0);
    assert_eq!(expr.evaluate(&binds(&[("c", 0.0)])).unwrap(), 20.0);
}

#[test]
fn conditional_interval_unions_when_undecided() {
    let expr = conditional("c", "1", "3");
    let out = evaluate_interval(&expr, "c", ArithmaInterval::new(-1.0, 1.0)).unwrap();
    assert_eq!(out.lo, 1.0);
    assert_eq!(out.hi, 3.0);
}

#[test]
fn iterative_derivative_returns_one_result() {
    let diff = differentiate_iterative(&parse("x^3"), "x").unwrap();
    assert_eq!(diff.evaluate(&binds(&[("x", 2.0)])).unwrap(), 12.0);
}

#[test]
fn library_denies_unwrap_and_expect() {
    let lib = include_str!("../src/lib.rs");
    assert!(lib.contains("deny(clippy::unwrap_used, clippy::expect_used)"));
}
