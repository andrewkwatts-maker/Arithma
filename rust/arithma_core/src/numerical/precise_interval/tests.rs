//====== Arithma/rust/arithma_core/src/numerical/precise_interval/tests.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! Behaviour tests for [`evaluate_precise`](super::evaluate_precise).

use std::cmp::Ordering;

use super::{evaluate_precise, Rational};
use crate::expression::ArithmaExpression;
use crate::function::ArithmaFunction;
use crate::integer::ArithmaInteger;

fn sqrt5_expr() -> ArithmaExpression {
    ArithmaExpression::sqrt(ArithmaExpression::from_i64(5))
}

fn pi_expr() -> ArithmaExpression {
    ArithmaExpression::constant("pi", Some("pi"), None, false)
}

#[test]
fn sqrt5_to_200_bits_brackets_known_value() {
    let iv = evaluate_precise(&sqrt5_expr(), &[], 200).expect("evaluate_precise must compute");
    let known = 5.0_f64.sqrt();
    assert!(
        iv.lo.to_f64() <= known + 1e-6,
        "lo = {}",
        iv.lo.to_f64()
    );
    assert!(
        iv.hi.to_f64() >= known - 1e-6,
        "hi = {}",
        iv.hi.to_f64()
    );
    // The exact square: lo^2 <= 5 <= hi^2.
    let five = Rational::from_i64(5).unwrap();
    assert_ne!(
        iv.lo.pow_u32(2).unwrap().cmp(&five).unwrap(),
        Ordering::Greater
    );
    assert_ne!(
        iv.hi.pow_u32(2).unwrap().cmp(&five).unwrap(),
        Ordering::Less
    );
}

#[test]
fn sin_pi_over_5_matches_radical_form_to_128_bits() {
    // sin(36 degrees) = sqrt(10 - 2*sqrt(5)) / 4.
    let five_rad = ArithmaExpression::sqrt(ArithmaExpression::from_i64(5));
    let inner = ArithmaExpression::sub(
        ArithmaExpression::from_i64(10),
        ArithmaExpression::mul(ArithmaExpression::from_i64(2), five_rad),
    );
    let radical = ArithmaExpression::div(
        ArithmaExpression::sqrt(inner),
        ArithmaExpression::from_i64(4),
    );

    let angle = ArithmaExpression::div(pi_expr(), ArithmaExpression::from_i64(5));
    let sin_expr = ArithmaExpression::sin(angle);

    let a = evaluate_precise(&sin_expr, &[], 128).expect("sin must compute");
    let b = evaluate_precise(&radical, &[], 128).expect("radical must compute");
    assert!(
        a.overlaps(&b).expect("overlap check must compute"),
        "sin(pi/5)={a:?} radical={b:?}"
    );
}

#[test]
fn cos_2pi_over_5_matches_radical_form_to_128_bits() {
    // cos(72 degrees) = (sqrt(5) - 1) / 4.
    let radical = ArithmaExpression::div(
        ArithmaExpression::sub(
            ArithmaExpression::sqrt(ArithmaExpression::from_i64(5)),
            ArithmaExpression::from_i64(1),
        ),
        ArithmaExpression::from_i64(4),
    );

    let angle = ArithmaExpression::div(
        ArithmaExpression::mul(ArithmaExpression::from_i64(2), pi_expr()),
        ArithmaExpression::from_i64(5),
    );
    let cos_expr = ArithmaExpression::cos(angle);

    let a = evaluate_precise(&cos_expr, &[], 128).expect("cos must compute");
    let b = evaluate_precise(&radical, &[], 128).expect("radical must compute");
    assert!(
        a.overlaps(&b).expect("overlap check must compute"),
        "cos(2pi/5)={a:?} radical={b:?}"
    );
}

#[test]
fn widths_shrink_as_bits_grows() {
    // Newton's quadratic convergence can satisfy two nearby bit targets
    // from the very same iteration, so adjacent requests are only
    // guaranteed non-increasing; widely spaced ones must strictly
    // shrink.
    let w = |bits: u32| {
        let iv =
            evaluate_precise(&sqrt5_expr(), &[], bits).expect("evaluate_precise must compute");
        iv.hi.sub(&iv.lo).unwrap().to_f64()
    };
    let w64 = w(64);
    let w128 = w(128);
    let w256 = w(256);
    assert!(w64 >= w128, "w64={w64} w128={w128}");
    assert!(w128 >= w256, "w128={w128} w256={w256}");
    assert!(w64 > w256, "w64={w64} w256={w256}");
}

#[test]
fn division_by_an_interval_containing_zero_is_an_error() {
    let expr = ArithmaExpression::div(
        ArithmaExpression::from_i64(1),
        ArithmaExpression::sub(ArithmaExpression::var("x"), ArithmaExpression::var("x")),
    );
    let x = Rational::from_i64(3).unwrap();
    let err = evaluate_precise(&expr, &[("x", x)], 64).unwrap_err();
    assert!(err.contains("division"), "unexpected error: {err}");
}

#[test]
fn unbound_variable_is_a_named_error() {
    let err = evaluate_precise(&ArithmaExpression::var("y"), &[], 64).unwrap_err();
    assert!(err.contains('y'), "unexpected error: {err}");
}

#[test]
fn unsupported_function_is_a_named_error() {
    let expr =
        ArithmaExpression::func(ArithmaFunction::Gamma, vec![ArithmaExpression::from_i64(2)]);
    let err = evaluate_precise(&expr, &[], 64).unwrap_err();
    assert!(err.contains("Gamma"), "unexpected error: {err}");
}

#[test]
fn rational_arithmetic_reduces() {
    let a = Rational::new(ArithmaInteger::from_i64(2), ArithmaInteger::from_i64(4)).unwrap();
    let b = Rational::from_i64(1)
        .unwrap()
        .div(&Rational::from_i64(2).unwrap())
        .unwrap();
    assert_eq!(a.cmp(&b).unwrap(), Ordering::Equal);
}
