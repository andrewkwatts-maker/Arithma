//====== Arithma/rust/arithma_core/src/expression/rational_fold.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Rational folding
//!
//! Exact folding of rational literals for the iterative simplifier.
//!
//! A *rational literal* is an [`ArithmaExpression::Number`] with no unit and a
//! plain integer value, or an [`ArithmaExpression::Rational`]. `Add`,
//! `Subtract`, `Multiply`, `Divide` and `Negate` over literals fold into one
//! reduced leaf: an integer result stays a `Number`, anything else becomes a
//! `Rational`. A fold that would divide by zero or overflow `ArithmaInteger`
//! is refused and the node is left as it was.

use crate::expression::ArithmaExpression;
use crate::function::ArithmaFunction;
use crate::rational::Rational;

/// The exact value of `expr` when it is a rational literal.
pub(crate) fn literal(expr: &ArithmaExpression) -> Option<Rational> {
    match expr {
        ArithmaExpression::Number(n) if n.unit.is_none() && n.value.is_plain_integer() => {
            Rational::from_integer(n.clone()).ok()
        }
        ArithmaExpression::Rational(r) => Some(r.clone()),
        _ => None,
    }
}

/// The leaf for `value`: a `Number` when the denominator is one, otherwise a
/// `Rational`. `None` when reduction overflows.
pub(crate) fn leaf(value: Rational) -> Option<ArithmaExpression> {
    let reduced = value.reduce().ok()?;
    debug_assert!(!reduced.denom().is_zero(), "a reduced rational has a denominator");
    if reduced.denom().is_one() {
        Some(ArithmaExpression::Number(reduced.numer().clone()))
    } else {
        Some(ArithmaExpression::Rational(reduced))
    }
}

/// One exact step of `op` on two values, reduced. `None` on a zero divisor or
/// an overflow.
fn step(op: &ArithmaFunction, left: &Rational, right: &Rational) -> Option<Rational> {
    let raw = match op {
        ArithmaFunction::Add => left.add(right),
        ArithmaFunction::Subtract => left.sub(right),
        ArithmaFunction::Multiply => left.mul(right),
        ArithmaFunction::Divide => left.div(right),
        _ => return None,
    };
    raw.ok()?.reduce().ok()
}

/// Fold `args` left to right with `op`.
fn fold_all(op: &ArithmaFunction, values: &[Rational]) -> Option<Rational> {
    let (first, rest) = values.split_first()?;
    let mut acc = first.clone();
    for next in rest {
        acc = step(op, &acc, next)?;
    }
    Some(acc)
}

/// Whether `func` takes exactly the operand count `count`.
fn arity_fits(func: &ArithmaFunction, count: usize) -> bool {
    match func {
        ArithmaFunction::Add | ArithmaFunction::Multiply => count >= 2,
        ArithmaFunction::Subtract | ArithmaFunction::Divide => count == 2,
        ArithmaFunction::Negate => count == 1,
        _ => false,
    }
}

/// The folded replacement for `node`, or `None` when it is not a foldable
/// shape.
///
/// A node folds when every operand is a rational literal and either one of
/// them is a `Rational` or the operator is `Divide` (so `1/3` becomes a
/// leaf). Integer-only `Add`, `Subtract`, `Multiply` and `Negate` stay with
/// the integer rules.
pub(crate) fn fold(node: &ArithmaExpression) -> Option<ArithmaExpression> {
    let (func, args) = match node {
        ArithmaExpression::Function(f, a) if arity_fits(f, a.len()) => (f, a),
        _ => return None,
    };
    let has_rational = args
        .iter()
        .any(|a| matches!(a, ArithmaExpression::Rational(_)));
    if !has_rational && !matches!(func, ArithmaFunction::Divide) {
        return None;
    }
    let values: Vec<Rational> = args.iter().map(literal).collect::<Option<_>>()?;
    let value = match func {
        ArithmaFunction::Negate => values[0].neg().ok()?,
        _ => fold_all(func, &values)?,
    };
    leaf(value)
}
