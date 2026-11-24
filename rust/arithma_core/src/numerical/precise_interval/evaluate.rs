//====== Arithma/rust/arithma_core/src/numerical/precise_interval/evaluate.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! Expression-tree evaluation: [`evaluate_precise`] and its node dispatch.

use std::cmp::Ordering;

use super::interval::{pow_int_interval, rational_max, PreciseInterval};
use super::rounding::{pow2_recip, round_outward, ARITHMA_PRECISE_ROUNDING_GUARD};
use super::series::{cos_interval, exp_bounds, ln_bounds, pi_bounds, sin_interval};
use super::sqrt::sqrt_interval;
use crate::expression::ArithmaExpression;
use crate::function::ArithmaFunction;
use crate::integer::ArithmaInteger;
use crate::rational::Rational;

/// Hard cap on visited expression nodes, mirroring
/// [`interval_analysis`](crate::numerical::interval_analysis)'s own bound.
const ARITHMA_PRECISE_NODE_CAP: usize = 1_048_576;

/// How many times [`evaluate_precise`] will retry at higher internal
/// precision before giving up.
const ARITHMA_PRECISE_RETRY_CAP: u32 = 8;

/// Extra bits of internal working precision added above the caller's
/// request, to absorb the dependency problem's over-estimation.
const ARITHMA_PRECISE_GUARD_BITS: u32 = 16;

/// The largest `pow` exponent magnitude this module will expand.
const ARITHMA_PRECISE_MAX_POWER_EXPONENT: i64 = 1024;

/// One stack frame of the iterative, explicit-stack tree walk (matches
/// `evaluate_interval`'s `Frame` shape).
enum Frame<'a> {
    Enter(&'a ArithmaExpression),
    Combine(&'a ArithmaFunction, usize),
}

/// `name`'s binding in `bindings`, as a point interval.
fn lookup_binding(name: &str, bindings: &[(&str, Rational)]) -> Result<PreciseInterval, String> {
    bindings
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, v)| PreciseInterval::point(v.clone()))
        .ok_or_else(|| format!("evaluate_precise: unbound variable '{name}'"))
}

/// The `pi` constant, or an error naming any other symbol.
fn constant_interval(symbol: &str, working_bits: u32) -> Result<PreciseInterval, String> {
    if symbol == "pi" || symbol == "π" {
        pi_bounds(working_bits)
    } else {
        Err(format!("evaluate_precise: unsupported constant '{symbol}'"))
    }
}

fn enter_node<'a>(
    node: &'a ArithmaExpression,
    bindings: &[(&str, Rational)],
    working_bits: u32,
    work: &mut Vec<Frame<'a>>,
    values: &mut Vec<PreciseInterval>,
) -> Result<(), String> {
    use ArithmaExpression as E;
    match node {
        E::Number(n) => values.push(PreciseInterval::point(Rational::from_integer(n.clone())?)),
        E::Rational(r) => values.push(PreciseInterval::point(r.clone())),
        E::Constant { symbol, .. } => values.push(constant_interval(symbol, working_bits)?),
        E::Variable(name) => values.push(lookup_binding(name, bindings)?),
        E::Function(func, args) => {
            work.push(Frame::Combine(func, args.len()));
            for a in args.iter().rev() {
                work.push(Frame::Enter(a));
            }
        }
        E::Sum { .. } => return Err("evaluate_precise: unsupported node Sum".into()),
        E::Limit { .. } => return Err("evaluate_precise: unsupported node Limit".into()),
        E::Product { .. } => return Err("evaluate_precise: unsupported node Product".into()),
        E::Conditional { .. } => {
            return Err("evaluate_precise: unsupported node Conditional".into())
        }
        E::CachedValue { .. } => {
            return Err("evaluate_precise: unsupported node CachedValue".into())
        }
        E::FourierOptimized { .. } => {
            return Err("evaluate_precise: unsupported node FourierOptimized".into())
        }
    }
    Ok(())
}

fn unary(args: &[PreciseInterval], name: &str) -> Result<PreciseInterval, String> {
    if args.len() != 1 {
        return Err(format!(
            "evaluate_precise: {name} expects 1 argument, got {}",
            args.len()
        ));
    }
    Ok(args[0].clone())
}

fn binary(
    args: &[PreciseInterval],
    name: &str,
) -> Result<(PreciseInterval, PreciseInterval), String> {
    if args.len() != 2 {
        return Err(format!(
            "evaluate_precise: {name} expects 2 arguments, got {}",
            args.len()
        ));
    }
    Ok((args[0].clone(), args[1].clone()))
}

fn fold_add(args: &[PreciseInterval]) -> Result<PreciseInterval, String> {
    let mut acc = PreciseInterval::point(Rational::zero());
    for a in args {
        acc = acc.add(a)?;
    }
    Ok(acc)
}

fn fold_mul(args: &[PreciseInterval]) -> Result<PreciseInterval, String> {
    let mut acc = PreciseInterval::point(Rational::from_i64(1)?);
    for a in args {
        acc = acc.mul(a)?;
    }
    Ok(acc)
}

/// `Power`: the exponent must evaluate to a single exact integer within
/// [`ARITHMA_PRECISE_MAX_POWER_EXPONENT`].
fn power_node(args: &[PreciseInterval]) -> Result<PreciseInterval, String> {
    let (base, exponent) = binary(args, "Power")?;
    if exponent.lo.cmp(&exponent.hi)? != Ordering::Equal {
        return Err("evaluate_precise: power exponent is not an exact integer".into());
    }
    let point = exponent.lo;
    if point.denom().cmp_integer(&ArithmaInteger::from_i64(1)) != Some(Ordering::Equal) {
        return Err("evaluate_precise: power exponent is not an exact integer".into());
    }
    let k = point
        .numer()
        .to_i64()
        .ok_or_else(|| "evaluate_precise: power exponent out of range".to_string())?;
    if k.unsigned_abs() > ARITHMA_PRECISE_MAX_POWER_EXPONENT as u64 {
        return Err("evaluate_precise: power exponent too large".into());
    }
    pow_int_interval(&base, k)
}

fn apply_function(
    func: &ArithmaFunction,
    args: &[PreciseInterval],
    working_bits: u32,
) -> Result<PreciseInterval, String> {
    use ArithmaFunction as F;
    match func {
        F::Add => fold_add(args),
        F::Subtract => {
            let (x, y) = binary(args, "Subtract")?;
            x.sub(&y)
        }
        F::Multiply => fold_mul(args),
        F::Divide => {
            let (x, y) = binary(args, "Divide")?;
            x.div(&y)
        }
        F::Power => power_node(args),
        F::Negate => unary(args, "Negate")?.neg(),
        F::Sqrt => {
            let x = unary(args, "Sqrt")?;
            sqrt_interval(&x, working_bits)
        }
        F::Exp => {
            let x = unary(args, "Exp")?;
            Ok(PreciseInterval {
                lo: exp_bounds(&x.lo, working_bits)?.lo,
                hi: exp_bounds(&x.hi, working_bits)?.hi,
            })
        }
        F::Ln => {
            let x = unary(args, "Ln")?;
            if x.lo.is_zero() || x.lo.is_negative() {
                return Err("evaluate_precise: ln of a non-positive interval".into());
            }
            Ok(PreciseInterval {
                lo: ln_bounds(&x.lo, working_bits)?.lo,
                hi: ln_bounds(&x.hi, working_bits)?.hi,
            })
        }
        F::Sin => {
            let x = unary(args, "Sin")?;
            sin_interval(&x, working_bits)
        }
        F::Cos => {
            let x = unary(args, "Cos")?;
            cos_interval(&x, working_bits)
        }
        other => Err(format!("evaluate_precise: unsupported function {other:?}")),
    }
}

/// Combine one node's already-evaluated arguments, then round the result
/// outward to a bounded size. Without this, a wide expression tree (many
/// `+`/`×` nodes each combining results that already carry a sizeable
/// denominator, e.g. several independent `sqrt` sub-expressions) compounds
/// those denominators multiplicatively with tree size, not just with
/// requested precision, and can blow past [`ArithmaInteger`]'s digit cap
/// even though every individual leaf was comfortably bounded. See
/// [`ARITHMA_PRECISE_ROUNDING_GUARD`].
fn combine_node(
    func: &ArithmaFunction,
    n: usize,
    working_bits: u32,
    values: &mut Vec<PreciseInterval>,
) -> Result<(), String> {
    if values.len() < n {
        return Err("evaluate_precise: value stack underflow".into());
    }
    let start = values.len() - n;
    let args: Vec<PreciseInterval> = values.drain(start..).collect();
    let result = apply_function(func, &args, working_bits)?;
    let budget = working_bits.saturating_add(ARITHMA_PRECISE_ROUNDING_GUARD);
    values.push(round_outward(&result, budget)?);
    Ok(())
}

/// Walk `expr` once at `working_bits` of internal precision for every
/// transcendental leaf. Composite operators (`+ − × ÷ pow`) are exact and
/// introduce no error of their own.
fn eval_tree(
    expr: &ArithmaExpression,
    bindings: &[(&str, Rational)],
    working_bits: u32,
) -> Result<PreciseInterval, String> {
    let mut work: Vec<Frame> = Vec::with_capacity(32);
    let mut values: Vec<PreciseInterval> = Vec::with_capacity(32);
    work.push(Frame::Enter(expr));
    let mut guard: usize = 0;

    while let Some(frame) = work.pop() {
        guard += 1;
        if guard > ARITHMA_PRECISE_NODE_CAP {
            return Err("evaluate_precise: node cap exceeded".into());
        }
        match frame {
            Frame::Enter(node) => enter_node(node, bindings, working_bits, &mut work, &mut values)?,
            Frame::Combine(func, n) => combine_node(func, n, working_bits, &mut values)?,
        }
    }

    if values.len() != 1 {
        return Err(format!(
            "evaluate_precise: final stack size {} != 1",
            values.len()
        ));
    }
    Ok(values.remove(0))
}

/// True iff `iv`'s width already meets `2^-bits * max(1, |value|)`.
fn interval_meets_width(iv: &PreciseInterval, bits: u32) -> Result<bool, String> {
    let width = iv.hi.sub(&iv.lo)?;
    let one = Rational::from_i64(1)?;
    let mag = rational_max(&rational_max(&one, &iv.lo.abs()?)?, &iv.hi.abs()?)?;
    let threshold = mag.mul(&pow2_recip(bits)?)?;
    Ok(width.cmp(&threshold)? != Ordering::Greater)
}

/// Evaluate `expr` to a [`PreciseInterval`] whose width is at most
/// `2^-bits * max(1, |value|)`, over the exact rational `bindings`.
///
/// Internally this walks the expression tree once at a generous internal
/// precision (`bits` plus guard bits) and checks the authoritative,
/// exact-rational width condition above; if that check fails (the
/// dependency problem ate more precision than the guard provided), it
/// retries at higher internal precision, up to
/// [`ARITHMA_PRECISE_RETRY_CAP`] times, before giving up with a named
/// error. The returned interval is never wider than the contract allows —
/// this function errors rather than silently returning a looser bound.
///
/// # Errors
///
/// An unbound variable, an unsupported node or function (named in the
/// message), division by an interval containing zero, a non-integer or
/// oversized `pow` exponent, or exhausting the retry budget.
pub fn evaluate_precise(
    expr: &ArithmaExpression,
    bindings: &[(&str, Rational)],
    bits: u32,
) -> Result<PreciseInterval, String> {
    let mut working_bits = bits.saturating_add(ARITHMA_PRECISE_GUARD_BITS);
    for _ in 0..ARITHMA_PRECISE_RETRY_CAP {
        let interval = eval_tree(expr, bindings, working_bits)?;
        if interval_meets_width(&interval, bits)? {
            return Ok(interval);
        }
        working_bits = working_bits.saturating_add(working_bits.max(ARITHMA_PRECISE_GUARD_BITS));
    }
    Err(format!(
        "evaluate_precise: could not reach {bits} bits of precision (cancellation, or an iteration cap was hit)"
    ))
}
