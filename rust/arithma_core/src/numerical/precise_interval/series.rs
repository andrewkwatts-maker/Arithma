//====== Arithma/rust/arithma_core/src/numerical/precise_interval/series.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! Taylor-series kernels with rigorous tail bounds: `exp`, `ln`, `sin`, `cos`,
//! `atan` (for `pi`), each over exact rationals.

use std::cmp::Ordering;

use super::interval::{interval_abs_max, rational_max, rational_min, PreciseInterval};
use super::rounding::{pow2_recip, round_outward, ARITHMA_PRECISE_ROUNDING_GUARD};
use crate::integer::ArithmaInteger;
use crate::rational::Rational;

/// Hard cap on Taylor-series terms for `exp`/`ln`/`sin`/`cos`/`atan`.
const ARITHMA_PRECISE_SERIES_CAP: i64 = 10_000;

/// Hard cap on halvings or doublings in `ln` argument reduction. Any finite
/// [`Rational`] lies within `2^±8192` (the 4,096-bit [`ArithmaInteger`] cap).
const ARITHMA_PRECISE_LN_REDUCTION_CAP: i64 = 8_200;

/// Extra bits for the `ln(2)` series, covering `|e| <= 8200` (14 bits).
const ARITHMA_PRECISE_LN_GUARD_BITS: u32 = 14;

/// If `|next|` is already at most half of `|prev_term|` (so the geometric
/// tail bound `2*|next|` is valid) and that tail meets `target_bits`
/// relative to `sum`, the tail bound; otherwise `None` (keep summing).
/// `next`/`prev_term`/`sum` are the rounded, bounded-size enclosures the
/// series loops track (see [`ARITHMA_PRECISE_ROUNDING_GUARD`]); only their
/// magnitude bounds matter here, so widening from rounding is harmless.
fn tail_bound_if_enough(
    next: &PreciseInterval,
    prev_term: &PreciseInterval,
    sum: &PreciseInterval,
    target_bits: u32,
) -> Result<Option<Rational>, String> {
    let half = Rational::new(ArithmaInteger::from_i64(1), ArithmaInteger::from_i64(2))?;
    let next_mag = interval_abs_max(next)?;
    let half_prev = interval_abs_max(prev_term)?.mul(&half)?;
    if next_mag.cmp(&half_prev)? == Ordering::Greater {
        return Ok(None);
    }
    let tail = next_mag.mul(&Rational::from_i64(2)?)?;
    let one = Rational::from_i64(1)?;
    let mag = rational_max(&one, &interval_abs_max(sum)?)?;
    let threshold = mag.mul(&pow2_recip(target_bits)?)?;
    if tail.cmp(&threshold)? == Ordering::Greater {
        return Ok(None);
    }
    Ok(Some(tail))
}

/// `exp(x)`, by Taylor series with a rigorous geometric tail bound. The
/// running `term`/`sum` are rounded outward each step to a bounded size —
/// see [`ARITHMA_PRECISE_ROUNDING_GUARD`].
pub(super) fn exp_bounds(x: &Rational, target_bits: u32) -> Result<PreciseInterval, String> {
    let budget = target_bits.saturating_add(ARITHMA_PRECISE_ROUNDING_GUARD);
    let x_iv = round_outward(&PreciseInterval::point(x.clone()), budget)?;
    let mut term = round_outward(&PreciseInterval::point(Rational::from_i64(1)?), budget)?;
    let mut sum = term.clone();
    let mut n: i64 = 0;
    loop {
        n += 1;
        if n > ARITHMA_PRECISE_SERIES_CAP {
            return Err("evaluate_precise: exp series cap exceeded".into());
        }
        let divisor = PreciseInterval::point(Rational::from_i64(n)?);
        let next = round_outward(&term.mul(&x_iv)?.div(&divisor)?, budget)?;
        if let Some(tail) = tail_bound_if_enough(&next, &term, &sum, target_bits)? {
            return Ok(PreciseInterval {
                lo: sum.lo.sub(&tail)?,
                hi: sum.hi.add(&tail)?,
            });
        }
        sum = round_outward(&sum.add(&next)?, budget)?;
        term = next;
    }
}

/// Splits `x > 0` as `y * 2^e` with `3/4 <= y < 3/2`, by halving or doubling.
/// `e = 0` when `x` is already in range, so there is no cancellation near
/// `x = 1`. Errors by name past [`ARITHMA_PRECISE_LN_REDUCTION_CAP`] steps.
fn ln_reduce(x: &Rational) -> Result<(Rational, i64), String> {
    let three_halves = Rational::new(ArithmaInteger::from_i64(3), ArithmaInteger::from_i64(2))?;
    let three_quarters = Rational::new(ArithmaInteger::from_i64(3), ArithmaInteger::from_i64(4))?;
    let half = Rational::new(ArithmaInteger::from_i64(1), ArithmaInteger::from_i64(2))?;
    let two = Rational::from_i64(2)?;
    let mut y = x.clone();
    let mut e: i64 = 0;
    while y.cmp(&three_halves)? != Ordering::Less {
        e += 1;
        if e > ARITHMA_PRECISE_LN_REDUCTION_CAP {
            return Err("evaluate_precise: ln argument reduction cap exceeded".into());
        }
        y = y.mul(&half)?;
    }
    while y.cmp(&three_quarters)? == Ordering::Less {
        e -= 1;
        if -e > ARITHMA_PRECISE_LN_REDUCTION_CAP {
            return Err("evaluate_precise: ln argument reduction cap exceeded".into());
        }
        y = y.mul(&two)?;
    }
    if e != 0 {
        y = y.reduce()?;
    }
    Ok((y, e))
}

/// `ln(x)` for `x > 0`, via argument reduction `x = y * 2^e` with
/// `y` in `[3/4, 3/2)`: `ln(x) = ln(y) + e*ln(2)`. `ln(2)` is summed with
/// [`ARITHMA_PRECISE_LN_GUARD_BITS`] extra bits to cover `|e| <= 8200`.
pub(super) fn ln_bounds(x: &Rational, target_bits: u32) -> Result<PreciseInterval, String> {
    if x.is_zero() || x.is_negative() {
        return Err("evaluate_precise: ln of a non-positive value".into());
    }
    let (y, e) = ln_reduce(x)?;
    let base = ln_series(&y, target_bits)?;
    if e == 0 {
        return Ok(base);
    }
    let two = Rational::from_i64(2)?;
    let ln2 = ln_series(
        &two,
        target_bits.saturating_add(ARITHMA_PRECISE_LN_GUARD_BITS),
    )?;
    let scaled = ln2.mul(&PreciseInterval::point(Rational::from_i64(e)?))?;
    base.add(&scaled)
}

/// `ln(y)` by `2*atanh((y-1)/(y+1))`. Converges within this module's
/// "halve each step" tail-bound test for `|t| = |(y-1)/(y+1)| < 0.17`, which
/// [`ln_reduce`] guarantees (`|t| <= 1/5` on `[3/4, 3/2)`).
fn ln_series(x: &Rational, target_bits: u32) -> Result<PreciseInterval, String> {
    let budget = target_bits.saturating_add(ARITHMA_PRECISE_ROUNDING_GUARD);
    let one = Rational::from_i64(1)?;
    let t = x.sub(&one)?.div(&x.add(&one)?)?;
    let t_iv = round_outward(&PreciseInterval::point(t), budget)?;
    let t2 = round_outward(&t_iv.mul(&t_iv)?, budget)?;
    let mut term = t_iv;
    let mut sum = term.clone();
    let mut k: i64 = 0;
    loop {
        if k > ARITHMA_PRECISE_SERIES_CAP {
            return Err("evaluate_precise: ln series cap exceeded".into());
        }
        let factor = PreciseInterval::point(
            Rational::from_i64(2 * k + 1)?.div(&Rational::from_i64(2 * k + 3)?)?,
        );
        let next = round_outward(&term.mul(&t2)?.mul(&factor)?, budget)?;
        if let Some(tail) = tail_bound_if_enough(&next, &term, &sum, target_bits)? {
            let two = Rational::from_i64(2)?;
            return Ok(PreciseInterval {
                lo: sum.lo.sub(&tail)?.mul(&two)?,
                hi: sum.hi.add(&tail)?.mul(&two)?,
            });
        }
        sum = round_outward(&sum.add(&next)?, budget)?;
        term = next;
        k += 1;
    }
}

/// `clamp(iv, -1, 1)`, sound since `sin`/`cos`'s true range is `[-1, 1]`.
fn clamp_unit(iv: PreciseInterval) -> Result<PreciseInterval, String> {
    let neg_one = Rational::from_i64(-1)?;
    let one = Rational::from_i64(1)?;
    Ok(PreciseInterval {
        lo: rational_max(&iv.lo, &neg_one)?,
        hi: rational_min(&iv.hi, &one)?,
    })
}

/// `sin(x)`, by Taylor series (`term_{k+1} = term_k * (-x^2) / ((2k+2)(2k+3))`).
/// See [`exp_bounds`] for why `term`/`sum` are rounded each step.
fn sin_bounds(x: &Rational, target_bits: u32) -> Result<PreciseInterval, String> {
    let budget = target_bits.saturating_add(ARITHMA_PRECISE_ROUNDING_GUARD);
    let x_iv = round_outward(&PreciseInterval::point(x.clone()), budget)?;
    let neg_x2 = round_outward(&x_iv.mul(&x_iv)?.neg()?, budget)?;
    let mut term = x_iv;
    let mut sum = term.clone();
    let mut k: i64 = 0;
    loop {
        if k > ARITHMA_PRECISE_SERIES_CAP {
            return Err("evaluate_precise: sin series cap exceeded".into());
        }
        let denom = PreciseInterval::point(Rational::from_i64((2 * k + 2) * (2 * k + 3))?);
        let next = round_outward(&term.mul(&neg_x2)?.div(&denom)?, budget)?;
        if let Some(tail) = tail_bound_if_enough(&next, &term, &sum, target_bits)? {
            return clamp_unit(PreciseInterval {
                lo: sum.lo.sub(&tail)?,
                hi: sum.hi.add(&tail)?,
            });
        }
        sum = round_outward(&sum.add(&next)?, budget)?;
        term = next;
        k += 1;
    }
}

/// `cos(x)`, by Taylor series (`term_{k+1} = term_k * (-x^2) / ((2k+1)(2k+2))`).
/// See [`exp_bounds`] for why `term`/`sum` are rounded each step.
fn cos_bounds(x: &Rational, target_bits: u32) -> Result<PreciseInterval, String> {
    let budget = target_bits.saturating_add(ARITHMA_PRECISE_ROUNDING_GUARD);
    let x_iv = round_outward(&PreciseInterval::point(x.clone()), budget)?;
    let neg_x2 = round_outward(&x_iv.mul(&x_iv)?.neg()?, budget)?;
    let mut term = round_outward(&PreciseInterval::point(Rational::from_i64(1)?), budget)?;
    let mut sum = term.clone();
    let mut k: i64 = 0;
    loop {
        if k > ARITHMA_PRECISE_SERIES_CAP {
            return Err("evaluate_precise: cos series cap exceeded".into());
        }
        let denom = PreciseInterval::point(Rational::from_i64((2 * k + 1) * (2 * k + 2))?);
        let next = round_outward(&term.mul(&neg_x2)?.div(&denom)?, budget)?;
        if let Some(tail) = tail_bound_if_enough(&next, &term, &sum, target_bits)? {
            return clamp_unit(PreciseInterval {
                lo: sum.lo.sub(&tail)?,
                hi: sum.hi.add(&tail)?,
            });
        }
        sum = round_outward(&sum.add(&next)?, budget)?;
        term = next;
        k += 1;
    }
}

/// `atan(x)` for `|x| < 1`, by its alternating Taylor series. See
/// [`exp_bounds`] for why `term`/`sum` are rounded each step.
fn arctan_bounds(x: &Rational, target_bits: u32) -> Result<PreciseInterval, String> {
    let budget = target_bits.saturating_add(ARITHMA_PRECISE_ROUNDING_GUARD);
    let x_iv = round_outward(&PreciseInterval::point(x.clone()), budget)?;
    let neg_x2 = round_outward(&x_iv.mul(&x_iv)?.neg()?, budget)?;
    let mut term = x_iv;
    let mut sum = term.clone();
    let mut k: i64 = 0;
    loop {
        if k > ARITHMA_PRECISE_SERIES_CAP {
            return Err("evaluate_precise: atan series cap exceeded".into());
        }
        let factor = PreciseInterval::point(
            Rational::from_i64(2 * k + 1)?.div(&Rational::from_i64(2 * k + 3)?)?,
        );
        let next = round_outward(&term.mul(&neg_x2)?.mul(&factor)?, budget)?;
        if let Some(tail) = tail_bound_if_enough(&next, &term, &sum, target_bits)? {
            return Ok(PreciseInterval {
                lo: sum.lo.sub(&tail)?,
                hi: sum.hi.add(&tail)?,
            });
        }
        sum = round_outward(&sum.add(&next)?, budget)?;
        term = next;
        k += 1;
    }
}

/// `pi`, via Machin's formula `pi = 16*atan(1/5) - 4*atan(1/239)`.
pub(super) fn pi_bounds(target_bits: u32) -> Result<PreciseInterval, String> {
    let guard = target_bits.saturating_add(8);
    let fifth = Rational::new(ArithmaInteger::from_i64(1), ArithmaInteger::from_i64(5))?;
    let small = Rational::new(ArithmaInteger::from_i64(1), ArithmaInteger::from_i64(239))?;
    let a = arctan_bounds(&fifth, guard)?;
    let b = arctan_bounds(&small, guard)?;
    let sixteen = PreciseInterval::point(Rational::from_i64(16)?);
    let four = PreciseInterval::point(Rational::from_i64(4)?);
    sixteen.mul(&a)?.sub(&four.mul(&b)?)
}

/// `sin` over an interval: 1-Lipschitz, so bound at the midpoint and widen
/// by the half-width.
pub(super) fn sin_interval(x: &PreciseInterval, working_bits: u32) -> Result<PreciseInterval, String> {
    let (mid, half_width) = midpoint_and_half_width(x)?;
    let core = sin_bounds(&mid, working_bits)?;
    clamp_unit(PreciseInterval {
        lo: core.lo.sub(&half_width)?,
        hi: core.hi.add(&half_width)?,
    })
}

/// `cos` over an interval: see [`sin_interval`].
pub(super) fn cos_interval(x: &PreciseInterval, working_bits: u32) -> Result<PreciseInterval, String> {
    let (mid, half_width) = midpoint_and_half_width(x)?;
    let core = cos_bounds(&mid, working_bits)?;
    clamp_unit(PreciseInterval {
        lo: core.lo.sub(&half_width)?,
        hi: core.hi.add(&half_width)?,
    })
}

fn midpoint_and_half_width(x: &PreciseInterval) -> Result<(Rational, Rational), String> {
    let two = Rational::from_i64(2)?;
    let mid = x.lo.add(&x.hi)?.div(&two)?;
    let half_width = x.hi.sub(&x.lo)?.div(&two)?.abs()?;
    Ok((mid, half_width))
}
