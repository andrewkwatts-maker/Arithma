//====== Arithma/rust/arithma_core/src/numerical/precise_interval/sqrt.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! `sqrt` of a [`Rational`] or a [`PreciseInterval`], bracketed by Newton and AM-GM.

use std::cmp::Ordering;

use super::interval::{rational_max, PreciseInterval};
use super::rounding::{pow2_recip, round_up, ARITHMA_PRECISE_ROUNDING_GUARD};
use crate::rational::Rational;

/// Hard cap on Newton iterations for `sqrt`. Quadratic convergence means
/// this is reached in practice after a few dozen steps at most.
const ARITHMA_PRECISE_NEWTON_CAP: usize = 4_096;

/// `sqrt(x)` for `x >= 0`, bracketed by exact rationals via Newton's method:
/// `hi` starts at `max(x, 1)` (provably `>= sqrt(x)`), and each step sets
/// `lo = x/hi` (provably `<= sqrt(x)` whenever `hi >= sqrt(x)`) then
/// advances `hi` to the AM-GM mean of `hi` and `lo` (provably still `>=
/// sqrt(x)`, by AM-GM, regardless of where `hi` started).
///
/// `hi` is rounded *up* to a bounded size each step (rounding up only ever
/// strengthens `hi >= sqrt(x)`, never breaks it), which in turn keeps the
/// freshly recomputed `lo = x/hi` a bounded size too — without this, the
/// Newton recurrence's own denominator roughly doubles in bit-length every
/// iteration even for a well-behaved `x`, which blows past
/// [`ArithmaInteger`](crate::integer::ArithmaInteger)'s digit cap long before 200+ bits of precision is
/// reached. See [`ARITHMA_PRECISE_ROUNDING_GUARD`].
fn sqrt_bounds(x: &Rational, target_bits: u32) -> Result<PreciseInterval, String> {
    if x.is_negative() {
        return Err("evaluate_precise: sqrt of a negative value".into());
    }
    if x.is_zero() {
        return Ok(PreciseInterval::point(Rational::zero()));
    }
    let budget = target_bits.saturating_add(ARITHMA_PRECISE_ROUNDING_GUARD);
    let one = Rational::from_i64(1)?;
    let mut hi = if x.cmp(&one)? == Ordering::Greater {
        round_up(x, budget)?
    } else {
        one.clone()
    };
    let two = Rational::from_i64(2)?;
    let threshold = pow2_recip(target_bits)?;
    for _ in 0..ARITHMA_PRECISE_NEWTON_CAP {
        let lo = x.div(&hi)?;
        let width = hi.sub(&lo)?;
        let bound = rational_max(&one, &hi)?.mul(&threshold)?;
        if width.cmp(&bound)? != Ordering::Greater {
            return Ok(PreciseInterval { lo, hi });
        }
        hi = round_up(&hi.add(&lo)?.div(&two)?, budget)?;
    }
    Err("evaluate_precise: sqrt Newton iteration cap exceeded".into())
}

/// `sqrt` over an interval: monotone increasing, clipped to `x >= 0` first.
pub(super) fn sqrt_interval(x: &PreciseInterval, working_bits: u32) -> Result<PreciseInterval, String> {
    if x.hi.is_negative() {
        return Err("evaluate_precise: sqrt of a negative interval".into());
    }
    let lo_arg = if x.lo.is_negative() {
        Rational::zero()
    } else {
        x.lo.clone()
    };
    let lo_bounds = sqrt_bounds(&lo_arg, working_bits)?;
    let hi_bounds = sqrt_bounds(&x.hi, working_bits)?;
    Ok(PreciseInterval {
        lo: lo_bounds.lo,
        hi: hi_bounds.hi,
    })
}
