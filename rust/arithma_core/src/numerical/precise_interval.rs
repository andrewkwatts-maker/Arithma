//====== Arithma/rust/arithma_core/src/numerical/precise_interval.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Precise interval evaluation
//!
//! [`evaluate_interval`](crate::numerical::interval_analysis::evaluate_interval)
//! narrows the image of a *one-variable* expression over an input interval,
//! but has no way to evaluate a fully-bound (closed-form constant)
//! expression to a requested number of bits: every leaf collapses to a
//! point `f64`, so the result is exactly the `f64` evaluation, no tighter.
//! This module closes that gap with [`evaluate_precise`], which evaluates
//! any supported expression over exact rational bindings to a caller-chosen
//! precision, bounded end to end by exact `ArithmaInteger` arithmetic — no
//! `f64` is read anywhere in the numeric path.
//!
//! ## Rational type
//!
//! [`Rational`] is a `(numerator, denominator)` pair of plain
//! [`ArithmaInteger`]s, denominator always positive, reduced by gcd (Euclid's
//! algorithm on `checked_rem_euclid`, capped). Every arithmetic operation is
//! exact; nothing is ever rounded to fit a bit width, because an
//! `ArithmaInteger` has none to fit. The "directed outward rounding" this
//! module's contract calls for is realised differently for each bounded
//! operation, never by truncating a rational:
//! - `sqrt` brackets the root with a Newton iterate and its AM-GM-derived
//!   partner (`y`, `x/y`), both *exact* rationals that bracket the root with
//!   no rounding step at all;
//! - `exp`/`ln`/`sin`/`cos`/`atan` sum a Taylor series to a partial sum `S`
//!   plus a rigorous tail bound `T` (itself an exact rational), returning
//!   `[S - T, S + T]` — the bound is added outward, never trimmed back in.
//!
//! Every one of these brackets is a sound enclosure of the true value by
//! construction, so composing them (via the same corner-hull rules
//! [`interval_analysis`](crate::numerical::interval_analysis) uses, just
//! over exact rationals instead of `f64`) never under-claims precision.
//!
//! ## The `pi` constant
//!
//! The only `Constant` this module recognises is `pi`: a `symbol` of
//! exactly `"pi"` or `"π"`. There is no other convention to match (nothing
//! else in the codebase evaluates `pi` symbolically), so this file's own
//! tests are the only contract for that spelling.
//!
//! ## Supported nodes
//!
//! `Number`, a bound `Variable`, the `pi` `Constant`, and `Function` nodes
//! for `+ − × ÷`, `pow` (small integer exponent only), `sqrt`, `exp`, `ln`,
//! `sin`, `cos`. Everything else — `Sum`, `Limit`, `Product`, `Conditional`,
//! `CachedValue`, `FourierOptimized`, and any `ArithmaFunction` not listed
//! above — is a named error, never a silent approximation.

use std::cmp::Ordering;

use crate::expression::ArithmaExpression;
use crate::function::ArithmaFunction;
use crate::integer::ArithmaInteger;

/// Hard cap on visited expression nodes, mirroring
/// [`interval_analysis`](crate::numerical::interval_analysis)'s own bound.
const ARITHMA_PRECISE_NODE_CAP: usize = 1_048_576;

/// Hard cap on Euclid's-algorithm steps when reducing a [`Rational`]. A gcd
/// needs `O(log)` steps, so this is generous headroom, never a tight fit.
const ARITHMA_PRECISE_GCD_CAP: usize = 4_096;

/// Hard cap on Taylor-series terms for `exp`/`ln`/`sin`/`cos`/`atan`.
const ARITHMA_PRECISE_SERIES_CAP: i64 = 10_000;

/// Hard cap on Newton iterations for `sqrt`. Quadratic convergence means
/// this is reached in practice after a few dozen steps at most.
const ARITHMA_PRECISE_NEWTON_CAP: usize = 4_096;

/// How many times [`evaluate_precise`] will retry at higher internal
/// precision before giving up.
const ARITHMA_PRECISE_RETRY_CAP: u32 = 8;

/// Extra bits of internal working precision added above the caller's
/// request, to absorb the dependency problem's over-estimation.
const ARITHMA_PRECISE_GUARD_BITS: u32 = 32;

/// The largest `pow` exponent magnitude this module will expand.
const ARITHMA_PRECISE_MAX_POWER_EXPONENT: i64 = 1024;

/// Extra bits of budget a Taylor series' running `term`/`sum` are rounded
/// to, above the precision the series is actually asked for. Without this,
/// repeatedly multiplying by a factor that itself has a large denominator
/// (e.g. a `pi` already computed to hundreds of bits) would multiply that
/// denominator's bit-length into the running term every iteration, blowing
/// past [`ArithmaInteger`]'s digit cap long before the series converges.
/// Rounding outward each step keeps every intermediate a bounded size, at
/// the cost of an extra, utterly negligible `iterations * 2^-budget` of
/// error alongside the already-conservative tail bound.
const ARITHMA_PRECISE_ROUNDING_GUARD: u32 = 64;

/// An exact rational number: `num / den`, `den` always positive and the
/// pair always reduced. See the module doc for why this never rounds.
#[derive(Debug, Clone)]
pub struct Rational {
    num: ArithmaInteger,
    den: ArithmaInteger,
}

impl Rational {
    /// The exact value zero.
    pub fn zero() -> Self {
        Self { num: ArithmaInteger::zero(), den: ArithmaInteger::from_i64(1) }
    }

    /// Build `num/den`, reducing by gcd. `den` must be non-zero.
    pub fn new(num: ArithmaInteger, den: ArithmaInteger) -> Result<Self, String> {
        if den.is_zero() {
            return Err("precise_interval: rational with zero denominator".into());
        }
        let (mut n, mut d) = (num, den);
        if d.value.is_negative() {
            n = negate(&n)?;
            d = negate(&d)?;
        }
        let g = gcd_nonnegative(&n, &d)?;
        if g.is_zero() {
            return Ok(Self::zero());
        }
        let err = || "precise_interval: reduction failed".to_string();
        let num_r = n.checked_div_exact(&g).ok_or_else(err)?;
        let den_r = d.checked_div_exact(&g).ok_or_else(err)?;
        Ok(Self { num: num_r, den: den_r })
    }

    /// Build the exact value `n`.
    pub fn from_i64(n: i64) -> Result<Self, String> {
        Self::new(ArithmaInteger::from_i64(n), ArithmaInteger::from_i64(1))
    }

    /// Build the exact value of a plain `ArithmaInteger`.
    pub fn from_integer(n: ArithmaInteger) -> Result<Self, String> {
        Self::new(n, ArithmaInteger::from_i64(1))
    }

    /// True iff this value is exactly zero.
    pub fn is_zero(&self) -> bool {
        self.num.is_zero()
    }

    /// True iff this value is strictly negative.
    pub fn is_negative(&self) -> bool {
        self.num.value.is_negative()
    }

    /// `|self|`.
    pub fn abs(&self) -> Result<Self, String> {
        if self.is_negative() { self.neg() } else { Ok(self.clone()) }
    }

    /// `-self`.
    pub fn neg(&self) -> Result<Self, String> {
        Ok(Self { num: negate(&self.num)?, den: self.den.clone() })
    }

    /// `self + other`.
    pub fn add(&self, other: &Self) -> Result<Self, String> {
        let err = || "precise_interval: addition overflow".to_string();
        let a = self.num.checked_mul(&other.den).ok_or_else(err)?;
        let b = other.num.checked_mul(&self.den).ok_or_else(err)?;
        let n = a.checked_add(&b).ok_or_else(err)?;
        let d = self.den.checked_mul(&other.den).ok_or_else(err)?;
        Self::new(n, d)
    }

    /// `self - other`.
    pub fn sub(&self, other: &Self) -> Result<Self, String> {
        self.add(&other.neg()?)
    }

    /// `self * other`.
    pub fn mul(&self, other: &Self) -> Result<Self, String> {
        let err = || "precise_interval: multiplication overflow".to_string();
        let n = self.num.checked_mul(&other.num).ok_or_else(err)?;
        let d = self.den.checked_mul(&other.den).ok_or_else(err)?;
        Self::new(n, d)
    }

    /// `self / other`. `Err` if `other` is zero.
    pub fn div(&self, other: &Self) -> Result<Self, String> {
        if other.is_zero() {
            return Err("precise_interval: division by zero".into());
        }
        let err = || "precise_interval: division overflow".to_string();
        let n = self.num.checked_mul(&other.den).ok_or_else(err)?;
        let d = self.den.checked_mul(&other.num).ok_or_else(err)?;
        Self::new(n, d)
    }

    /// `self` raised to a non-negative integer power.
    pub fn pow_u32(&self, exponent: u32) -> Result<Self, String> {
        let err = || "precise_interval: power overflow".to_string();
        let n = self.num.checked_pow(exponent).ok_or_else(err)?;
        let d = self.den.checked_pow(exponent).ok_or_else(err)?;
        Self::new(n, d)
    }

    /// Exact three-way comparison, by cross-multiplying (both denominators
    /// are positive, so the sign of the cross product is the sign of the
    /// difference).
    pub fn cmp(&self, other: &Self) -> Result<Ordering, String> {
        let err = || "precise_interval: comparison overflow".to_string();
        let a = self.num.checked_mul(&other.den).ok_or_else(err)?;
        let b = other.num.checked_mul(&self.den).ok_or_else(err)?;
        a.cmp_integer(&b).ok_or_else(err)
    }

    #[cfg(test)]
    fn approx_f64(&self) -> f64 {
        self.num.to_f64() / self.den.to_f64()
    }
}

/// `-v`, for a plain integer. `Err` only if `v` somehow is not plain, which
/// cannot happen for a value built by this module.
fn negate(v: &ArithmaInteger) -> Result<ArithmaInteger, String> {
    v.checked_neg().ok_or_else(|| "precise_interval: negation failed".to_string())
}

/// `|v|`, for a plain integer.
fn abs_integer(v: &ArithmaInteger) -> Result<ArithmaInteger, String> {
    if v.value.is_negative() { negate(v) } else { Ok(v.clone()) }
}

/// `gcd(|a|, |b|)` via Euclid's algorithm, capped.
fn gcd_nonnegative(a: &ArithmaInteger, b: &ArithmaInteger) -> Result<ArithmaInteger, String> {
    let mut x = abs_integer(a)?;
    let mut y = abs_integer(b)?;
    let mut guard = 0usize;
    while !y.is_zero() {
        guard += 1;
        if guard > ARITHMA_PRECISE_GCD_CAP {
            return Err("precise_interval: gcd iteration cap exceeded".into());
        }
        let r = x
            .checked_rem_euclid(&y)
            .ok_or_else(|| "precise_interval: gcd remainder failed".to_string())?;
        x = y;
        y = r;
    }
    Ok(x)
}

fn rational_max(a: &Rational, b: &Rational) -> Result<Rational, String> {
    Ok(if a.cmp(b)? == Ordering::Less { b.clone() } else { a.clone() })
}

fn rational_min(a: &Rational, b: &Rational) -> Result<Rational, String> {
    Ok(if a.cmp(b)? == Ordering::Greater { b.clone() } else { a.clone() })
}

fn rational_max4(a: &Rational, b: &Rational, c: &Rational, d: &Rational) -> Result<Rational, String> {
    rational_max(&rational_max(a, b)?, &rational_max(c, d)?)
}

fn rational_min4(a: &Rational, b: &Rational, c: &Rational, d: &Rational) -> Result<Rational, String> {
    rational_min(&rational_min(a, b)?, &rational_min(c, d)?)
}

/// `2^bits`, exact, by repeated squaring on `checked_mul` alone.
///
/// `ArithmaInteger::checked_pow` refuses any exponent past its own
/// `MAX_POW_EXPONENT` (1024) as a deliberate policy cap unrelated to
/// whether the result fits — and this module legitimately asks for larger
/// bit budgets (the rounding guard alone adds
/// [`ARITHMA_PRECISE_ROUNDING_GUARD`] on top of the caller's request).
/// Building the power this way is bounded only by the real limit,
/// `ArithmaInteger`'s digit cap, via `checked_mul`'s own `None`.
fn pow2(bits: u32) -> Result<ArithmaInteger, String> {
    let err = || "precise_interval: 2^bits overflowed".to_string();
    let mut result = ArithmaInteger::from_i64(1);
    let mut base = ArithmaInteger::from_i64(2);
    let mut remaining = bits;
    while remaining > 0 {
        if remaining & 1 == 1 {
            result = result.checked_mul(&base).ok_or_else(err)?;
        }
        remaining >>= 1;
        if remaining > 0 {
            base = base.checked_mul(&base).ok_or_else(err)?;
        }
    }
    Ok(result)
}

/// `1 / 2^bits`, exact.
fn pow2_recip(bits: u32) -> Result<Rational, String> {
    Rational::new(ArithmaInteger::from_i64(1), pow2(bits)?)
}

/// The largest multiple of `1/2^bits` that does not exceed `r`: rounds
/// toward negative infinity, never past the true value.
fn round_down(r: &Rational, bits: u32) -> Result<Rational, String> {
    let err = || "precise_interval: rounding overflow".to_string();
    let scale = pow2(bits)?;
    let scaled = r.num.checked_mul(&scale).ok_or_else(err)?;
    let floored = scaled.checked_div_floor(&r.den).ok_or_else(err)?;
    Rational::new(floored, scale)
}

/// The smallest multiple of `1/2^bits` that is not less than `r`: rounds
/// toward positive infinity, never past the true value. `ceil(r) =
/// -floor(-r)`.
fn round_up(r: &Rational, bits: u32) -> Result<Rational, String> {
    round_down(&r.neg()?, bits)?.neg()
}

/// Round `iv`'s bounds outward (down/up) to a bounded `1/2^bits` grid,
/// never narrowing the enclosure.
fn round_outward(iv: &PreciseInterval, bits: u32) -> Result<PreciseInterval, String> {
    Ok(PreciseInterval { lo: round_down(&iv.lo, bits)?, hi: round_up(&iv.hi, bits)? })
}

/// `max(|lo|, |hi|)`: a conservative magnitude bound for `iv`.
fn interval_abs_max(iv: &PreciseInterval) -> Result<Rational, String> {
    rational_max(&iv.lo.abs()?, &iv.hi.abs()?)
}

/// Closed interval `[lo, hi]` over [`Rational`]s: a guaranteed, exact
/// enclosure of a real value.
#[derive(Debug, Clone)]
pub struct PreciseInterval {
    /// The lower bound.
    pub lo: Rational,
    /// The upper bound.
    pub hi: Rational,
}

impl PreciseInterval {
    /// A degenerate interval holding the single exact value `v`.
    pub fn point(v: Rational) -> Self {
        Self { lo: v.clone(), hi: v }
    }

    /// True iff `0` lies in `[lo, hi]`.
    pub fn contains_zero(&self) -> bool {
        let lo_le_zero = self.lo.is_negative() || self.lo.is_zero();
        let hi_ge_zero = !self.hi.is_negative() || self.hi.is_zero();
        lo_le_zero && hi_ge_zero
    }

    /// `self + other`.
    pub fn add(&self, other: &Self) -> Result<Self, String> {
        Ok(Self { lo: self.lo.add(&other.lo)?, hi: self.hi.add(&other.hi)? })
    }

    /// `self - other`.
    pub fn sub(&self, other: &Self) -> Result<Self, String> {
        Ok(Self { lo: self.lo.sub(&other.hi)?, hi: self.hi.sub(&other.lo)? })
    }

    /// `-self`.
    pub fn neg(&self) -> Result<Self, String> {
        Ok(Self { lo: self.hi.neg()?, hi: self.lo.neg()? })
    }

    /// `self * other`, by the corner hull.
    pub fn mul(&self, other: &Self) -> Result<Self, String> {
        let p1 = self.lo.mul(&other.lo)?;
        let p2 = self.lo.mul(&other.hi)?;
        let p3 = self.hi.mul(&other.lo)?;
        let p4 = self.hi.mul(&other.hi)?;
        Ok(Self { lo: rational_min4(&p1, &p2, &p3, &p4)?, hi: rational_max4(&p1, &p2, &p3, &p4)? })
    }

    /// `self / other`. `Err` if `other` contains zero.
    pub fn div(&self, other: &Self) -> Result<Self, String> {
        if other.contains_zero() {
            return Err("evaluate_precise: division by an interval containing zero".into());
        }
        let q1 = self.lo.div(&other.lo)?;
        let q2 = self.lo.div(&other.hi)?;
        let q3 = self.hi.div(&other.lo)?;
        let q4 = self.hi.div(&other.hi)?;
        Ok(Self { lo: rational_min4(&q1, &q2, &q3, &q4)?, hi: rational_max4(&q1, &q2, &q3, &q4)? })
    }

    /// True iff every value `self` can take is strictly greater than every
    /// value `other` can take: the two enclosures do not overlap, and
    /// `self` is the one entirely above.
    pub fn is_entirely_above(&self, other: &Self) -> Result<bool, String> {
        Ok(self.lo.cmp(&other.hi)? == Ordering::Greater)
    }

    /// True iff `self` is entirely below `other`. See
    /// [`is_entirely_above`](Self::is_entirely_above).
    pub fn is_entirely_below(&self, other: &Self) -> Result<bool, String> {
        other.is_entirely_above(self)
    }

    /// True iff neither interval is entirely above the other: their
    /// enclosures might describe the same real value.
    pub fn overlaps(&self, other: &Self) -> Result<bool, String> {
        Ok(!self.is_entirely_above(other)? && !self.is_entirely_below(other)?)
    }
}

/// `base` raised to the exact integer power `k`, by exact rational interval
/// arithmetic (mirrors
/// [`interval_analysis::pow_int`](crate::numerical::interval_analysis)).
fn pow_int_interval(base: &PreciseInterval, k: i64) -> Result<PreciseInterval, String> {
    if k == 0 {
        return Ok(PreciseInterval::point(Rational::from_i64(1)?));
    }
    if k < 0 {
        if base.contains_zero() {
            return Err("evaluate_precise: division by an interval containing zero".into());
        }
        let positive = pow_int_interval(base, -k)?;
        return PreciseInterval::point(Rational::from_i64(1)?).div(&positive);
    }
    let k_u = k as u32;
    if k % 2 == 0 {
        let a = base.lo.abs()?;
        let b = base.hi.abs()?;
        let outer = rational_max(&a, &b)?.pow_u32(k_u)?;
        if base.contains_zero() {
            Ok(PreciseInterval { lo: Rational::zero(), hi: outer })
        } else {
            Ok(PreciseInterval { lo: rational_min(&a, &b)?.pow_u32(k_u)?, hi: outer })
        }
    } else {
        Ok(PreciseInterval { lo: base.lo.pow_u32(k_u)?, hi: base.hi.pow_u32(k_u)? })
    }
}

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
/// [`ArithmaInteger`]'s digit cap long before 200+ bits of precision is
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
    let mut hi = if x.cmp(&one)? == Ordering::Greater { round_up(x, budget)? } else { one.clone() };
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

/// `exp(x)`, by Taylor series with a rigorous geometric tail bound. The
/// running `term`/`sum` are rounded outward each step to a bounded size —
/// see [`ARITHMA_PRECISE_ROUNDING_GUARD`].
fn exp_bounds(x: &Rational, target_bits: u32) -> Result<PreciseInterval, String> {
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
            return Ok(PreciseInterval { lo: sum.lo.sub(&tail)?, hi: sum.hi.add(&tail)? });
        }
        sum = round_outward(&sum.add(&next)?, budget)?;
        term = next;
    }
}

/// `ln(x)` for `x > 0`, via `ln(x) = 2*atanh((x-1)/(x+1))`. Converges for
/// every `x > 0`, but only quickly (within this module's "halve each step"
/// tail-bound test) while `|t| = |(x-1)/(x+1)|` is not too close to 1 —
/// i.e. `x` not too far from 1. Errors with the series cap rather than
/// looping unboundedly outside that range.
fn ln_bounds(x: &Rational, target_bits: u32) -> Result<PreciseInterval, String> {
    if x.is_zero() || x.is_negative() {
        return Err("evaluate_precise: ln of a non-positive value".into());
    }
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
        let factor = PreciseInterval::point(Rational::from_i64(2 * k + 1)?.div(&Rational::from_i64(2 * k + 3)?)?);
        let next = round_outward(&term.mul(&t2)?.mul(&factor)?, budget)?;
        if let Some(tail) = tail_bound_if_enough(&next, &term, &sum, target_bits)? {
            let two = Rational::from_i64(2)?;
            return Ok(PreciseInterval { lo: sum.lo.sub(&tail)?.mul(&two)?, hi: sum.hi.add(&tail)?.mul(&two)? });
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
    Ok(PreciseInterval { lo: rational_max(&iv.lo, &neg_one)?, hi: rational_min(&iv.hi, &one)? })
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
            return clamp_unit(PreciseInterval { lo: sum.lo.sub(&tail)?, hi: sum.hi.add(&tail)? });
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
            return clamp_unit(PreciseInterval { lo: sum.lo.sub(&tail)?, hi: sum.hi.add(&tail)? });
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
        let factor = PreciseInterval::point(Rational::from_i64(2 * k + 1)?.div(&Rational::from_i64(2 * k + 3)?)?);
        let next = round_outward(&term.mul(&neg_x2)?.mul(&factor)?, budget)?;
        if let Some(tail) = tail_bound_if_enough(&next, &term, &sum, target_bits)? {
            return Ok(PreciseInterval { lo: sum.lo.sub(&tail)?, hi: sum.hi.add(&tail)? });
        }
        sum = round_outward(&sum.add(&next)?, budget)?;
        term = next;
        k += 1;
    }
}

/// `pi`, via Machin's formula `pi = 16*atan(1/5) - 4*atan(1/239)`.
fn pi_bounds(target_bits: u32) -> Result<PreciseInterval, String> {
    let guard = target_bits.saturating_add(8);
    let fifth = Rational::new(ArithmaInteger::from_i64(1), ArithmaInteger::from_i64(5))?;
    let small = Rational::new(ArithmaInteger::from_i64(1), ArithmaInteger::from_i64(239))?;
    let a = arctan_bounds(&fifth, guard)?;
    let b = arctan_bounds(&small, guard)?;
    let sixteen = PreciseInterval::point(Rational::from_i64(16)?);
    let four = PreciseInterval::point(Rational::from_i64(4)?);
    sixteen.mul(&a)?.sub(&four.mul(&b)?)
}

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
        E::Conditional { .. } => return Err("evaluate_precise: unsupported node Conditional".into()),
        E::CachedValue { .. } => return Err("evaluate_precise: unsupported node CachedValue".into()),
        E::FourierOptimized { .. } => {
            return Err("evaluate_precise: unsupported node FourierOptimized".into())
        }
    }
    Ok(())
}

fn unary(args: &[PreciseInterval], name: &str) -> Result<PreciseInterval, String> {
    if args.len() != 1 {
        return Err(format!("evaluate_precise: {name} expects 1 argument, got {}", args.len()));
    }
    Ok(args[0].clone())
}

fn binary(args: &[PreciseInterval], name: &str) -> Result<(PreciseInterval, PreciseInterval), String> {
    if args.len() != 2 {
        return Err(format!("evaluate_precise: {name} expects 2 arguments, got {}", args.len()));
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
    if point.den.cmp_integer(&ArithmaInteger::from_i64(1)) != Some(Ordering::Equal) {
        return Err("evaluate_precise: power exponent is not an exact integer".into());
    }
    let k = point
        .num
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
            Ok(PreciseInterval { lo: exp_bounds(&x.lo, working_bits)?.lo, hi: exp_bounds(&x.hi, working_bits)?.hi })
        }
        F::Ln => {
            let x = unary(args, "Ln")?;
            if x.lo.is_zero() || x.lo.is_negative() {
                return Err("evaluate_precise: ln of a non-positive interval".into());
            }
            Ok(PreciseInterval { lo: ln_bounds(&x.lo, working_bits)?.lo, hi: ln_bounds(&x.hi, working_bits)?.hi })
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

/// `sqrt` over an interval: monotone increasing, clipped to `x >= 0` first.
fn sqrt_interval(x: &PreciseInterval, working_bits: u32) -> Result<PreciseInterval, String> {
    if x.hi.is_negative() {
        return Err("evaluate_precise: sqrt of a negative interval".into());
    }
    let lo_arg = if x.lo.is_negative() { Rational::zero() } else { x.lo.clone() };
    let lo_bounds = sqrt_bounds(&lo_arg, working_bits)?;
    let hi_bounds = sqrt_bounds(&x.hi, working_bits)?;
    Ok(PreciseInterval { lo: lo_bounds.lo, hi: hi_bounds.hi })
}

/// `sin` over an interval: 1-Lipschitz, so bound at the midpoint and widen
/// by the half-width.
fn sin_interval(x: &PreciseInterval, working_bits: u32) -> Result<PreciseInterval, String> {
    let (mid, half_width) = midpoint_and_half_width(x)?;
    let core = sin_bounds(&mid, working_bits)?;
    clamp_unit(PreciseInterval { lo: core.lo.sub(&half_width)?, hi: core.hi.add(&half_width)? })
}

/// `cos` over an interval: see [`sin_interval`].
fn cos_interval(x: &PreciseInterval, working_bits: u32) -> Result<PreciseInterval, String> {
    let (mid, half_width) = midpoint_and_half_width(x)?;
    let core = cos_bounds(&mid, working_bits)?;
    clamp_unit(PreciseInterval { lo: core.lo.sub(&half_width)?, hi: core.hi.add(&half_width)? })
}

fn midpoint_and_half_width(x: &PreciseInterval) -> Result<(Rational, Rational), String> {
    let two = Rational::from_i64(2)?;
    let mid = x.lo.add(&x.hi)?.div(&two)?;
    let half_width = x.hi.sub(&x.lo)?.div(&two)?.abs()?;
    Ok((mid, half_width))
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
        return Err(format!("evaluate_precise: final stack size {} != 1", values.len()));
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

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(iv.lo.approx_f64() <= known + 1e-6, "lo = {}", iv.lo.approx_f64());
        assert!(iv.hi.approx_f64() >= known - 1e-6, "hi = {}", iv.hi.approx_f64());
        // The exact square: lo^2 <= 5 <= hi^2.
        let five = Rational::from_i64(5).unwrap();
        assert_ne!(iv.lo.pow_u32(2).unwrap().cmp(&five).unwrap(), Ordering::Greater);
        assert_ne!(iv.hi.pow_u32(2).unwrap().cmp(&five).unwrap(), Ordering::Less);
    }

    #[test]
    fn sin_pi_over_5_matches_radical_form_to_128_bits() {
        // sin(36 degrees) = sqrt(10 - 2*sqrt(5)) / 4.
        let five_rad = ArithmaExpression::sqrt(ArithmaExpression::from_i64(5));
        let inner = ArithmaExpression::sub(
            ArithmaExpression::from_i64(10),
            ArithmaExpression::mul(ArithmaExpression::from_i64(2), five_rad),
        );
        let radical = ArithmaExpression::div(ArithmaExpression::sqrt(inner), ArithmaExpression::from_i64(4));

        let angle = ArithmaExpression::div(pi_expr(), ArithmaExpression::from_i64(5));
        let sin_expr = ArithmaExpression::sin(angle);

        let a = evaluate_precise(&sin_expr, &[], 128).expect("sin must compute");
        let b = evaluate_precise(&radical, &[], 128).expect("radical must compute");
        assert!(a.overlaps(&b).expect("overlap check must compute"), "sin(pi/5)={a:?} radical={b:?}");
    }

    #[test]
    fn cos_2pi_over_5_matches_radical_form_to_128_bits() {
        // cos(72 degrees) = (sqrt(5) - 1) / 4.
        let radical = ArithmaExpression::div(
            ArithmaExpression::sub(ArithmaExpression::sqrt(ArithmaExpression::from_i64(5)), ArithmaExpression::from_i64(1)),
            ArithmaExpression::from_i64(4),
        );

        let angle = ArithmaExpression::div(
            ArithmaExpression::mul(ArithmaExpression::from_i64(2), pi_expr()),
            ArithmaExpression::from_i64(5),
        );
        let cos_expr = ArithmaExpression::cos(angle);

        let a = evaluate_precise(&cos_expr, &[], 128).expect("cos must compute");
        let b = evaluate_precise(&radical, &[], 128).expect("radical must compute");
        assert!(a.overlaps(&b).expect("overlap check must compute"), "cos(2pi/5)={a:?} radical={b:?}");
    }

    #[test]
    fn widths_shrink_as_bits_grows() {
        // Newton's quadratic convergence can satisfy two nearby bit targets
        // from the very same iteration, so adjacent requests are only
        // guaranteed non-increasing; widely spaced ones must strictly
        // shrink.
        let w = |bits: u32| {
            let iv = evaluate_precise(&sqrt5_expr(), &[], bits).expect("evaluate_precise must compute");
            iv.hi.sub(&iv.lo).unwrap().approx_f64()
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
        let expr = ArithmaExpression::func(ArithmaFunction::Gamma, vec![ArithmaExpression::from_i64(2)]);
        let err = evaluate_precise(&expr, &[], 64).unwrap_err();
        assert!(err.contains("Gamma"), "unexpected error: {err}");
    }

    #[test]
    fn rational_arithmetic_reduces() {
        let a = Rational::new(ArithmaInteger::from_i64(2), ArithmaInteger::from_i64(4)).unwrap();
        let b = Rational::from_i64(1).unwrap().div(&Rational::from_i64(2).unwrap()).unwrap();
        assert_eq!(a.cmp(&b).unwrap(), Ordering::Equal);
    }
}
