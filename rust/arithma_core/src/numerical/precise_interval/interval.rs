//====== Arithma/rust/arithma_core/src/numerical/precise_interval/interval.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! [`PreciseInterval`]: a closed interval over exact [`Rational`]s, with
//! corner-hull arithmetic and integer powers.

use std::cmp::Ordering;

use crate::rational::Rational;

pub(super) fn rational_max(a: &Rational, b: &Rational) -> Result<Rational, String> {
    Ok(if a.cmp(b)? == Ordering::Less {
        b.clone()
    } else {
        a.clone()
    })
}

pub(super) fn rational_min(a: &Rational, b: &Rational) -> Result<Rational, String> {
    Ok(if a.cmp(b)? == Ordering::Greater {
        b.clone()
    } else {
        a.clone()
    })
}

fn rational_max4(
    a: &Rational,
    b: &Rational,
    c: &Rational,
    d: &Rational,
) -> Result<Rational, String> {
    rational_max(&rational_max(a, b)?, &rational_max(c, d)?)
}

fn rational_min4(
    a: &Rational,
    b: &Rational,
    c: &Rational,
    d: &Rational,
) -> Result<Rational, String> {
    rational_min(&rational_min(a, b)?, &rational_min(c, d)?)
}

/// `max(|lo|, |hi|)`: a conservative magnitude bound for `iv`.
pub(super) fn interval_abs_max(iv: &PreciseInterval) -> Result<Rational, String> {
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
        Self {
            lo: v.clone(),
            hi: v,
        }
    }

    /// The interval midpoint as an `f64`, via [`Rational::to_f64`]; the
    /// rounding of the two bounds dominates its error, not the width.
    pub fn mid_f64(&self) -> f64 {
        0.5 * (self.lo.to_f64() + self.hi.to_f64())
    }

    /// True iff `0` lies in `[lo, hi]`.
    pub fn contains_zero(&self) -> bool {
        let lo_le_zero = self.lo.is_negative() || self.lo.is_zero();
        let hi_ge_zero = !self.hi.is_negative() || self.hi.is_zero();
        lo_le_zero && hi_ge_zero
    }

    /// `self + other`.
    pub fn add(&self, other: &Self) -> Result<Self, String> {
        Ok(Self {
            lo: self.lo.add(&other.lo)?,
            hi: self.hi.add(&other.hi)?,
        })
    }

    /// `self - other`.
    pub fn sub(&self, other: &Self) -> Result<Self, String> {
        Ok(Self {
            lo: self.lo.sub(&other.hi)?,
            hi: self.hi.sub(&other.lo)?,
        })
    }

    /// `-self`.
    pub fn neg(&self) -> Result<Self, String> {
        Ok(Self {
            lo: self.hi.neg()?,
            hi: self.lo.neg()?,
        })
    }

    /// `self * other`, by the corner hull.
    pub fn mul(&self, other: &Self) -> Result<Self, String> {
        let p1 = self.lo.mul(&other.lo)?;
        let p2 = self.lo.mul(&other.hi)?;
        let p3 = self.hi.mul(&other.lo)?;
        let p4 = self.hi.mul(&other.hi)?;
        Ok(Self {
            lo: rational_min4(&p1, &p2, &p3, &p4)?,
            hi: rational_max4(&p1, &p2, &p3, &p4)?,
        })
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
        Ok(Self {
            lo: rational_min4(&q1, &q2, &q3, &q4)?,
            hi: rational_max4(&q1, &q2, &q3, &q4)?,
        })
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
pub(super) fn pow_int_interval(base: &PreciseInterval, k: i64) -> Result<PreciseInterval, String> {
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
            Ok(PreciseInterval {
                lo: Rational::zero(),
                hi: outer,
            })
        } else {
            Ok(PreciseInterval {
                lo: rational_min(&a, &b)?.pow_u32(k_u)?,
                hi: outer,
            })
        }
    } else {
        Ok(PreciseInterval {
            lo: base.lo.pow_u32(k_u)?,
            hi: base.hi.pow_u32(k_u)?,
        })
    }
}
