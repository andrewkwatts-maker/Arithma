//====== Arithma/rust/arithma_core/src/rational.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Exact rational numbers
//!
//! [`Rational`] is a `(numerator, denominator)` pair of plain
//! [`ArithmaInteger`]s, denominator always positive, reduced by gcd (Euclid's
//! algorithm on `checked_rem_euclid`, capped). Every arithmetic operation is
//! exact; nothing is ever rounded to fit a bit width, because an
//! `ArithmaInteger` has none to fit.
//!
//! The arithmetic combinators (`add`, `sub`, `mul`, `div`, `pow_u32`) return
//! *unreduced* values; [`Rational::new`] and [`Rational::reduce`] reduce.
//! [`PartialEq`] compares by cross-multiplication, so unreduced and reduced
//! forms of one value are equal. Serialisation goes through a raw
//! `(num, den)` pair and deserialisation through [`Rational::new`], so a
//! deserialised value is always reduced and never has a zero denominator.
//!
//! [`precise_interval`](crate::numerical::precise_interval) builds its
//! exact interval evaluation on this type.

use std::cmp::Ordering;

use serde::{Deserialize, Serialize};

use crate::integer::ArithmaInteger;

/// Hard cap on Euclid's-algorithm steps when reducing a [`Rational`]. A gcd
/// needs `O(log)` steps, so this is generous headroom, never a tight fit.
const ARITHMA_PRECISE_GCD_CAP: usize = 4_096;

/// The serialised shape of a [`Rational`]: the raw `(num, den)` pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RationalParts {
    num: ArithmaInteger,
    den: ArithmaInteger,
}

impl TryFrom<RationalParts> for Rational {
    type Error = String;

    fn try_from(parts: RationalParts) -> Result<Self, String> {
        Rational::new(parts.num, parts.den)
    }
}

impl From<Rational> for RationalParts {
    fn from(r: Rational) -> Self {
        Self {
            num: r.num,
            den: r.den,
        }
    }
}

/// An exact rational number: `num / den`, `den` always positive and the
/// pair always reduced. See the module doc for why this never rounds.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "RationalParts", into = "RationalParts")]
pub struct Rational {
    num: ArithmaInteger,
    den: ArithmaInteger,
}

impl Rational {
    /// The exact value zero.
    pub fn zero() -> Self {
        Self {
            num: ArithmaInteger::zero(),
            den: ArithmaInteger::from_i64(1),
        }
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
        Ok(Self {
            num: num_r,
            den: den_r,
        })
    }

    /// Build the exact value `n`.
    pub fn from_i64(n: i64) -> Result<Self, String> {
        Self::new(ArithmaInteger::from_i64(n), ArithmaInteger::from_i64(1))
    }

    /// Build the exact value of a plain `ArithmaInteger`.
    pub fn from_integer(n: ArithmaInteger) -> Result<Self, String> {
        Self::new(n, ArithmaInteger::from_i64(1))
    }

    /// Build `num/den` from parts the caller has already reduced (positive
    /// `den`, no common factor). No check is made; only the bounded
    /// power-of-two rounding in `precise_interval` uses this.
    pub(crate) fn from_reduced_parts(num: ArithmaInteger, den: ArithmaInteger) -> Self {
        Self { num, den }
    }

    /// The numerator (carries the sign).
    pub fn numer(&self) -> &ArithmaInteger {
        &self.num
    }

    /// The denominator (always positive).
    pub fn denom(&self) -> &ArithmaInteger {
        &self.den
    }

    /// This value reduced by gcd.
    pub fn reduce(self) -> Result<Self, String> {
        Self::new(self.num, self.den)
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
        if self.is_negative() {
            self.neg()
        } else {
            Ok(self.clone())
        }
    }

    /// `-self`.
    pub fn neg(&self) -> Result<Self, String> {
        Ok(Self {
            num: negate(&self.num)?,
            den: self.den.clone(),
        })
    }

    /// `self + other`, *unreduced* (see [`Self::raw`]).
    pub fn add(&self, other: &Self) -> Result<Self, String> {
        let err = || "precise_interval: addition overflow".to_string();
        let a = self.num.checked_mul(&other.den).ok_or_else(err)?;
        let b = other.num.checked_mul(&self.den).ok_or_else(err)?;
        let n = a.checked_add(&b).ok_or_else(err)?;
        let d = self.den.checked_mul(&other.den).ok_or_else(err)?;
        Self::raw(n, d)
    }

    /// `self - other`, *unreduced*.
    pub fn sub(&self, other: &Self) -> Result<Self, String> {
        self.add(&other.neg()?)
    }

    /// `self * other`, *unreduced*.
    pub fn mul(&self, other: &Self) -> Result<Self, String> {
        let err = || "precise_interval: multiplication overflow".to_string();
        let n = self.num.checked_mul(&other.num).ok_or_else(err)?;
        let d = self.den.checked_mul(&other.den).ok_or_else(err)?;
        Self::raw(n, d)
    }

    /// `self / other`, *unreduced*. `Err` if `other` is zero.
    pub fn div(&self, other: &Self) -> Result<Self, String> {
        if other.is_zero() {
            return Err("precise_interval: division by zero".into());
        }
        let err = || "precise_interval: division overflow".to_string();
        let n = self.num.checked_mul(&other.den).ok_or_else(err)?;
        let d = self.den.checked_mul(&other.num).ok_or_else(err)?;
        Self::raw(n, d)
    }

    /// `self` raised to a non-negative integer power, *unreduced*.
    pub fn pow_u32(&self, exponent: u32) -> Result<Self, String> {
        let err = || "precise_interval: power overflow".to_string();
        let n = self.num.checked_pow(exponent).ok_or_else(err)?;
        let d = self.den.checked_pow(exponent).ok_or_else(err)?;
        Self::raw(n, d)
    }

    /// Build `num/den`, fixing the sign so `den > 0`, but *without*
    /// reducing by gcd. [`Self::new`] is the always-reduced public
    /// constructor; the arithmetic combinators above use this instead,
    /// because they run on essentially every expression-tree node (see
    /// [`combine_node`]) and reducing every one of them is the single
    /// largest cost in this module — general Euclidean reduction needs a
    /// full-width division per step. Skipping it here is sound because
    /// reduction is never required for *correctness* (every operation here
    /// is exact and sign-aware regardless of common factors); size is
    /// instead kept bounded by the explicit outward rounding
    /// [`combine_node`] and the bounded loops apply after every step (see
    /// [`ARITHMA_PRECISE_ROUNDING_GUARD`]), which *does* reduce (cheaply,
    /// against a known power of two: [`reduce_against_pow2`]).
    fn raw(num: ArithmaInteger, den: ArithmaInteger) -> Result<Self, String> {
        if den.is_zero() {
            return Err("precise_interval: rational with zero denominator".into());
        }
        if den.value.is_negative() {
            Ok(Self {
                num: negate(&num)?,
                den: negate(&den)?,
            })
        } else {
            Ok(Self { num, den })
        }
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

    /// The value rounded to the nearest-ish `f64` (`num / den` after each
    /// part is converted, so two roundings). `NaN` when either part is out of
    /// `f64` range; never silently clamped.
    pub fn to_f64(&self) -> f64 {
        self.num.to_f64() / self.den.to_f64()
    }
}

impl PartialEq for Rational {
    /// Exact equality via [`Rational::cmp`]; a comparison that overflows
    /// `ArithmaInteger` is not equal.
    fn eq(&self, other: &Self) -> bool {
        matches!(self.cmp(other), Ok(Ordering::Equal))
    }
}

/// `-v`, for a plain integer. `Err` only if `v` somehow is not plain, which
/// cannot happen for a value built by this module.
fn negate(v: &ArithmaInteger) -> Result<ArithmaInteger, String> {
    v.checked_neg()
        .ok_or_else(|| "precise_interval: negation failed".to_string())
}

/// `|v|`, for a plain integer.
fn abs_integer(v: &ArithmaInteger) -> Result<ArithmaInteger, String> {
    if v.value.is_negative() {
        negate(v)
    } else {
        Ok(v.clone())
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn int(n: i64) -> ArithmaInteger {
        ArithmaInteger::from_i64(n)
    }

    #[test]
    fn unreduced_and_reduced_forms_are_equal() {
        let sum = Rational::new(int(1), int(3))
            .and_then(|a| Rational::new(int(1), int(6)).and_then(|b| a.add(&b)));
        let half = Rational::new(int(1), int(2));
        assert_eq!(sum.ok(), half.ok());
    }

    #[test]
    fn serde_round_trip_reduces_and_refuses_zero_denominator() {
        let r = Rational::new(int(2), int(4)).ok();
        let json = serde_json::to_string(&r).ok();
        let back: Option<Rational> = json.and_then(|j| serde_json::from_str(&j).ok());
        assert_eq!(r, back);

        let unreduced = RationalParts {
            num: int(2),
            den: int(4),
        };
        let json = serde_json::to_string(&unreduced).unwrap_or_default();
        let reduced: Option<Rational> = serde_json::from_str(&json).ok();
        assert_eq!(reduced.map(|q| q.denom().to_i64()), Some(Some(2)));

        let zero_den = RationalParts {
            num: int(1),
            den: int(0),
        };
        let json = serde_json::to_string(&zero_den).unwrap_or_default();
        assert!(serde_json::from_str::<Rational>(&json).is_err());
    }
}
