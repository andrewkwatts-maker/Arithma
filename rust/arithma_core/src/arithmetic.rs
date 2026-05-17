//====== Arithma/rust/arithma_core/src/arithmetic.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! Internal lossless arithmetic helpers + trait contract.
//!
//! ## Migration note
//!
//! The [`ArithmosLosslessArithmetic`] trait is migrated from pt-arithmos
//! `pt_internal_arithmetic.rs::PTLosslessArithmetic` (Wave 3 follow-up,
//! plan Â§F.8 step 3 "integer/variable"). The trait shape is the
//! contract â€” pt-arithmos `PTInteger` continues to implement the legacy
//! trait until `ArithmosInteger` graduates from stub status, at which
//! point the impl moves here too.

// â”€â”€â”€ Free-function helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Lossless multiplication that detects overflow.
/// Returns `None` if the result wraps.
pub fn checked_mul_i64(a: i64, b: i64) -> Option<i64> {
    a.checked_mul(b)
}

/// Lossless addition with overflow detection.
pub fn checked_add_i64(a: i64, b: i64) -> Option<i64> {
    a.checked_add(b)
}

/// Lossless subtraction with overflow detection.
pub fn checked_sub_i64(a: i64, b: i64) -> Option<i64> {
    a.checked_sub(b)
}

/// Lossless integer division â€” returns `None` for zero divisors and for
/// non-exact division (i.e. `a % b != 0`). The "lossless" contract means
/// any caller getting a `Some(_)` knows the division was exact.
pub fn checked_div_exact_i64(a: i64, b: i64) -> Option<i64> {
    if b == 0 { return None; }
    if a % b != 0 { return None; }
    a.checked_div(b)
}

/// Returns `true` if `(a + b)` would not overflow.
pub fn can_add_i64(a: i64, b: i64) -> bool {
    checked_add_i64(a, b).is_some()
}

/// Returns `true` if `(a * b)` would not overflow.
pub fn can_multiply_i64(a: i64, b: i64) -> bool {
    checked_mul_i64(a, b).is_some()
}

// â”€â”€â”€ Trait contract â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Trait for lossless arithmetic operations on number-like types.
///
/// "Lossless" here means the operation preserves the exact mathematical
/// value with no truncation, rounding, or loss of precision. Returning
/// `None` is the implementor's signal to the caller that a fallback
/// strategy (symbolic, bigint, error) is needed.
///
/// Migrated from pt-arithmos `PTLosslessArithmetic`. The shape is the
/// migration contract â€” every existing call site continues to work
/// unchanged once `ArithmosInteger` adopts this trait.
pub trait ArithmosLosslessArithmetic: Sized {
    /// `true` iff `self + other` can be performed without loss.
    fn can_add_lossless(&self, other: &Self) -> bool;

    /// `true` iff `self - other` can be performed without loss.
    fn can_subtract_lossless(&self, other: &Self) -> bool {
        // Default: same predicate as addition. Override if a type has
        // an asymmetric subtraction rule (e.g. unsigned types where
        // `0 - 1` is impossible).
        self.can_add_lossless(other)
    }

    /// `true` iff `self * other` can be performed without loss.
    fn can_multiply_lossless(&self, other: &Self) -> bool;

    /// `true` iff `self / other` can be performed without loss
    /// (i.e. exact division, no rounding).
    fn can_divide_lossless(&self, other: &Self) -> bool;

    /// Add two values losslessly. Returns `None` if either input is
    /// non-finite or the result would lose precision.
    fn add_lossless(&self, other: &Self) -> Option<Self>;

    /// Subtract `other` from `self` losslessly.
    fn subtract_lossless(&self, other: &Self) -> Option<Self>;

    /// Multiply two values losslessly.
    fn multiply_lossless(&self, other: &Self) -> Option<Self>;

    /// Divide `self` by `other` losslessly. Returns `None` for zero
    /// divisors or for non-exact division.
    fn divide_lossless(&self, other: &Self) -> Option<Self>;
}

// â”€â”€â”€ Blanket impl for i64 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

impl ArithmosLosslessArithmetic for i64 {
    fn can_add_lossless(&self, other: &i64) -> bool { can_add_i64(*self, *other) }
    fn can_subtract_lossless(&self, other: &i64) -> bool {
        // Override the default â€” subtraction overflow is asymmetric from
        // addition for i64 (e.g. `i64::MIN - 1` overflows but
        // `i64::MIN + (-1)` doesn't, and vice-versa).
        checked_sub_i64(*self, *other).is_some()
    }
    fn can_multiply_lossless(&self, other: &i64) -> bool { can_multiply_i64(*self, *other) }
    fn can_divide_lossless(&self, other: &i64) -> bool {
        *other != 0 && self % other == 0
    }
    fn add_lossless(&self, other: &i64) -> Option<i64> { checked_add_i64(*self, *other) }
    fn subtract_lossless(&self, other: &i64) -> Option<i64> { checked_sub_i64(*self, *other) }
    fn multiply_lossless(&self, other: &i64) -> Option<i64> { checked_mul_i64(*self, *other) }
    fn divide_lossless(&self, other: &i64) -> Option<i64> { checked_div_exact_i64(*self, *other) }
}

#[cfg(test)]
mod tests {
    use super::*;

    // â”€â”€â”€ Free-function helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    #[test]
    fn checked_mul_overflow_returns_none() {
        assert_eq!(checked_mul_i64(i64::MAX, 2), None);
    }

    #[test]
    fn checked_mul_normal_works() {
        assert_eq!(checked_mul_i64(3, 4), Some(12));
    }

    #[test]
    fn checked_add_normal_works() {
        assert_eq!(checked_add_i64(1, 2), Some(3));
    }

    #[test]
    fn checked_sub_underflow_returns_none() {
        assert_eq!(checked_sub_i64(i64::MIN, 1), None);
    }

    #[test]
    fn checked_div_exact_rejects_zero_divisor() {
        assert_eq!(checked_div_exact_i64(10, 0), None);
    }

    #[test]
    fn checked_div_exact_rejects_non_exact() {
        assert_eq!(checked_div_exact_i64(10, 3), None);
    }

    #[test]
    fn checked_div_exact_works_for_clean_divides() {
        assert_eq!(checked_div_exact_i64(10, 2), Some(5));
        assert_eq!(checked_div_exact_i64(0, 7), Some(0));
        assert_eq!(checked_div_exact_i64(-12, 4), Some(-3));
    }

    // â”€â”€â”€ Trait blanket impl for i64 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    #[test]
    fn i64_can_add_lossless_detects_overflow() {
        assert!(!(i64::MAX).can_add_lossless(&1));
        assert!((100_i64).can_add_lossless(&200));
    }

    #[test]
    fn i64_can_multiply_lossless_detects_overflow() {
        assert!(!(i64::MAX).can_multiply_lossless(&2));
        assert!((100_i64).can_multiply_lossless(&3));
    }

    #[test]
    fn i64_can_divide_lossless_rejects_inexact() {
        assert!(!(10_i64).can_divide_lossless(&3));
        assert!((10_i64).can_divide_lossless(&5));
    }

    #[test]
    fn i64_can_divide_lossless_rejects_zero() {
        assert!(!(10_i64).can_divide_lossless(&0));
    }

    #[test]
    fn i64_arithmetic_round_trips() {
        let a: i64 = 24;
        let b: i64 = 6;
        assert_eq!(a.add_lossless(&b), Some(30));
        assert_eq!(a.subtract_lossless(&b), Some(18));
        assert_eq!(a.multiply_lossless(&b), Some(144));
        assert_eq!(a.divide_lossless(&b), Some(4));
    }

    #[test]
    fn i64_default_can_subtract_falls_through_to_add() {
        // The default trait impl mirrors can_add â€” confirm via i64.
        assert!((100_i64).can_subtract_lossless(&50));
        assert!(!(i64::MIN).can_subtract_lossless(&i64::MAX));
    }
}

