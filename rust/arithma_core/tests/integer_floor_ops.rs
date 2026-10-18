//====== Arithma/rust/arithma_core/tests/integer_floor_ops.rs ======//
//! Acceptance tests for the Phase 0 `ArithmaInteger` additions needed by
//! `topos_grid::AxisIndex::Big`: floor division/remainder, integer
//! comparison, `to_i64` and the canonical signed-bytes form.
//!
//! Floor division here rounds the quotient toward negative infinity and the
//! remainder shares the *divisor's* sign (so `-7 / 2` floors to `-4`,
//! remainder `1`, and `7 / -2` floors to `-4`, remainder `-1`). The oracle
//! below computes that convention directly over `i128` rather than relying
//! on `i128::rem_euclid` (which always returns a non-negative remainder and
//! so disagrees with floor rounding when the divisor is negative).

use arithma_core::integer::ArithmaInteger;

fn i(n: i64) -> ArithmaInteger {
    ArithmaInteger::from_i64(n)
}

/// Floor-division oracle: `q` rounds toward negative infinity, `r` shares
/// `b`'s sign (or is zero), and `q * b + r == a`.
fn floor_divmod_i128(a: i128, b: i128) -> (i128, i128) {
    let q = a / b;
    let r = a % b;
    if r != 0 && (r < 0) != (b < 0) {
        (q - 1, r + b)
    } else {
        (q, r)
    }
}

#[test]
fn floor_div_and_rem_match_i128_over_a_grid() {
    for a in -1000i64..1000 {
        for d in 1i64..=17 {
            for &divisor in &[d, -d] {
                let (expected_q, expected_r) = floor_divmod_i128(a as i128, divisor as i128);
                let q = i(a)
                    .checked_div_floor(&i(divisor))
                    .expect("non-zero divisor must fold");
                let r = i(a)
                    .checked_rem_euclid(&i(divisor))
                    .expect("non-zero divisor must fold");
                assert_eq!(
                    q.to_i64(),
                    Some(expected_q as i64),
                    "div_floor({a}, {divisor})"
                );
                assert_eq!(
                    r.to_i64(),
                    Some(expected_r as i64),
                    "rem_euclid({a}, {divisor})"
                );
            }
        }
    }
}

#[test]
fn division_by_zero_is_refused() {
    assert!(i(7).checked_div_floor(&i(0)).is_none());
    assert!(i(7).checked_rem_euclid(&i(0)).is_none());
    assert!(i(0).checked_div_floor(&i(0)).is_none());
}

#[test]
fn floor_div_and_rem_beyond_two_pow_64() {
    // 2^65 = 36_893_488_147_419_103_232, built via checked_mul so the value
    // actually exceeds a u64/i64, and hand-computed against a divisor of 3.
    let two = i(2);
    let big = two.checked_pow(65).expect("2^65 must fold");
    // 36_893_488_147_419_103_232 / 3 = 12_297_829_382_473_034_410 remainder 2.
    let three = i(3);
    let q = big.checked_div_floor(&three).expect("big / 3 must fold");
    let r = big.checked_rem_euclid(&three).expect("big % 3 must fold");
    assert_eq!(q.to_f64(), 12_297_829_382_473_034_410.0);
    assert_eq!(r.to_f64(), 2.0);

    // Negate it by hand (big is beyond i64 range, so negation must go
    // through the raw representation rather than `checked_neg`'s i64 path).
    let mut neg_big = big.clone();
    neg_big.value.set_negative(true);
    // -36_893_488_147_419_103_232 / 3 floors to -12_297_829_382_473_034_411,
    // remainder 1 (same sign as the positive divisor).
    let neg_q = neg_big
        .checked_div_floor(&three)
        .expect("-big / 3 must fold");
    let neg_r = neg_big
        .checked_rem_euclid(&three)
        .expect("-big % 3 must fold");
    assert_eq!(neg_q.to_f64(), -12_297_829_382_473_034_411.0);
    assert_eq!(neg_r.to_f64(), 1.0);
}

#[test]
fn to_i64_round_trips_at_the_boundaries() {
    assert_eq!(i(i64::MAX).to_i64(), Some(i64::MAX));
    assert_eq!(i(i64::MIN).to_i64(), Some(i64::MIN));
    assert_eq!(i(0).to_i64(), Some(0));
    assert_eq!(i(-1).to_i64(), Some(-1));
}

#[test]
fn to_i64_refuses_values_that_do_not_fit() {
    let two = i(2);
    // 2^63 is one past i64::MAX.
    let just_over = two.checked_pow(63).expect("2^63 must fold");
    assert_eq!(just_over.to_i64(), None);

    // -(2^63 + 1) is one past i64::MIN.
    let mut too_negative = two.checked_pow(63).expect("2^63 must fold");
    too_negative.value.set_negative(true);
    let too_negative = too_negative
        .checked_sub(&i(1))
        .expect("-2^63 - 1 must fold");
    assert_eq!(too_negative.to_i64(), None);
}

#[test]
fn cmp_integer_orders_signed_values() {
    use core::cmp::Ordering;
    assert_eq!(i(3).cmp_integer(&i(5)), Some(Ordering::Less));
    assert_eq!(i(5).cmp_integer(&i(3)), Some(Ordering::Greater));
    assert_eq!(i(-5).cmp_integer(&i(-5)), Some(Ordering::Equal));
    assert_eq!(i(-5).cmp_integer(&i(3)), Some(Ordering::Less));
    assert_eq!(i(0).cmp_integer(&i(0)), Some(Ordering::Equal));
    assert_eq!(i(0).cmp_integer(&i(-1)), Some(Ordering::Greater));

    let big = i(2).checked_pow(65).expect("2^65 must fold");
    let bigger = i(2).checked_pow(66).expect("2^66 must fold");
    assert_eq!(big.cmp_integer(&bigger), Some(Ordering::Less));

    assert_eq!(ArithmaInteger::nan().cmp_integer(&i(1)), None);
    assert_eq!(ArithmaInteger::infinity().cmp_integer(&i(1)), None);
}

#[test]
fn to_le_bytes_signed_is_canonical() {
    assert_eq!(i(0).to_le_bytes_signed(), Some((false, Vec::new())));
    assert_eq!(i(1).to_le_bytes_signed(), Some((false, vec![1])));
    assert_eq!(i(-1).to_le_bytes_signed(), Some((true, vec![1])));
    assert_eq!(i(256).to_le_bytes_signed(), Some((false, vec![0, 1])));

    // No trailing zero byte, even if the internal representation carries one.
    let big = i(2).checked_pow(64).expect("2^64 must fold");
    let (negative, bytes) = big.to_le_bytes_signed().expect("2^64 must give bytes");
    assert!(!negative);
    assert_eq!(*bytes.last().expect("non-empty"), 1, "no trailing zero byte");

    assert_eq!(ArithmaInteger::nan().to_le_bytes_signed(), None);
    assert_eq!(ArithmaInteger::infinity().to_le_bytes_signed(), None);
}
