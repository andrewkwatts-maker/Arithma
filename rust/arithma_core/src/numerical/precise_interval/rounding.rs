//====== Arithma/rust/arithma_core/src/numerical/precise_interval/rounding.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! Power-of-two grids and directed rounding for the exact rational path.

use super::interval::PreciseInterval;
use crate::integer::ArithmaInteger;
use crate::rational::Rational;

/// Extra bits of budget a Taylor series' running `term`/`sum` are rounded
/// to, above the precision the series is actually asked for. Without this,
/// repeatedly multiplying by a factor that itself has a large denominator
/// (e.g. a `pi` already computed to hundreds of bits) would multiply that
/// denominator's bit-length into the running term every iteration, blowing
/// past [`ArithmaInteger`]'s digit cap long before the series converges.
/// Rounding outward each step keeps every intermediate a bounded size, at
/// the cost of an extra, utterly negligible `iterations * 2^-budget` of
/// error alongside the already-conservative tail bound.
pub(super) const ARITHMA_PRECISE_ROUNDING_GUARD: u32 = 24;

/// `2^bits`, exact, by repeated squaring on `checked_mul` alone.
///
/// `ArithmaInteger::checked_pow` refuses any exponent past its own
/// `MAX_POW_EXPONENT` (1024) as a deliberate policy cap unrelated to
/// whether the result fits — and this module legitimately asks for larger
/// bit budgets (the rounding guard alone adds
/// [`ARITHMA_PRECISE_ROUNDING_GUARD`] on top of the caller's request).
/// Building the power this way is bounded only by the real limit,
/// `ArithmaInteger`'s digit cap, via `checked_mul`'s own `None`.
pub(super) fn pow2(bits: u32) -> Result<ArithmaInteger, String> {
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
pub(super) fn pow2_recip(bits: u32) -> Result<Rational, String> {
    Rational::new(ArithmaInteger::from_i64(1), pow2(bits)?)
}

/// The largest multiple of `1/2^bits` that does not exceed `r`: rounds
/// toward negative infinity, never past the true value.
pub(super) fn round_down(r: &Rational, bits: u32) -> Result<Rational, String> {
    let err = || "precise_interval: rounding overflow".to_string();
    let scale = pow2(bits)?;
    let scaled = r.numer().checked_mul(&scale).ok_or_else(err)?;
    let floored = scaled.checked_div_floor(r.denom()).ok_or_else(err)?;
    reduce_against_pow2(floored, bits)
}

/// `floored / 2^bits`, reduced. The denominator here is *known* to be a
/// power of two (every caller is [`round_down`]/[`round_up`]), so the
/// reduction is `2^v` where `v` is `floored`'s 2-adic valuation (capped at
/// `bits`): found by peeling off factors of two one at a time with cheap
/// `div_floor`/`rem_euclid` by the small constant 2.
///
/// This exists because rounding runs on every combined node (see
/// [`combine_node`](super::evaluate)) and every bounded-loop iteration, so it dominates how
/// often this module reduces a fraction. General Euclid
/// (`gcd_nonnegative`, kept for [`Rational::new`]'s ordinary arithmetic)
/// needs a full-width division per step and, for two ~200-bit operands, can
/// take on the order of a hundred such steps — measurably slow at this
/// call volume. Peeling factors of two is `O(bits)` cheap divisions
/// instead, with no general division at all.
fn reduce_against_pow2(numerator: ArithmaInteger, bits: u32) -> Result<Rational, String> {
    if numerator.is_zero() {
        return Ok(Rational::zero());
    }
    // `numerator`'s 2-adic valuation, capped at `bits`: found by scanning
    // its base-256, least-significant-byte-first magnitude for trailing
    // zero bits -- no division at all, let alone one per bit, is needed
    // just to *count* them.
    let mut zero_bits: u32 = 0;
    for &byte in numerator.value.value.iter() {
        if zero_bits >= bits {
            break;
        }
        if byte == 0 {
            zero_bits += 8;
            continue;
        }
        zero_bits += byte.trailing_zeros();
        break;
    }
    let shift = zero_bits.min(bits);
    let err = || "precise_interval: rounding reduction failed".to_string();
    let num = if shift == 0 {
        numerator
    } else {
        numerator.checked_div_exact(&pow2(shift)?).ok_or_else(err)?
    };
    Ok(Rational::from_reduced_parts(num, pow2(bits - shift)?))
}

/// The smallest multiple of `1/2^bits` that is not less than `r`: rounds
/// toward positive infinity, never past the true value. `ceil(r) =
/// -floor(-r)`.
pub(super) fn round_up(r: &Rational, bits: u32) -> Result<Rational, String> {
    round_down(&r.neg()?, bits)?.neg()
}

/// Round `iv`'s bounds outward (down/up) to a bounded `1/2^bits` grid,
/// never narrowing the enclosure.
pub(super) fn round_outward(iv: &PreciseInterval, bits: u32) -> Result<PreciseInterval, String> {
    Ok(PreciseInterval {
        lo: round_down(&iv.lo, bits)?,
        hi: round_up(&iv.hi, bits)?,
    })
}
