//====== Arithma/rust/arithma_core/src/integer.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Integer
//!
//! `ArithmaInteger` — exact, unlimited-precision integer used in place of f64
//! and i64 throughout the AST. Mirrors `pt_arithmos::PTInteger` and inherits
//! its split design:
//!
//! - [`ArithmaInternalInteger`] holds the bytes-and-flags representation.
//! - [`ArithmaInteger`] adds an optional base SI unit.
//!
//! Special values (NaN, infinity, imaginary, infinitesimal, exact-rational)
//! are tracked via flag bits inside the internal representation so the same
//! type can express both ordinary integers and the engine's full numeric tower.

use serde::{Deserialize, Serialize};

/// Special-value flag bits for [`ArithmaInternalInteger`]. Public-but-stable so
/// downstream backends can route based on flag inspection without exposing the
/// internal byte layout.
pub mod flag {
    pub const NEGATIVE: u16 = 0x0001;
    pub const INFINITY: u16 = 0x0004;
    pub const NAN: u16 = 0x0008;
    pub const IMAGINARY: u16 = 0x0010;
    pub const INFINITESIMAL: u16 = 0x0020;
    pub const RATIONAL: u16 = 0x0080;
}

/// Tunable behaviour for `ArithmaInteger`. Mirrors `PTIntegerConfig`.
#[derive(Debug, Clone)]
pub struct ArithmaIntegerConfig {
    /// Try to extract symbolic constants from f64 inputs when possible.
    pub extract_constants_from_floats: bool,
}

impl Default for ArithmaIntegerConfig {
    fn default() -> Self {
        Self {
            extract_constants_from_floats: true,
        }
    }
}

/// Unlimited-precision integer with bit flags for special values.
///
/// Bytes are stored little-endian (least-significant byte first). The vector
/// is never empty — `[0]` represents zero and is the canonical empty form.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ArithmaInternalInteger {
    /// Special-value flags. See [`flag`].
    pub flags: u16,
    /// Base-256 little-endian digits.
    pub value: Vec<u8>,
}

impl ArithmaInternalInteger {
    /// Construct a fresh zero. Flags are cleared.
    pub fn new() -> Self {
        Self {
            flags: 0,
            value: vec![0],
        }
    }

    /// Construct from a u64.
    ///
    /// Stores the magnitude as little-endian base-256 bytes. Zero is canonicalised
    /// to a single `0` byte so the value vector is never empty.
    pub fn from_u64(value: u64) -> Self {
        if value == 0 {
            return Self::new();
        }
        let mut bytes: Vec<u8> = Vec::with_capacity(8);
        let mut v = value;
        let mut guard: usize = 0;
        // Bounded loop: u64 has at most 8 bytes; cap at 9 for safety.
        while v > 0 && guard < 9 {
            bytes.push((v & 0xFF) as u8);
            v >>= 8;
            guard += 1;
        }
        debug_assert!(!bytes.is_empty(), "from_u64 produced empty bytes");
        debug_assert!(guard <= 9, "from_u64 exceeded byte cap");
        Self {
            flags: 0,
            value: bytes,
        }
    }

    /// Construct from an i64. Sign goes into the [`flag::NEGATIVE`] bit.
    pub fn from_i64(value: i64) -> Self {
        let (magnitude, negative) = if value < 0 {
            // Use unsigned absolute value to handle i64::MIN safely.
            ((value as i128).unsigned_abs() as u64, true)
        } else {
            (value as u64, false)
        };
        let mut out = Self::from_u64(magnitude);
        if negative {
            out.flags |= flag::NEGATIVE;
        }
        debug_assert!(out.is_negative() == negative, "sign flag inconsistent");
        debug_assert!(!out.value.is_empty(), "from_i64 produced empty bytes");
        out
    }

    /// Convert the little-endian magnitude to an f64.
    ///
    /// Special-value flags (NaN, infinity) take precedence over the raw bytes
    /// so the conversion is consistent with [`ArithmaInteger::to_f64`].
    pub fn to_f64(&self) -> f64 {
        if self.is_nan() {
            return f64::NAN;
        }
        if self.is_infinity() {
            return if self.is_negative() {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
        }
        // Horner from the most significant non-zero byte downwards.
        //
        // This used to read a capped PREFIX -- the low 32 bytes -- which
        // silently returns 0.0 for any value whose low 32 bytes are zero.
        // 2^360 is exactly that shape (one set bit, 45 bytes up), and it is
        // the denominator of every small float built by `from_f64`, so
        // 1.9e-93 evaluated as a division by zero. The cap was justified as
        // keeping the loop finite, but `self.value.len()` is already finite;
        // it only bounded correctness.
        //
        // Accumulating downwards also removes the overflow hazard in the old
        // ascending `scale *= 256.0`, which reaches infinity after about 128
        // bytes and then produces NaN on the next zero byte. Here a genuinely
        // out-of-range magnitude saturates to infinity, which is the honest
        // answer for a value f64 cannot hold.
        let top = match self.value.iter().rposition(|&byte| byte != 0) {
            Some(index) => index,
            None => return 0.0,
        };
        let mut acc: f64 = 0.0;
        for index in (0..=top).rev() {
            acc = acc * 256.0 + f64::from(self.value[index]);
            if acc.is_infinite() {
                break;
            }
        }
        debug_assert!(!acc.is_nan(), "to_f64 produced NaN from finite bytes");
        if self.is_negative() {
            -acc
        } else {
            acc
        }
    }

    /// Set the negative flag.
    pub fn set_negative(&mut self, neg: bool) {
        if neg {
            self.flags |= flag::NEGATIVE;
        } else {
            self.flags &= !flag::NEGATIVE;
        }
    }

    /// Read the negative flag.
    pub fn is_negative(&self) -> bool {
        self.flags & flag::NEGATIVE != 0
    }

    /// Read the rational flag.
    pub fn is_rational(&self) -> bool {
        self.flags & flag::RATIONAL != 0
    }

    /// Read the infinity flag.
    pub fn is_infinity(&self) -> bool {
        self.flags & flag::INFINITY != 0
    }

    /// Read the NaN flag.
    pub fn is_nan(&self) -> bool {
        self.flags & flag::NAN != 0
    }
}

/// Upper bound on base-256 digits in any computed result.
///
/// Safety-critical standard 2 (all loops have fixed bounds) and standard 3 (no
/// unbounded allocation after initialisation): every arithmetic routine below
/// refuses rather than grows past this. 512 bytes is a 4096-bit integer, far
/// beyond anything symbolic simplification produces, so hitting the cap means
/// something has gone wrong and `None` is the honest answer.
pub const MAX_DIGITS: usize = 512;

/// Largest exponent [`ArithmaInternalInteger::checked_pow`] will evaluate.
///
/// Chosen so even a two-digit base stays well inside [`MAX_DIGITS`]; larger
/// powers are left symbolic rather than attempted.
pub const MAX_POW_EXPONENT: u32 = 1024;

// ---------------------------------------------------------------------------
// Magnitude helpers. These operate on bare little-endian base-256 slices and
// know nothing about sign or flags -- sign handling lives in the signed
// wrappers below, which is what keeps each routine inside one printed page.
// ---------------------------------------------------------------------------

/// Strip redundant leading zeros (high-order, i.e. the tail of the LE vector).
///
/// The representation guarantees a non-empty vector, so zero canonicalises to
/// `[0]` rather than the empty slice.
fn mag_trim(v: &mut Vec<u8>) {
    debug_assert!(!v.is_empty(), "magnitude vector must never be empty");
    let mut guard: usize = 0;
    while v.len() > 1 && v[v.len() - 1] == 0 && guard <= MAX_DIGITS {
        v.pop();
        guard += 1;
    }
    debug_assert!(!v.is_empty(), "trim must leave at least one digit");
}

/// Compare two magnitudes. Neither slice's sign is considered.
fn mag_cmp(a: &[u8], b: &[u8]) -> core::cmp::Ordering {
    use core::cmp::Ordering;
    debug_assert!(!a.is_empty(), "empty left magnitude");
    debug_assert!(!b.is_empty(), "empty right magnitude");
    // Trailing zeros are possible on un-trimmed input, so compare by
    // significant length rather than raw length.
    let sig = |x: &[u8]| -> usize {
        let mut n = x.len();
        while n > 1 && x[n - 1] == 0 {
            n -= 1;
        }
        n
    };
    let (na, nb) = (sig(a), sig(b));
    if na != nb {
        return na.cmp(&nb);
    }
    let mut i = na;
    while i > 0 {
        i -= 1;
        match a[i].cmp(&b[i]) {
            Ordering::Equal => {}
            other => return other,
        }
    }
    Ordering::Equal
}

/// `a + b`. Returns `None` if the result would exceed [`MAX_DIGITS`].
fn mag_add(a: &[u8], b: &[u8]) -> Option<Vec<u8>> {
    debug_assert!(!a.is_empty(), "empty left magnitude");
    debug_assert!(!b.is_empty(), "empty right magnitude");
    let n = a.len().max(b.len());
    if n >= MAX_DIGITS {
        return None;
    }
    let mut out = Vec::with_capacity(n + 1);
    let mut carry: u16 = 0;
    for i in 0..n {
        let lhs = u16::from(*a.get(i).unwrap_or(&0));
        let rhs = u16::from(*b.get(i).unwrap_or(&0));
        let sum = lhs + rhs + carry;
        out.push((sum & 0xFF) as u8);
        carry = sum >> 8;
    }
    if carry > 0 {
        out.push(carry as u8);
    }
    debug_assert!(out.len() <= MAX_DIGITS, "mag_add exceeded the digit cap");
    debug_assert!(!out.is_empty(), "mag_add produced no digits");
    mag_trim(&mut out);
    Some(out)
}

/// `a - b`, requiring `a >= b`. Returns `None` if that precondition is broken.
fn mag_sub(a: &[u8], b: &[u8]) -> Option<Vec<u8>> {
    debug_assert!(!a.is_empty(), "empty left magnitude");
    debug_assert!(!b.is_empty(), "empty right magnitude");
    if mag_cmp(a, b) == core::cmp::Ordering::Less {
        return None;
    }
    let mut out = Vec::with_capacity(a.len());
    let mut borrow: i16 = 0;
    for (i, &digit) in a.iter().enumerate() {
        let lhs = i16::from(digit);
        let rhs = i16::from(*b.get(i).unwrap_or(&0));
        let mut diff = lhs - rhs - borrow;
        if diff < 0 {
            diff += 256;
            borrow = 1;
        } else {
            borrow = 0;
        }
        out.push(diff as u8);
    }
    debug_assert_eq!(borrow, 0, "mag_sub borrowed out of the top digit");
    debug_assert!(!out.is_empty(), "mag_sub produced no digits");
    mag_trim(&mut out);
    Some(out)
}

/// `a * b` by long multiplication. `None` if the product exceeds [`MAX_DIGITS`].
fn mag_mul(a: &[u8], b: &[u8]) -> Option<Vec<u8>> {
    debug_assert!(!a.is_empty(), "empty left magnitude");
    debug_assert!(!b.is_empty(), "empty right magnitude");
    if a.len() + b.len() > MAX_DIGITS {
        return None;
    }
    let mut out = vec![0u8; a.len() + b.len()];
    for (i, &lhs) in a.iter().enumerate() {
        if lhs == 0 {
            continue;
        }
        let mut carry: u32 = 0;
        for (j, &rhs) in b.iter().enumerate() {
            let idx = i + j;
            let acc = u32::from(out[idx]) + u32::from(lhs) * u32::from(rhs) + carry;
            out[idx] = (acc & 0xFF) as u8;
            carry = acc >> 8;
        }
        // Propagate the tail carry. Bounded by the pre-sized output.
        let mut k = i + b.len();
        while carry > 0 && k < out.len() {
            let acc = u32::from(out[k]) + carry;
            out[k] = (acc & 0xFF) as u8;
            carry = acc >> 8;
            k += 1;
        }
        debug_assert_eq!(carry, 0, "mag_mul carry escaped the pre-sized buffer");
    }
    debug_assert!(!out.is_empty(), "mag_mul produced no digits");
    mag_trim(&mut out);
    Some(out)
}

/// Exact `a / b`. `None` when `b` is zero **or the division is not exact** --
/// an inexact quotient has no representation here, and rounding it would make
/// the simplifier silently lossy.
fn mag_div_exact(a: &[u8], b: &[u8]) -> Option<Vec<u8>> {
    debug_assert!(!a.is_empty(), "empty dividend magnitude");
    debug_assert!(!b.is_empty(), "empty divisor magnitude");
    if b.iter().all(|&d| d == 0) {
        return None;
    }
    // Schoolbook long division, most-significant digit first.
    let mut quotient = vec![0u8; a.len()];
    let mut remainder: Vec<u8> = vec![0];
    let mut i = a.len();
    while i > 0 {
        i -= 1;
        // remainder = remainder * 256 + a[i]
        remainder.insert(0, a[i]);
        mag_trim(&mut remainder);
        // At most 255 subtractions, so the loop is bounded by construction.
        let mut digit: u16 = 0;
        while digit < 256 && mag_cmp(&remainder, b) != core::cmp::Ordering::Less {
            remainder = mag_sub(&remainder, b)?;
            digit += 1;
        }
        debug_assert!(digit < 256, "quotient digit overflowed a byte");
        quotient[i] = digit as u8;
    }
    if !remainder.iter().all(|&d| d == 0) {
        return None; // inexact
    }
    mag_trim(&mut quotient);
    debug_assert!(!quotient.is_empty(), "division produced no digits");
    Some(quotient)
}

impl ArithmaInternalInteger {
    /// True when the value is exactly zero, ignoring the sign bit.
    ///
    /// Negative zero is reachable through [`set_negative`](Self::set_negative),
    /// so this deliberately ignores the flag rather than trusting it.
    pub fn is_zero(&self) -> bool {
        debug_assert!(!self.value.is_empty(), "empty magnitude");
        debug_assert!(self.value.len() <= MAX_DIGITS, "magnitude past the cap");
        !self.is_nan() && !self.is_infinity() && self.value.iter().all(|&d| d == 0)
    }

    /// True when the value is exactly `+1`.
    pub fn is_one(&self) -> bool {
        debug_assert!(!self.value.is_empty(), "empty magnitude");
        if self.is_negative() || self.is_nan() || self.is_infinity() {
            return false;
        }
        debug_assert!(!self.value.is_empty(), "magnitude vanished after guards");
        self.value[0] == 1 && self.value[1..].iter().all(|&d| d == 0)
    }

    /// True when this is an ordinary finite integer -- no special flags other
    /// than the sign bit.
    ///
    /// Arithmetic below refuses anything else rather than guessing: folding
    /// `NaN + 1`, or treating a rational as an integer, produces a wrong answer
    /// that looks right.
    pub fn is_plain_integer(&self) -> bool {
        debug_assert!(!self.value.is_empty(), "empty magnitude");
        debug_assert!(self.value.len() <= MAX_DIGITS, "magnitude past the cap");
        self.flags & !flag::NEGATIVE == 0
    }

    /// Build from a sign and magnitude, canonicalising negative zero to `+0`.
    fn from_parts(magnitude: Vec<u8>, negative: bool) -> Self {
        debug_assert!(!magnitude.is_empty(), "empty magnitude");
        debug_assert!(magnitude.len() <= MAX_DIGITS, "magnitude past the cap");
        let is_zero = magnitude.iter().all(|&d| d == 0);
        let mut out = Self {
            flags: 0,
            value: magnitude,
        };
        if negative && !is_zero {
            out.flags |= flag::NEGATIVE;
        }
        out
    }

    /// Exact `self + other`.
    ///
    /// Returns `None` if either operand carries a special flag (NaN, infinity,
    /// rational, imaginary, infinitesimal) or the result exceeds
    /// [`MAX_DIGITS`]. Callers read `None` as "leave it symbolic".
    pub fn checked_add(&self, other: &Self) -> Option<Self> {
        if !self.is_plain_integer() || !other.is_plain_integer() {
            return None;
        }
        debug_assert!(!self.value.is_empty(), "empty left magnitude");
        debug_assert!(!other.value.is_empty(), "empty right magnitude");
        let out = if self.is_negative() == other.is_negative() {
            Self::from_parts(mag_add(&self.value, &other.value)?, self.is_negative())
        } else {
            // Opposite signs: subtract the smaller magnitude from the larger
            // and take the larger operand's sign.
            match mag_cmp(&self.value, &other.value) {
                core::cmp::Ordering::Less => {
                    Self::from_parts(mag_sub(&other.value, &self.value)?, other.is_negative())
                }
                _ => Self::from_parts(mag_sub(&self.value, &other.value)?, self.is_negative()),
            }
        };
        debug_assert!(out.is_plain_integer(), "sum acquired a special flag");
        Some(out)
    }

    /// Exact `self - other`. Same refusal rules as [`checked_add`](Self::checked_add).
    pub fn checked_sub(&self, other: &Self) -> Option<Self> {
        if !self.is_plain_integer() || !other.is_plain_integer() {
            return None;
        }
        debug_assert!(!other.value.is_empty(), "empty right magnitude");
        let mut negated = other.clone();
        negated.set_negative(!other.is_negative());
        self.checked_add(&negated)
    }

    /// Exact `self * other`. Same refusal rules as [`checked_add`](Self::checked_add).
    pub fn checked_mul(&self, other: &Self) -> Option<Self> {
        if !self.is_plain_integer() || !other.is_plain_integer() {
            return None;
        }
        debug_assert!(!self.value.is_empty(), "empty left magnitude");
        debug_assert!(!other.value.is_empty(), "empty right magnitude");
        let magnitude = mag_mul(&self.value, &other.value)?;
        let out = Self::from_parts(magnitude, self.is_negative() != other.is_negative());
        debug_assert!(out.is_plain_integer(), "product acquired a special flag");
        Some(out)
    }

    /// Exact `self / other`.
    ///
    /// `None` when the divisor is zero or the quotient is not exact. Inexact
    /// division stays symbolic so no precision is lost silently.
    pub fn checked_div_exact(&self, other: &Self) -> Option<Self> {
        if !self.is_plain_integer() || !other.is_plain_integer() || other.is_zero() {
            return None;
        }
        debug_assert!(!other.is_zero(), "divisor must be non-zero here");
        debug_assert!(!self.value.is_empty(), "empty dividend magnitude");
        let magnitude = mag_div_exact(&self.value, &other.value)?;
        let out = Self::from_parts(magnitude, self.is_negative() != other.is_negative());
        debug_assert!(out.is_plain_integer(), "quotient acquired a special flag");
        Some(out)
    }

    /// Exact `self ^ exponent` for a small non-negative exponent.
    ///
    /// The exponent is capped at [`MAX_POW_EXPONENT`] and the result at
    /// [`MAX_DIGITS`], so a crafted input such as `2 ^ 10000000` cannot be used
    /// to exhaust memory.
    pub fn checked_pow(&self, exponent: u32) -> Option<Self> {
        if !self.is_plain_integer() || exponent > MAX_POW_EXPONENT {
            return None;
        }
        debug_assert!(exponent <= MAX_POW_EXPONENT, "exponent cap not enforced");
        debug_assert!(!self.value.is_empty(), "empty base magnitude");
        let mut acc = Self::from_u64(1);
        for _ in 0..exponent {
            acc = acc.checked_mul(self)?;
        }
        debug_assert!(acc.is_plain_integer(), "power acquired a special flag");
        Some(acc)
    }

    /// Read the value as a `u32`, for use as an exponent. `None` if it is
    /// negative, flagged, or too large.
    pub fn to_u32(&self) -> Option<u32> {
        if !self.is_plain_integer() || self.is_negative() {
            return None;
        }
        debug_assert!(!self.value.is_empty(), "empty magnitude");
        // Anything above the low 4 bytes cannot fit a u32.
        if self.value.len() > 4 && self.value[4..].iter().any(|&d| d != 0) {
            return None;
        }
        let mut acc: u32 = 0;
        for i in 0..self.value.len().min(4) {
            acc |= u32::from(self.value[i]) << (8 * i);
        }
        debug_assert!(self.value.len() <= MAX_DIGITS, "magnitude past the cap");
        Some(acc)
    }
}

impl ArithmaInteger {
    /// True when the value is exactly zero.
    pub fn is_zero(&self) -> bool {
        self.value.is_zero()
    }

    /// True when the value is exactly `+1`.
    pub fn is_one(&self) -> bool {
        self.value.is_one()
    }

    /// Unit-aware exact addition.
    ///
    /// Refuses to combine operands with differing units -- adding metres to
    /// seconds is a modelling error, and returning a unitless number would hide
    /// it. Matching units (including both `None`) propagate.
    pub fn checked_add(&self, other: &Self) -> Option<Self> {
        let unit = combine_units(self, other)?;
        let value = self.value.checked_add(&other.value)?;
        debug_assert!(value.is_plain_integer(), "sum is not a plain integer");
        debug_assert!(!value.value.is_empty(), "sum has no digits");
        Some(Self { value, unit })
    }

    /// Unit-aware exact subtraction. See [`checked_add`](Self::checked_add).
    pub fn checked_sub(&self, other: &Self) -> Option<Self> {
        let unit = combine_units(self, other)?;
        let value = self.value.checked_sub(&other.value)?;
        debug_assert!(
            value.is_plain_integer(),
            "difference is not a plain integer"
        );
        debug_assert!(!value.value.is_empty(), "difference has no digits");
        Some(Self { value, unit })
    }

    /// Exact multiplication.
    ///
    /// Unitless operands only: `m * m` is m^2, which this type cannot express,
    /// so the fold is refused rather than mislabelled.
    pub fn checked_mul(&self, other: &Self) -> Option<Self> {
        if self.unit.is_some() || other.unit.is_some() {
            return None;
        }
        debug_assert!(self.unit.is_none(), "unit slipped past the guard");
        let value = self.value.checked_mul(&other.value)?;
        debug_assert!(value.is_plain_integer(), "product is not a plain integer");
        Some(Self { value, unit: None })
    }

    /// Exact division. Unitless only, and only when the quotient is exact.
    pub fn checked_div_exact(&self, other: &Self) -> Option<Self> {
        if self.unit.is_some() || other.unit.is_some() {
            return None;
        }
        debug_assert!(other.unit.is_none(), "unit slipped past the guard");
        let value = self.value.checked_div_exact(&other.value)?;
        debug_assert!(value.is_plain_integer(), "quotient is not a plain integer");
        Some(Self { value, unit: None })
    }

    /// Exact exponentiation by a small non-negative exponent. Unitless only.
    pub fn checked_pow(&self, exponent: u32) -> Option<Self> {
        if self.unit.is_some() {
            return None;
        }
        debug_assert!(self.unit.is_none(), "unit slipped past the guard");
        let value = self.value.checked_pow(exponent)?;
        debug_assert!(value.is_plain_integer(), "power is not a plain integer");
        Some(Self { value, unit: None })
    }

    /// Negation. Keeps the unit; never fails for a plain integer.
    pub fn checked_neg(&self) -> Option<Self> {
        if !self.value.is_plain_integer() {
            return None;
        }
        debug_assert!(!self.value.value.is_empty(), "empty magnitude");
        let mut value = self.value.clone();
        let negative = !value.is_negative();
        value.set_negative(negative && !value.is_zero());
        debug_assert!(value.is_plain_integer(), "negation acquired a special flag");
        Some(Self {
            value,
            unit: self.unit.clone(),
        })
    }

    /// Read as a `u32` exponent, if it is unitless, non-negative and fits.
    pub fn to_u32(&self) -> Option<u32> {
        if self.unit.is_some() {
            return None;
        }
        self.value.to_u32()
    }
}

/// Unit for the result of an additive operation, or `None` if incompatible.
///
/// The nested `Option` is deliberate: the outer layer is "can these combine",
/// the inner is "does the result carry a unit".
fn combine_units(a: &ArithmaInteger, b: &ArithmaInteger) -> Option<Option<String>> {
    match (&a.unit, &b.unit) {
        (None, None) => Some(None),
        (Some(x), Some(y)) if x == y => Some(Some(x.clone())),
        _ => None,
    }
}

impl Default for ArithmaInternalInteger {
    fn default() -> Self {
        Self::new()
    }
}

/// Public-facing integer with optional base-SI-unit attribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArithmaInteger {
    /// Underlying integer value.
    pub value: ArithmaInternalInteger,
    /// Optional base SI unit (only the seven base units: m, kg, s, A, K, mol, cd).
    pub unit: Option<String>,
}

impl ArithmaInteger {
    /// Construct zero.
    pub fn zero() -> Self {
        Self {
            value: ArithmaInternalInteger::new(),
            unit: None,
        }
    }

    /// Construct from u64.
    pub fn from_u64(value: u64) -> Self {
        Self {
            value: ArithmaInternalInteger::from_u64(value),
            unit: None,
        }
    }

    /// Construct from i64.
    pub fn from_i64(value: i64) -> Self {
        Self {
            value: ArithmaInternalInteger::from_i64(value),
            unit: None,
        }
    }

    /// IEEE infinity sentinel.
    pub fn infinity() -> Self {
        let mut v = ArithmaInternalInteger::new();
        v.flags |= flag::INFINITY;
        Self {
            value: v,
            unit: None,
        }
    }

    /// IEEE NaN sentinel.
    pub fn nan() -> Self {
        let mut v = ArithmaInternalInteger::new();
        v.flags |= flag::NAN;
        Self {
            value: v,
            unit: None,
        }
    }

    /// Convert to f64 by delegating to the internal representation.
    ///
    /// Unit attribution is ignored — this is a numeric-only conversion.
    pub fn to_f64(&self) -> f64 {
        self.value.to_f64()
    }
}

impl Default for ArithmaInteger {
    fn default() -> Self {
        Self::zero()
    }
}

// ---------------------------------------------------------------------------
// Backward-compatibility aliases for the pre-rename `Arithmos*` names.
// Retained for one release; downstream (eml-math, eml-spectral, metaphysica,
// periodica) should migrate to the `Arithma*` names above.
// ---------------------------------------------------------------------------
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaInteger`")]
#[allow(unused)]
pub use self::ArithmaInteger as ArithmosInteger;
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaIntegerConfig`")]
#[allow(unused)]
pub use self::ArithmaIntegerConfig as ArithmosIntegerConfig;
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaInternalInteger`")]
#[allow(unused)]
pub use self::ArithmaInternalInteger as ArithmosInternalInteger;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_has_no_flags() {
        let z = ArithmaInteger::zero();
        assert_eq!(z.value.flags, 0);
    }

    #[test]
    fn infinity_sets_flag() {
        let i = ArithmaInteger::infinity();
        assert!(i.value.is_infinity());
    }

    #[test]
    fn nan_sets_flag() {
        let n = ArithmaInteger::nan();
        assert!(n.value.is_nan());
    }

    #[test]
    fn negative_flag_round_trip() {
        let mut v = ArithmaInternalInteger::new();
        assert!(!v.is_negative());
        v.set_negative(true);
        assert!(v.is_negative());
        v.set_negative(false);
        assert!(!v.is_negative());
    }

    #[test]
    fn from_u64_round_trips_through_f64() {
        let one = ArithmaInternalInteger::from_u64(1);
        assert_eq!(one.to_f64(), 1.0);
        let big = ArithmaInternalInteger::from_u64(1_000_000);
        assert_eq!(big.to_f64(), 1_000_000.0);
        let zero = ArithmaInternalInteger::from_u64(0);
        assert_eq!(zero.to_f64(), 0.0);
    }

    #[test]
    fn from_i64_handles_sign() {
        let pos = ArithmaInternalInteger::from_i64(42);
        assert_eq!(pos.to_f64(), 42.0);
        let neg = ArithmaInternalInteger::from_i64(-42);
        assert!(neg.is_negative());
        assert_eq!(neg.to_f64(), -42.0);
    }

    #[test]
    fn arithma_integer_to_f64_handles_specials() {
        assert!(ArithmaInteger::nan().to_f64().is_nan());
        assert_eq!(ArithmaInteger::infinity().to_f64(), f64::INFINITY);
        assert_eq!(ArithmaInteger::from_i64(-7).to_f64(), -7.0);
    }
    // ─── exact arithmetic ──────────────────────────────────────────────────
    //
    // `ArithmaInteger` had no arithmetic at all before this; the AST could
    // hold unlimited-precision integers but never combine two of them. These
    // pin the representation-level behaviour that constant folding relies on.

    fn i(n: i64) -> ArithmaInteger {
        ArithmaInteger::from_i64(n)
    }

    #[test]
    fn addition_crosses_the_byte_boundary() {
        // 255 + 1 must carry into a second base-256 digit.
        let sum = i(255).checked_add(&i(1)).expect("255 + 1 must fold");
        assert_eq!(sum.to_f64(), 256.0);
        assert_eq!(sum.value.value.len(), 2, "carry did not extend the digits");
    }

    #[test]
    fn addition_handles_mixed_signs_both_ways() {
        assert_eq!(i(5).checked_add(&i(-3)).map(|n| n.to_f64()), Some(2.0));
        assert_eq!(i(3).checked_add(&i(-5)).map(|n| n.to_f64()), Some(-2.0));
        assert_eq!(i(-3).checked_add(&i(-5)).map(|n| n.to_f64()), Some(-8.0));
    }

    #[test]
    fn a_value_plus_its_negation_is_positive_zero() {
        // Negative zero would compare unequal to zero in the simplifier's
        // identity rules, so the canonicalisation matters.
        let zero = i(42).checked_add(&i(-42)).expect("must fold");
        assert!(zero.is_zero(), "42 + -42 must be zero");
        assert!(
            !zero.value.is_negative(),
            "zero must not carry the negative flag"
        );
        assert_eq!(zero.to_f64(), 0.0);
    }

    #[test]
    fn subtraction_is_addition_of_the_negation() {
        assert_eq!(i(10).checked_sub(&i(4)).map(|n| n.to_f64()), Some(6.0));
        assert_eq!(i(4).checked_sub(&i(10)).map(|n| n.to_f64()), Some(-6.0));
        assert_eq!(i(-4).checked_sub(&i(-10)).map(|n| n.to_f64()), Some(6.0));
    }

    #[test]
    fn multiplication_exceeds_what_an_i64_could_hold() {
        // 2^64 is the whole point of an unlimited-precision type: it must be
        // exact, not saturated and not wrapped.
        let two = i(2);
        let big = two.checked_pow(64).expect("2^64 must fold");
        assert_eq!(big.to_f64(), 18_446_744_073_709_551_616.0);
        assert_eq!(big.value.value.len(), 9, "2^64 needs 9 base-256 digits");
    }

    #[test]
    fn multiplication_carries_signs() {
        assert_eq!(i(-6).checked_mul(&i(7)).map(|n| n.to_f64()), Some(-42.0));
        assert_eq!(i(-6).checked_mul(&i(-7)).map(|n| n.to_f64()), Some(42.0));
        assert!(
            i(0).checked_mul(&i(-7)).map(|n| n.value.is_negative()) == Some(false),
            "zero must not come out negative"
        );
    }

    #[test]
    fn division_folds_only_when_it_is_exact() {
        assert_eq!(
            i(12).checked_div_exact(&i(4)).map(|n| n.to_f64()),
            Some(3.0)
        );
        assert_eq!(
            i(-12).checked_div_exact(&i(4)).map(|n| n.to_f64()),
            Some(-3.0)
        );
        assert!(
            i(7).checked_div_exact(&i(2)).is_none(),
            "7/2 is not an integer and must be refused, not rounded"
        );
        assert!(
            i(1).checked_div_exact(&i(0)).is_none(),
            "division by zero must be refused"
        );
    }

    #[test]
    fn division_is_exact_for_large_multi_digit_values() {
        let big = i(2).checked_pow(64).expect("2^64");
        let half = big.checked_div_exact(&i(2)).expect("2^64 / 2");
        assert_eq!(half.to_f64(), 9_223_372_036_854_775_808.0);
        let round_trip = half.checked_mul(&i(2)).expect("re-multiply");
        assert_eq!(round_trip.to_f64(), big.to_f64());
    }

    #[test]
    fn exponent_and_digit_caps_are_enforced() {
        assert!(
            i(2).checked_pow(MAX_POW_EXPONENT + 1).is_none(),
            "an exponent past the cap must be refused, not attempted"
        );
        // 2^4096 needs 513 bytes, one past MAX_DIGITS.
        assert!(
            i(2).checked_pow(4096).is_none(),
            "a result past the digit cap must be refused"
        );
        assert!(i(2).checked_pow(0).map(|n| n.is_one()) == Some(true));
    }

    #[test]
    fn special_values_are_refused_rather_than_folded() {
        let nan = ArithmaInteger::nan();
        let inf = ArithmaInteger::infinity();
        assert!(nan.checked_add(&i(1)).is_none(), "NaN + 1 must not fold");
        assert!(inf.checked_add(&i(1)).is_none(), "inf + 1 must not fold");
        assert!(inf.checked_mul(&i(0)).is_none(), "inf * 0 must not fold");
        assert!(!nan.value.is_plain_integer());
    }

    #[test]
    fn additive_operations_require_matching_units() {
        let metres = ArithmaInteger {
            value: ArithmaInternalInteger::from_i64(3),
            unit: Some("m".to_string()),
        };
        let seconds = ArithmaInteger {
            value: ArithmaInternalInteger::from_i64(4),
            unit: Some("s".to_string()),
        };
        let more_metres = ArithmaInteger {
            value: ArithmaInternalInteger::from_i64(4),
            unit: Some("m".to_string()),
        };
        assert!(
            metres.checked_add(&seconds).is_none(),
            "metres plus seconds is a modelling error and must not fold"
        );
        let sum = metres
            .checked_add(&more_metres)
            .expect("matching units must fold");
        assert_eq!(sum.to_f64(), 7.0);
        assert_eq!(sum.unit.as_deref(), Some("m"));
        assert!(
            metres.checked_mul(&more_metres).is_none(),
            "m * m is m^2, which this type cannot express, so it must refuse"
        );
    }

    #[test]
    fn to_u32_rejects_values_that_do_not_fit() {
        assert_eq!(i(1024).to_u32(), Some(1024));
        assert_eq!(i(-1).to_u32(), None, "negative exponents are not supported");
        let big = i(2).checked_pow(40).expect("2^40");
        assert_eq!(big.to_u32(), None, "2^40 does not fit a u32");
    }
}
