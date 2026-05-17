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
//! `ArithmosInteger` â€” exact, unlimited-precision integer used in place of f64
//! and i64 throughout the AST. Mirrors `pt_arithmos::PTInteger` and inherits
//! its split design:
//!
//! - [`ArithmosInternalInteger`] holds the bytes-and-flags representation.
//! - [`ArithmosInteger`] adds an optional base SI unit.
//!
//! Special values (NaN, infinity, imaginary, infinitesimal, exact-rational)
//! are tracked via flag bits inside the internal representation so the same
//! type can express both ordinary integers and the engine's full numeric tower.

use serde::{Deserialize, Serialize};

/// Special-value flag bits for [`ArithmosInternalInteger`]. Public-but-stable so
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

/// Tunable behaviour for `ArithmosInteger`. Mirrors `PTIntegerConfig`.
#[derive(Debug, Clone)]
pub struct ArithmosIntegerConfig {
    /// Try to extract symbolic constants from f64 inputs when possible.
    pub extract_constants_from_floats: bool,
}

impl Default for ArithmosIntegerConfig {
    fn default() -> Self {
        Self {
            extract_constants_from_floats: true,
        }
    }
}

/// Unlimited-precision integer with bit flags for special values.
///
/// Bytes are stored little-endian (least-significant byte first). The vector
/// is never empty â€” `[0]` represents zero and is the canonical empty form.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ArithmosInternalInteger {
    /// Special-value flags. See [`flag`].
    pub flags: u16,
    /// Base-256 little-endian digits.
    pub value: Vec<u8>,
}

impl ArithmosInternalInteger {
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
    /// so the conversion is consistent with [`ArithmosInteger::to_f64`].
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
        let mut acc: f64 = 0.0;
        let mut scale: f64 = 1.0;
        // Bounded loop: magnitude is at most a handful of bytes for realistic
        // inputs; cap at 32 (matches a 256-bit integer) to keep this finite.
        let cap: usize = self.value.len().min(32);
        for i in 0..cap {
            acc += (self.value[i] as f64) * scale;
            scale *= 256.0;
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

impl Default for ArithmosInternalInteger {
    fn default() -> Self {
        Self::new()
    }
}

/// Public-facing integer with optional base-SI-unit attribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArithmosInteger {
    /// Underlying integer value.
    pub value: ArithmosInternalInteger,
    /// Optional base SI unit (only the seven base units: m, kg, s, A, K, mol, cd).
    pub unit: Option<String>,
}

impl ArithmosInteger {
    /// Construct zero.
    pub fn zero() -> Self {
        Self {
            value: ArithmosInternalInteger::new(),
            unit: None,
        }
    }

    /// Construct from u64.
    pub fn from_u64(value: u64) -> Self {
        Self {
            value: ArithmosInternalInteger::from_u64(value),
            unit: None,
        }
    }

    /// Construct from i64.
    pub fn from_i64(value: i64) -> Self {
        Self {
            value: ArithmosInternalInteger::from_i64(value),
            unit: None,
        }
    }

    /// IEEE infinity sentinel.
    pub fn infinity() -> Self {
        let mut v = ArithmosInternalInteger::new();
        v.flags |= flag::INFINITY;
        Self {
            value: v,
            unit: None,
        }
    }

    /// IEEE NaN sentinel.
    pub fn nan() -> Self {
        let mut v = ArithmosInternalInteger::new();
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

impl Default for ArithmosInteger {
    fn default() -> Self {
        Self::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_has_no_flags() {
        let z = ArithmosInteger::zero();
        assert_eq!(z.value.flags, 0);
    }

    #[test]
    fn infinity_sets_flag() {
        let i = ArithmosInteger::infinity();
        assert!(i.value.is_infinity());
    }

    #[test]
    fn nan_sets_flag() {
        let n = ArithmosInteger::nan();
        assert!(n.value.is_nan());
    }

    #[test]
    fn negative_flag_round_trip() {
        let mut v = ArithmosInternalInteger::new();
        assert!(!v.is_negative());
        v.set_negative(true);
        assert!(v.is_negative());
        v.set_negative(false);
        assert!(!v.is_negative());
    }

    #[test]
    fn from_u64_round_trips_through_f64() {
        let one = ArithmosInternalInteger::from_u64(1);
        assert_eq!(one.to_f64(), 1.0);
        let big = ArithmosInternalInteger::from_u64(1_000_000);
        assert_eq!(big.to_f64(), 1_000_000.0);
        let zero = ArithmosInternalInteger::from_u64(0);
        assert_eq!(zero.to_f64(), 0.0);
    }

    #[test]
    fn from_i64_handles_sign() {
        let pos = ArithmosInternalInteger::from_i64(42);
        assert_eq!(pos.to_f64(), 42.0);
        let neg = ArithmosInternalInteger::from_i64(-42);
        assert!(neg.is_negative());
        assert_eq!(neg.to_f64(), -42.0);
    }

    #[test]
    fn arithmos_integer_to_f64_handles_specials() {
        assert!(ArithmosInteger::nan().to_f64().is_nan());
        assert_eq!(ArithmosInteger::infinity().to_f64(), f64::INFINITY);
        assert_eq!(ArithmosInteger::from_i64(-7).to_f64(), -7.0);
    }
}

