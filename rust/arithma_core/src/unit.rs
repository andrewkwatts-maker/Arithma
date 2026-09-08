//====== Arithma/rust/arithma_core/src/unit.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! Unit-of-measure types and dimensional analysis.
//!
//! [`ArithmaDimension`] is the exponent vector over the seven SI base
//! dimensions. It answers the question the type system cannot: may these two
//! quantities be added? Adding metres to seconds is a modelling error, and
//! without a dimension there is nothing to catch it.
//!
//! This complements [`crate::si_units`] rather than duplicating it. That
//! catalogue holds the seven base *units* by design; this module knows the
//! *dimensions* of the standard derived units too, so `N` resolves here even
//! though `si_lookup("N")` is `None`.
//!
//! A dimension alone still cannot answer "how many metres is a kilometre?",
//! because a dimension is deliberately scale-free: `km` and `m` share one
//! exponent vector. [`SI_PREFIXES`], [`split_unit_symbol`],
//! [`ArithmaUnit::scale_to_base`] and [`ArithmaUnit::convert`] add the missing
//! magnitude, and the dimension is what lets `convert` refuse metres-to-seconds
//! instead of quietly returning a number.
//!
//! Everything here is ASCII by project rule: the micro prefix is spelled `u`
//! (`um`, `uF`) and resistance is `ohm`, never a Greek glyph.

use serde::{Deserialize, Serialize};

use crate::expression::ArithmaSIPrefix;

/// Number of SI base dimensions.
pub const BASE_DIMENSION_COUNT: usize = 7;

/// The seven SI base unit symbols, in the canonical exponent-vector order.
///
/// This order is the type's wire format: [`ArithmaDimension`] is an array of
/// exponents indexed by it, so changing it changes the meaning of every stored
/// dimension.
pub const BASE_SYMBOLS: [&str; BASE_DIMENSION_COUNT] = ["m", "kg", "s", "A", "K", "mol", "cd"];

/// Long-form names of the base dimensions, parallel to [`BASE_SYMBOLS`].
pub const BASE_QUANTITIES: [&str; BASE_DIMENSION_COUNT] = [
    "length",
    "mass",
    "time",
    "electric current",
    "thermodynamic temperature",
    "amount of substance",
    "luminous intensity",
];

/// Largest magnitude allowed for any single dimension exponent.
///
/// Safety-critical standard 2: exponent arithmetic is bounded and checked
/// rather than allowed to wrap. Real physical quantities live well inside
/// this; anything outside it is a modelling error worth reporting.
pub const MAX_EXPONENT: i8 = 32;

/// A physical dimension as a vector of exponents over the SI base units.
///
/// `m/s` is length^1 time^-1; `N` is mass^1 length^1 time^-2. Two quantities
/// may be added only if their dimensions are equal, which is the check that
/// catches "metres plus seconds" before it reaches a result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct ArithmaDimension {
    /// Exponent per base dimension, indexed by [`BASE_SYMBOLS`].
    exponents: [i8; BASE_DIMENSION_COUNT],
}

impl ArithmaDimension {
    /// The dimensionless quantity: every exponent zero.
    pub fn dimensionless() -> Self {
        Self {
            exponents: [0; BASE_DIMENSION_COUNT],
        }
    }

    /// Construct directly from an exponent vector.
    ///
    /// Returns `None` if any exponent exceeds [`MAX_EXPONENT`] in magnitude.
    pub fn from_exponents(exponents: [i8; BASE_DIMENSION_COUNT]) -> Option<Self> {
        for e in exponents.iter() {
            if e.saturating_abs() > MAX_EXPONENT {
                return None;
            }
        }
        debug_assert!(exponents.len() == BASE_DIMENSION_COUNT, "wrong arity");
        debug_assert!(
            exponents.iter().all(|e| e.saturating_abs() <= MAX_EXPONENT),
            "exponent cap not enforced"
        );
        Some(Self { exponents })
    }

    /// The dimension of a single base unit raised to the first power.
    pub fn base(symbol: &str) -> Option<Self> {
        debug_assert!(!symbol.is_empty(), "base symbol must not be empty");
        let index = BASE_SYMBOLS.iter().position(|s| *s == symbol)?;
        debug_assert!(index < BASE_DIMENSION_COUNT, "index out of range");
        let mut out = Self::dimensionless();
        out.exponents[index] = 1;
        Some(out)
    }

    /// Read the exponent of one base dimension.
    pub fn exponent(&self, symbol: &str) -> Option<i8> {
        debug_assert!(!symbol.is_empty(), "base symbol must not be empty");
        let index = BASE_SYMBOLS.iter().position(|s| *s == symbol)?;
        debug_assert!(index < BASE_DIMENSION_COUNT, "index out of range");
        Some(self.exponents[index])
    }

    /// The raw exponent vector, in [`BASE_SYMBOLS`] order.
    pub fn exponents(&self) -> [i8; BASE_DIMENSION_COUNT] {
        self.exponents
    }

    /// True when every exponent is zero.
    pub fn is_dimensionless(&self) -> bool {
        debug_assert!(self.exponents.len() == BASE_DIMENSION_COUNT, "wrong arity");
        self.exponents.iter().all(|e| *e == 0)
    }

    /// Dimension of a product: exponents add.
    ///
    /// `None` if any resulting exponent exceeds [`MAX_EXPONENT`].
    pub fn multiply(&self, other: &Self) -> Option<Self> {
        let mut out = [0i8; BASE_DIMENSION_COUNT];
        for (slot, (a, b)) in out
            .iter_mut()
            .zip(self.exponents.iter().zip(other.exponents.iter()))
        {
            *slot = a.checked_add(*b)?;
        }
        debug_assert!(out.len() == BASE_DIMENSION_COUNT, "wrong arity");
        Self::from_exponents(out)
    }

    /// Dimension of a quotient: exponents subtract.
    pub fn divide(&self, other: &Self) -> Option<Self> {
        let mut out = [0i8; BASE_DIMENSION_COUNT];
        for (slot, (a, b)) in out
            .iter_mut()
            .zip(self.exponents.iter().zip(other.exponents.iter()))
        {
            *slot = a.checked_sub(*b)?;
        }
        debug_assert!(out.len() == BASE_DIMENSION_COUNT, "wrong arity");
        Self::from_exponents(out)
    }

    /// Dimension raised to an integer power: exponents multiply.
    pub fn powi(&self, power: i8) -> Option<Self> {
        let mut out = [0i8; BASE_DIMENSION_COUNT];
        for (slot, e) in out.iter_mut().zip(self.exponents.iter()) {
            *slot = e.checked_mul(power)?;
        }
        debug_assert!(
            out.iter().all(|e| e.saturating_abs() <= MAX_EXPONENT) || power != 0,
            "a zero power must yield the dimensionless result"
        );
        Self::from_exponents(out)
    }

    /// Dimension of an n-th root, if it divides evenly.
    ///
    /// `None` when a root would need a fractional exponent -- the square root
    /// of a length is not expressible here, and returning a truncated exponent
    /// would silently produce the wrong dimension.
    pub fn nth_root(&self, n: i8) -> Option<Self> {
        if n == 0 {
            return None;
        }
        let mut out = [0i8; BASE_DIMENSION_COUNT];
        for (slot, e) in out.iter_mut().zip(self.exponents.iter()) {
            if e % n != 0 {
                return None;
            }
            *slot = e / n;
        }
        debug_assert!(n != 0, "zero root slipped past the guard");
        Self::from_exponents(out)
    }

    /// True when two quantities may be added or compared.
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self == other
    }

    /// The derived-unit symbol for this dimension, if it has a standard one.
    ///
    /// This is the reverse of [`Self::of_unit`], and is what lets a computed
    /// dimension be reported as `N` rather than `kg*m*s^-2`.
    pub fn derived_symbol(&self) -> Option<&'static str> {
        DERIVED_DIMENSIONS
            .iter()
            .find(|(_, _, exps)| *exps == self.exponents)
            .map(|(symbol, _, _)| *symbol)
    }

    /// The quantity this dimension measures, if it is a named one.
    pub fn quantity(&self) -> Option<&'static str> {
        if let Some(index) = self.single_base_index() {
            return Some(BASE_QUANTITIES[index]);
        }
        DERIVED_DIMENSIONS
            .iter()
            .find(|(_, _, exps)| *exps == self.exponents)
            .map(|(_, quantity, _)| *quantity)
    }

    /// Index of the sole base dimension with exponent 1, if that is the shape.
    fn single_base_index(&self) -> Option<usize> {
        let mut found: Option<usize> = None;
        for (i, e) in self.exponents.iter().enumerate() {
            match *e {
                0 => {}
                1 if found.is_none() => found = Some(i),
                _ => return None,
            }
        }
        found
    }

    /// Dimension of a unit symbol, base or derived.
    ///
    /// This is the bridge to [`crate::si_units`]: the SI catalogue holds the
    /// seven base units only, so `si_lookup("N")` is `None` -- but `N` is
    /// still a well-defined dimension, and this resolves it.
    pub fn of_unit(symbol: &str) -> Option<Self> {
        debug_assert!(!symbol.is_empty(), "unit symbol must not be empty");
        if let Some(base) = Self::base(symbol) {
            return Some(base);
        }
        let entry = DERIVED_DIMENSIONS.iter().find(|(s, _, _)| *s == symbol)?;
        debug_assert!(!entry.0.is_empty(), "derived table holds an empty symbol");
        Self::from_exponents(entry.2)
    }

    /// Dimension of a unit symbol that may carry one SI decimal prefix.
    ///
    /// `km` and `m` have the same dimension: a prefix changes the magnitude,
    /// never the exponent vector. [`Self::of_unit`] stays prefix-blind on
    /// purpose so that existing callers keep the exact symbol table they were
    /// written against; this is the prefix-aware entry point.
    pub fn of_prefixed_unit(symbol: &str) -> Option<Self> {
        debug_assert!(!symbol.is_empty(), "unit symbol must not be empty");
        debug_assert!(symbol.is_ascii(), "unit symbols are ASCII by project rule");
        let (power, base) = split_unit_symbol(symbol)?;
        debug_assert!(
            power.saturating_abs() <= MAX_PREFIX_POWER,
            "split returned a power outside the SI prefix range"
        );
        debug_assert!(!base.is_empty(), "split left no base symbol");
        // The power is deliberately discarded here: scale is not dimension.
        Self::of_unit(base)
    }
}

/// Standard derived dimensions, as `(symbol, quantity, exponent vector)`.
///
/// Exponents are in [`BASE_SYMBOLS`] order: m, kg, s, A, K, mol, cd. These are
/// dimensions, not units: they say what `N` *is*, which is what makes a
/// dimensional check on a derived quantity possible at all.
const DERIVED_DIMENSIONS: &[(&str, &str, [i8; BASE_DIMENSION_COUNT])] = &[
    ("Bq", "activity", [0, 0, -1, 0, 0, 0, 0]),
    ("C", "electric charge", [0, 0, 1, 1, 0, 0, 0]),
    ("F", "capacitance", [-2, -1, 4, 2, 0, 0, 0]),
    ("Gy", "absorbed dose", [2, 0, -2, 0, 0, 0, 0]),
    ("H", "inductance", [2, 1, -2, -2, 0, 0, 0]),
    ("Hz", "frequency", [0, 0, -1, 0, 0, 0, 0]),
    ("J", "energy", [2, 1, -2, 0, 0, 0, 0]),
    ("N", "force", [1, 1, -2, 0, 0, 0, 0]),
    ("Pa", "pressure", [-1, 1, -2, 0, 0, 0, 0]),
    ("S", "conductance", [-2, -1, 3, 2, 0, 0, 0]),
    ("Sv", "dose equivalent", [2, 0, -2, 0, 0, 0, 0]),
    ("T", "magnetic flux density", [0, 1, -2, -1, 0, 0, 0]),
    ("V", "electric potential", [2, 1, -3, -1, 0, 0, 0]),
    ("W", "power", [2, 1, -3, 0, 0, 0, 0]),
    ("Wb", "magnetic flux", [2, 1, -2, -1, 0, 0, 0]),
    ("kat", "catalytic activity", [0, 0, -1, 0, 0, 1, 0]),
    ("lm", "luminous flux", [0, 0, 0, 0, 0, 0, 1]),
    ("lx", "illuminance", [-2, 0, 0, 0, 0, 0, 1]),
    ("ohm", "electric resistance", [2, 1, -3, -2, 0, 0, 0]),
];

// ---------------------------------------------------------------------------
// SI decimal prefixes.
//
// The crate already owns a prefix *type*: `crate::expression::ArithmaSIPrefix`,
// an enum spanning yocto..yotta that the constants and function modules use to
// preserve symbolic intent (`5 km` is not `5000 m` in the output). This module
// reuses that enum rather than declaring a rival one -- but it cannot be the
// whole story here, for two reasons:
//
//   1. It stops at yotta/yocto. The 2022 CGPM additions (quetta, ronna, ronto,
//      quecto) have no variant, and this file may not edit that module.
//   2. Its `symbol()` returns a non-ASCII glyph for micro, and its
//      `multiplier()` is a rounded `f64`. Conversion here must be exact, which
//      means arithmetic on an integer power of ten, not on a float multiplier.
//
// So the table below is data, not a second type: it carries the ASCII symbol,
// the long name, the integer power of ten, and the matching `ArithmaSIPrefix`
// variant where one exists. `None` in the last column means "this prefix
// postdates that enum", which is a fact worth being able to read off.
// ---------------------------------------------------------------------------

/// Number of SI decimal prefixes, quetta (10^30) through quecto (10^-30).
pub const SI_PREFIX_COUNT: usize = 24;

/// Largest magnitude of any SI prefix power of ten.
pub const MAX_PREFIX_POWER: i32 = 30;

/// The SI decimal prefixes as `(ASCII symbol, name, power of ten, legacy enum)`.
///
/// Ordered from largest to smallest power. Symbols are ASCII by project rule:
/// micro is `u`, never a Greek glyph, so `um` and `uF` are the spellings this
/// crate accepts. Every power in the table is distinct, which is what makes
/// [`si_prefix_for_power`] a well-defined inverse of [`si_prefix_power`].
pub const SI_PREFIXES: [(&str, &str, i32, Option<ArithmaSIPrefix>); SI_PREFIX_COUNT] = [
    ("Q", "quetta", 30, None),
    ("R", "ronna", 27, None),
    ("Y", "yotta", 24, Some(ArithmaSIPrefix::Yotta)),
    ("Z", "zetta", 21, Some(ArithmaSIPrefix::Zetta)),
    ("E", "exa", 18, Some(ArithmaSIPrefix::Exa)),
    ("P", "peta", 15, Some(ArithmaSIPrefix::Peta)),
    ("T", "tera", 12, Some(ArithmaSIPrefix::Tera)),
    ("G", "giga", 9, Some(ArithmaSIPrefix::Giga)),
    ("M", "mega", 6, Some(ArithmaSIPrefix::Mega)),
    ("k", "kilo", 3, Some(ArithmaSIPrefix::Kilo)),
    ("h", "hecto", 2, Some(ArithmaSIPrefix::Hecto)),
    ("da", "deca", 1, Some(ArithmaSIPrefix::Deca)),
    ("d", "deci", -1, Some(ArithmaSIPrefix::Deci)),
    ("c", "centi", -2, Some(ArithmaSIPrefix::Centi)),
    ("m", "milli", -3, Some(ArithmaSIPrefix::Milli)),
    ("u", "micro", -6, Some(ArithmaSIPrefix::Micro)),
    ("n", "nano", -9, Some(ArithmaSIPrefix::Nano)),
    ("p", "pico", -12, Some(ArithmaSIPrefix::Pico)),
    ("f", "femto", -15, Some(ArithmaSIPrefix::Femto)),
    ("a", "atto", -18, Some(ArithmaSIPrefix::Atto)),
    ("z", "zepto", -21, Some(ArithmaSIPrefix::Zepto)),
    ("y", "yocto", -24, Some(ArithmaSIPrefix::Yocto)),
    ("r", "ronto", -27, None),
    ("q", "quecto", -30, None),
];

/// The power of ten of a prefix symbol, e.g. `k` is 3 and `u` is -6.
///
/// `None` for anything that is not a prefix. Note that the empty string is not
/// a prefix: "no prefix" is the absence of an entry, not a zero-power row, so
/// that a lookup cannot silently succeed on a typo that stripped to nothing.
pub fn si_prefix_power(symbol: &str) -> Option<i32> {
    debug_assert!(!symbol.is_empty(), "a prefix symbol must not be empty");
    debug_assert!(
        symbol.is_ascii(),
        "prefix symbols are ASCII by project rule"
    );
    SI_PREFIXES
        .iter()
        .find(|(s, _, _, _)| *s == symbol)
        .map(|(_, _, power, _)| *power)
}

/// The long-form name of a prefix symbol, e.g. `k` is "kilo".
pub fn si_prefix_name(symbol: &str) -> Option<&'static str> {
    debug_assert!(!symbol.is_empty(), "a prefix symbol must not be empty");
    debug_assert!(
        symbol.is_ascii(),
        "prefix symbols are ASCII by project rule"
    );
    SI_PREFIXES
        .iter()
        .find(|(s, _, _, _)| *s == symbol)
        .map(|(_, name, _, _)| *name)
}

/// The prefix symbol for a power of ten, the inverse of [`si_prefix_power`].
///
/// `None` for a power no prefix names (10^4 has none) and for 0, which is the
/// unprefixed case rather than a prefix.
pub fn si_prefix_for_power(power: i32) -> Option<&'static str> {
    debug_assert!(
        SI_PREFIXES.len() == SI_PREFIX_COUNT,
        "prefix table arity drifted from SI_PREFIX_COUNT"
    );
    debug_assert!(
        power != i32::MIN,
        "i32::MIN has no negation and cannot be a prefix power"
    );
    SI_PREFIXES
        .iter()
        .find(|(_, _, p, _)| *p == power)
        .map(|(s, _, _, _)| *s)
}

/// The [`ArithmaSIPrefix`] variant for a prefix symbol, where one exists.
///
/// This is the bridge to the pre-existing enum in [`crate::expression`], which
/// is what the AST stores. `Some(None)` means "a real SI prefix, but one that
/// postdates that enum" (quetta, ronna, ronto, quecto); the outer `None` means
/// "not a prefix at all". The two are different answers and are kept apart.
#[allow(clippy::option_option)]
pub fn si_prefix_variant(symbol: &str) -> Option<Option<ArithmaSIPrefix>> {
    debug_assert!(!symbol.is_empty(), "a prefix symbol must not be empty");
    debug_assert!(
        symbol.is_ascii(),
        "prefix symbols are ASCII by project rule"
    );
    SI_PREFIXES
        .iter()
        .find(|(s, _, _, _)| *s == symbol)
        .map(|(_, _, _, variant)| *variant)
}

/// Split a unit symbol into `(power of ten, base symbol)`.
///
/// `km` is `(3, "m")`, `ms` is `(-3, "s")`, `m` is `(0, "m")`. `None` when the
/// symbol names no unit this crate knows, which is distinct from `(0, ...)`.
///
/// Two resolution rules make this deterministic, and both matter:
///
///   1. **A whole symbol that is itself a unit wins outright.** This is what
///      keeps `kg` from parsing as kilo + grams. `kg` is an SI *base* unit and
///      `g` is not in [`BASE_SYMBOLS`] at all, so splitting it would produce a
///      base symbol this crate cannot resolve and a mass off by 1000. The same
///      rule protects `cd` (candela, not centi-day), `mol` (not milli-ol),
///      `Pa`, `Gy`, `kat` and `T` (tesla, not tera).
///   2. **Otherwise the longest prefix whose remainder is itself a unit wins.**
///      Only `da` is two characters, so this decides exactly one family:
///      `dam` is deca-metre, because the `d` reading would leave `am`, which
///      is not a unit. No known unit symbol begins with `a`, so no `daX` case
///      is genuinely ambiguous today; the longest-match rule fixes the answer
///      in advance if one ever appears.
///
/// Consequence worth stating: because `g` is not a base symbol here, `mg` and
/// `ng` do not resolve. Mass is prefixed off `kg`, and adding gram as a
/// non-coherent unit is a separate change from adding prefixes.
pub fn split_unit_symbol(symbol: &str) -> Option<(i32, &str)> {
    debug_assert!(!symbol.is_empty(), "unit symbol must not be empty");
    debug_assert!(symbol.is_ascii(), "unit symbols are ASCII by project rule");

    // Rule 1: the unprefixed reading is preferred whenever it is a real unit.
    if ArithmaDimension::of_unit(symbol).is_some() {
        return Some((0, symbol));
    }

    // Rule 2: longest valid prefix. Bounded by the table, no recursion.
    let mut best: Option<(i32, &str)> = None;
    let mut best_len = 0usize;
    for (prefix, _, power, _) in SI_PREFIXES.iter() {
        let rest = match symbol.strip_prefix(*prefix) {
            Some(rest) => rest,
            None => continue,
        };
        if rest.is_empty() || ArithmaDimension::of_unit(rest).is_none() {
            continue;
        }
        if prefix.len() > best_len {
            best_len = prefix.len();
            best = Some((*power, rest));
        }
    }
    debug_assert!(
        best.is_none() || best_len >= 1,
        "a match must have consumed at least one prefix character"
    );
    debug_assert!(
        best.is_none_or(|(_, base)| base.len() < symbol.len()),
        "a prefixed base symbol must be shorter than the whole symbol"
    );
    best
}

/// Multiply or divide `value` by an exact integer power of ten.
///
/// Multiplying by `10^-3` would mean multiplying by a rounded reciprocal;
/// dividing by `10^3` instead keeps the common cases exact, so `1000 m` is
/// exactly `1 km` and not `0.9999999999999999 km`. Powers up to 10^22 are
/// exactly representable in `f64`, which covers every prefix pair that occurs
/// in practice; beyond that the result is the nearest double.
fn apply_power_of_ten(value: f64, power: i32) -> f64 {
    debug_assert!(
        value.is_finite(),
        "a non-finite magnitude must be refused before scaling"
    );
    debug_assert!(
        power.saturating_abs() <= 2 * MAX_PREFIX_POWER,
        "power of ten outside the range any prefix pair can produce"
    );
    let magnitude = 10f64.powi(power.saturating_abs());
    if power >= 0 {
        value * magnitude
    } else {
        value / magnitude
    }
}

/// What can go wrong when converting between units.
///
/// Refusal is the point of carrying a dimension, so these are returned rather
/// than papered over: no panic, and no sentinel value that a caller could mistake
/// for a converted magnitude.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArithmaUnitError {
    /// The symbol names no unit this crate knows.
    ///
    /// Kept distinct from [`Self::IncompatibleDimensions`]: "I do not know that
    /// unit" and "those two units cannot be converted" call for different fixes.
    UnknownUnit {
        /// The symbol that failed to resolve.
        symbol: String,
    },
    /// The two units measure different quantities -- metres to seconds.
    IncompatibleDimensions {
        /// Source unit symbol.
        from: String,
        /// Target unit symbol.
        to: String,
    },
    /// The magnitude to convert was NaN or infinite.
    NonFiniteValue,
    /// A finite magnitude scaled past the `f64` range.
    ///
    /// Returning the infinity would report a definite, wrong number.
    Overflow {
        /// Source unit symbol.
        from: String,
        /// Target unit symbol.
        to: String,
    },
}

impl core::fmt::Display for ArithmaUnitError {
    /// ASCII-only messages, matching the rest of this module's rendering.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnknownUnit { symbol } => {
                write!(f, "unknown unit symbol '{symbol}'")
            }
            Self::IncompatibleDimensions { from, to } => write!(
                f,
                "cannot convert '{from}' to '{to}': the dimensions differ"
            ),
            Self::NonFiniteValue => {
                write!(f, "cannot convert a non-finite magnitude")
            }
            Self::Overflow { from, to } => {
                write!(f, "converting '{from}' to '{to}' overflowed the f64 range")
            }
        }
    }
}

impl std::error::Error for ArithmaUnitError {}

impl core::fmt::Display for ArithmaDimension {
    /// Render as a product of base symbols with exponents, e.g. `kg*m*s^-2`.
    ///
    /// Dimensionless renders as `1`. ASCII only -- the project's naming rule
    /// is Latin letters, so no superscript glyphs.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.is_dimensionless() {
            return write!(f, "1");
        }
        let mut first = true;
        for (i, e) in self.exponents.iter().enumerate() {
            if *e == 0 {
                continue;
            }
            if !first {
                write!(f, "*")?;
            }
            first = false;
            if *e == 1 {
                write!(f, "{}", BASE_SYMBOLS[i])?;
            } else {
                write!(f, "{}^{}", BASE_SYMBOLS[i], e)?;
            }
        }
        debug_assert!(!first, "a non-dimensionless value printed nothing");
        Ok(())
    }
}

/// A unit of measure (e.g. "meter", "kilogram"). Composite units like
/// `m*s^-1` are stored as a `Vec<(ArithmaUnit, i32)>` exponent list elsewhere;
/// this struct represents one base or derived unit, optionally carrying a
/// single SI decimal prefix (`km`, `ms`, `uF`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ArithmaUnit {
    /// SI symbol (e.g. "m", "kg").
    pub symbol: String,
    /// Long-form name (e.g. "meter").
    pub name: String,
}

impl ArithmaUnit {
    pub fn new(symbol: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            symbol: symbol.into(),
            name: name.into(),
        }
    }

    /// The physical dimension of this unit, base or derived, prefix or not.
    ///
    /// `km` reports the dimension of length: a prefix scales a quantity, it
    /// does not change what the quantity is.
    ///
    /// `None` for a symbol this crate does not recognise. That is deliberately
    /// distinct from "dimensionless": an unknown unit and a pure number are
    /// not the same answer.
    pub fn dimension(&self) -> Option<ArithmaDimension> {
        debug_assert!(!self.symbol.is_empty(), "a unit must carry a symbol");
        debug_assert!(self.symbol.len() <= 16, "implausible unit symbol");
        ArithmaDimension::of_prefixed_unit(&self.symbol)
    }

    /// The factor from this unit to its coherent SI base unit.
    ///
    /// `km` is 1000.0, `mm` is 0.001, `m` and `kg` are 1.0. `None` for a symbol
    /// this crate cannot resolve, which is not the same answer as 1.0.
    ///
    /// This is the scale only. It says nothing about whether two units may be
    /// converted -- that is [`Self::convert`]'s job, and it needs the dimension.
    pub fn scale_to_base(&self) -> Option<f64> {
        debug_assert!(!self.symbol.is_empty(), "a unit must carry a symbol");
        debug_assert!(self.symbol.len() <= 16, "implausible unit symbol");
        let (power, base) = split_unit_symbol(&self.symbol)?;
        debug_assert!(!base.is_empty(), "split left no base symbol");
        Some(apply_power_of_ten(1.0, power))
    }

    /// Convert a magnitude from one unit to another.
    ///
    /// Refuses when the dimensions differ: metres to seconds is a modelling
    /// error, and a converted number would hide it. Refusing is the whole
    /// reason this module carries dimensions at all.
    ///
    /// The scaling is done as a single net power of ten rather than as
    /// `value * from_scale / to_scale`, so the common cases are exact:
    /// `convert(1.0, km, m)` is exactly `1000.0` and `convert(1000.0, m, km)`
    /// is exactly `1.0`.
    pub fn convert(value: f64, from: &Self, to: &Self) -> Result<f64, ArithmaUnitError> {
        debug_assert!(
            !from.symbol.is_empty(),
            "the source unit must carry a symbol"
        );
        debug_assert!(!to.symbol.is_empty(), "the target unit must carry a symbol");

        // A NaN or infinite magnitude cannot be scaled into a meaningful
        // answer, and propagating it would launder the caller's bad input.
        if !value.is_finite() {
            return Err(ArithmaUnitError::NonFiniteValue);
        }

        let (from_power, from_base) =
            split_unit_symbol(&from.symbol).ok_or_else(|| ArithmaUnitError::UnknownUnit {
                symbol: from.symbol.clone(),
            })?;
        let (to_power, to_base) =
            split_unit_symbol(&to.symbol).ok_or_else(|| ArithmaUnitError::UnknownUnit {
                symbol: to.symbol.clone(),
            })?;

        // `split_unit_symbol` only returns a base symbol it already resolved,
        // so these lookups cannot fail; the `ok_or_else` keeps the promise of
        // never panicking if that invariant is ever loosened.
        let from_dimension =
            ArithmaDimension::of_unit(from_base).ok_or_else(|| ArithmaUnitError::UnknownUnit {
                symbol: from.symbol.clone(),
            })?;
        let to_dimension =
            ArithmaDimension::of_unit(to_base).ok_or_else(|| ArithmaUnitError::UnknownUnit {
                symbol: to.symbol.clone(),
            })?;
        if !from_dimension.is_compatible_with(&to_dimension) {
            return Err(ArithmaUnitError::IncompatibleDimensions {
                from: from.symbol.clone(),
                to: to.symbol.clone(),
            });
        }

        let net = from_power.saturating_sub(to_power);
        let converted = apply_power_of_ten(value, net);
        if !converted.is_finite() {
            return Err(ArithmaUnitError::Overflow {
                from: from.symbol.clone(),
                to: to.symbol.clone(),
            });
        }
        debug_assert!(
            net != 0 || converted == value,
            "units of equal scale must leave the magnitude untouched"
        );
        debug_assert!(
            net.saturating_abs() <= 2 * MAX_PREFIX_POWER,
            "net power of ten outside the range two prefixes can produce"
        );
        Ok(converted)
    }

    /// The quantity this unit measures ("force", "length"), if it is known.
    pub fn quantity(&self) -> Option<&'static str> {
        self.dimension()?.quantity()
    }

    /// Whether two quantities in these units may be added or compared.
    ///
    /// `None` when either symbol is unrecognised -- "I cannot tell" must not
    /// be reported as "incompatible", because a caller would treat the latter
    /// as a definite error.
    pub fn is_compatible_with(&self, other: &Self) -> Option<bool> {
        debug_assert!(!self.symbol.is_empty(), "a unit must carry a symbol");
        let (a, b) = (self.dimension()?, other.dimension()?);
        Some(a.is_compatible_with(&b))
    }
}

// ---------------------------------------------------------------------------
// Backward-compatibility aliases for the pre-rename `Arithmos*` names.
// Retained for one release; downstream (eml-math, eml-spectral, metaphysica,
// periodica) should migrate to the `Arithma*` names above.
// ---------------------------------------------------------------------------
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaUnit`")]
#[allow(unused)]
pub use self::ArithmaUnit as ArithmosUnit;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_constructs_unit() {
        let u = ArithmaUnit::new("m", "meter");
        assert_eq!(u.symbol, "m");
        assert_eq!(u.name, "meter");
    }
    // ----- dimensional analysis --------------------------------------------
    //
    // `ArithmaUnit` carried a symbol and a name and nothing else; there was no
    // way to ask whether two quantities could be added. These pin the algebra
    // that makes that check possible.

    #[test]
    fn base_dimensions_resolve() {
        for symbol in BASE_SYMBOLS {
            let d = ArithmaDimension::base(symbol).expect("base unit must resolve");
            assert_eq!(d.exponent(symbol), Some(1));
            assert!(!d.is_dimensionless());
        }
        assert!(
            ArithmaDimension::base("N").is_none(),
            "N is not a base unit"
        );
    }

    #[test]
    fn dimensionless_is_the_multiplicative_identity() {
        let one = ArithmaDimension::dimensionless();
        assert!(one.is_dimensionless());
        let m = ArithmaDimension::base("m").expect("m");
        assert_eq!(m.multiply(&one), Some(m));
        assert_eq!(m.divide(&m), Some(one));
    }

    #[test]
    fn force_is_mass_times_acceleration() {
        let kg = ArithmaDimension::base("kg").expect("kg");
        let m = ArithmaDimension::base("m").expect("m");
        let s = ArithmaDimension::base("s").expect("s");
        let acceleration = m.divide(&s.powi(2).expect("s^2")).expect("m/s^2");
        let force = kg.multiply(&acceleration).expect("kg*m/s^2");
        assert_eq!(
            force,
            ArithmaDimension::of_unit("N").expect("N"),
            "kg*m/s^2 must be the dimension of a newton"
        );
        assert_eq!(force.derived_symbol(), Some("N"));
        assert_eq!(force.quantity(), Some("force"));
    }

    #[test]
    fn energy_is_force_times_distance() {
        let force = ArithmaDimension::of_unit("N").expect("N");
        let m = ArithmaDimension::base("m").expect("m");
        assert_eq!(
            force.multiply(&m),
            Some(ArithmaDimension::of_unit("J").expect("J"))
        );
    }

    #[test]
    fn power_is_energy_per_time() {
        let joule = ArithmaDimension::of_unit("J").expect("J");
        let s = ArithmaDimension::base("s").expect("s");
        assert_eq!(
            joule.divide(&s),
            Some(ArithmaDimension::of_unit("W").expect("W"))
        );
    }

    #[test]
    fn ohms_law_is_dimensionally_consistent() {
        // V = I * R
        let volt = ArithmaDimension::of_unit("V").expect("V");
        let amp = ArithmaDimension::base("A").expect("A");
        let ohm = ArithmaDimension::of_unit("ohm").expect("ohm");
        assert_eq!(amp.multiply(&ohm), Some(volt));
    }

    #[test]
    fn incompatible_dimensions_are_reported_as_such() {
        let m = ArithmaDimension::base("m").expect("m");
        let s = ArithmaDimension::base("s").expect("s");
        assert!(
            !m.is_compatible_with(&s),
            "metres and seconds must not be addable"
        );
        assert!(m.is_compatible_with(&m));
    }

    #[test]
    fn roots_are_refused_when_they_would_be_fractional() {
        let area = ArithmaDimension::base("m")
            .expect("m")
            .powi(2)
            .expect("m^2");
        assert_eq!(
            area.nth_root(2),
            Some(ArithmaDimension::base("m").expect("m")),
            "the square root of an area is a length"
        );
        let m = ArithmaDimension::base("m").expect("m");
        assert!(
            m.nth_root(2).is_none(),
            "sqrt(m) has no integer exponent and must be refused, not truncated"
        );
        assert!(m.nth_root(0).is_none(), "a zeroth root is undefined");
    }

    #[test]
    fn exponent_overflow_is_refused_rather_than_wrapped() {
        let m = ArithmaDimension::base("m").expect("m");
        assert!(
            m.powi(MAX_EXPONENT + 1).is_none(),
            "an exponent past the cap must be refused"
        );
        let huge =
            ArithmaDimension::from_exponents([MAX_EXPONENT, 0, 0, 0, 0, 0, 0]).expect("at the cap");
        assert!(
            huge.multiply(&huge).is_none(),
            "a product past the cap must be refused, not wrapped"
        );
    }

    #[test]
    fn display_is_ascii_and_readable() {
        assert_eq!(ArithmaDimension::dimensionless().to_string(), "1");
        assert_eq!(ArithmaDimension::base("m").expect("m").to_string(), "m");
        let force = ArithmaDimension::of_unit("N").expect("N");
        assert_eq!(force.to_string(), "m*kg*s^-2");
        assert!(
            force.to_string().is_ascii(),
            "dimension rendering must stay ASCII"
        );
    }

    #[test]
    fn units_expose_their_dimension() {
        let metre = ArithmaUnit::new("m", "meter");
        assert_eq!(metre.dimension(), ArithmaDimension::base("m"));
        // A derived unit has no entry in the SI catalogue but does have a
        // dimension -- that gap is exactly what this closes.
        let newton = ArithmaUnit::new("N", "newton");
        assert_eq!(newton.dimension(), ArithmaDimension::of_unit("N"));
        assert_eq!(newton.quantity(), Some("force"));

        let nonsense = ArithmaUnit::new("zz", "not a unit");
        assert_eq!(nonsense.dimension(), None);
    }

    #[test]
    fn units_report_addition_compatibility() {
        let metre = ArithmaUnit::new("m", "meter");
        let second = ArithmaUnit::new("s", "second");
        assert_eq!(metre.is_compatible_with(&metre), Some(true));
        assert_eq!(metre.is_compatible_with(&second), Some(false));
        // An unknown symbol yields None rather than a guess: "I don't know"
        // and "not compatible" are different answers.
        let unknown = ArithmaUnit::new("zz", "not a unit");
        assert_eq!(metre.is_compatible_with(&unknown), None);
    }

    // ----- SI prefixes and conversion ---------------------------------------
    //
    // A dimension is scale-free, so before these there was no way to say that
    // a kilometre is a thousand metres. These pin the prefix table, the split
    // rules that make `kg` behave, and the exactness of the conversion.

    #[test]
    fn every_prefix_round_trips_through_symbol_lookup() {
        assert_eq!(
            SI_PREFIXES.len(),
            SI_PREFIX_COUNT,
            "the table must hold every prefix from quetta to quecto"
        );
        for (symbol, name, power, _) in SI_PREFIXES.iter() {
            assert_eq!(
                si_prefix_power(symbol),
                Some(*power),
                "symbol {symbol} must resolve to its own power"
            );
            assert_eq!(
                si_prefix_for_power(*power),
                Some(*symbol),
                "power {power} must resolve back to its own symbol"
            );
            assert_eq!(si_prefix_name(symbol), Some(*name));
            assert!(
                si_prefix_variant(symbol).is_some(),
                "{symbol} must be present in the variant bridge"
            );
        }
        assert_eq!(si_prefix_power("zz"), None, "zz is not a prefix");
        assert_eq!(
            si_prefix_for_power(0),
            None,
            "no prefix means 10^0, not a row"
        );
        assert_eq!(si_prefix_for_power(4), None, "10^4 has no SI prefix");
    }

    #[test]
    fn the_prefix_table_is_ascii_and_free_of_duplicates() {
        for (i, (symbol, name, power, _)) in SI_PREFIXES.iter().enumerate() {
            assert!(symbol.is_ascii(), "prefix symbol {symbol} must be ASCII");
            assert!(name.is_ascii(), "prefix name {name} must be ASCII");
            assert!(!symbol.is_empty(), "a prefix must have a symbol");
            assert!(
                power.abs() <= MAX_PREFIX_POWER,
                "{name} exceeds the declared prefix range"
            );
            for (other_symbol, _, other_power, _) in SI_PREFIXES.iter().skip(i + 1) {
                assert_ne!(symbol, other_symbol, "duplicate prefix symbol");
                assert_ne!(power, other_power, "duplicate prefix power");
            }
        }
        // Micro is `u`, not a Greek glyph: the whole reason this table exists
        // rather than reusing the expression enum's own `symbol()`.
        assert_eq!(si_prefix_power("u"), Some(-6));
    }

    #[test]
    fn the_prefix_table_agrees_with_the_expression_module_enum() {
        // The bridge to the pre-existing `ArithmaSIPrefix`. Symbols are not
        // compared, because that enum renders micro with a non-ASCII glyph;
        // the magnitudes are what must agree. `None` in the last column marks
        // the four 2022 prefixes that enum predates.
        let mut bridged = 0usize;
        for (_, name, power, variant) in SI_PREFIXES.iter() {
            match variant {
                Some(v) => {
                    let mine = apply_power_of_ten(1.0, *power);
                    let theirs = v.multiplier();
                    assert!(
                        (mine - theirs).abs() <= 1e-9 * theirs.abs(),
                        "{name} disagrees with ArithmaSIPrefix::multiplier"
                    );
                    bridged += 1;
                }
                None => assert!(
                    power.abs() > 24,
                    "only the post-2022 prefixes may lack an enum variant"
                ),
            }
        }
        assert_eq!(bridged, 20, "the legacy enum covers yocto..yotta");
    }

    #[test]
    fn kilogram_does_not_parse_as_kilo_plus_grams() {
        // The classic trap. `kg` is itself the SI base unit of mass, and `g`
        // is not in BASE_SYMBOLS at all, so a kilo+gram reading would give a
        // scale of 1000 against a base symbol this crate cannot resolve.
        assert_eq!(split_unit_symbol("kg"), Some((0, "kg")));
        assert_eq!(
            ArithmaDimension::of_unit("g"),
            None,
            "gram is deliberately not a known symbol here"
        );

        let kilogram = ArithmaUnit::new("kg", "kilogram");
        assert_eq!(
            kilogram.scale_to_base(),
            Some(1.0),
            "the kilogram is already the coherent base unit"
        );
        assert_eq!(kilogram.dimension(), ArithmaDimension::base("kg"));
        assert_eq!(kilogram.quantity(), Some("mass"));

        // And the gram-prefixed spellings stay unresolved rather than guessing.
        assert_eq!(split_unit_symbol("mg"), None);
        assert!(matches!(
            ArithmaUnit::convert(1.0, &ArithmaUnit::new("mg", "milligram"), &kilogram),
            Err(ArithmaUnitError::UnknownUnit { .. })
        ));
    }

    #[test]
    fn a_whole_symbol_that_is_a_unit_beats_any_prefix_reading() {
        // Every one of these has a tempting prefix split: centi-day, milli-ol,
        // tera-?, peta-?. Rule 1 settles all of them the same way.
        for symbol in [
            "m", "kg", "s", "A", "K", "mol", "cd", "T", "Pa", "Gy", "kat", "Hz",
        ] {
            assert_eq!(
                split_unit_symbol(symbol),
                Some((0, symbol)),
                "{symbol} must read as itself, unprefixed"
            );
        }
    }

    #[test]
    fn the_longest_prefix_wins_when_two_readings_are_possible() {
        // `dam` could be d+am or da+m. `am` is not a unit, and the longest
        // prefix whose remainder is a unit wins, so both rules agree on deca.
        assert_eq!(split_unit_symbol("dam"), Some((1, "m")));
        assert_eq!(split_unit_symbol("dm"), Some((-1, "m")));
        assert_eq!(split_unit_symbol("daN"), Some((1, "N")));
    }

    #[test]
    fn prefixed_units_report_the_dimension_of_their_base_unit() {
        for (prefixed, base) in [
            ("km", "m"),
            ("mm", "m"),
            ("ms", "s"),
            ("mA", "A"),
            ("uF", "F"),
            ("nN", "N"),
            ("kmol", "mol"),
            ("Mohm", "ohm"),
            ("hPa", "Pa"),
        ] {
            let unit = ArithmaUnit::new(prefixed, "prefixed unit");
            assert_eq!(
                unit.dimension(),
                ArithmaDimension::of_unit(base),
                "{prefixed} must carry the dimension of {base}"
            );
        }
        let kilometre = ArithmaUnit::new("km", "kilometer");
        assert_eq!(kilometre.quantity(), Some("length"));
        assert_eq!(
            kilometre.is_compatible_with(&ArithmaUnit::new("m", "meter")),
            Some(true),
            "a prefix does not change what may be added to what"
        );
        // Still unknown when the base is unknown, not silently dimensionless.
        assert_eq!(ArithmaUnit::new("kzz", "nonsense").dimension(), None);
    }

    #[test]
    fn scale_to_base_is_exact_for_the_common_prefixes() {
        for (symbol, expected) in [
            ("m", 1.0),
            ("kg", 1.0),
            ("km", 1000.0),
            ("mm", 0.001),
            ("cm", 0.01),
            ("us", 1e-6),
            ("MHz", 1e6),
            ("GHz", 1e9),
        ] {
            assert_eq!(
                ArithmaUnit::new(symbol, "probe").scale_to_base(),
                Some(expected),
                "{symbol} must scale exactly"
            );
        }
        assert_eq!(
            ArithmaUnit::new("zz", "not a unit").scale_to_base(),
            None,
            "an unknown unit has no scale, which is not the same as 1.0"
        );
    }

    #[test]
    fn kilometres_round_trip_through_metres_exactly() {
        let km = ArithmaUnit::new("km", "kilometer");
        let m = ArithmaUnit::new("m", "meter");
        let mm = ArithmaUnit::new("mm", "millimeter");

        let metres = ArithmaUnit::convert(1.0, &km, &m).expect("km to m");
        assert_eq!(metres, 1000.0, "one kilometre is exactly a thousand metres");
        let back = ArithmaUnit::convert(metres, &m, &km).expect("m to km");
        assert_eq!(back, 1.0, "the round trip must not drift by an ulp");

        // Scaling by a net power of ten rather than by a rounded reciprocal is
        // what keeps these exact in both directions.
        assert_eq!(ArithmaUnit::convert(1.0, &m, &mm).expect("m to mm"), 1000.0);
        assert_eq!(ArithmaUnit::convert(1.0, &mm, &m).expect("mm to m"), 0.001);
        assert_eq!(
            ArithmaUnit::convert(2.5, &km, &mm).expect("km to mm"),
            2_500_000.0
        );
        assert_eq!(
            ArithmaUnit::convert(7.0, &m, &m).expect("m to m"),
            7.0,
            "converting a unit to itself must be the identity"
        );
    }

    #[test]
    fn converting_metres_to_seconds_is_an_error() {
        let m = ArithmaUnit::new("m", "meter");
        let s = ArithmaUnit::new("s", "second");
        let err = ArithmaUnit::convert(1.0, &m, &s).expect_err("metres are not seconds");
        assert_eq!(
            err,
            ArithmaUnitError::IncompatibleDimensions {
                from: "m".to_string(),
                to: "s".to_string(),
            }
        );
        assert!(err.to_string().is_ascii(), "error text must stay ASCII");
        // A prefix does not rescue an incompatible pair.
        let ms = ArithmaUnit::new("ms", "millisecond");
        assert!(ArithmaUnit::convert(1.0, &ArithmaUnit::new("km", "kilometer"), &ms).is_err());
    }

    #[test]
    fn an_unknown_unit_is_a_different_error_from_an_incompatible_one() {
        let m = ArithmaUnit::new("m", "meter");
        let s = ArithmaUnit::new("s", "second");
        let nonsense = ArithmaUnit::new("zz", "not a unit");

        let unknown = ArithmaUnit::convert(1.0, &nonsense, &m).expect_err("zz is unknown");
        let incompatible = ArithmaUnit::convert(1.0, &m, &s).expect_err("m is not s");
        assert_eq!(
            unknown,
            ArithmaUnitError::UnknownUnit {
                symbol: "zz".to_string()
            }
        );
        assert_ne!(
            unknown, incompatible,
            "'I do not know that unit' and 'those cannot convert' need different fixes"
        );
        // The unknown unit is reported whichever side it appears on.
        assert_eq!(
            ArithmaUnit::convert(1.0, &m, &nonsense).expect_err("zz is unknown"),
            ArithmaUnitError::UnknownUnit {
                symbol: "zz".to_string()
            }
        );
    }

    #[test]
    fn non_finite_and_overflowing_conversions_are_refused() {
        let m = ArithmaUnit::new("m", "meter");
        let km = ArithmaUnit::new("km", "kilometer");
        assert_eq!(
            ArithmaUnit::convert(f64::NAN, &km, &m),
            Err(ArithmaUnitError::NonFiniteValue)
        );
        assert_eq!(
            ArithmaUnit::convert(f64::INFINITY, &km, &m),
            Err(ArithmaUnitError::NonFiniteValue)
        );

        // A finite input that scales past f64 is reported, not returned as
        // infinity -- an infinity reads as a definite, wrong answer.
        let quetta = ArithmaUnit::new("Qm", "quettameter");
        let quecto = ArithmaUnit::new("qm", "quectometer");
        assert_eq!(
            ArithmaUnit::convert(1.0e300, &quetta, &quecto),
            Err(ArithmaUnitError::Overflow {
                from: "Qm".to_string(),
                to: "qm".to_string(),
            })
        );
        assert!(ArithmaUnit::convert(1.0, &quetta, &quecto).is_ok());
    }

    #[test]
    fn this_module_renders_nothing_but_ascii() {
        // The project's naming rule, asserted rather than trusted.
        let errors = [
            ArithmaUnitError::UnknownUnit {
                symbol: "zz".to_string(),
            },
            ArithmaUnitError::IncompatibleDimensions {
                from: "m".to_string(),
                to: "s".to_string(),
            },
            ArithmaUnitError::NonFiniteValue,
            ArithmaUnitError::Overflow {
                from: "Qm".to_string(),
                to: "qm".to_string(),
            },
        ];
        for e in errors.iter() {
            assert!(e.to_string().is_ascii(), "error rendering must stay ASCII");
            assert!(!e.to_string().is_empty(), "an error must say something");
        }
    }
    #[test]
    fn the_two_prefix_spellings_of_micro_agree() {
        // `crate::expression::ArithmaSIPrefix` and this module's table are two
        // views of the same concept, and they used to disagree: the enum
        // returned the micro sign while this table uses "u". That made
        // `si_prefix_power(Micro.symbol())` return None, so a caller round-
        // tripping through the enum silently lost the prefix.
        for variant in [
            crate::expression::ArithmaSIPrefix::Micro,
            crate::expression::ArithmaSIPrefix::Kilo,
            crate::expression::ArithmaSIPrefix::Milli,
            crate::expression::ArithmaSIPrefix::Nano,
        ] {
            let symbol = variant.symbol();
            assert!(symbol.is_ascii(), "prefix symbol {symbol:?} must be ASCII");
            assert!(
                si_prefix_power(symbol).is_some(),
                "{symbol:?} from the expression enum must resolve in this table"
            );
        }
    }
}
