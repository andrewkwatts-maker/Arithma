//====== Arithma/rust/arithma_core/src/pyfacade/units.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! PyO3 facade for the unit-of-measure domain: `ArithmaUnit` and the
//! `ArithmaSIUnits` registry.
//!
//! ## Scope -- read this before adding to the file
//!
//! This note used to say the opposite of what is true, and it is worth keeping
//! the correction visible. It described `ArithmaUnit` as *"a Wave-2 placeholder
//! carrying two strings"* with *"no dimension vector, no scale factor, no
//! dimensional-consistency check, and no conversion between units"*, and said
//! this facade deliberately exposed none of those.
//!
//! All of it now exists. [`crate::unit::ArithmaDimension`] is a seven-exponent
//! vector over the SI base dimensions with `multiply`, `divide`, `powi`,
//! `nth_root`, `is_compatible_with` and a table of derived symbols;
//! `ArithmaUnit` has `dimension()`, `scale_to_base()`, `quantity()`,
//! `is_compatible_with()` and `convert()`; and there are 24 SI prefixes with
//! `split_unit_symbol` to separate a prefix from its base symbol. The facade
//! exposes all of them -- `Unit.dimension`, `Unit.dimension_exponents`,
//! `Unit.quantity`, `Unit.scale_to_base`, `Unit.is_compatible_with`,
//! `Unit.split_symbol`, and the module-level `convert`, `dimension_of`,
//! `base_dimensions`, `prefix_power` and `si_prefixes`.
//!
//! The original judgement was right and stays: a wrapper that makes an absent
//! implementation look present is worse than no wrapper. The failure mode this
//! file demonstrates is the mirror of it -- a scope note that goes on saying
//! something is absent after it lands. Nothing fails; the reader simply never
//! looks for the method, and a docstring is read at precisely the moment
//! someone is deciding whether what they want exists.
//!
//! ## The registry only contains the seven base units
//!
//! `si_units.rs` parses `si_units.json` into `base_units` and `derived_units`
//! lists. The shipped JSON has no `derived_units` key at all -- it has a
//! `derived_units_info` note saying derived units "should be represented as
//! PTExpressions". Because that field is `#[serde(default)]`, the
//! `derived_units` half of the parse silently yields an empty vector, so the
//! live registry holds exactly the seven SI base units: `m`, `kg`, `s`, `A`,
//! `K`, `mol`, `cd`. [`si_lookup`] returning `None` for `N` or `Hz` is the
//! catalogue being thin, not a bug in this binding.
//!
//! The JSON `quantity` and `definition` fields are also dropped: `SiUnitDef`
//! deserialises only `symbol` and `name`, so no dimensional metadata reaches
//! `ArithmaUnit` even though the data file carries it. [`si_units_json`]
//! exposes the raw catalogue for callers that need those fields today.
//!
//! ## Failure modes surfaced as exceptions
//!
//! A malformed catalogue does not panic in `si_units.rs`: the `Lazy` block
//! swallows the `serde_json` error and leaves an empty `HashMap`, with
//! `len() == 0` as the only signal. Silently returning `0` or an empty list
//! from Python would hide a broken build, so [`si_unit_count`] and
//! [`si_base_units`] both raise `RuntimeError` in that state.

// `#[pymethods]` / `#[pyfunction]` on pyo3 0.22 expand `-> PyResult<T>` into a
// `PyErr::from` round-trip that clippy reads as a no-op conversion in *our*
// signature span. `pyfacade::core` already carries these warnings; scoping the
// allow to this file keeps the new surface from adding more without editing the
// crate-wide lint policy in `lib.rs`. Matches `calculus.rs` / `geometry.rs`.
#![allow(clippy::useless_conversion)]

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::pyfacade::MAX_SEQUENCE_LEN;
use crate::si_units::ArithmaSIUnits;
use crate::unit::{ArithmaDimension, ArithmaUnit, BASE_DIMENSION_COUNT, BASE_SYMBOLS};

/// The seven SI base units, in the order the BIPM lists them. Fixed-size array
/// so every loop over it has a compile-time bound (CLAUDE.md standard 3).
const SI_BASE_SYMBOLS: [&str; 7] = ["m", "kg", "s", "A", "K", "mol", "cd"];

/// Longest unit symbol or name accepted from Python. Far above any real unit
/// string, but it bounds every character scan below.
const MAX_UNIT_TEXT_LEN: usize = 256;

// ============================================================================
// Helpers.
// ============================================================================

/// Validate a unit symbol. A symbol is a compact token (`m`, `kg`, `mol`), so
/// empty strings and embedded whitespace are rejected: both would produce a
/// registry key that could never be looked up again.
fn validate_symbol(symbol: &str) -> PyResult<()> {
    if symbol.is_empty() {
        return Err(PyValueError::new_err("unit symbol must not be empty"));
    }
    if symbol.len() > MAX_UNIT_TEXT_LEN {
        return Err(PyValueError::new_err(format!(
            "unit symbol of {} bytes exceeds the {}-byte limit",
            symbol.len(),
            MAX_UNIT_TEXT_LEN
        )));
    }
    assert!(
        symbol.len() <= MAX_SEQUENCE_LEN,
        "symbol length bounded above, so the scan below is bounded"
    );
    if symbol.chars().any(char::is_whitespace) {
        return Err(PyValueError::new_err(format!(
            "unit symbol {symbol:?} must not contain whitespace"
        )));
    }
    Ok(())
}

/// Validate a long-form unit name. Unlike a symbol, a name may contain spaces
/// ("degree Celsius"), so only emptiness and length are constrained.
fn validate_name(name: &str) -> PyResult<()> {
    if name.trim().is_empty() {
        return Err(PyValueError::new_err(
            "unit name must not be empty or whitespace",
        ));
    }
    if name.len() > MAX_UNIT_TEXT_LEN {
        return Err(PyValueError::new_err(format!(
            "unit name of {} bytes exceeds the {}-byte limit",
            name.len(),
            MAX_UNIT_TEXT_LEN
        )));
    }
    assert!(
        name.len() <= MAX_SEQUENCE_LEN,
        "name length bounded above by MAX_UNIT_TEXT_LEN"
    );
    Ok(())
}

/// Number of units currently in the registry, or an error describing the
/// empty-registry failure mode. Shared by [`si_unit_count`] and
/// [`si_base_units`] so both report the same diagnosis.
fn registry_len_checked() -> PyResult<usize> {
    let count = ArithmaSIUnits::len();
    if count == 0 {
        return Err(PyRuntimeError::new_err(
            "SI registry is empty: the embedded si_units.json failed to parse. \
             arithma_core::si_units leaves an empty map rather than panicking, \
             so this is a build-time authoring error in the catalogue",
        ));
    }
    assert!(count > 0, "zero-count case returned immediately above");
    Ok(count)
}

// ============================================================================
// `Unit` pyclass.
// ============================================================================

/// A unit of measure -- Python wrapper around `ArithmaUnit`.
///
/// Carries an SI `symbol` and a long-form `name`, and resolves both against the
/// dimensional tables: `dimension()` gives the derived symbol (`m*kg*s^-2`),
/// `dimension_exponents()` the raw seven-vector, `quantity()` the named
/// quantity, `scale_to_base()` the factor to SI base units, and
/// `is_compatible_with()` whether two units measure the same thing.
/// `split_symbol()` separates an SI prefix from its base symbol, so `km` is
/// `(3, "m")`.
///
/// Those resolve through a lookup, so they return `None` for a symbol the
/// tables do not carry rather than guessing. Equality and hashing follow the
/// Rust `PartialEq` / `Hash` derives: both fields must match, so `Unit("m",
/// "meter")` and `Unit("m", "metre")` are different units.
#[pyclass(name = "Unit", module = "arithma")]
#[derive(Clone)]
pub struct Unit {
    pub(crate) inner: ArithmaUnit,
}

#[pymethods]
impl Unit {
    /// Construct from an SI symbol and a long-form name.
    ///
    /// Raises `ValueError` for an empty symbol or name, for a symbol containing
    /// whitespace, or for either string exceeding 256 bytes.
    #[new]
    fn new(symbol: &str, name: &str) -> PyResult<Self> {
        validate_symbol(symbol)?;
        validate_name(name)?;
        let inner = ArithmaUnit::new(symbol, name);
        assert_eq!(
            inner.symbol, symbol,
            "ArithmaUnit::new must keep the symbol"
        );
        assert_eq!(inner.name, name, "ArithmaUnit::new must keep the name");
        Ok(Self { inner })
    }

    /// SI symbol, e.g. `"kg"`.
    #[getter]
    fn symbol(&self) -> String {
        self.inner.symbol.clone()
    }

    /// Long-form name, e.g. `"kilogram"`.
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// True when this unit is one of the seven SI base units by symbol.
    ///
    /// A membership test against a fixed list -- it does not consult the
    /// registry and does not validate the accompanying name.
    fn is_base_unit(&self) -> bool {
        assert!(
            SI_BASE_SYMBOLS.len() == 7,
            "the SI defines exactly seven base units"
        );
        SI_BASE_SYMBOLS.contains(&self.inner.symbol.as_str())
    }

    /// Equality follows the Rust derive: symbol **and** name must match.
    /// Comparing against a non-`Unit` yields `False` rather than raising, which
    /// is what Python expects of `__eq__`.
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        match other.extract::<PyRef<Unit>>() {
            Ok(rhs) => self.inner == rhs.inner,
            Err(_) => false,
        }
    }

    /// Hash consistent with [`Unit::__eq__`]: both fields feed the hasher, via
    /// the same `Hash` derive that `ArithmaUnit` carries.
    fn __hash__(&self) -> isize {
        let mut hasher = DefaultHasher::new();
        self.inner.hash(&mut hasher);
        let digest = hasher.finish();
        assert!(
            std::mem::size_of::<isize>() <= std::mem::size_of::<u64>(),
            "isize must fit in the u64 digest on every supported target"
        );
        digest as isize
    }

    // -------- dimensional analysis --------
    //
    // `Unit` carried a symbol and a name and nothing else, so there was no way
    // to ask from Python whether two quantities could be added. These expose
    // `ArithmaDimension`.

    /// The physical dimension as a string, e.g. `"m*kg*s^-2"` for a newton.
    ///
    /// Returns `None` for a symbol this crate does not recognise. That is
    /// deliberately distinct from `"1"` (dimensionless): an unknown unit and a
    /// pure number are not the same answer.
    fn dimension(&self) -> Option<String> {
        self.inner.dimension().map(|d| d.to_string())
    }

    /// The exponent vector over the seven SI base dimensions, in the order
    /// `m, kg, s, A, K, mol, cd`.
    fn dimension_exponents(&self) -> Option<Vec<i8>> {
        self.inner.dimension().map(|d| d.exponents().to_vec())
    }

    /// The quantity this unit measures (`"force"`, `"length"`), if known.
    fn quantity(&self) -> Option<String> {
        self.inner.quantity().map(str::to_string)
    }

    /// True when this unit is dimensionless.
    ///
    /// Raises `ValueError` for an unrecognised symbol rather than answering
    /// `False`, which a caller would read as "known, and not dimensionless".
    fn is_dimensionless(&self) -> PyResult<bool> {
        let d = self.inner.dimension().ok_or_else(|| {
            PyValueError::new_err(format!("unknown unit symbol {:?}", self.inner.symbol))
        })?;
        Ok(d.is_dimensionless())
    }

    /// Whether quantities in these two units may be added or compared.
    ///
    /// Returns `None` when either symbol is unrecognised -- "I cannot tell"
    /// must not be reported as "incompatible".
    fn is_compatible_with(&self, other: &Self) -> Option<bool> {
        self.inner.is_compatible_with(&other.inner)
    }

    // -------- prefixes and conversion --------

    /// The factor from this unit to its coherent SI base unit.
    ///
    /// `km` gives `1000.0`, `mm` gives `0.001`, `m` gives `1.0`. `None` for a
    /// symbol this crate does not recognise.
    fn scale_to_base(&self) -> Option<f64> {
        self.inner.scale_to_base()
    }

    /// Split this unit into `(prefix_power_of_ten, base_symbol)`.
    ///
    /// `km` gives `(3, "m")`. `kg` gives `(0, "kg")` -- it is the SI base unit
    /// for mass, not kilo-grams, and a whole symbol that is itself a unit
    /// always beats a prefix reading.
    fn split_symbol(&self) -> Option<(i32, String)> {
        crate::unit::split_unit_symbol(&self.inner.symbol)
            .map(|(power, base)| (power, base.to_string()))
    }

    fn __repr__(&self) -> String {
        format!("Unit({:?}, {:?})", self.inner.symbol, self.inner.name)
    }

    fn __str__(&self) -> String {
        self.inner.symbol.clone()
    }
}

// ============================================================================
// SI registry functions.
// ============================================================================

/// Look up a unit by SI symbol, returning `None` when it is not in the
/// catalogue. Wraps `ArithmaSIUnits::lookup`.
///
/// The shipped catalogue holds only the seven base units (see the module
/// docstring), so derived symbols such as `"N"`, `"J"`, or `"Hz"` return
/// `None`. That is the state of `si_units.json`, not a lookup failure.
#[pyfunction]
fn si_lookup(symbol: &str) -> PyResult<Option<Unit>> {
    validate_symbol(symbol)?;
    let found = match ArithmaSIUnits::lookup(symbol) {
        Some(unit) => unit,
        None => return Ok(None),
    };
    assert_eq!(
        found.symbol, symbol,
        "registry key must equal the symbol of the unit it stores"
    );
    Ok(Some(Unit {
        inner: found.clone(),
    }))
}

/// Number of units in the SI registry.
///
/// Raises `RuntimeError` when the registry is empty, because `si_units.rs`
/// treats a parse failure as an empty map rather than a panic and returning `0`
/// would disguise a broken build as an ordinary answer.
#[pyfunction]
fn si_unit_count() -> PyResult<usize> {
    let count = registry_len_checked()?;
    assert!(
        count >= SI_BASE_SYMBOLS.len(),
        "base units must all be present"
    );
    Ok(count)
}

/// The seven SI base units, resolved through the registry.
///
/// Raises `RuntimeError` if the registry is empty or missing any base unit --
/// either means the embedded catalogue did not parse or was edited badly.
#[pyfunction]
fn si_base_units() -> PyResult<Vec<Unit>> {
    let _ = registry_len_checked()?;
    let mut out: Vec<Unit> = Vec::with_capacity(SI_BASE_SYMBOLS.len());
    for symbol in SI_BASE_SYMBOLS {
        let found = ArithmaSIUnits::lookup(symbol).ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "SI registry is missing base unit {:?} ({} units loaded)",
                symbol,
                ArithmaSIUnits::len()
            ))
        })?;
        assert_eq!(found.symbol, symbol, "registry key must match its unit");
        out.push(Unit {
            inner: found.clone(),
        });
    }
    assert_eq!(out.len(), SI_BASE_SYMBOLS.len(), "bounded loop fills all 7");
    Ok(out)
}

/// The raw embedded `si_units.json` source. Wraps
/// `ArithmaSIUnits::embedded_json`.
///
/// Useful because the parsed registry drops the `quantity` and `definition`
/// fields the catalogue carries: a caller that needs them can parse this string
/// in Python until `ArithmaUnit` grows the fields itself.
#[pyfunction]
fn si_units_json() -> PyResult<&'static str> {
    let json = ArithmaSIUnits::embedded_json();
    if json.is_empty() {
        return Err(PyRuntimeError::new_err(
            "embedded si_units.json is empty; the crate was built from a broken tree",
        ));
    }
    assert!(!json.is_empty(), "empty case returned immediately above");
    Ok(json)
}

// ============================================================================
// Registration
// ============================================================================

/// The dimension of a unit symbol, base or derived, as a string.
///
/// This is the bridge over the SI catalogue's deliberate base-units-only
/// scope: `si_lookup("N")` is `None`, but `N` is still a well-defined
/// dimension and resolves here.
#[pyfunction]
fn dimension_of(symbol: &str) -> PyResult<Option<String>> {
    validate_symbol(symbol)?;
    Ok(ArithmaDimension::of_unit(symbol).map(|d| d.to_string()))
}

/// The standard derived-unit symbol for a dimension given as an exponent
/// vector over `m, kg, s, A, K, mol, cd`.
///
/// `dimension_symbol([1, 1, -2, 0, 0, 0, 0])` is `"N"`.
#[pyfunction]
fn dimension_symbol(exponents: Vec<i8>) -> PyResult<Option<String>> {
    if exponents.len() != BASE_DIMENSION_COUNT {
        return Err(PyValueError::new_err(format!(
            "expected {BASE_DIMENSION_COUNT} exponents, got {}",
            exponents.len()
        )));
    }
    let mut fixed = [0i8; BASE_DIMENSION_COUNT];
    fixed.copy_from_slice(&exponents);
    let dimension = ArithmaDimension::from_exponents(fixed)
        .ok_or_else(|| PyValueError::new_err("an exponent exceeds the supported range"))?;
    Ok(dimension.derived_symbol().map(str::to_string))
}

/// The seven SI base dimension symbols, in exponent-vector order.
#[pyfunction]
fn base_dimensions() -> Vec<String> {
    BASE_SYMBOLS.iter().map(|s| (*s).to_string()).collect()
}

/// Register this module's surface on the extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Unit>()?;
    m.add_function(wrap_pyfunction!(si_lookup, m)?)?;
    m.add_function(wrap_pyfunction!(si_unit_count, m)?)?;
    m.add_function(wrap_pyfunction!(si_base_units, m)?)?;
    m.add_function(wrap_pyfunction!(si_units_json, m)?)?;
    m.add_function(wrap_pyfunction!(dimension_of, m)?)?;
    m.add_function(wrap_pyfunction!(dimension_symbol, m)?)?;
    m.add_function(wrap_pyfunction!(base_dimensions, m)?)?;
    m.add_function(wrap_pyfunction!(convert, m)?)?;
    m.add_function(wrap_pyfunction!(prefix_power, m)?)?;
    m.add_function(wrap_pyfunction!(si_prefixes, m)?)?;
    m.add_function(wrap_pyfunction!(split_unit_symbol, m)?)?;
    Ok(())
}

/// Convert a magnitude between two units.
///
/// Raises `ValueError` when the dimensions differ -- metres do not convert to
/// seconds, and refusing is the whole point of tracking dimensions -- when a
/// symbol is unrecognised, or when the result would not be finite.
///
/// Exact where it can be: the net power of ten is applied as a single
/// multiply or divide, so `convert(1.0, "km", "m")` is exactly `1000.0` and
/// converting back is exactly `1.0`.
#[pyfunction]
fn convert(value: f64, from_unit: &str, to_unit: &str) -> PyResult<f64> {
    let from = validated_unit(from_unit)?;
    let to = validated_unit(to_unit)?;
    debug_assert!(!from.symbol.is_empty(), "empty source symbol");
    debug_assert!(!to.symbol.is_empty(), "empty target symbol");
    ArithmaUnit::convert(value, &from, &to).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// The power of ten for an SI prefix symbol, or `None` if it is not one.
///
/// Micro is `"u"`, not the micro sign: this project's symbols are ASCII.
#[pyfunction]
fn prefix_power(symbol: &str) -> Option<i32> {
    crate::unit::si_prefix_power(symbol)
}

/// Every SI prefix as `(symbol, name, power_of_ten)`.
#[pyfunction]
fn si_prefixes() -> Vec<(String, String, i32)> {
    crate::unit::SI_PREFIXES
        .iter()
        .map(|(symbol, name, power, _)| ((*symbol).to_string(), (*name).to_string(), *power))
        .collect()
}

/// Split a unit symbol into `(prefix_power_of_ten, base_symbol)`.
#[pyfunction]
fn split_unit_symbol(symbol: &str) -> PyResult<Option<(i32, String)>> {
    validate_symbol(symbol)?;
    Ok(crate::unit::split_unit_symbol(symbol).map(|(power, base)| (power, base.to_string())))
}

/// Build a validated `ArithmaUnit` from a caller-supplied symbol.
fn validated_unit(symbol: &str) -> PyResult<ArithmaUnit> {
    validate_symbol(symbol)?;
    debug_assert!(
        !symbol.is_empty(),
        "validate_symbol let an empty symbol pass"
    );
    debug_assert!(symbol.len() <= MAX_UNIT_TEXT_LEN, "symbol length unchecked");
    Ok(ArithmaUnit::new(symbol, symbol))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_symbol_accepts_base_symbols() {
        for symbol in SI_BASE_SYMBOLS {
            assert!(validate_symbol(symbol).is_ok(), "{symbol} must validate");
        }
    }

    #[test]
    fn validate_symbol_rejects_empty_and_whitespace() {
        assert!(validate_symbol("").is_err());
        assert!(validate_symbol("k g").is_err());
        assert!(validate_symbol("m\t").is_err());
    }

    #[test]
    fn validate_symbol_rejects_oversize() {
        let long = "m".repeat(MAX_UNIT_TEXT_LEN + 1);
        assert!(validate_symbol(&long).is_err());
    }

    #[test]
    fn validate_name_allows_spaces_but_not_blank() {
        assert!(validate_name("degree Celsius").is_ok());
        assert!(validate_name("").is_err());
        assert!(validate_name("   ").is_err());
    }

    #[test]
    fn unit_new_round_trips_fields() {
        let u = Unit::new("m", "meter").expect("valid unit");
        assert_eq!(u.symbol(), "m");
        assert_eq!(u.name(), "meter");
        assert!(u.is_base_unit());
    }

    #[test]
    fn unit_new_rejects_bad_input() {
        assert!(Unit::new("", "meter").is_err());
        assert!(Unit::new("m", "").is_err());
        assert!(Unit::new("m s", "meter second").is_err());
    }

    #[test]
    fn non_base_symbol_is_not_a_base_unit() {
        let u = Unit::new("N", "newton").expect("valid unit");
        assert!(!u.is_base_unit());
    }

    #[test]
    fn si_lookup_resolves_every_base_unit() {
        for symbol in SI_BASE_SYMBOLS {
            let found = si_lookup(symbol)
                .expect("lookup must not error")
                .unwrap_or_else(|| panic!("base unit {symbol} must be registered"));
            assert_eq!(found.symbol(), symbol);
        }
    }

    #[test]
    fn si_lookup_rejects_an_invalid_symbol() {
        assert!(si_lookup("").is_err());
    }

    #[test]
    fn si_lookup_misses_return_none() {
        assert!(si_lookup("zzz_not_a_unit")
            .expect("lookup must not error")
            .is_none());
    }

    #[test]
    fn derived_units_are_absent_from_the_catalogue() {
        // Documents a live defect rather than a design choice: si_units.json
        // ships a `derived_units_info` note instead of the `derived_units`
        // array that `si_units.rs` deserialises, so the derived half of the
        // registry is always empty. Update this test when the catalogue grows
        // a real `derived_units` array.
        for symbol in ["N", "J", "Hz", "Pa", "W"] {
            assert!(
                si_lookup(symbol).expect("lookup must not error").is_none(),
                "{symbol} unexpectedly present -- the catalogue gained derived units"
            );
        }
    }

    #[test]
    fn si_base_units_returns_seven() {
        let units = si_base_units().expect("registry must be populated");
        assert_eq!(units.len(), 7);
        assert!(units.iter().all(Unit::is_base_unit));
    }

    #[test]
    fn si_unit_count_is_at_least_the_base_units() {
        assert!(si_unit_count().expect("registry must be populated") >= 7);
    }

    #[test]
    fn si_units_json_is_non_empty() {
        assert!(si_units_json()
            .expect("embedded json")
            .contains("base_units"));
    }

    #[test]
    fn equal_units_hash_equally() {
        let a = Unit::new("m", "meter").expect("valid unit");
        let b = Unit::new("m", "meter").expect("valid unit");
        let c = Unit::new("s", "second").expect("valid unit");
        assert_eq!(a.__hash__(), b.__hash__());
        assert!(a.inner == b.inner);
        assert!(a.inner != c.inner);
    }
}
