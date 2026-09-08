//====== Arithma/rust/arithma_core/src/si_units.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! SI-units registry. Loads `si_units.json` (embedded via `include_str!`) at
//! first access and exposes lookups.

use once_cell::sync::Lazy;
use serde::Deserialize;
use std::collections::HashMap;

use crate::unit::ArithmaUnit;

/// Embedded SI units catalogue. Read once, parsed lazily.
const SI_UNITS_JSON: &str = include_str!("si_units.json");

/// JSON shape of `si_units.json`.
#[derive(Debug, Deserialize)]
struct SiUnitsDoc {
    si_units: SiUnitGroups,
}

#[derive(Debug, Deserialize)]
struct SiUnitGroups {
    #[serde(default)]
    base_units: Vec<SiUnitDef>,
    /// The catalogue's own design note. It is deserialised rather than ignored
    /// so that `si_units_scope_note()` can quote it back to a caller whose
    /// lookup of a derived unit came back empty.
    #[serde(default)]
    derived_units_info: Option<DerivedUnitsInfo>,
}

/// The `derived_units_info` block in `si_units.json`.
#[derive(Debug, Deserialize)]
struct DerivedUnitsInfo {
    #[serde(default)]
    note: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SiUnitDef {
    symbol: String,
    name: String,
}

/// Lazy-built map of symbol → [`ArithmaUnit`], parsed from the embedded JSON.
///
/// A malformed catalogue is a build-time authoring error, not a runtime
/// condition — but panicking inside a `Lazy` would poison every later lookup,
/// so a parse failure yields an empty registry and `len() == 0` is the signal.
/// `si_units_parse_is_ok()` asserts the shipped file parses.
static REGISTRY: Lazy<HashMap<String, ArithmaUnit>> = Lazy::new(|| {
    let mut map = HashMap::new();
    if let Ok(doc) = serde_json::from_str::<SiUnitsDoc>(SI_UNITS_JSON) {
        // Base units only, by design -- see `si_units_scope_note`.
        for def in doc.si_units.base_units.into_iter() {
            map.insert(def.symbol.clone(), ArithmaUnit::new(def.symbol, def.name));
        }
    }
    map
});

/// Public-facing SI-units registry.
pub struct ArithmaSIUnits;

impl ArithmaSIUnits {
    /// Try to find a unit by SI symbol.
    ///
    /// # Scope
    ///
    /// The catalogue holds the **seven SI base units only**. Derived units
    /// (`N`, `J`, `Hz`, `m/s`, ...) return `None` — that is deliberate, not a
    /// missing-data bug. `si_units.json` states the intent directly:
    ///
    /// > Derived units like m/s, kg*m/s^2, etc. should be represented as
    /// > expressions
    ///
    /// The loader previously also read a `derived_units` array that the
    /// catalogue has never contained. Because the field was
    /// `#[serde(default)]` the absence was silent, which made a deliberate
    /// design decision look like a data gap. That field is gone; use
    /// [`Self::scope_note`] to explain an empty result to a caller.
    pub fn lookup(symbol: &str) -> Option<&'static ArithmaUnit> {
        debug_assert!(!symbol.is_empty(), "lookup requires a non-empty symbol");
        debug_assert!(!REGISTRY.is_empty(), "SI registry failed to parse");
        REGISTRY.get(symbol)
    }

    /// The catalogue's own note on why derived units are absent.
    ///
    /// Returns `None` only if the embedded JSON is malformed, which
    /// `si_units_parse_is_ok` asserts against.
    pub fn scope_note() -> Option<String> {
        let doc = serde_json::from_str::<SiUnitsDoc>(SI_UNITS_JSON).ok()?;
        doc.si_units.derived_units_info?.note
    }

    /// Number of registered units.
    pub fn len() -> usize {
        REGISTRY.len()
    }

    /// Returns the embedded JSON source. Useful for testing and tooling.
    pub fn embedded_json() -> &'static str {
        SI_UNITS_JSON
    }
}

// ---------------------------------------------------------------------------
// Backward-compatibility aliases for the pre-rename `Arithmos*` names.
// Retained for one release; downstream (eml-math, eml-spectral, metaphysica,
// periodica) should migrate to the `Arithma*` names above.
// ---------------------------------------------------------------------------
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaSIUnits`")]
#[allow(unused)]
pub use self::ArithmaSIUnits as ArithmosSIUnits;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_json_is_non_empty() {
        assert!(!ArithmaSIUnits::embedded_json().is_empty());
    }

    #[test]
    fn lookup_unknown_returns_none() {
        assert!(ArithmaSIUnits::lookup("zzz_unknown").is_none());
    }

    #[test]
    fn si_units_parse_is_ok() {
        // Guards the silent-empty failure mode: a malformed catalogue yields an
        // empty REGISTRY rather than a panic, so assert the parse independently.
        serde_json::from_str::<SiUnitsDoc>(SI_UNITS_JSON)
            .expect("embedded si_units.json must parse");
    }

    #[test]
    fn registry_is_populated() {
        assert!(ArithmaSIUnits::len() > 0, "SI registry must not be empty");
    }

    #[test]
    fn base_units_resolve() {
        for (sym, name) in [("m", "meter"), ("kg", "kilogram"), ("s", "second")] {
            let u =
                ArithmaSIUnits::lookup(sym).unwrap_or_else(|| panic!("base unit '{sym}' missing"));
            assert_eq!(u.symbol, sym);
            assert_eq!(u.name, name);
        }
    }
    #[test]
    fn catalogue_holds_the_seven_base_units() {
        assert_eq!(
            ArithmaSIUnits::len(),
            7,
            "the SI catalogue is base units only, by design"
        );
        for sym in ["m", "kg", "s", "A", "K", "mol", "cd"] {
            assert!(
                ArithmaSIUnits::lookup(sym).is_some(),
                "base unit {sym} must be present"
            );
        }
    }

    #[test]
    fn derived_units_are_absent_by_design_and_the_catalogue_says_why() {
        // Not a data gap: si_units.json declares that derived units are meant
        // to be expressions. If someone later adds a derived-unit table, this
        // test should be updated deliberately rather than deleted.
        for sym in ["N", "J", "Hz", "W", "Pa"] {
            assert!(
                ArithmaSIUnits::lookup(sym).is_none(),
                "{sym} is a derived unit and is out of scope for the catalogue"
            );
        }
        let note = ArithmaSIUnits::scope_note().expect("catalogue must carry its scope note");
        assert!(
            note.to_lowercase().contains("derived"),
            "scope note should explain the derived-unit policy, got {note:?}"
        );
    }
}
