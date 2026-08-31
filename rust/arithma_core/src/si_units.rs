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

use crate::unit::ArithmosUnit;

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
    #[serde(default)]
    derived_units: Vec<SiUnitDef>,
}

#[derive(Debug, Deserialize)]
struct SiUnitDef {
    symbol: String,
    name: String,
}

/// Lazy-built map of symbol → [`ArithmosUnit`], parsed from the embedded JSON.
///
/// A malformed catalogue is a build-time authoring error, not a runtime
/// condition — but panicking inside a `Lazy` would poison every later lookup,
/// so a parse failure yields an empty registry and `len() == 0` is the signal.
/// `si_units_parse_is_ok()` asserts the shipped file parses.
static REGISTRY: Lazy<HashMap<String, ArithmosUnit>> = Lazy::new(|| {
    let mut map = HashMap::new();
    if let Ok(doc) = serde_json::from_str::<SiUnitsDoc>(SI_UNITS_JSON) {
        for def in doc
            .si_units
            .base_units
            .into_iter()
            .chain(doc.si_units.derived_units)
        {
            map.insert(def.symbol.clone(), ArithmosUnit::new(def.symbol, def.name));
        }
    }
    map
});

/// Public-facing SI-units registry.
pub struct ArithmosSIUnits;

impl ArithmosSIUnits {
    /// Try to find a unit by SI symbol.
    pub fn lookup(symbol: &str) -> Option<&'static ArithmosUnit> {
        REGISTRY.get(symbol)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_json_is_non_empty() {
        assert!(!ArithmosSIUnits::embedded_json().is_empty());
    }

    #[test]
    fn lookup_unknown_returns_none() {
        assert!(ArithmosSIUnits::lookup("zzz_unknown").is_none());
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
        assert!(ArithmosSIUnits::len() > 0, "SI registry must not be empty");
    }

    #[test]
    fn base_units_resolve() {
        for (sym, name) in [("m", "meter"), ("kg", "kilogram"), ("s", "second")] {
            let u =
                ArithmosSIUnits::lookup(sym).unwrap_or_else(|| panic!("base unit '{sym}' missing"));
            assert_eq!(u.symbol, sym);
            assert_eq!(u.name, name);
        }
    }
}
