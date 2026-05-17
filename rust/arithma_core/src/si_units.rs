//====== Arithma/rust/arithma_core/src/si_units.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! SI-units registry. Loads `si_units.json` (embedded via `include_str!`) at
//! first access and exposes lookups. Wave-2 placeholder.

use once_cell::sync::Lazy;
use std::collections::HashMap;

use crate::unit::ArithmosUnit;

/// Embedded SI units catalogue. Read once, parsed lazily.
const SI_UNITS_JSON: &str = include_str!("../src/si_units.json");

/// Lazy-built map of "symbol" â†’ `ArithmosUnit`. The Wave-2 stub does not parse
/// the embedded JSON yet; it returns an empty map. Wave 3 wires up `serde_json`.
static REGISTRY: Lazy<HashMap<String, ArithmosUnit>> = Lazy::new(|| HashMap::new());

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
}

