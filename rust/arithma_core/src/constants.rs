//====== Arithma/rust/arithma_core/src/constants.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Constants
//!
//! Global registry of symbolic constants (π, e, c, h, …). Mirrors
//! `pt_arithmos::pt_constants` and embeds `default_constants.json` at compile
//! time via `include_str!` so the binary is self-contained for PyPI shipping.
//!
//! Per CLAUDE.md §11 (Constants Management):
//! - Mathematical constants live in `default_constants.json`.
//! - Domain-specific constants get their own JSON and are loaded via
//!   [`load_constants_from_json`].
//! - Access constants via [`lookup_symbol`] — no magic numbers.

use once_cell::sync::Lazy;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::expression::ArithmaExpression;

/// `default_constants.json`, embedded into the binary at compile time. The
/// engine and PyPI consumers never need to ship the JSON separately.
pub const DEFAULT_CONSTANTS_JSON: &str = include_str!("default_constants.json");

/// JSON shape used by `default_constants.json`. Mirrors `PTConstantDef`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArithmaConstantDef {
    /// Friendly name (e.g. "Pi"). Optional.
    #[serde(default)]
    pub name: Option<String>,
    /// Symbol used in expressions (e.g. "π"). Required.
    pub symbol: String,
    /// Optional symbolic expression form.
    #[serde(default)]
    pub expression: Option<serde_json::Value>,
    /// Optional pre-computed f64.
    #[serde(default)]
    pub cached_value: Option<f64>,
    /// Whether the simplifier may collapse this constant to its cached value.
    #[serde(default)]
    pub allow_simplification: bool,
    /// Optional unit string.
    #[serde(default)]
    pub unit: Option<String>,
    /// Optional SI prefix.
    #[serde(default)]
    pub prefix: Option<String>,
    /// Whether the constant is enabled by default.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Whether to expand the constant to its expression form when looked up.
    #[serde(default)]
    pub use_expression: bool,
}

fn default_true() -> bool {
    true
}

/// Global symbol registry.
///
/// Constants and variables both live here; lookup-key uniqueness is enforced
/// at registration time. Access goes through the helper functions in this
/// module — never via `SYMBOL_REGISTRY.write()` directly from outside Arithma.
pub static SYMBOL_REGISTRY: Lazy<RwLock<HashMap<String, ArithmaExpression>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

/// User-tunable enable flags. Maps symbol → `true|false`. Off-by-default
/// constants are skipped during symbol resolution.
#[derive(Debug, Clone, Default)]
pub struct ArithmaConstantConfig {
    /// Per-symbol enable map.
    pub enabled_constants: HashMap<String, bool>,
    /// Default for unknown symbols.
    pub enable_all_by_default: bool,
}

/// Façade type re-exported as `arithma_core::ArithmaConstants` — a static-only
/// service for downstream code that prefers method calls over free functions.
pub struct ArithmaConstants;

impl ArithmaConstants {
    /// Look up a symbol in the registry. Equivalent to [`lookup_symbol`].
    pub fn lookup(symbol: &str) -> Option<ArithmaExpression> {
        lookup_symbol(symbol)
    }

    /// Register a symbol. Errors if the symbol is already present.
    pub fn register(symbol: String, expr: ArithmaExpression) -> Result<(), String> {
        register_symbol(symbol, expr)
    }

    /// Initialise the registry from `default_constants.json`. Idempotent — safe
    /// to call multiple times. Returns the number of symbols registered.
    pub fn initialize_defaults() -> Result<usize, String> {
        load_constants_from_json(DEFAULT_CONSTANTS_JSON)
    }
}

/// ASCII spellings for the constants whose canonical symbol is not ASCII.
///
/// 21 of the 30 default constants are keyed by their mathematical glyph --
/// `pi`, `tau`, `phi`, `sqrt2`, `zeta3` and friends are stored as the actual
/// characters. That is correct for rendering, and hostile for lookup: a caller
/// has to produce the glyph to ask for the value, which is awkward from a
/// keyboard, fragile across source encodings, and impossible in an ASCII-only
/// identifier position.
///
/// This table adds a plain-letter spelling for each. The canonical glyph keeps
/// working -- nothing is renamed or removed; these are additional entry points.
///
/// Ordered pairs of `(ascii_alias, canonical_symbol)`. Kept as a sorted `const`
/// slice rather than a lazily-built map: it is small, fixed at compile time,
/// and avoids a heap allocation on the lookup path (safety-critical standard 3,
/// no heap allocation after initialisation).
const ASCII_ALIASES: &[(&str, &str)] = &[
    ("alphaF", "\u{3b1}_F"),
    ("conway", "\u{3bb}"),
    ("delta", "\u{3b4}"),
    ("digamma1", "\u{3c8}(1)"),
    ("euler_mascheroni", "\u{3b3}"),
    ("expPi", "eToPow\u{3c0}"),
    ("feigenbaum_alpha", "\u{3b1}_F"),
    ("feigenbaum_delta", "\u{3b4}"),
    ("gamma", "\u{3b3}"),
    ("golden_ratio", "\u{3c6}"),
    ("lambda", "\u{3bb}"),
    ("phi", "\u{3c6}"),
    ("pi", "\u{3c0}"),
    ("pi_fourThirds", "\u{3c0}fourThirds"),
    ("pi_half", "\u{3c0}half"),
    ("pi_quarter", "\u{3c0}quarter"),
    ("pi_squaredHalf", "\u{3c0}SquaredHalf"),
    ("pi_third", "\u{3c0}third"),
    ("plastic", "\u{3c1}"),
    ("psi1", "\u{3c8}(1)"),
    ("rho", "\u{3c1}"),
    ("salem", "\u{3c3}"),
    ("sigma", "\u{3c3}"),
    ("sqrt2", "\u{221a}2"),
    ("sqrt3", "\u{221a}3"),
    ("sqrt5", "\u{221a}5"),
    ("sqrtPi", "\u{221a}\u{3c0}"),
    ("tau", "\u{3c4}"),
    ("zeta3", "\u{3b6}(3)"),
];

/// Resolve an ASCII alias to its canonical symbol, if one exists.
///
/// Linear scan over a fixed 29-entry table: bounded by construction
/// (safety-critical standard 2), no allocation, and faster than hashing for
/// this size.
pub fn resolve_alias(symbol: &str) -> Option<&'static str> {
    debug_assert!(!ASCII_ALIASES.is_empty(), "alias table must not be empty");
    for (alias, canonical) in ASCII_ALIASES {
        if *alias == symbol {
            debug_assert!(!canonical.is_empty(), "alias must map to a real symbol");
            return Some(canonical);
        }
    }
    None
}

/// Every ASCII alias, for discovery and documentation.
pub fn ascii_aliases() -> &'static [(&'static str, &'static str)] {
    ASCII_ALIASES
}

/// Look up a symbol in the global registry.
///
/// Tries the symbol exactly as given, then falls back to the ASCII alias table
/// so `lookup_symbol("pi")` finds the constant stored under its glyph. Exact
/// matches always win, so a caller who registers their own `pi` shadows the
/// alias rather than being silently overridden by it.
pub fn lookup_symbol(symbol: &str) -> Option<ArithmaExpression> {
    debug_assert!(
        !symbol.is_empty(),
        "lookup_symbol requires a non-empty symbol"
    );
    let registry = SYMBOL_REGISTRY.read();
    if let Some(found) = registry.get(symbol) {
        return Some(found.clone());
    }
    let canonical = resolve_alias(symbol)?;
    debug_assert_ne!(canonical, symbol, "alias must differ from its input");
    registry.get(canonical).cloned()
}

/// Register a symbol. Errors if the symbol is already present (use
/// [`reregister_symbol`] for hot-reload paths that intentionally overwrite).
pub fn register_symbol(symbol: String, expr: ArithmaExpression) -> Result<(), String> {
    let mut registry = SYMBOL_REGISTRY.write();
    if registry.contains_key(&symbol) {
        return Err(format!("Symbol '{symbol}' is already registered"));
    }
    registry.insert(symbol, expr);
    Ok(())
}

/// Replace an existing symbol or insert a fresh one. Used by hot-reload.
pub fn reregister_symbol(symbol: String, expr: ArithmaExpression) {
    SYMBOL_REGISTRY.write().insert(symbol, expr);
}

/// Number of currently-registered symbols.
pub fn registered_count() -> usize {
    SYMBOL_REGISTRY.read().len()
}

/// Strip leading `//`-style comment lines from a JSON-with-comments string
/// so the strict JSON parser accepts it. The PlayTow datasheet convention
/// (carried over from pt-arithmos) prepends a copyright banner to every
/// shipped JSON file; rather than maintain a JSONC parser, we strip the
/// banner here. Bounded by line count for safety-critical §2.
fn strip_jsonc_header(jsonc: &str) -> String {
    let mut out = String::with_capacity(jsonc.len());
    let mut header_done = false;
    for line in jsonc.lines() {
        if header_done {
            out.push_str(line);
            out.push('\n');
            continue;
        }
        let trimmed = line.trim_start();
        if trimmed.starts_with("//") || trimmed.is_empty() {
            // still in the banner
            continue;
        }
        header_done = true;
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// Load constants from a JSON string into [`SYMBOL_REGISTRY`].
///
/// The document is an array of [`ArithmaConstantDef`]. Entries with
/// `enabled: false` are skipped. Registration uses [`reregister_symbol`] so the
/// call is idempotent — [`ArithmaConstants::initialize_defaults`] documents
/// itself as safe to call repeatedly, which a duplicate-key error would break.
///
/// Returns the number of symbols registered.
pub fn load_constants_from_json(json: &str) -> Result<usize, String> {
    let cleaned = strip_jsonc_header(json);
    let defs: Vec<ArithmaConstantDef> = serde_json::from_str(&cleaned)
        .map_err(|e| format!("Failed to parse constants JSON: {e}"))?;

    let mut registered = 0_usize;
    for def in defs {
        if !def.enabled {
            continue;
        }
        if def.symbol.is_empty() {
            return Err("constant entry has an empty symbol".to_string());
        }
        // A constant with neither a cached value nor an expression cannot be
        // evaluated, and silently registering it would resurrect the class of
        // bug this function is fixing.
        if def.cached_value.is_none() && def.expression.is_none() {
            return Err(format!(
                "constant '{}' has neither cached_value nor expression",
                def.symbol
            ));
        }
        let expr = ArithmaExpression::Constant {
            name: def.name.clone(),
            symbol: def.symbol.clone(),
            cached_value: def.cached_value,
            allow_simplification: def.allow_simplification,
            unit: def.unit.clone(),
            prefix: None,
        };
        reregister_symbol(def.symbol, expr);
        registered += 1;
    }
    Ok(registered)
}

// ---------------------------------------------------------------------------
// Backward-compatibility aliases for the pre-rename `Arithmos*` names.
// Retained for one release; downstream (eml-math, eml-spectral, metaphysica,
// periodica) should migrate to the `Arithma*` names above.
// ---------------------------------------------------------------------------
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaConstantConfig`")]
#[allow(unused)]
pub use self::ArithmaConstantConfig as ArithmosConstantConfig;
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaConstantDef`")]
#[allow(unused)]
pub use self::ArithmaConstantDef as ArithmosConstantDef;
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaConstants`")]
#[allow(unused)]
pub use self::ArithmaConstants as ArithmosConstants;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_constants_json_is_embedded() {
        assert!(!DEFAULT_CONSTANTS_JSON.is_empty());
    }

    #[test]
    fn default_constants_json_parses() {
        load_constants_from_json(DEFAULT_CONSTANTS_JSON)
            .expect("default_constants.json must parse cleanly");
    }

    #[test]
    fn lookup_unknown_returns_none() {
        assert!(lookup_symbol("__definitely_not_registered__").is_none());
    }

    #[test]
    fn defaults_actually_register() {
        // The regression this guards: load_constants_from_json used to parse
        // the document and register nothing, leaving SYMBOL_REGISTRY empty so
        // every constant lookup returned None.
        let n = ArithmaConstants::initialize_defaults().expect("defaults must load");
        assert!(n >= 30, "expected the full catalogue, registered {n}");
        assert!(registered_count() >= n);
    }

    #[test]
    fn pi_resolves_to_its_value() {
        ArithmaConstants::initialize_defaults().expect("defaults must load");
        let pi = lookup_symbol("\u{3c0}").expect("π must be registered");
        let v = pi.to_f64().expect("π must carry a cached value");
        assert!(
            (v - std::f64::consts::PI).abs() < 1e-12,
            "π resolved to {v}"
        );
    }

    #[test]
    fn e_and_phi_resolve() {
        ArithmaConstants::initialize_defaults().expect("defaults must load");
        let e = lookup_symbol("e").expect("e must be registered");
        assert!((e.to_f64().unwrap() - std::f64::consts::E).abs() < 1e-12);
        // φ carries use_expression: true but still ships a cached value.
        let phi = lookup_symbol("\u{3c6}").expect("φ must be registered");
        assert!((phi.to_f64().unwrap() - 1.618_033_988_749_895).abs() < 1e-12);
    }

    #[test]
    fn initialize_defaults_is_idempotent() {
        let first = ArithmaConstants::initialize_defaults().expect("first load");
        let second = ArithmaConstants::initialize_defaults().expect("second load");
        assert_eq!(
            first, second,
            "reloading must not error or change the count"
        );
    }

    #[test]
    fn entry_without_value_or_expression_is_rejected() {
        let bad = r#"[{"symbol":"zz","enabled":true}]"#;
        assert!(load_constants_from_json(bad).is_err());
    }
    // ─── ASCII aliases ─────────────────────────────────────────────────────

    #[test]
    fn every_alias_target_is_a_real_default_constant() {
        // Guards against a typo in the alias table silently producing a
        // lookup that always returns None.
        let loaded = ArithmaConstants::initialize_defaults().expect("defaults must load");
        assert!(loaded > 0, "no default constants were registered");
        for (alias, canonical) in ascii_aliases() {
            assert!(
                lookup_symbol(canonical).is_some(),
                "alias {alias:?} points at {canonical:?}, which is not registered"
            );
        }
    }

    #[test]
    fn ascii_aliases_resolve_to_the_same_value_as_the_glyph() {
        let _ = ArithmaConstants::initialize_defaults().expect("defaults must load");
        let via_alias = lookup_symbol("pi").expect("pi must resolve via alias");
        let via_glyph = lookup_symbol("\u{3c0}").expect("glyph must resolve");
        assert_eq!(
            via_alias.to_f64(),
            via_glyph.to_f64(),
            "alias and glyph must yield the same constant"
        );
    }

    #[test]
    fn the_common_ascii_spellings_all_resolve() {
        let _ = ArithmaConstants::initialize_defaults().expect("defaults must load");
        for name in [
            "pi", "tau", "phi", "gamma", "sqrt2", "sqrt3", "sqrt5", "zeta3",
        ] {
            assert!(
                lookup_symbol(name).is_some(),
                "{name} should be reachable without typing a glyph"
            );
        }
    }

    /// Serialises the tests that mutate the process-wide `SYMBOL_REGISTRY`.
    /// `cargo test` runs tests in parallel, so a test that shadows a real
    /// symbol would otherwise be visible to any test looking that symbol up.
    static REGISTRY_TEST_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

    #[test]
    fn an_exact_match_beats_an_alias() {
        let _guard = REGISTRY_TEST_LOCK.lock();
        // A caller registering their own `pi` must shadow the alias, not be
        // silently overridden by it.
        let _ = ArithmaConstants::initialize_defaults().expect("defaults must load");
        reregister_symbol("pi".to_string(), ArithmaExpression::from_f64(42.0));
        let got = lookup_symbol("pi").expect("exact entry must win");
        assert_eq!(got.to_f64(), Some(42.0));
        // Leave the registry as we found it for other tests.
        SYMBOL_REGISTRY.write().remove("pi");
    }

    #[test]
    fn alias_table_is_sorted_and_has_no_duplicate_aliases() {
        let table = ascii_aliases();
        for pair in table.windows(2) {
            assert!(
                pair[0].0 < pair[1].0,
                "alias table must stay sorted and unique: {:?} then {:?}",
                pair[0].0,
                pair[1].0
            );
        }
    }

    #[test]
    fn an_unknown_alias_resolves_to_nothing() {
        assert!(resolve_alias("definitely_not_a_constant").is_none());
        assert!(lookup_symbol("definitely_not_a_constant").is_none());
    }
}
