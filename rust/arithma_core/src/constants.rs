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

/// Look up a symbol in the global registry.
pub fn lookup_symbol(symbol: &str) -> Option<ArithmaExpression> {
    SYMBOL_REGISTRY.read().get(symbol).cloned()
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
}
