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
//! Global registry of symbolic constants (Ï€, e, c, h, â€¦). Mirrors
//! `pt_arithmos::pt_constants` and embeds `default_constants.json` at compile
//! time via `include_str!` so the binary is self-contained for PyPI shipping.
//!
//! Per CLAUDE.md Â§11 (Constants Management):
//! - Mathematical constants live in `default_constants.json`.
//! - Domain-specific constants get their own JSON and are loaded via
//!   [`load_constants_from_json`].
//! - Access constants via [`lookup_symbol`] â€” no magic numbers.

use once_cell::sync::Lazy;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::expression::ArithmosExpression;

/// `default_constants.json`, embedded into the binary at compile time. The
/// engine and PyPI consumers never need to ship the JSON separately.
pub const DEFAULT_CONSTANTS_JSON: &str = include_str!("default_constants.json");

/// JSON shape used by `default_constants.json`. Mirrors `PTConstantDef`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArithmosConstantDef {
    /// Friendly name (e.g. "Pi"). Optional.
    #[serde(default)]
    pub name: Option<String>,
    /// Symbol used in expressions (e.g. "Ï€"). Required.
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
/// module â€” never via `SYMBOL_REGISTRY.write()` directly from outside Arithmos.
pub static SYMBOL_REGISTRY: Lazy<RwLock<HashMap<String, ArithmosExpression>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

/// User-tunable enable flags. Maps symbol â†’ `true|false`. Off-by-default
/// constants are skipped during symbol resolution.
#[derive(Debug, Clone, Default)]
pub struct ArithmosConstantConfig {
    /// Per-symbol enable map.
    pub enabled_constants: HashMap<String, bool>,
    /// Default for unknown symbols.
    pub enable_all_by_default: bool,
}

/// FaÃ§ade type re-exported as `arithmos_core::ArithmosConstants` â€” a static-only
/// service for downstream code that prefers method calls over free functions.
pub struct ArithmosConstants;

impl ArithmosConstants {
    /// Look up a symbol in the registry. Equivalent to [`lookup_symbol`].
    pub fn lookup(symbol: &str) -> Option<ArithmosExpression> {
        lookup_symbol(symbol)
    }

    /// Register a symbol. Errors if the symbol is already present.
    pub fn register(symbol: String, expr: ArithmosExpression) -> Result<(), String> {
        register_symbol(symbol, expr)
    }

    /// Initialise the registry from `default_constants.json`. Idempotent â€” safe
    /// to call multiple times.
    pub fn initialize_defaults() -> Result<(), String> {
        load_constants_from_json(DEFAULT_CONSTANTS_JSON)
    }
}

/// Look up a symbol in the global registry.
pub fn lookup_symbol(symbol: &str) -> Option<ArithmosExpression> {
    SYMBOL_REGISTRY.read().get(symbol).cloned()
}

/// Register a symbol. Errors if the symbol is already present (use
/// [`reregister_symbol`] for hot-reload paths that intentionally overwrite).
pub fn register_symbol(symbol: String, expr: ArithmosExpression) -> Result<(), String> {
    let mut registry = SYMBOL_REGISTRY.write();
    if registry.contains_key(&symbol) {
        return Err(format!("Symbol '{}' is already registered", symbol));
    }
    registry.insert(symbol, expr);
    Ok(())
}

/// Replace an existing symbol or insert a fresh one. Used by hot-reload.
pub fn reregister_symbol(symbol: String, expr: ArithmosExpression) {
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
/// banner here. Bounded by line count for safety-critical Â§2.
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

/// Load constants from a JSON string. Stub for Wave 2; Wave 3 wires up the
/// real expression-tree parser.
pub fn load_constants_from_json(json: &str) -> Result<(), String> {
    // Wave 2: validate that the JSON parses as an array-or-object form so the
    // call surface is correct, even though we don't yet register anything.
    let cleaned = strip_jsonc_header(json);
    let _: serde_json::Value = serde_json::from_str(&cleaned)
        .map_err(|e| format!("Failed to parse constants JSON: {e}"))?;
    Ok(())
}

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
}

