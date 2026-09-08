//====== Arithma/rust/arithma_core/src/pyfacade/constants.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! PyO3 facade for [`crate::constants`] — the global symbol registry.
//!
//! Surface
//! -------
//!
//! | Python | Rust |
//! |---|---|
//! | `lookup(symbol)` | [`crate::constants::ArithmaConstants::lookup`] / [`crate::constants::lookup_symbol`] |
//! | `lookup_value(symbol)` | `lookup_symbol` then `ArithmaExpression::to_f64` |
//! | `is_registered(symbol)` | `lookup_symbol(...).is_some()` |
//! | `register(symbol, value)` | [`crate::constants::ArithmaConstants::register`] / [`crate::constants::register_symbol`] |
//! | `reregister(symbol, value)` | [`crate::constants::reregister_symbol`] |
//! | `register_many(mapping, overwrite=False)` | bounded loop over `register_symbol` / `reregister_symbol` |
//! | `initialize_defaults()` | [`crate::constants::ArithmaConstants::initialize_defaults`] |
//! | `registered_count()` | [`crate::constants::registered_count`] |
//! | `load_constants_from_json(json)` | [`crate::constants::load_constants_from_json`] |
//! | `default_constants_json()` | [`crate::constants::DEFAULT_CONSTANTS_JSON`] |
//!
//! Notes on the wrapped API
//! ------------------------
//!
//! - `register_symbol` errors on a duplicate key while `reregister_symbol`
//!   returns `()` and overwrites. The two are exposed separately rather than
//!   behind one `overwrite` flag on `register`, so a Python caller cannot lose
//!   an existing definition by forgetting a keyword. `register_many` does take
//!   the flag, because a bulk load has a legitimate reason to want either.
//! - `load_constants_from_json` is itself idempotent (it registers through
//!   `reregister_symbol`), so repeated calls to `initialize_defaults` neither
//!   error nor change the count. That is the documented contract, and it is
//!   why the JSON loader here does not need an `overwrite` flag.
//! - Registering a plain number stores it as an `ArithmaExpression::Constant`
//!   carrying the symbol and a cached `f64` — not a bare `Number` literal —
//!   because the registry is a *constants* table and downstream consumers
//!   match on the `Constant` variant to render the symbol.
//! - `Expression::from_inner` in [`crate::pyfacade::core`] is private and
//!   `core.rs` is not ours to edit, so the wrapper is built with a struct
//!   literal in [`to_py_expression`]: `Expression::inner` is `pub(crate)` and
//!   this module is in the same crate. No change to `core.rs` was required.

// ── Lint policy ──────────────────────────────────────────────────────────────
// Every `#[pyfunction]` returning `PyResult<T>` expands to a `PyErr -> PyErr`
// conversion inside the generated trampoline, which `clippy::useless_conversion`
// flags at the *return type* of our source function. There is nothing to remove
// at the call site — the conversion is not ours — so the lint is silenced here
// rather than left to fire once per exported function. Scoped to this module,
// per lib.rs's policy of justifying every exception where it is taken.
#![allow(clippy::useless_conversion)]

use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict};

use crate::constants::{
    load_constants_from_json as rust_load_constants_from_json, lookup_symbol,
    register_symbol as rust_register_symbol, registered_count as rust_registered_count,
    reregister_symbol as rust_reregister_symbol, ArithmaConstants, DEFAULT_CONSTANTS_JSON,
};
use crate::expression::ArithmaExpression;
use crate::pyfacade::core::Expression;
use crate::pyfacade::MAX_SEQUENCE_LEN;

// ============================================================================
// Bounds.
// ============================================================================

/// Upper bound on the byte length of a registry symbol. Symbols are short
/// identifiers such as `pi`, `c` or `h_bar`; the bound also fixes the iteration
/// count of the character scan inside [`check_symbol`].
const MAX_SYMBOL_LEN: usize = 256;

// ============================================================================
// Pure validation helpers (unit-tested below without a Python interpreter).
// ============================================================================

/// Validate a symbol arriving from Python.
///
/// The Rust registry is a `HashMap<String, _>` and will happily accept `""` or
/// `" pi "` as distinct keys, which would then be unreachable from any parsed
/// expression. Rejecting them here is the only place that can happen.
fn check_symbol(symbol: &str) -> Result<(), String> {
    if symbol.is_empty() {
        return Err("symbol must not be empty".to_string());
    }
    if symbol.len() > MAX_SYMBOL_LEN {
        return Err(format!(
            "symbol is {} bytes, exceeding the {MAX_SYMBOL_LEN}-byte limit",
            symbol.len()
        ));
    }
    // Bounded by the length check immediately above.
    if symbol.chars().any(char::is_whitespace) {
        return Err(format!("symbol {symbol:?} must not contain whitespace"));
    }
    Ok(())
}

/// Validate a JSON payload before it reaches `serde_json`.
///
/// The length check is the "bounded input" rule applied to a string: the loader
/// walks every entry of whatever the document decodes to, so the document
/// itself is the caller-supplied sequence that needs a cap.
fn check_json_payload(json: &str) -> Result<(), String> {
    if json.trim().is_empty() {
        return Err("JSON payload must not be empty".to_string());
    }
    if json.len() > MAX_SEQUENCE_LEN {
        return Err(format!(
            "JSON payload of {} bytes exceeds the maximum of {MAX_SEQUENCE_LEN}",
            json.len()
        ));
    }
    Ok(())
}

/// Check a caller-supplied mapping length against [`MAX_SEQUENCE_LEN`] before
/// any iteration over it begins.
fn check_mapping_len(len: usize, what: &str) -> Result<(), String> {
    if len > MAX_SEQUENCE_LEN {
        return Err(format!(
            "{what}: mapping of {len} entries exceeds the maximum of {MAX_SEQUENCE_LEN}"
        ));
    }
    Ok(())
}

// ============================================================================
// Python <-> Rust conversion.
// ============================================================================

/// Wrap an `ArithmaExpression` in the `Expression` pyclass. See the module
/// docs for why this is a struct literal rather than `Expression::from_inner`.
fn to_py_expression(inner: ArithmaExpression) -> Expression {
    let wrapped = Expression { inner };
    debug_assert!(
        !matches!(&wrapped.inner, ArithmaExpression::Constant { symbol, .. } if symbol.is_empty()),
        "to_py_expression: wrapped a constant with an empty symbol"
    );
    wrapped
}

/// Convert the value side of a registry entry.
///
/// An :class:`Expression` is stored verbatim; an ``int`` / ``float`` becomes a
/// named `Constant` carrying `symbol` and the cached value, which is what the
/// simplifier and the LaTeX walker expect from the registry. ``bool`` is
/// rejected so ``True`` cannot silently register as ``1``.
fn coerce_registry_value(symbol: &str, obj: &Bound<'_, PyAny>) -> PyResult<ArithmaExpression> {
    debug_assert!(!symbol.is_empty(), "coerce_registry_value: empty symbol");
    if obj.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(
            "constant value must be an Expression, int, or float; bool is not accepted",
        ));
    }
    if let Ok(py_expr) = obj.extract::<PyRef<Expression>>() {
        return Ok(py_expr.inner.clone());
    }
    if let Ok(value) = obj.extract::<f64>() {
        if !value.is_finite() {
            return Err(PyValueError::new_err(format!(
                "constant {symbol:?} must have a finite value, got {value}"
            )));
        }
        return Ok(ArithmaExpression::constant(symbol, None, Some(value), true));
    }
    Err(PyTypeError::new_err(format!(
        "constant {symbol:?}: expected Expression, int, or float"
    )))
}

/// Validate a symbol and surface a failure as `ValueError`.
fn symbol_or_err(symbol: &str, what: &str) -> PyResult<()> {
    check_symbol(symbol).map_err(|e| PyValueError::new_err(format!("{what}: {e}")))
}

// ============================================================================
// Lookup.
// ============================================================================

/// Look up `symbol` in the global registry.
///
/// Returns the registered :class:`Expression`, or ``None`` if the symbol is not
/// registered. Raises :class:`ValueError` for a malformed symbol — an empty or
/// padded key can never be registered, so asking for one is a caller bug rather
/// than a miss.
#[pyfunction]
#[pyo3(signature = (symbol))]
fn lookup(symbol: &str) -> PyResult<Option<Expression>> {
    symbol_or_err(symbol, "lookup")?;
    debug_assert!(!symbol.is_empty(), "lookup: empty symbol");
    debug_assert!(
        symbol.len() <= MAX_SYMBOL_LEN,
        "lookup: oversized symbol survived validation"
    );
    Ok(ArithmaConstants::lookup(symbol).map(to_py_expression))
}

/// Numeric value of `symbol`, or ``None`` if it is unregistered.
///
/// Raises :class:`ValueError` for a malformed symbol, and :class:`KeyError` if
/// the symbol is registered but carries no numerically reducible value — a
/// silent ``None`` there would be indistinguishable from "not registered".
#[pyfunction]
#[pyo3(signature = (symbol))]
fn lookup_value(symbol: &str) -> PyResult<Option<f64>> {
    symbol_or_err(symbol, "lookup_value")?;
    debug_assert!(!symbol.is_empty(), "lookup_value: empty symbol");
    let Some(expr) = lookup_symbol(symbol) else {
        return Ok(None);
    };
    let value = expr.to_f64().ok_or_else(|| {
        PyKeyError::new_err(format!(
            "constant {symbol:?} is registered but has no numeric value"
        ))
    })?;
    debug_assert!(
        lookup_symbol(symbol).is_some(),
        "lookup_value: symbol vanished from the registry mid-call"
    );
    Ok(Some(value))
}

/// Whether `symbol` currently resolves in the registry.
#[pyfunction]
#[pyo3(signature = (symbol))]
fn is_registered(symbol: &str) -> PyResult<bool> {
    symbol_or_err(symbol, "is_registered")?;
    debug_assert!(!symbol.is_empty(), "is_registered: empty symbol");
    debug_assert!(
        symbol.len() <= MAX_SYMBOL_LEN,
        "is_registered: oversized symbol survived validation"
    );
    Ok(lookup_symbol(symbol).is_some())
}

/// Number of symbols currently registered.
///
/// # The assertion that used to be here made this function panic
///
/// It read `debug_assert!(lookup_symbol("").is_none())`, meaning to check that
/// the empty symbol never reached the registry. But `lookup_symbol` opens with
/// `debug_assert!(!symbol.is_empty())` -- so the probe tripped the assertion in
/// the callee, and `arithma.registered_count()` panicked **unconditionally** on
/// every debug build, before returning a number it had already computed.
///
/// It was also checking the wrong thing in the wrong place. Counting entries
/// looks nothing up, so the empty symbol has no way to be involved; whether it
/// can be *registered* is a property of `register`, and it is asserted there
/// and tested below. There is no precondition or postcondition here beyond
/// "the registry knows its own length", which is `HashMap::len`.
#[pyfunction]
fn registered_count() -> usize {
    rust_registered_count()
}

// ============================================================================
// Registration.
// ============================================================================

/// Register `symbol` with `value`, failing if the symbol already exists.
///
/// `value` may be an :class:`Expression`, an ``int`` or a ``float``; a number
/// is stored as a named constant carrying that cached value. Raises
/// :class:`ValueError` on a malformed symbol, a non-finite value, or a
/// duplicate key. Use :func:`reregister` to overwrite deliberately.
/// Exposed to Python as `register`. The Rust name differs because standard 8
/// reserves `register` in this module for the module registrar at the bottom of
/// the file, and two functions cannot share a name.
#[pyfunction]
#[pyo3(name = "register", signature = (symbol, value))]
fn register_constant(symbol: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
    symbol_or_err(symbol, "register")?;
    let expr = coerce_registry_value(symbol, value)?;
    debug_assert!(!symbol.is_empty(), "register: empty symbol");
    debug_assert!(
        symbol.len() <= MAX_SYMBOL_LEN,
        "register: oversized symbol survived validation"
    );
    ArithmaConstants::register(symbol.to_string(), expr)
        .map_err(|e| PyValueError::new_err(format!("register: {e}")))
}

/// Register `symbol` with `value`, replacing any existing entry.
///
/// The hot-reload path. Same argument handling as :func:`register`, but a
/// duplicate key is overwritten instead of raising.
#[pyfunction]
#[pyo3(signature = (symbol, value))]
fn reregister(symbol: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
    symbol_or_err(symbol, "reregister")?;
    let expr = coerce_registry_value(symbol, value)?;
    debug_assert!(!symbol.is_empty(), "reregister: empty symbol");
    debug_assert!(
        symbol.len() <= MAX_SYMBOL_LEN,
        "reregister: oversized symbol survived validation"
    );
    rust_reregister_symbol(symbol.to_string(), expr);
    if lookup_symbol(symbol).is_none() {
        return Err(PyValueError::new_err(format!(
            "reregister: {symbol:?} did not survive insertion into the registry"
        )));
    }
    Ok(())
}

/// Register every entry of `mapping`, a ``{str: Expression | int | float}``
/// dict.
///
/// Returns the number of symbols registered. With ``overwrite=False`` (the
/// default) a key that already exists raises :class:`ValueError`; with
/// ``overwrite=True`` it is replaced. The dict length is checked against
/// ``MAX_SEQUENCE_LEN`` before any iteration begins.
///
/// Registration is *not* transactional: entries before a failing one stay
/// registered, matching the underlying registry, which has no rollback.
#[pyfunction]
#[pyo3(signature = (mapping, overwrite = false))]
fn register_many(mapping: &Bound<'_, PyDict>, overwrite: bool) -> PyResult<usize> {
    let len = mapping.len();
    check_mapping_len(len, "register_many").map_err(PyValueError::new_err)?;
    debug_assert!(
        len <= MAX_SEQUENCE_LEN,
        "register_many: mapping passed validation but exceeds the cap"
    );
    let mut registered: usize = 0;
    // Fixed bound: `len` was checked against MAX_SEQUENCE_LEN above, and the
    // counter below fails loudly if the dict mutates under iteration.
    for (key, value) in mapping.iter() {
        if registered >= len {
            return Err(PyValueError::new_err(
                "register_many: mapping grew during iteration",
            ));
        }
        let symbol: String = key
            .extract()
            .map_err(|_| PyTypeError::new_err("register_many: every key must be str"))?;
        symbol_or_err(&symbol, "register_many")?;
        let expr = coerce_registry_value(&symbol, &value)?;
        if overwrite {
            rust_reregister_symbol(symbol, expr);
        } else {
            rust_register_symbol(symbol, expr)
                .map_err(|e| PyValueError::new_err(format!("register_many: {e}")))?;
        }
        registered += 1;
    }
    debug_assert_eq!(registered, len, "register_many: entry count drifted");
    Ok(registered)
}

// ============================================================================
// Bulk loading.
// ============================================================================

/// Load the embedded `default_constants.json` catalogue into the registry.
///
/// Idempotent — safe to call repeatedly; the count does not change and no
/// duplicate-key error is raised. Returns the number of symbols registered.
/// Raises :class:`ValueError` if the embedded catalogue fails to parse.
#[pyfunction]
fn initialize_defaults() -> PyResult<usize> {
    let count = ArithmaConstants::initialize_defaults()
        .map_err(|e| PyValueError::new_err(format!("initialize_defaults: {e}")))?;
    if count == 0 {
        return Err(PyValueError::new_err(
            "initialize_defaults: the embedded catalogue registered no symbols",
        ));
    }
    debug_assert!(
        rust_registered_count() >= count,
        "initialize_defaults: registry smaller than the load it just accepted"
    );
    Ok(count)
}

/// Load constants from a JSON document into the registry.
///
/// The document is an array of constant definitions; a leading ``//`` comment
/// banner is tolerated. Entries with ``"enabled": false`` are skipped, and
/// existing symbols are replaced, so the call is idempotent. Returns the number
/// of symbols registered. Raises :class:`ValueError` for an empty or oversized
/// payload, a parse failure, or an entry with neither a cached value nor an
/// expression.
#[pyfunction]
#[pyo3(signature = (json))]
fn load_constants_from_json(json: &str) -> PyResult<usize> {
    check_json_payload(json)
        .map_err(|e| PyValueError::new_err(format!("load_constants_from_json: {e}")))?;
    debug_assert!(
        !json.is_empty(),
        "load_constants_from_json: empty payload survived validation"
    );
    debug_assert!(
        json.len() <= MAX_SEQUENCE_LEN,
        "load_constants_from_json: oversized payload survived validation"
    );
    let count = rust_load_constants_from_json(json)
        .map_err(|e| PyValueError::new_err(format!("load_constants_from_json: {e}")))?;
    Ok(count)
}

/// The embedded `default_constants.json` source text.
///
/// Exposed so Python-side tooling can inspect or re-emit the catalogue without
/// locating the file inside the installed wheel.
#[pyfunction]
fn default_constants_json() -> &'static str {
    debug_assert!(
        !DEFAULT_CONSTANTS_JSON.is_empty(),
        "default_constants_json: the embedded catalogue is empty"
    );
    DEFAULT_CONSTANTS_JSON
}

// ============================================================================
// Registration
// ============================================================================

/// Register this module's surface on the extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lookup, m)?)?;
    m.add_function(wrap_pyfunction!(lookup_value, m)?)?;
    m.add_function(wrap_pyfunction!(is_registered, m)?)?;
    m.add_function(wrap_pyfunction!(registered_count, m)?)?;
    m.add_function(wrap_pyfunction!(register_constant, m)?)?;
    m.add_function(wrap_pyfunction!(reregister, m)?)?;
    m.add_function(wrap_pyfunction!(register_many, m)?)?;
    m.add_function(wrap_pyfunction!(initialize_defaults, m)?)?;
    m.add_function(wrap_pyfunction!(load_constants_from_json, m)?)?;
    m.add_function(wrap_pyfunction!(default_constants_json, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbol_rejects_empty() {
        assert!(check_symbol("").is_err());
    }

    /// `registered_count()` returns a number instead of panicking.
    ///
    /// It panicked on every debug build: its own `debug_assert!` probed
    /// `lookup_symbol("")`, and `lookup_symbol` asserts its argument is
    /// non-empty. Nothing caught it because nothing called this function from
    /// Rust -- it is a `#[pyfunction]`, and the binding tests were unrunnable
    /// while `extension-module` was welded to the pyo3 dependency.
    ///
    /// The registry is process-global and these tests run in parallel, so this
    /// deliberately asserts nothing about the count's *value*. An earlier
    /// version compared it before and after a registration and failed
    /// intermittently, which would have been a flaky test guarding a bug that
    /// is not intermittent at all.
    #[test]
    fn counting_the_registry_does_not_panic() {
        let symbol = "__arithma_count_probe";
        crate::constants::reregister_symbol(symbol.to_string(), ArithmaExpression::from_i64(1));
        assert!(is_registered(symbol).unwrap());
        // The whole bug: this line used to panic before returning.
        assert!(
            registered_count() >= 1,
            "a registered symbol must be counted"
        );
    }

    /// The property the removed assertion was reaching for, checked where it
    /// belongs: the public entry points are what must refuse the empty symbol.
    ///
    /// They refuse with an error rather than an assertion, which is the right
    /// split -- `lookup_symbol`'s `debug_assert!` states an internal
    /// precondition, and these functions are the boundary that guarantees it.
    #[test]
    fn the_empty_symbol_is_refused_at_every_entry_point() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let one = 1.0_f64.into_py(py);
            assert!(register_constant("", one.bind(py)).is_err());
            assert!(reregister("", one.bind(py)).is_err());
        });
        assert!(is_registered("").is_err());
        assert!(lookup_value("").is_err());
    }

    #[test]
    fn symbol_accepts_registry_keys() {
        assert!(check_symbol("e").is_ok());
        assert!(check_symbol("h_bar").is_ok());
        // The shipped catalogue keys on the Greek letters themselves.
        assert!(check_symbol("\u{3c0}").is_ok());
        assert!(check_symbol("\u{3c6}").is_ok());
    }

    #[test]
    fn symbol_rejects_padding_and_whitespace() {
        assert!(check_symbol(" pi").is_err());
        assert!(check_symbol("pi ").is_err());
        assert!(check_symbol("h bar").is_err());
        assert!(check_symbol("\n").is_err());
    }

    #[test]
    fn symbol_rejects_oversized() {
        let ok = "s".repeat(MAX_SYMBOL_LEN);
        assert!(check_symbol(&ok).is_ok());
        let too_long = "s".repeat(MAX_SYMBOL_LEN + 1);
        assert!(check_symbol(&too_long).is_err());
    }

    #[test]
    fn json_payload_rejects_blank() {
        assert!(check_json_payload("").is_err());
        assert!(check_json_payload("   \n\t ").is_err());
    }

    #[test]
    fn json_payload_accepts_the_embedded_catalogue() {
        assert!(check_json_payload(DEFAULT_CONSTANTS_JSON).is_ok());
    }

    #[test]
    fn json_payload_is_capped() {
        let too_long = "x".repeat(MAX_SEQUENCE_LEN + 1);
        assert!(check_json_payload(&too_long).is_err());
    }

    #[test]
    fn mapping_len_is_capped() {
        assert!(check_mapping_len(0, "t").is_ok());
        assert!(check_mapping_len(MAX_SEQUENCE_LEN, "t").is_ok());
        assert!(check_mapping_len(MAX_SEQUENCE_LEN + 1, "t").is_err());
    }

    #[test]
    fn mapping_len_error_names_the_caller() {
        let err = check_mapping_len(MAX_SEQUENCE_LEN + 1, "register_many").unwrap_err();
        assert!(err.contains("register_many"), "message was {err:?}");
    }

    #[test]
    fn to_py_expression_preserves_the_constant() {
        let e = ArithmaExpression::constant("pi", None, Some(std::f64::consts::PI), true);
        let wrapped = to_py_expression(e.clone());
        assert_eq!(format!("{:?}", wrapped.inner), format!("{e:?}"));
    }
}
