//====== Arithma/rust/arithma_core/src/pyfacade.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! PyO3 facade for the `arithma` Python package.
//!
//! Gated behind the `python` feature so the engine workspace builds without
//! a Python interpreter on the path. When `pip install arithmos` triggers a
//! maturin build, this module is the entry point that exposes the Rust core
//! to Python under the module name `arithma._arithma_core`.
//!
//! Wave-2 scaffold: registers the module name, the version string, and the
//! `_HAS_RUST` flag analogue (`is_rust_backend()`). Wave 3 wires up the
//! ArithmosExpression / ArithmosFunction / ArithmosInteger PyO3 wrapper types.

use pyo3::prelude::*;

/// Returns the underlying Rust crate version. The Python facade matches this
/// against `arithmos.__version__` to detect maturin/wheel/python desyncs.
#[pyfunction]
fn version_rust() -> &'static str {
    crate::version()
}

/// Sentinel that the Python facade can probe to confirm the Rust backend
/// loaded successfully (analogous to `_HAS_RUST = True` in pure-Python).
#[pyfunction]
fn is_rust_backend() -> bool {
    true
}

/// `arithma._arithma_core` module entry point. Maturin invokes this through
/// the `[tool.maturin] module-name` setting in `pyproject.toml`.
#[pymodule]
fn _arithmos_core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version_rust, m)?)?;
    m.add_function(wrap_pyfunction!(is_rust_backend, m)?)?;
    Ok(())
}

