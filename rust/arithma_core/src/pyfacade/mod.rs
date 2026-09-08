//====== Arithma/rust/arithma_core/src/pyfacade/mod.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! PyO3 facade for the `arithma` Python package.
//!
//! Gated behind the `python` feature so the crate builds standalone without a
//! Python interpreter on the path. **Rust does not depend on Python**: this
//! module is additive, and `arithma_core` is a complete library without it.
//! Python is the thin wrapper, never the implementation.
//!
//! ## Layout
//!
//! Split into one file per domain. This was a single 1,393-line file, which
//! fought the "restrict functions to a single printed page" standard and made
//! parallel work impossible.
//!
//! | Module | Surface |
//! |---|---|
//! | [`core`] | `Expression`, `Integer`, `Variable` -- the base types |
//! | [`calculus`] | differentiation and integration |
//! | [`constants`] | the symbol registry |
//! | [`geometry`] | vectors, lines, planes, spheres, intersection |
//! | [`linalg`] | matrices and tensors |
//! | [`units`] | units, the SI registry and dimensional analysis |
//! | [`analysis`] | equation solving and Fourier transforms |
//! | [`numerical`] | root finding, critical points, interval analysis |
//! | [`stats`] | distributions, moments, confidence intervals |
//!
//! ## Conventions for everything under this directory
//!
//! Every wrapper follows the project's safety-critical standards:
//!
//! - **Two runtime assertions minimum** per function, checking preconditions
//!   that the Python boundary cannot enforce through types alone.
//! - **Bounded loops.** Any iteration over Python-supplied data carries an
//!   explicit upper bound, so a hostile or malformed input cannot spin.
//! - **Every non-void return is checked.** No silently discarded `Result`.
//! - **Errors surface as Python exceptions**, never as a default value. A
//!   wrapper that swallows an error is worse than one that does not exist,
//!   because it makes the backend look healthy while it is not.

use pyo3::prelude::*;

pub mod analysis;
pub mod calculus;
pub mod constants;
pub mod core;
pub mod geometry;
pub mod linalg;
pub mod numerical;
pub mod stats;
pub mod units;

pub use core::{Expression, Integer, Variable};

/// Upper bound on elements accepted from a Python sequence in one call.
///
/// Safety-critical standard 2: all loops must have fixed bounds. Every wrapper
/// that iterates a caller-supplied sequence checks length against this first,
/// so a malformed or hostile input fails fast with a clear error instead of
/// allocating unboundedly.
pub const MAX_SEQUENCE_LEN: usize = 1_048_576;

/// `arithma._arithma_core` module entry point. Maturin invokes this through
/// the `[tool.maturin] module-name` setting in `pyproject.toml`.
#[pymodule]
fn _arithma_core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    core::register(m)?;
    calculus::register(m)?;
    constants::register(m)?;
    geometry::register(m)?;
    linalg::register(m)?;
    units::register(m)?;
    analysis::register(m)?;
    numerical::register(m)?;
    stats::register(m)?;
    Ok(())
}
