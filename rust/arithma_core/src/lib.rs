//====== Arithma/rust/arithma_core/src/lib.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # arithma_core
//!
//! Arithma is the foundation symbolic mathematics engine. It is the bottom of the
//! dependency chain for the entire ecosystem (eml-math, eml-spectral, metaphysica,
//! periodica, pt-arithmos engine wrapper) and is published as a standalone PyPI
//! library so it can be reused outside the engine.
//!
//! ## Module map
//!
//! - [`expression`] â€” `ArithmosExpression` AST plus iterative and runtime simplifier
//!   passes. The substrate every other module ultimately emits or consumes.
//! - [`function`] â€” `ArithmosFunction` enum: arithmetic, transcendental, calculus
//!   and statistical operators that appear inside `Expression::Function` nodes.
//! - [`integer`] â€” `ArithmosInteger` exact unlimited-precision integer.
//! - [`variable`] â€” `ArithmosVariable` named variable with optional bound value.
//! - [`constants`] â€” Global constants registry. Loads `default_constants.json` at
//!   first access (embedded via `include_str!`).
//! - [`calculus`] â€” Symbolic and iterative differentiation, integration.
//! - [`fourier`] â€” Fourier transform configuration and pipeline.
//! - [`equation_solver`] â€” Solving algebraic equations symbolically and numerically.
//! - [`geometry`] â€” Vector / line / plane / sphere / intersection types.
//! - [`probabilities`] â€” Distributions, quantiles, moments, samplers.
//! - [`numerical`] â€” Root-finding, critical points, interval analysis.
//! - [`matrix`] / [`tensor`] â€” Linear algebra.
//! - [`unit`] / [`si_units`] â€” Unit-of-measure system. Loads `si_units.json` at
//!   first access (embedded via `include_str!`).
//! - [`lookup`] â€” Trig and general math hash-lookup tables.
//! - [`fallback`] â€” Fallback dispatch system when a primary backend cannot evaluate.
//! - [`external`] â€” External-function registry (the Arithmos / eml-math / engine
//!   three-way routing entry point) plus C++ and Rust dynamic executors.
//! - [`arithmetic`] â€” Lossless internal arithmetic helpers.
//! - [`pyfacade`] â€” PyO3 wrapper structs (gated by the `python` feature).
//!
//! ## Cross-library interop
//!
//! Downstream libraries (eml-math, eml-spectral, metaphysica, periodica) opt into
//! Arithmos by enabling their `with-arithma` feature flag and implementing
//! [`ArithmosInterop`] on their expression-bearing types. This lets every library
//! carry an `ArithmosExpression` payload so the engine can keep bitwise-deterministic
//! semantics across the entire stack while PyPI consumers stay independent.

#![allow(clippy::module_inception)]

pub mod arithmetic;
pub mod calculus;
pub mod constants;
pub mod equation_solver;
pub mod expression;
pub mod external;
pub mod fallback;
pub mod fourier;
pub mod function;
pub mod geometry;
pub mod integer;
pub mod lookup;
pub mod matrix;
pub mod numerical;
pub mod probabilities;
pub mod si_units;
pub mod tensor;
pub mod unit;
pub mod variable;

#[cfg(feature = "python")]
pub mod pyfacade;

// Re-export the most commonly used types at the crate root for ergonomic access.
pub use crate::constants::ArithmosConstants;
pub use crate::expression::ArithmosExpression;
pub use crate::external::registry::{
    ArithmosExternalFunctionError, ArithmosExternalFunctionRegistry,
};
pub use crate::fourier::ArithmosFourierConfig;
pub use crate::function::ArithmosFunction;
pub use crate::integer::ArithmosInteger;
pub use crate::si_units::ArithmosSIUnits;
pub use crate::unit::ArithmosUnit;
pub use crate::variable::ArithmosVariable;

/// Returns the library name. Used by the `arithmos` Python package facade and by
/// engine init logging.
pub fn hello() -> &'static str {
    "arithma"
}

/// Returns the package version from Cargo.toml.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Cross-library interop trait.
///
/// Downstream libraries (eml-math, eml-spectral, metaphysica, periodica) opt into
/// Arithmos by implementing this trait on their expression-bearing types behind a
/// `with-arithma` feature flag. The engine then has a single, uniform substrate
/// (`ArithmosExpression`) over which every library can be inspected, simplified,
/// differentiated or compared.
///
/// The trait is intentionally narrow â€” only conversion methods plus an associated
/// error type â€” so libraries can implement it without taking on the rest of
/// Arithmos's surface as a hard dependency.
pub trait ArithmosInterop {
    /// Error type returned by a failed conversion. Libraries are free to use
    /// their own error type or `String` for simplicity.
    type Error;

    /// Convert a downstream-library expression into an `ArithmosExpression`.
    ///
    /// Implementations should preserve symbolic structure where possible and only
    /// fall through to numeric literals when the source has no symbolic meaning.
    fn to_arithma(&self) -> Result<ArithmosExpression, Self::Error>;

    /// Convert an `ArithmosExpression` into a downstream-library expression. Used
    /// by the engine when results need to flow back into the originating library
    /// (for re-evaluation or rendering).
    fn from_arithma(expr: &ArithmosExpression) -> Result<Self, Self::Error>
    where
        Self: Sized;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hello_returns_arithma() {
        assert_eq!(hello(), "arithma");
    }

    #[test]
    fn version_is_non_empty() {
        assert!(!version().is_empty());
    }

    #[test]
    fn re_exports_resolve() {
        // Compile-time smoke: each re-export path must resolve.
        let _: Option<ArithmosExpression> = None;
        let _: Option<ArithmosFunction> = None;
        let _: Option<ArithmosInteger> = None;
    }
}

