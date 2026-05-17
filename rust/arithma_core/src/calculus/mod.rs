//====== Arithma/rust/arithma_core/src/calculus/mod.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Calculus
//!
//! Symbolic calculus operators applied to `ArithmosExpression`. Implementations
//! are split between the recursive (legacy) and iterative (preferred) forms so
//! engine-side callers can choose stack-safe variants when expressions are deep.
//!
//! ## Submodules
//!
//! - [`differentiation`] â€” direct differentiation rules (sum, product, chain).
//! - [`differentiation_iterative`] â€” stack-based version, no recursion (per the
//!   engine's safety-critical standard rule 1).
//! - [`integration`] â€” symbolic integration (table lookup, integration by parts,
//!   substitution).

pub mod differentiation;
pub mod differentiation_iterative;
pub mod integration;

pub use differentiation::differentiate;
pub use differentiation_iterative::differentiate_iterative;
pub use integration::{integrate, integrate_definite};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::ArithmosExpression;

    #[test]
    fn calculus_module_exports_resolve() {
        // Compile-time smoke: every re-export is callable. We don't invoke the
        // stubs (they panic) â€” just take function pointers to prove the paths.
        let _: fn(&ArithmosExpression, &str) -> _ = differentiate;
        let _: fn(&ArithmosExpression, &str) -> _ = differentiate_iterative;
        let _: fn(&ArithmosExpression, &str) -> _ = integrate;
    }
}

