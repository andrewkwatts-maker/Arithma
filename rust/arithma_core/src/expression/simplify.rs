//====== Arithma/rust/arithma_core/src/expression/simplify.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Simplify
//!
//! Compile-time and runtime simplification rules.
//!
//! Compile-time rules are pure rewrites we can apply at expression construction
//! time (e.g. `0 + x â†’ x`, `1 * x â†’ x`, `0 * x â†’ 0`). Runtime rules require a
//! [`SimplificationConfig`] policy and are dispatched from [`run`].

use crate::expression::{ArithmosExpression, SimplificationConfig};

/// Simplification rule classification, used by the cost model and by the
/// router to decide between Arithmos's own simplifier and an external backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArithmosSimplificationMethod {
    /// Compile-time / always-on identities (idempotents, units, zeros).
    Compiletime,
    /// Iterative pattern-matched rewrites.
    Iterative,
    /// Costed search over candidate rewrites â€” most expensive.
    Costed,
}

/// Apply the configured simplification rules to `expr`.
///
/// This is the single entry point downstream modules call when they want a
/// simplified expression. Currently a stub; Wave 3 wires in the real rule set.
pub fn run(expr: &mut ArithmosExpression, _config: &SimplificationConfig) -> bool {
    let _ = expr;
    false
}

/// Apply the always-on compile-time identities. Returns `true` if anything
/// changed. This is the path the AST constructors call internally.
pub fn run_compile_time(_expr: &mut ArithmosExpression) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_returns_false_on_atom() {
        let mut expr = ArithmosExpression::var("x");
        let cfg = SimplificationConfig::default();
        assert!(!run(&mut expr, &cfg));
    }

    #[test]
    fn compile_time_pass_returns_false_on_atom() {
        let mut expr = ArithmosExpression::zero();
        assert!(!run_compile_time(&mut expr));
    }
}

