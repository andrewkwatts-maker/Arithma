//====== Arithma/rust/arithma_core/src/expression/iterative.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Iterative simplifier passes
//!
//! Stack-based traversal of `ArithmosExpression` trees. The engine's
//! safety-critical standards forbid recursion (rule 1: avoid recursion); every
//! pass in this module is implemented with an explicit work stack so the call
//! depth stays O(1) regardless of expression depth.
//!
//! Two flavours:
//!
//! - [`ArithmosIterativeSimplifier`] â€” stateful simplifier with a work queue.
//! - [`simplify_iterative`] â€” convenience function that owns its simplifier.

use crate::expression::{ArithmosExpression, SimplificationConfig};

/// Iterative, stack-based simplifier.
///
/// Owns its own work stack so multiple invocations can reuse the allocation.
/// Implementations must respect `config.max_iterations` and bail rather than
/// loop forever (CLAUDE.md safety rule 2: all loops have fixed bounds).
#[derive(Debug, Default)]
pub struct ArithmosIterativeSimplifier {
    /// Re-usable work stack. Avoids per-call allocations.
    stack: Vec<ArithmosExpression>,
    /// Iterations consumed by the most recent run.
    last_iterations: usize,
}

impl ArithmosIterativeSimplifier {
    /// Create a fresh simplifier with empty state.
    pub fn new() -> Self {
        Self {
            stack: Vec::new(),
            last_iterations: 0,
        }
    }

    /// Number of iterations the last `simplify` consumed.
    pub fn last_iterations(&self) -> usize {
        self.last_iterations
    }

    /// Simplify `expr` in place using the configured policy.
    ///
    /// Wave-2 stub returns the input unchanged. Wave 3 fills in the real
    /// pattern-matched rewrites.
    pub fn simplify(
        &mut self,
        expr: &mut ArithmosExpression,
        _config: &SimplificationConfig,
    ) -> bool {
        self.stack.clear();
        self.last_iterations = 0;
        let _ = expr; // silence unused for now
        false
    }
}

/// Convenience entry point: build a simplifier, run it, and discard it.
pub fn simplify_iterative(expr: &mut ArithmosExpression, config: &SimplificationConfig) -> bool {
    let mut simplifier = ArithmosIterativeSimplifier::new();
    simplifier.simplify(expr, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_simplifier_starts_empty() {
        let s = ArithmosIterativeSimplifier::new();
        assert_eq!(s.last_iterations(), 0);
    }

    #[test]
    fn iterative_pass_returns_false_on_atom() {
        let mut expr = ArithmosExpression::zero();
        let cfg = SimplificationConfig::default();
        assert!(!simplify_iterative(&mut expr, &cfg));
    }
}

