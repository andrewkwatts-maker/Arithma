//====== Arithma/rust/arithma_core/src/equation_solver.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Equation solver
//!
//! Symbolic and numeric equation solving. Mirrors
//! `pt_arithmos::pt_equation_solver`. Wave 2 ships type signatures only; Wave 3
//! ports the real solver passes (linear, quadratic, polynomial root,
//! transcendental, system-of-equations).

use serde::{Deserialize, Serialize};

use crate::expression::ArithmaExpression;

/// Strategy hint for the solver. Implementations may ignore it and use their
/// own heuristics, but this gives callers a way to express priorities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ArithmaSolverStrategy {
    /// Auto-detect (default).
    #[default]
    Auto,
    /// Force the algebraic / closed-form path.
    Algebraic,
    /// Force the numeric (root-finding) path.
    Numeric,
    /// Try algebraic, fall back to numeric.
    Hybrid,
}

/// One root or solution branch returned by the solver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArithmaSolution {
    /// Symbolic form of the solution (e.g. `(-b ± √(b²-4ac)) / (2a)`).
    pub expression: ArithmaExpression,
    /// Optional cached numeric value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached: Option<f64>,
    /// Whether this branch is real-valued.
    pub is_real: bool,
}

/// Solve `expr = 0` for `var`. Returns every branch the solver finds.
pub fn solve(
    _expr: &ArithmaExpression,
    _var: &str,
    _strategy: ArithmaSolverStrategy,
) -> Result<Vec<ArithmaSolution>, String> {
    unimplemented!("solve — populated in Wave 3")
}

/// Solve `lhs = rhs` for `var`. Convenience that internally rewrites to
/// `lhs - rhs = 0`.
pub fn solve_equation(
    _lhs: &ArithmaExpression,
    _rhs: &ArithmaExpression,
    _var: &str,
    _strategy: ArithmaSolverStrategy,
) -> Result<Vec<ArithmaSolution>, String> {
    unimplemented!("solve_equation — populated in Wave 3")
}

/// Solve a system of equations for the listed variables.
pub fn solve_system(
    _equations: &[(ArithmaExpression, ArithmaExpression)],
    _vars: &[&str],
    _strategy: ArithmaSolverStrategy,
) -> Result<Vec<Vec<ArithmaSolution>>, String> {
    unimplemented!("solve_system — populated in Wave 3")
}

// ---------------------------------------------------------------------------
// Backward-compatibility aliases for the pre-rename `Arithmos*` names.
// Retained for one release; downstream (eml-math, eml-spectral, metaphysica,
// periodica) should migrate to the `Arithma*` names above.
// ---------------------------------------------------------------------------
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaSolution`")]
#[allow(unused)]
pub use self::ArithmaSolution as ArithmosSolution;
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaSolverStrategy`")]
#[allow(unused)]
pub use self::ArithmaSolverStrategy as ArithmosSolverStrategy;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_strategy_is_auto() {
        assert_eq!(
            ArithmaSolverStrategy::default(),
            ArithmaSolverStrategy::Auto
        );
    }
}
