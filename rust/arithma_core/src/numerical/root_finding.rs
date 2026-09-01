//====== Arithma/rust/arithma_core/src/numerical/root_finding.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Root finding
//!
//! Bisection / Newton-Raphson / secant root finders.

use crate::expression::ArithmaExpression;

/// Termination criterion configuration.
#[derive(Debug, Clone, Copy)]
pub struct ArithmaRootFindingConfig {
    /// Relative tolerance on `|f(x)|`.
    pub tol: f64,
    /// Hard cap on iterations (CLAUDE.md safety rule 2: bounded loops).
    pub max_iterations: usize,
}

impl Default for ArithmaRootFindingConfig {
    fn default() -> Self {
        Self {
            tol: 1e-12,
            max_iterations: 1024,
        }
    }
}

/// Outcome of a single root-finding run.
#[derive(Debug, Clone, Copy)]
pub struct ArithmaRootFindingResult {
    /// The root estimate.
    pub root: f64,
    /// Iterations consumed.
    pub iterations: usize,
    /// Whether the run converged within tolerance.
    pub converged: bool,
}

/// Bisection root finder. Wave-2 stub.
pub fn find_root_bisection(
    _expr: &ArithmaExpression,
    _var: &str,
    _lo: f64,
    _hi: f64,
    _config: &ArithmaRootFindingConfig,
) -> Result<ArithmaRootFindingResult, String> {
    unimplemented!("find_root_bisection — populated in Wave 3")
}

/// Newton-Raphson root finder. Wave-2 stub.
pub fn find_root_newton_raphson(
    _expr: &ArithmaExpression,
    _var: &str,
    _initial: f64,
    _config: &ArithmaRootFindingConfig,
) -> Result<ArithmaRootFindingResult, String> {
    unimplemented!("find_root_newton_raphson — populated in Wave 3")
}

/// Secant-method root finder. Wave-2 stub.
pub fn find_root_secant(
    _expr: &ArithmaExpression,
    _var: &str,
    _x0: f64,
    _x1: f64,
    _config: &ArithmaRootFindingConfig,
) -> Result<ArithmaRootFindingResult, String> {
    unimplemented!("find_root_secant — populated in Wave 3")
}

// ---------------------------------------------------------------------------
// Backward-compatibility aliases for the pre-rename `Arithmos*` names.
// Retained for one release; downstream (eml-math, eml-spectral, metaphysica,
// periodica) should migrate to the `Arithma*` names above.
// ---------------------------------------------------------------------------
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaRootFindingConfig`")]
#[allow(unused)]
pub use self::ArithmaRootFindingConfig as ArithmosRootFindingConfig;
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaRootFindingResult`")]
#[allow(unused)]
pub use self::ArithmaRootFindingResult as ArithmosRootFindingResult;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_sane() {
        let cfg = ArithmaRootFindingConfig::default();
        assert!(cfg.tol > 0.0);
        assert!(cfg.max_iterations > 0);
    }
}
