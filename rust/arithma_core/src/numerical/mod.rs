//====== Arithma/rust/arithma_core/src/numerical/mod.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Numerical methods
//!
//! Bisection, Newton-Raphson, secant; critical-point and interval analysis.
//! Mirrors `pt_arithmos::math::numerical`.
//!
//! ## Submodules
//!
//! - [`methods`] â€” the generic `solve_with_method` dispatcher and the
//!   underlying iterative solvers.
//! - [`critical_points`] â€” local maxima/minima/saddles.
//! - [`interval_analysis`] â€” interval arithmetic for guaranteed bounds.
//! - [`root_finding`] â€” bisection / Newton-Raphson / secant.

pub mod critical_points;
pub mod interval_analysis;
pub mod methods;
pub mod root_finding;

pub use critical_points::{ArithmosCriticalPoint, ArithmosCriticalPointKind};
pub use interval_analysis::ArithmosInterval;
pub use methods::{ArithmosNumericalMethod, solve_with_method};
pub use root_finding::{
    ArithmosRootFindingConfig, ArithmosRootFindingResult, find_root_bisection,
    find_root_newton_raphson, find_root_secant,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numerical_re_exports_resolve() {
        let _: Option<ArithmosInterval> = None;
        let _: Option<ArithmosNumericalMethod> = None;
    }
}

