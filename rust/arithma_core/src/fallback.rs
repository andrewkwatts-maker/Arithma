//====== Arithma/rust/arithma_core/src/fallback.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! Fallback dispatch system.
//!
//! When a primary backend (Arithmos, EML, an external Python/C++ executor)
//! cannot evaluate an expression, the fallback chain decides what to do:
//! try a different backend, drop to numeric f64, return the expression
//! unchanged, or surface an error.
//!
//! ## Migration note
//!
//! The trait + stats migrated from pt-arithmos
//! `pt_fallback_system.rs::PTFallbackFunction` and
//! `PTFallbackStats` (Wave 3 follow-up, plan Â§F.8 step 2). The
//! `PTFallbackRegistry` orchestrator stays in pt-arithmos until
//! `ArithmosExternalFunctionRegistry` can host it directly â€” its concrete
//! shape depends on the engine's symbol-resolver chain that pt-arithmos
//! still owns.

use crate::expression::ArithmosExpression;

// â”€â”€â”€ Strategy enum â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Strategy used when a backend fails or refuses an expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithmosFallbackStrategy {
    /// Drop to numeric f64 if all leaves can be evaluated to f64.
    Numeric,
    /// Return the expression unchanged.
    Symbolic,
    /// Surface the error to the caller.
    Error,
}

impl Default for ArithmosFallbackStrategy {
    fn default() -> Self {
        Self::Symbolic
    }
}

/// Try the next strategy in a chain. Wave-2 stub: returns the strategy
/// unchanged. Wave 3 wires up the actual chain (Numeric â†’ Symbolic â†’ Error).
pub fn try_fallback(
    _expr: &ArithmosExpression,
    strategy: ArithmosFallbackStrategy,
) -> ArithmosFallbackStrategy {
    strategy
}

// â”€â”€â”€ Trait contract â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// A function that can have one or more external implementations with a
/// pure-Rust fallback. Concrete impls live in pt-arithmos
/// (`PTFallbackFunction`) and other engine plugins; the trait shape
/// stays here so multiple registries can share a uniform contract.
///
/// Generic over the expression type so downstream consumers can plug in
/// `ArithmosExpression`, pt-arithmos `PTExpression`, eml-math `EMLPoint`,
/// or any other carrier without committing to one in the foundation.
pub trait ArithmosFallbackFunction<Expr> {
    /// Stable function identifier (e.g. `"sin"`, `"erf"`, `"my_lib::foo"`).
    fn function_id(&self) -> &str;

    /// Pure-Rust implementation. Always available â€” this is the floor of
    /// the fallback chain.
    fn rust_implementation(&self, args: &[Expr]) -> Result<Expr, String>;

    /// Optional external-implementation lookup name. When `Some(name)`
    /// the registry first asks the external registry by `name`; if
    /// that fails the chain falls through to `rust_implementation`.
    fn external_function_name(&self) -> Option<&str> { None }

    /// Whether this function should prefer external implementation
    /// when both paths are available. Defaults to `true` â€” external
    /// libraries usually win on speed for ops they specialise in.
    fn prefer_external(&self) -> bool { true }
}

// â”€â”€â”€ Per-function statistics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Telemetry for a single registered function. The registry surfaces
/// these for debug dashboards and adaptive routing decisions.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ArithmosFallbackStats {
    /// External-backend invocations attempted.
    pub external_calls: u64,
    /// External invocations that returned a value (no error).
    pub external_successes: u64,
    /// Times the rust fallback ran (either external failed or wasn't
    /// available).
    pub fallback_calls: u64,
    /// Total `evaluate` calls including both external and rust paths.
    pub total_calls: u64,
}

impl ArithmosFallbackStats {
    /// Record a successful external invocation.
    pub fn record_external_success(&mut self) {
        self.total_calls = self.total_calls.saturating_add(1);
        self.external_calls = self.external_calls.saturating_add(1);
        self.external_successes = self.external_successes.saturating_add(1);
    }

    /// Record a failed external invocation that fell through to the
    /// Rust fallback.
    pub fn record_external_failure_then_fallback(&mut self) {
        self.total_calls = self.total_calls.saturating_add(1);
        self.external_calls = self.external_calls.saturating_add(1);
        self.fallback_calls = self.fallback_calls.saturating_add(1);
    }

    /// Record a rust-fallback-only invocation (no external attempted).
    pub fn record_fallback_only(&mut self) {
        self.total_calls = self.total_calls.saturating_add(1);
        self.fallback_calls = self.fallback_calls.saturating_add(1);
    }

    /// External success ratio in `[0.0, 1.0]`. Returns `None` when no
    /// external calls have happened yet (avoids 0/0 NaN).
    pub fn external_success_ratio(&self) -> Option<f64> {
        if self.external_calls == 0 { None }
        else { Some(self.external_successes as f64 / self.external_calls as f64) }
    }

    /// Reset all counters.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // â”€â”€â”€ strategy â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    #[test]
    fn default_strategy_is_symbolic() {
        assert_eq!(ArithmosFallbackStrategy::default(), ArithmosFallbackStrategy::Symbolic);
    }

    // â”€â”€â”€ stats â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    #[test]
    fn stats_default_is_zero() {
        let s = ArithmosFallbackStats::default();
        assert_eq!(s.total_calls, 0);
        assert_eq!(s.external_calls, 0);
        assert_eq!(s.external_successes, 0);
        assert_eq!(s.fallback_calls, 0);
        assert_eq!(s.external_success_ratio(), None);
    }

    #[test]
    fn stats_record_external_success() {
        let mut s = ArithmosFallbackStats::default();
        s.record_external_success();
        s.record_external_success();
        assert_eq!(s.total_calls, 2);
        assert_eq!(s.external_calls, 2);
        assert_eq!(s.external_successes, 2);
        assert_eq!(s.fallback_calls, 0);
        assert_eq!(s.external_success_ratio(), Some(1.0));
    }

    #[test]
    fn stats_record_external_failure_then_fallback() {
        let mut s = ArithmosFallbackStats::default();
        s.record_external_success();
        s.record_external_failure_then_fallback();
        assert_eq!(s.total_calls, 2);
        assert_eq!(s.external_calls, 2);
        assert_eq!(s.external_successes, 1);
        assert_eq!(s.fallback_calls, 1);
        assert!((s.external_success_ratio().unwrap() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn stats_record_fallback_only() {
        let mut s = ArithmosFallbackStats::default();
        s.record_fallback_only();
        assert_eq!(s.total_calls, 1);
        assert_eq!(s.external_calls, 0);
        assert_eq!(s.fallback_calls, 1);
        assert_eq!(s.external_success_ratio(), None);
    }

    #[test]
    fn stats_saturating_add_does_not_overflow() {
        let mut s = ArithmosFallbackStats {
            total_calls: u64::MAX, external_calls: u64::MAX,
            external_successes: u64::MAX, fallback_calls: u64::MAX,
        };
        s.record_external_success();
        // saturating_add caps at u64::MAX rather than wrapping.
        assert_eq!(s.total_calls, u64::MAX);
        assert_eq!(s.external_calls, u64::MAX);
        assert_eq!(s.external_successes, u64::MAX);
    }

    #[test]
    fn stats_reset_zeros_everything() {
        let mut s = ArithmosFallbackStats::default();
        s.record_external_success();
        s.record_fallback_only();
        s.reset();
        assert_eq!(s, ArithmosFallbackStats::default());
    }

    // â”€â”€â”€ trait â€” stub impl â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Pretend "sin" function â€” sums all f64 args (intentionally trivial)
    /// to exercise the trait surface.
    struct StubSin;
    impl ArithmosFallbackFunction<f64> for StubSin {
        fn function_id(&self) -> &str { "stub_sin" }
        fn rust_implementation(&self, args: &[f64]) -> Result<f64, String> {
            if args.len() != 1 {
                return Err("stub_sin expects 1 arg".into());
            }
            Ok(args[0].sin())
        }
    }

    #[test]
    fn trait_default_external_name_is_none() {
        let s = StubSin;
        assert!(s.external_function_name().is_none());
    }

    #[test]
    fn trait_default_prefers_external() {
        let s = StubSin;
        assert!(s.prefer_external());
    }

    #[test]
    fn trait_function_id_is_stable() {
        let s = StubSin;
        assert_eq!(s.function_id(), "stub_sin");
    }

    #[test]
    fn trait_rust_implementation_works() {
        let s = StubSin;
        let r = s.rust_implementation(&[0.0]).unwrap();
        assert!(r.abs() < 1e-12);
        let r = s.rust_implementation(&[std::f64::consts::FRAC_PI_2]).unwrap();
        assert!((r - 1.0).abs() < 1e-12);
    }

    #[test]
    fn trait_rust_implementation_validates_arity() {
        let s = StubSin;
        assert!(s.rust_implementation(&[]).is_err());
        assert!(s.rust_implementation(&[1.0, 2.0]).is_err());
    }
}

