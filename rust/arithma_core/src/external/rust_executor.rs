//====== Arithma/rust/arithma_core/src/external/rust_executor.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! In-process Rust backend for the external-function registry.
//!
//! Enabled by the `rust-support` feature. This is the simplest possible
//! [`ArithmosBackend`]: it wraps a caller-supplied closure, so a host crate
//! (EML-Math, pt-arithmos engine glue) can plug its own evaluator into the
//! registry without this crate depending on it.
//!
//! Deliberately closure-based rather than operator-table-based: an operator
//! table needs a stable `ArithmosFunction -> &str` mapping, and that mapping
//! currently lives behind the `python` feature in `pyfacade`. Promoting it into
//! `crate::function::tag` is tracked separately; until then a closure keeps this
//! backend honest and fully functional.

use crate::expression::ArithmosExpression;
use crate::external::registry::{ArithmosBackend, ArithmosExternalFunctionError};

/// The signature a host supplies to evaluate an expression.
pub type ArithmosRustHandler = Box<
    dyn Fn(&ArithmosExpression) -> Result<ArithmosExpression, ArithmosExternalFunctionError>
        + Send
        + Sync,
>;

/// A registry backend backed by an in-process Rust closure.
pub struct ArithmosRustExecutor {
    name: &'static str,
    handler: Option<ArithmosRustHandler>,
}

impl ArithmosRustExecutor {
    /// An executor with no handler bound. Every `try_evaluate` reports
    /// [`ArithmosExternalFunctionError::BackendUnavailable`], which the router
    /// reads as "fall through to the next backend".
    pub fn new(name: &'static str) -> Self {
        Self { name, handler: None }
    }

    /// Bind an evaluation handler.
    pub fn with_handler(name: &'static str, handler: ArithmosRustHandler) -> Self {
        Self { name, handler: Some(handler) }
    }

    /// Whether a handler is bound.
    pub fn is_bound(&self) -> bool {
        self.handler.is_some()
    }
}

impl ArithmosBackend for ArithmosRustExecutor {
    fn name(&self) -> &'static str {
        self.name
    }

    fn try_evaluate(
        &self,
        expr: &ArithmosExpression,
    ) -> Result<ArithmosExpression, ArithmosExternalFunctionError> {
        match &self.handler {
            Some(h) => h(expr),
            None => Err(ArithmosExternalFunctionError::BackendUnavailable(
                self.name.to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unbound_executor_reports_unavailable() {
        let exec = ArithmosRustExecutor::new("rust-test");
        assert!(!exec.is_bound());
        let expr = ArithmosExpression::var("x");
        match exec.try_evaluate(&expr) {
            Err(ArithmosExternalFunctionError::BackendUnavailable(n)) => {
                assert_eq!(n, "rust-test")
            }
            other => panic!("expected BackendUnavailable, got {other:?}"),
        }
    }

    #[test]
    fn bound_executor_calls_handler() {
        let exec = ArithmosRustExecutor::with_handler(
            "rust-test",
            Box::new(|_e| Ok(ArithmosExpression::from_i64(42))),
        );
        assert!(exec.is_bound());
        let out = exec.try_evaluate(&ArithmosExpression::var("x")).unwrap();
        assert_eq!(out.to_f64(), Some(42.0));
    }

    #[test]
    fn handler_errors_propagate() {
        let exec = ArithmosRustExecutor::with_handler(
            "rust-test",
            Box::new(|_e| {
                Err(ArithmosExternalFunctionError::EvaluationFailed("boom".into()))
            }),
        );
        match exec.try_evaluate(&ArithmosExpression::var("x")) {
            Err(ArithmosExternalFunctionError::EvaluationFailed(m)) => assert_eq!(m, "boom"),
            other => panic!("expected EvaluationFailed, got {other:?}"),
        }
    }

    #[test]
    fn registers_into_the_registry() {
        use crate::external::registry::ArithmosExternalFunctionRegistry;
        let mut r = ArithmosExternalFunctionRegistry::new();
        r.register(Box::new(ArithmosRustExecutor::new("rust-test")));
        assert_eq!(r.backends().len(), 1);
        assert_eq!(r.backends()[0].name(), "rust-test");
    }
}
