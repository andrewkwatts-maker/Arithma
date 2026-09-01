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
//! [`ArithmaBackend`]: it wraps a caller-supplied closure, so a host crate
//! (EML-Math, pt-arithmos engine glue) can plug its own evaluator into the
//! registry without this crate depending on it.
//!
//! Deliberately closure-based rather than operator-table-based: an operator
//! table needs a stable `ArithmaFunction -> &str` mapping, and that mapping
//! currently lives behind the `python` feature in `pyfacade`. Promoting it into
//! `crate::function::tag` is tracked separately; until then a closure keeps this
//! backend honest and fully functional.

use crate::expression::ArithmaExpression;
use crate::external::registry::{ArithmaBackend, ArithmaExternalFunctionError};

/// The signature a host supplies to evaluate an expression.
pub type ArithmaRustHandler = Box<
    dyn Fn(&ArithmaExpression) -> Result<ArithmaExpression, ArithmaExternalFunctionError>
        + Send
        + Sync,
>;

/// A registry backend backed by an in-process Rust closure.
pub struct ArithmaRustExecutor {
    name: &'static str,
    handler: Option<ArithmaRustHandler>,
}

impl ArithmaRustExecutor {
    /// An executor with no handler bound. Every `try_evaluate` reports
    /// [`ArithmaExternalFunctionError::BackendUnavailable`], which the router
    /// reads as "fall through to the next backend".
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            handler: None,
        }
    }

    /// Bind an evaluation handler.
    pub fn with_handler(name: &'static str, handler: ArithmaRustHandler) -> Self {
        Self {
            name,
            handler: Some(handler),
        }
    }

    /// Whether a handler is bound.
    pub fn is_bound(&self) -> bool {
        self.handler.is_some()
    }
}

impl ArithmaBackend for ArithmaRustExecutor {
    fn name(&self) -> &'static str {
        self.name
    }

    fn try_evaluate(
        &self,
        expr: &ArithmaExpression,
    ) -> Result<ArithmaExpression, ArithmaExternalFunctionError> {
        match &self.handler {
            Some(h) => h(expr),
            None => Err(ArithmaExternalFunctionError::BackendUnavailable(
                self.name.to_string(),
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Backward-compatibility aliases for the pre-rename `Arithmos*` names.
// Retained for one release; downstream (eml-math, eml-spectral, metaphysica,
// periodica) should migrate to the `Arithma*` names above.
// ---------------------------------------------------------------------------
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaRustExecutor`")]
#[allow(unused)]
pub use self::ArithmaRustExecutor as ArithmosRustExecutor;
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaRustHandler`")]
#[allow(unused)]
pub use self::ArithmaRustHandler as ArithmosRustHandler;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unbound_executor_reports_unavailable() {
        let exec = ArithmaRustExecutor::new("rust-test");
        assert!(!exec.is_bound());
        let expr = ArithmaExpression::var("x");
        match exec.try_evaluate(&expr) {
            Err(ArithmaExternalFunctionError::BackendUnavailable(n)) => {
                assert_eq!(n, "rust-test")
            }
            other => panic!("expected BackendUnavailable, got {other:?}"),
        }
    }

    #[test]
    fn bound_executor_calls_handler() {
        let exec = ArithmaRustExecutor::with_handler(
            "rust-test",
            Box::new(|_e| Ok(ArithmaExpression::from_i64(42))),
        );
        assert!(exec.is_bound());
        let out = exec.try_evaluate(&ArithmaExpression::var("x")).unwrap();
        assert_eq!(out.to_f64(), Some(42.0));
    }

    #[test]
    fn handler_errors_propagate() {
        let exec = ArithmaRustExecutor::with_handler(
            "rust-test",
            Box::new(|_e| {
                Err(ArithmaExternalFunctionError::EvaluationFailed(
                    "boom".into(),
                ))
            }),
        );
        match exec.try_evaluate(&ArithmaExpression::var("x")) {
            Err(ArithmaExternalFunctionError::EvaluationFailed(m)) => assert_eq!(m, "boom"),
            other => panic!("expected EvaluationFailed, got {other:?}"),
        }
    }

    #[test]
    fn registers_into_the_registry() {
        use crate::external::registry::ArithmaExternalFunctionRegistry;
        let mut r = ArithmaExternalFunctionRegistry::new();
        r.register(Box::new(ArithmaRustExecutor::new("rust-test")));
        assert_eq!(r.backends().len(), 1);
        assert_eq!(r.backends()[0].name(), "rust-test");
    }
}
