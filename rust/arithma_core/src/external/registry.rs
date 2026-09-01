//====== Arithma/rust/arithma_core/src/external/registry.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! Pluggable external-function registry.
//!
//! Wave-2 stub: defines the trait surface so pt-eml-bridge can compile against
//! it; the in-memory implementation populates in Wave 3 when the operator
//! mapping table is finalised.

use thiserror::Error;

use crate::expression::ArithmaExpression;

/// Failures that can arise when invoking a registered external function.
#[derive(Debug, Error)]
pub enum ArithmaExternalFunctionError {
    /// Backend is registered by name but not currently loaded.
    #[error("backend `{0}` is not available")]
    BackendUnavailable(String),

    /// Backend is loaded but does not implement the requested operator.
    #[error("backend `{backend}` does not implement `{op}`")]
    OperatorUnsupported { backend: String, op: String },

    /// Argument count mismatch.
    #[error("`{op}` expected {expected} args, got {got}")]
    ArityMismatch {
        op: String,
        expected: usize,
        got: usize,
    },

    /// Free-form runtime failure inside the backend.
    #[error("backend evaluation failed: {0}")]
    EvaluationFailed(String),
}

/// Implemented by every external math backend (pt-arithmos engine glue,
/// EML-Math, future C++/Python executors).
///
/// Per CLAUDE.md §10: this is the *single* registration point — never invent a
/// parallel symbol table.
pub trait ArithmaBackend: Send + Sync {
    /// Stable, unique backend identifier. Used by the capability router to
    /// memoise routing decisions.
    fn name(&self) -> &'static str;

    /// Try to evaluate an expression with this backend.
    ///
    /// Returning `Err(BackendUnavailable)` or `Err(OperatorUnsupported)` is the
    /// router's signal to fall through to the next backend; an
    /// `EvaluationFailed` is fatal for the request.
    fn try_evaluate(
        &self,
        expr: &ArithmaExpression,
    ) -> Result<ArithmaExpression, ArithmaExternalFunctionError>;
}

/// The registry singleton.
///
/// Backends register themselves at engine init; the router queries by name.
/// Wave-2 stub uses a `Vec<Box<dyn ArithmaBackend>>` ordered by registration;
/// Wave 3 swaps to a `DashMap` keyed by name.
pub struct ArithmaExternalFunctionRegistry {
    backends: Vec<Box<dyn ArithmaBackend>>,
}

impl ArithmaExternalFunctionRegistry {
    /// New empty registry. The engine seeds this with the canonical backends.
    pub fn new() -> Self {
        Self {
            backends: Vec::new(),
        }
    }

    /// Register a backend. Last-registered wins ties — the engine should
    /// register pt-arithmos engine glue first, then Arithma core, then EML.
    pub fn register(&mut self, backend: Box<dyn ArithmaBackend>) {
        self.backends.push(backend);
    }

    /// Iterate the registered backends in registration order.
    pub fn backends(&self) -> &[Box<dyn ArithmaBackend>] {
        &self.backends
    }
}

impl Default for ArithmaExternalFunctionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Backward-compatibility aliases for the pre-rename `Arithmos*` names.
// Retained for one release; downstream (eml-math, eml-spectral, metaphysica,
// periodica) should migrate to the `Arithma*` names above.
// ---------------------------------------------------------------------------
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaBackend`")]
#[allow(unused)]
pub use self::ArithmaBackend as ArithmosBackend;
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaExternalFunctionError`")]
#[allow(unused)]
pub use self::ArithmaExternalFunctionError as ArithmosExternalFunctionError;
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaExternalFunctionRegistry`")]
#[allow(unused)]
pub use self::ArithmaExternalFunctionRegistry as ArithmosExternalFunctionRegistry;

#[cfg(test)]
mod tests {
    use super::*;

    struct StubBackend;
    impl ArithmaBackend for StubBackend {
        fn name(&self) -> &'static str {
            "stub"
        }
        fn try_evaluate(
            &self,
            _expr: &ArithmaExpression,
        ) -> Result<ArithmaExpression, ArithmaExternalFunctionError> {
            Err(ArithmaExternalFunctionError::BackendUnavailable(
                "stub".into(),
            ))
        }
    }

    #[test]
    fn empty_registry_has_no_backends() {
        let r = ArithmaExternalFunctionRegistry::new();
        assert_eq!(r.backends().len(), 0);
    }

    #[test]
    fn register_appends_backend() {
        let mut r = ArithmaExternalFunctionRegistry::new();
        r.register(Box::new(StubBackend));
        assert_eq!(r.backends().len(), 1);
        assert_eq!(r.backends()[0].name(), "stub");
    }
}
