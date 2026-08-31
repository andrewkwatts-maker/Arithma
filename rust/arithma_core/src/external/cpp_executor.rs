//====== Arithma/rust/arithma_core/src/external/cpp_executor.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! C-ABI backend for the external-function registry.
//!
//! Enabled by the `cpp-support` feature. The wire format is JSON: the
//! expression is serialised with the AST's existing `serde` derives, handed to
//! a caller-supplied `extern "C"` function, and the reply is parsed back into
//! an [`ArithmosExpression`]. JSON rather than a repr-C struct graph because
//! the AST is a recursive enum with owned `String`s and `Vec`s — mirroring that
//! across the FFI boundary would mean hand-managing the lifetime of every node.
//!
//! The host is responsible for the C++ side; this module is only the seam.

use std::ffi::{c_char, c_int, CStr, CString};

use crate::expression::ArithmosExpression;
use crate::external::registry::{ArithmosBackend, ArithmosExternalFunctionError};

/// Status codes the C side returns.
pub mod status {
    /// Evaluation succeeded; the output buffer holds JSON.
    pub const OK: i32 = 0;
    /// The backend does not implement this operator — the router falls through.
    pub const UNSUPPORTED: i32 = 1;
    /// The output buffer was too small; nothing was written.
    pub const BUFFER_TOO_SMALL: i32 = 2;
    /// The backend failed; the output buffer may hold a UTF-8 message.
    pub const FAILED: i32 = 3;
}

/// The C entry point.
///
/// `input` is a NUL-terminated JSON expression. The callee writes a
/// NUL-terminated JSON result (or, for [`status::FAILED`], a plain-text
/// message) into `output`, which is `output_len` bytes long. The return value
/// is one of the [`status`] codes.
///
/// # Safety
/// The implementation must not write more than `output_len` bytes to `output`,
/// must NUL-terminate whatever it writes, and must not retain either pointer
/// after returning.
pub type ArithmosCppEvalFn =
    unsafe extern "C" fn(input: *const c_char, output: *mut c_char, output_len: usize) -> c_int;

/// Default reply buffer. Expressions serialise small; the callee can signal
/// [`status::BUFFER_TOO_SMALL`] and the caller retries with a larger buffer.
pub const DEFAULT_BUFFER_BYTES: usize = 64 * 1024;

/// A registry backend that dispatches across a C ABI.
pub struct ArithmosCppExecutor {
    name: &'static str,
    eval_fn: Option<ArithmosCppEvalFn>,
    buffer_bytes: usize,
}

impl ArithmosCppExecutor {
    /// An executor with no C function bound. Reports `BackendUnavailable`.
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            eval_fn: None,
            buffer_bytes: DEFAULT_BUFFER_BYTES,
        }
    }

    /// Bind a C evaluation function.
    ///
    /// # Safety
    /// `eval_fn` must uphold the contract documented on [`ArithmosCppEvalFn`].
    pub unsafe fn with_handler(name: &'static str, eval_fn: ArithmosCppEvalFn) -> Self {
        Self {
            name,
            eval_fn: Some(eval_fn),
            buffer_bytes: DEFAULT_BUFFER_BYTES,
        }
    }

    /// Override the reply buffer size.
    pub fn with_buffer_bytes(mut self, bytes: usize) -> Self {
        self.buffer_bytes = bytes.max(1);
        self
    }

    /// Whether a C function is bound.
    pub fn is_bound(&self) -> bool {
        self.eval_fn.is_some()
    }
}

impl ArithmosBackend for ArithmosCppExecutor {
    fn name(&self) -> &'static str {
        self.name
    }

    fn try_evaluate(
        &self,
        expr: &ArithmosExpression,
    ) -> Result<ArithmosExpression, ArithmosExternalFunctionError> {
        let eval_fn = self.eval_fn.ok_or_else(|| {
            ArithmosExternalFunctionError::BackendUnavailable(self.name.to_string())
        })?;

        let json = serde_json::to_string(expr).map_err(|e| {
            ArithmosExternalFunctionError::EvaluationFailed(format!("serialise: {e}"))
        })?;
        let input = CString::new(json).map_err(|e| {
            ArithmosExternalFunctionError::EvaluationFailed(format!("interior NUL: {e}"))
        })?;

        let mut buf = vec![0_u8; self.buffer_bytes];

        // SAFETY: `input` is a live NUL-terminated CString for the duration of
        // the call; `buf` is a live allocation of exactly `buffer_bytes`, and
        // that same length is passed as `output_len`. Neither pointer is
        // retained by the callee (documented on ArithmosCppEvalFn). No Rust
        // object is aliased while the call is in flight.
        let code = unsafe {
            eval_fn(
                input.as_ptr(),
                buf.as_mut_ptr() as *mut c_char,
                self.buffer_bytes,
            )
        };

        // Read back as a C string so a short reply doesn't drag trailing NULs.
        let reply = {
            // SAFETY: the callee contract requires a NUL within `output_len`
            // bytes; `buf` is zero-initialised, so even a callee that writes
            // nothing leaves a valid empty C string.
            let cstr = unsafe { CStr::from_ptr(buf.as_ptr() as *const c_char) };
            cstr.to_string_lossy().into_owned()
        };

        match code {
            status::OK => serde_json::from_str(&reply).map_err(|e| {
                ArithmosExternalFunctionError::EvaluationFailed(format!("parse reply: {e}"))
            }),
            status::UNSUPPORTED => Err(ArithmosExternalFunctionError::OperatorUnsupported {
                backend: self.name.to_string(),
                op: reply,
            }),
            status::BUFFER_TOO_SMALL => Err(ArithmosExternalFunctionError::EvaluationFailed(
                format!("reply exceeded {} byte buffer", self.buffer_bytes),
            )),
            _ => Err(ArithmosExternalFunctionError::EvaluationFailed(
                if reply.is_empty() {
                    format!("backend returned status {code}")
                } else {
                    reply
                },
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes `src` into `out` (capacity `cap`) with a NUL, mimicking a
    /// well-behaved C callee.
    unsafe fn write_reply(out: *mut c_char, cap: usize, src: &str) -> bool {
        let bytes = src.as_bytes();
        if bytes.len() + 1 > cap {
            return false;
        }
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out as *mut u8, bytes.len());
        *out.add(bytes.len()) = 0;
        true
    }

    unsafe extern "C" fn echo_42(_i: *const c_char, o: *mut c_char, n: usize) -> c_int {
        let expr = ArithmosExpression::from_i64(42);
        let json = serde_json::to_string(&expr).unwrap();
        if write_reply(o, n, &json) {
            status::OK
        } else {
            status::BUFFER_TOO_SMALL
        }
    }

    unsafe extern "C" fn unsupported(_i: *const c_char, o: *mut c_char, n: usize) -> c_int {
        write_reply(o, n, "gamma");
        status::UNSUPPORTED
    }

    unsafe extern "C" fn fails(_i: *const c_char, o: *mut c_char, n: usize) -> c_int {
        write_reply(o, n, "divide by zero");
        status::FAILED
    }

    unsafe extern "C" fn too_small(_i: *const c_char, _o: *mut c_char, _n: usize) -> c_int {
        status::BUFFER_TOO_SMALL
    }

    #[test]
    fn unbound_executor_reports_unavailable() {
        let exec = ArithmosCppExecutor::new("cpp-test");
        assert!(!exec.is_bound());
        match exec.try_evaluate(&ArithmosExpression::var("x")) {
            Err(ArithmosExternalFunctionError::BackendUnavailable(n)) => assert_eq!(n, "cpp-test"),
            other => panic!("expected BackendUnavailable, got {other:?}"),
        }
    }

    #[test]
    fn round_trips_a_successful_reply() {
        let exec = unsafe { ArithmosCppExecutor::with_handler("cpp-test", echo_42) };
        let out = exec.try_evaluate(&ArithmosExpression::var("x")).unwrap();
        assert_eq!(out.to_f64(), Some(42.0));
    }

    #[test]
    fn unsupported_carries_the_operator_name() {
        let exec = unsafe { ArithmosCppExecutor::with_handler("cpp-test", unsupported) };
        match exec.try_evaluate(&ArithmosExpression::var("x")) {
            Err(ArithmosExternalFunctionError::OperatorUnsupported { backend, op }) => {
                assert_eq!(backend, "cpp-test");
                assert_eq!(op, "gamma");
            }
            other => panic!("expected OperatorUnsupported, got {other:?}"),
        }
    }

    #[test]
    fn failure_surfaces_the_backend_message() {
        let exec = unsafe { ArithmosCppExecutor::with_handler("cpp-test", fails) };
        match exec.try_evaluate(&ArithmosExpression::var("x")) {
            Err(ArithmosExternalFunctionError::EvaluationFailed(m)) => {
                assert_eq!(m, "divide by zero")
            }
            other => panic!("expected EvaluationFailed, got {other:?}"),
        }
    }

    #[test]
    fn buffer_too_small_is_reported_not_silently_truncated() {
        let exec = unsafe { ArithmosCppExecutor::with_handler("cpp-test", too_small) };
        match exec.try_evaluate(&ArithmosExpression::var("x")) {
            Err(ArithmosExternalFunctionError::EvaluationFailed(m)) => {
                assert!(m.contains("byte buffer"), "unexpected message: {m}")
            }
            other => panic!("expected EvaluationFailed, got {other:?}"),
        }
    }

    #[test]
    fn registers_into_the_registry() {
        use crate::external::registry::ArithmosExternalFunctionRegistry;
        let mut r = ArithmosExternalFunctionRegistry::new();
        r.register(Box::new(ArithmosCppExecutor::new("cpp-test")));
        assert_eq!(r.backends()[0].name(), "cpp-test");
    }
}
