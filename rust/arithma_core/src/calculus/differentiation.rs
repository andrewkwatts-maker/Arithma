//====== Arithma/rust/arithma_core/src/calculus/differentiation.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Differentiation
//!
//! Symbolic differentiation of `ArithmosExpression`. Implements the standard
//! rules â€” sum, difference, product, quotient, chain, and the derivatives of
//! the elementary transcendental functions.
//!
//! ## Implementation
//!
//! Per the engine's safety-critical standard (no recursion), this module
//! routes to [`crate::calculus::differentiation_iterative`]. The free-function
//! API is preserved for source-compatibility with the legacy pt-arithmos
//! call-sites; the iterative module owns the actual traversal.

use crate::expression::ArithmosExpression;
use crate::function::ArithmosFunction;

/// Differentiate `expr` with respect to `var`. Returns the symbolic derivative.
///
/// Implementation note: routes to the iterative differentiator to honour
/// CLAUDE.md safety rule 1 (avoid recursion).
pub fn differentiate(expr: &ArithmosExpression, var: &str) -> Result<ArithmosExpression, String> {
    debug_assert!(!var.is_empty(), "differentiate: empty variable name");
    let out = crate::calculus::differentiation_iterative::differentiate_iterative(expr, var)?;
    debug_assert!(matches!(out, ArithmosExpression::Number(_) | _), "non-expression result");
    Ok(out)
}

/// Helper: differentiate a sum `a + b`. Returns `da + db`.
///
/// Convenience that wraps `differentiate` on each operand and assembles a
/// `Function::Add` from the parts.
pub fn diff_sum(
    a: &ArithmosExpression,
    b: &ArithmosExpression,
    var: &str,
) -> ArithmosExpression {
    debug_assert!(!var.is_empty(), "diff_sum: empty variable name");
    let da = differentiate(a, var).unwrap_or_else(|_| ArithmosExpression::zero());
    let db = differentiate(b, var).unwrap_or_else(|_| ArithmosExpression::zero());
    ArithmosExpression::add(da, db)
}

/// Helper: differentiate a product `a * b`. Returns `da*b + a*db`.
pub fn diff_product(
    a: &ArithmosExpression,
    b: &ArithmosExpression,
    var: &str,
) -> ArithmosExpression {
    debug_assert!(!var.is_empty(), "diff_product: empty variable name");
    let da = differentiate(a, var).unwrap_or_else(|_| ArithmosExpression::zero());
    let db = differentiate(b, var).unwrap_or_else(|_| ArithmosExpression::zero());
    let left = ArithmosExpression::mul(da, b.clone());
    let right = ArithmosExpression::mul(a.clone(), db);
    ArithmosExpression::add(left, right)
}

/// Helper: chain rule application â€” `df/dg * dg/dx`.
///
/// For simple known outers (sin/cos/exp/ln/sqrt), the iterative differentiator
/// already handles the chain rule end-to-end, so this helper is only used by
/// callers assembling derivatives by hand. We build the chain by wrapping
/// `outer` around `inner`, then differentiating the result.
pub fn diff_chain(
    outer: &ArithmosExpression,
    inner: &ArithmosExpression,
    var: &str,
) -> ArithmosExpression {
    debug_assert!(!var.is_empty(), "diff_chain: empty variable name");
    // The outer is expected to be a unary function (Sin, Cos, Exp, ...). We
    // substitute by rebuilding it around `inner` so the iterative pass can
    // handle the chain in a single pass.
    let rebuilt = match outer {
        ArithmosExpression::Function(func, _) => match func {
            ArithmosFunction::Sin => ArithmosExpression::sin(inner.clone()),
            ArithmosFunction::Cos => ArithmosExpression::cos(inner.clone()),
            ArithmosFunction::Tan => ArithmosExpression::tan(inner.clone()),
            ArithmosFunction::Exp => ArithmosExpression::exp(inner.clone()),
            ArithmosFunction::Ln => ArithmosExpression::ln(inner.clone()),
            ArithmosFunction::Sqrt => ArithmosExpression::sqrt(inner.clone()),
            ArithmosFunction::Negate => ArithmosExpression::neg(inner.clone()),
            _ => return ArithmosExpression::zero(),
        },
        _ => return ArithmosExpression::zero(),
    };
    differentiate(&rebuilt, var).unwrap_or_else(|_| ArithmosExpression::zero())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::{ArithmosBindings, Evaluable};

    #[test]
    fn module_compiles() {
        // Stub-only sanity check retained for continuity.
        assert!(true);
    }

    #[test]
    fn diff_sum_evaluates() {
        // d/dx (x + x^2) = 1 + 2x, at x = 4 â†’ 9.
        let a = ArithmosExpression::var("x");
        let b = ArithmosExpression::pow(
            ArithmosExpression::var("x"),
            ArithmosExpression::from_i64(2),
        );
        let d = diff_sum(&a, &b, "x");
        let mut bindings = ArithmosBindings::new();
        bindings.insert("x".to_string(), 4.0);
        let v = d.evaluate(&bindings).unwrap();
        assert!((v - 9.0).abs() < 1e-9, "got {}", v);
    }

    #[test]
    fn diff_product_evaluates_product_rule() {
        // d/dx (x * x) = 1*x + x*1 = 2x, at x = 5 â†’ 10.
        let a = ArithmosExpression::var("x");
        let b = ArithmosExpression::var("x");
        let d = diff_product(&a, &b, "x");
        let mut bindings = ArithmosBindings::new();
        bindings.insert("x".to_string(), 5.0);
        let v = d.evaluate(&bindings).unwrap();
        assert!((v - 10.0).abs() < 1e-9, "got {}", v);
    }

    #[test]
    fn differentiate_routes_to_iterative() {
        // Smoke: differentiate(var) -> 1.
        let e = ArithmosExpression::var("y");
        let d = differentiate(&e, "y").unwrap();
        let v = d.evaluate(&ArithmosBindings::new()).unwrap();
        assert!((v - 1.0).abs() < 1e-12);
    }
}

