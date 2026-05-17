//====== Arithma/rust/arithma_core/src/calculus/differentiation_iterative.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Iterative differentiation
//!
//! Stack-based, non-recursive differentiation. The preferred entry point for
//! the engine because it satisfies the safety-critical rule "avoid recursion"
//! while remaining behaviourally identical to [`super::differentiation`].
//!
//! ## Algorithm
//!
//! We perform a post-order traversal of the expression tree using two stacks:
//!
//! 1. A *work* stack of `Frame` values driving the traversal.
//! 2. A *result* stack of derivative sub-expressions and their original
//!    operands (kept side-by-side so combine frames can read both).
//!
//! Each `Enter` frame either produces an atomic derivative (for `Number`,
//! `Constant`, `Variable`) or schedules child-`Enter` frames followed by a
//! single `Combine` frame describing how to assemble the parent's derivative
//! from the child derivatives that the work stack will leave behind.

use crate::expression::ArithmosExpression;
use crate::function::ArithmosFunction;

/// Hard cap on visited tree nodes. Bounded loops per CLAUDE.md safety rule 2.
const ARITHMOS_DIFF_NODE_CAP: usize = 1_048_576;

/// What to do once a node's child derivatives are on the result stack.
#[derive(Debug, Clone)]
enum Combine {
    /// d/dx (a + b) = a' + b'
    Add,
    /// d/dx (a - b) = a' - b'
    Sub,
    /// d/dx (a * b) = a'*b + a*b'
    Mul { a: ArithmosExpression, b: ArithmosExpression },
    /// d/dx (a / b) = (a'*b - a*b') / b^2
    Div { a: ArithmosExpression, b: ArithmosExpression },
    /// d/dx (a ^ k) = k * a^(k-1) * a'   when k is constant w.r.t. var
    /// We only support constant-exponent for the iterative pass; the general
    /// rule produces a `Function::Power` shape callers can simplify later.
    Pow { base: ArithmosExpression, exponent: ArithmosExpression },
    /// d/dx (-a) = -a'
    Neg,
    /// d/dx sqrt(a) = a' / (2 sqrt(a))
    Sqrt { inner: ArithmosExpression },
    /// d/dx exp(a) = a' * exp(a)
    Exp { inner: ArithmosExpression },
    /// d/dx ln(a) = a' / a
    Ln { inner: ArithmosExpression },
    /// d/dx sin(a) = a' * cos(a)
    Sin { inner: ArithmosExpression },
    /// d/dx cos(a) = -a' * sin(a)
    Cos { inner: ArithmosExpression },
    /// d/dx tan(a) = a' / cos(a)^2
    Tan { inner: ArithmosExpression },
}

/// Worklist frame.
enum Frame<'a> {
    Enter(&'a ArithmosExpression),
    Combine(Combine),
}

/// Iterative implementation of symbolic differentiation. Uses an explicit work
/// stack so depth is bounded by available heap, not by the OS thread stack.
pub fn differentiate_iterative(
    expr: &ArithmosExpression,
    var: &str,
) -> Result<ArithmosExpression, String> {
    let mut diff = ArithmosIterativeDifferentiator::new();
    diff.differentiate(expr, var)
}

/// Stateful iterative differentiator that reuses its own work stack.
#[derive(Debug, Default)]
pub struct ArithmosIterativeDifferentiator {
    /// Re-usable stack of (expression, variable) tuples. Empty between calls.
    stack: Vec<(ArithmosExpression, String)>,
}

impl ArithmosIterativeDifferentiator {
    /// Construct a fresh differentiator.
    pub fn new() -> Self {
        Self { stack: Vec::new() }
    }

    /// Differentiate `expr` with respect to `var`.
    pub fn differentiate(
        &mut self,
        expr: &ArithmosExpression,
        var: &str,
    ) -> Result<ArithmosExpression, String> {
        self.stack.clear();
        debug_assert!(self.stack.is_empty(), "stack not cleared");
        debug_assert!(!var.is_empty(), "differentiate: empty variable name");

        let mut work: Vec<Frame> = Vec::with_capacity(32);
        let mut results: Vec<ArithmosExpression> = Vec::with_capacity(32);
        work.push(Frame::Enter(expr));

        let mut guard: usize = 0;
        while let Some(frame) = work.pop() {
            guard += 1;
            if guard > ARITHMOS_DIFF_NODE_CAP {
                return Err("differentiate: node cap exceeded".into());
            }
            match frame {
                Frame::Enter(node) => enter_node(node, var, &mut work, &mut results)?,
                Frame::Combine(c) => combine_frame(c, &mut results)?,
            }
        }
        if results.len() != 1 {
            return Err(format!("differentiate: final stack size {}", results.len()));
        }
        Ok(results.pop().unwrap())
    }
}

/// Handle one `Enter` frame: either emit an atomic derivative or schedule
/// child traversal followed by a `Combine` frame.
fn enter_node<'a>(
    node: &'a ArithmosExpression,
    var: &str,
    work: &mut Vec<Frame<'a>>,
    results: &mut Vec<ArithmosExpression>,
) -> Result<(), String> {
    debug_assert!(!var.is_empty(), "enter_node: empty variable");
    match node {
        ArithmosExpression::Number(_) | ArithmosExpression::Constant { .. } => {
            results.push(ArithmosExpression::zero());
            Ok(())
        }
        ArithmosExpression::Variable(name) => {
            let dv = if name == var {
                ArithmosExpression::from_i64(1)
            } else {
                ArithmosExpression::zero()
            };
            results.push(dv);
            Ok(())
        }
        ArithmosExpression::Function(func, args) => {
            schedule_function(func, args, work)
        }
        _ => Err(format!(
            "differentiate: unsupported variant for variable '{}'",
            var
        )),
    }
}

/// Schedule child evaluations and a matching `Combine` for a function node.
fn schedule_function<'a>(
    func: &ArithmosFunction,
    args: &'a [ArithmosExpression],
    work: &mut Vec<Frame<'a>>,
) -> Result<(), String> {
    debug_assert!(!args.is_empty(), "schedule_function: empty args");
    match func {
        ArithmosFunction::Add => {
            if args.len() != 2 {
                return Err("Add: expected 2 args".into());
            }
            work.push(Frame::Combine(Combine::Add));
            work.push(Frame::Enter(&args[1]));
            work.push(Frame::Enter(&args[0]));
            Ok(())
        }
        ArithmosFunction::Subtract => {
            if args.len() != 2 {
                return Err("Subtract: expected 2 args".into());
            }
            work.push(Frame::Combine(Combine::Sub));
            work.push(Frame::Enter(&args[1]));
            work.push(Frame::Enter(&args[0]));
            Ok(())
        }
        ArithmosFunction::Multiply => {
            if args.len() != 2 {
                return Err("Multiply: expected 2 args".into());
            }
            work.push(Frame::Combine(Combine::Mul {
                a: args[0].clone(),
                b: args[1].clone(),
            }));
            work.push(Frame::Enter(&args[1]));
            work.push(Frame::Enter(&args[0]));
            Ok(())
        }
        ArithmosFunction::Divide => {
            if args.len() != 2 {
                return Err("Divide: expected 2 args".into());
            }
            work.push(Frame::Combine(Combine::Div {
                a: args[0].clone(),
                b: args[1].clone(),
            }));
            work.push(Frame::Enter(&args[1]));
            work.push(Frame::Enter(&args[0]));
            Ok(())
        }
        ArithmosFunction::Power => {
            if args.len() != 2 {
                return Err("Power: expected 2 args".into());
            }
            work.push(Frame::Combine(Combine::Pow {
                base: args[0].clone(),
                exponent: args[1].clone(),
            }));
            work.push(Frame::Enter(&args[0]));
            Ok(())
        }
        ArithmosFunction::Negate => {
            work.push(Frame::Combine(Combine::Neg));
            work.push(Frame::Enter(&args[0]));
            Ok(())
        }
        ArithmosFunction::Sqrt => {
            work.push(Frame::Combine(Combine::Sqrt { inner: args[0].clone() }));
            work.push(Frame::Enter(&args[0]));
            Ok(())
        }
        ArithmosFunction::Exp => {
            work.push(Frame::Combine(Combine::Exp { inner: args[0].clone() }));
            work.push(Frame::Enter(&args[0]));
            Ok(())
        }
        ArithmosFunction::Ln => {
            work.push(Frame::Combine(Combine::Ln { inner: args[0].clone() }));
            work.push(Frame::Enter(&args[0]));
            Ok(())
        }
        ArithmosFunction::Sin => {
            work.push(Frame::Combine(Combine::Sin { inner: args[0].clone() }));
            work.push(Frame::Enter(&args[0]));
            Ok(())
        }
        ArithmosFunction::Cos => {
            work.push(Frame::Combine(Combine::Cos { inner: args[0].clone() }));
            work.push(Frame::Enter(&args[0]));
            Ok(())
        }
        ArithmosFunction::Tan => {
            work.push(Frame::Combine(Combine::Tan { inner: args[0].clone() }));
            work.push(Frame::Enter(&args[0]));
            Ok(())
        }
        other => Err(format!("differentiate: unsupported function {:?}", other)),
    }
}

/// Pop one or two child derivatives and push the assembled parent derivative.
fn combine_frame(c: Combine, results: &mut Vec<ArithmosExpression>) -> Result<(), String> {
    debug_assert!(!results.is_empty(), "combine_frame: empty results");
    match c {
        Combine::Add => {
            let db = pop_one(results)?;
            let da = pop_one(results)?;
            results.push(ArithmosExpression::add(da, db));
        }
        Combine::Sub => {
            let db = pop_one(results)?;
            let da = pop_one(results)?;
            results.push(ArithmosExpression::sub(da, db));
        }
        Combine::Mul { a, b } => {
            let db = pop_one(results)?;
            let da = pop_one(results)?;
            let left = ArithmosExpression::mul(da, b);
            let right = ArithmosExpression::mul(a, db);
            results.push(ArithmosExpression::add(left, right));
        }
        Combine::Div { a, b } => {
            let db = pop_one(results)?;
            let da = pop_one(results)?;
            let num_left = ArithmosExpression::mul(da, b.clone());
            let num_right = ArithmosExpression::mul(a, db);
            let num = ArithmosExpression::sub(num_left, num_right);
            let denom = ArithmosExpression::mul(b.clone(), b);
            results.push(ArithmosExpression::div(num, denom));
        }
        Combine::Pow { base, exponent } => {
            // We currently support the constant-exponent power rule.
            let da = pop_one(results)?;
            let new_exp = ArithmosExpression::sub(exponent.clone(), ArithmosExpression::from_i64(1));
            let base_pow = ArithmosExpression::pow(base, new_exp);
            let scaled = ArithmosExpression::mul(exponent, base_pow);
            results.push(ArithmosExpression::mul(scaled, da));
        }
        Combine::Neg => {
            let da = pop_one(results)?;
            results.push(ArithmosExpression::neg(da));
        }
        Combine::Sqrt { inner } => {
            let da = pop_one(results)?;
            let two = ArithmosExpression::from_i64(2);
            let denom = ArithmosExpression::mul(two, ArithmosExpression::sqrt(inner));
            results.push(ArithmosExpression::div(da, denom));
        }
        Combine::Exp { inner } => {
            let da = pop_one(results)?;
            results.push(ArithmosExpression::mul(da, ArithmosExpression::exp(inner)));
        }
        Combine::Ln { inner } => {
            let da = pop_one(results)?;
            results.push(ArithmosExpression::div(da, inner));
        }
        Combine::Sin { inner } => {
            let da = pop_one(results)?;
            results.push(ArithmosExpression::mul(da, ArithmosExpression::cos(inner)));
        }
        Combine::Cos { inner } => {
            let da = pop_one(results)?;
            let neg_sin = ArithmosExpression::neg(ArithmosExpression::sin(inner));
            results.push(ArithmosExpression::mul(da, neg_sin));
        }
        Combine::Tan { inner } => {
            let da = pop_one(results)?;
            let cos_inner = ArithmosExpression::cos(inner);
            let cos_sq = ArithmosExpression::mul(cos_inner.clone(), cos_inner);
            results.push(ArithmosExpression::div(da, cos_sq));
        }
    }
    Ok(())
}

fn pop_one(results: &mut Vec<ArithmosExpression>) -> Result<ArithmosExpression, String> {
    results
        .pop()
        .ok_or_else(|| "differentiate: result stack underflow".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::{ArithmosBindings, Evaluable};

    #[test]
    fn new_differentiator_starts_empty() {
        let d = ArithmosIterativeDifferentiator::new();
        assert!(d.stack.is_empty());
    }

    #[test]
    fn derivative_of_variable_is_one() {
        let expr = ArithmosExpression::var("x");
        let d = differentiate_iterative(&expr, "x").unwrap();
        let v = d.evaluate(&ArithmosBindings::new()).unwrap();
        assert!((v - 1.0).abs() < 1e-12);
    }

    #[test]
    fn derivative_of_constant_is_zero() {
        let expr = ArithmosExpression::from_i64(42);
        let d = differentiate_iterative(&expr, "x").unwrap();
        let v = d.evaluate(&ArithmosBindings::new()).unwrap();
        assert!(v.abs() < 1e-12);
    }

    #[test]
    fn derivative_of_x_squared_at_three_is_six() {
        // d/dx x^2 = 2x, evaluated at x=3 gives 6.
        let expr = ArithmosExpression::pow(
            ArithmosExpression::var("x"),
            ArithmosExpression::from_i64(2),
        );
        let d = differentiate_iterative(&expr, "x").unwrap();
        let mut bindings = ArithmosBindings::new();
        bindings.insert("x".to_string(), 3.0);
        let v = d.evaluate(&bindings).unwrap();
        assert!((v - 6.0).abs() < 1e-9, "got {}", v);
    }

    #[test]
    fn derivative_of_sin_x_is_cos_x() {
        let expr = ArithmosExpression::sin(ArithmosExpression::var("x"));
        let d = differentiate_iterative(&expr, "x").unwrap();
        let mut bindings = ArithmosBindings::new();
        bindings.insert("x".to_string(), 0.0);
        let v = d.evaluate(&bindings).unwrap();
        assert!((v - 1.0).abs() < 1e-9, "got {}", v);
    }
}

