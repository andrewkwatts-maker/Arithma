//! `CompiledExpr`: the evaluation step of an expression, resolved once to slots.
//!
//! The expression stays the source of truth (`source()`); this type only runs it.
//! `new` walks the expression exactly as `Evaluable::evaluate` does and emits
//! post-order ops over a variable list; `eval` then runs those ops on a fixed
//! stack array with no heap allocation on the `Ok` path, and returns the same
//! bits `evaluate` would for the same values.

use super::{arithma_apply_function, ArithmaExpression, ARITHMA_EVALUATE_NODE_CAP};
use crate::function::ArithmaFunction;

/// Value-stack capacity of [`CompiledExpr::eval`]; deeper expressions are refused.
pub const COMPILED_STACK: usize = 64;

/// Why an expression could not be compiled.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CompileError {
    /// A variant, function or arity that `evaluate` refuses or cannot be slotted.
    #[error("unsupported: {0}")]
    Unsupported(&'static str),
    /// A `Constant` with no cached value.
    #[error("constant has no cached value")]
    UncachedConstant,
    /// A variable that is not in the variable list.
    #[error("unbound variable `{0}`")]
    Unbound(String),
    /// The variable list names one variable twice.
    #[error("duplicate variable name")]
    DuplicateName,
    /// The value stack would exceed [`COMPILED_STACK`].
    #[error("stack depth {0} exceeds {COMPILED_STACK}")]
    TooDeep(usize),
    /// More ops or variables than the node cap or slot width allows.
    #[error("expression too large")]
    TooLarge,
}

#[derive(Debug, Clone)]
enum Op {
    Const(f64),
    Var(u16),
    Call(ArithmaFunction, u8),
    Select,
}

/// An expression compiled against an ordered variable list.
#[derive(Debug, Clone)]
pub struct CompiledExpr {
    ops: Vec<Op>,
    names: Vec<String>,
    depth: usize,
    source: ArithmaExpression,
}

enum Frame<'a> {
    Enter(&'a ArithmaExpression),
    Func(&'a ArithmaFunction, usize),
    Cond,
}

struct Builder<'a> {
    vars: &'a [&'a str],
    ops: Vec<Op>,
    work: Vec<Frame<'a>>,
    height: usize,
    depth: usize,
}

impl<'a> Builder<'a> {
    fn new(vars: &'a [&'a str]) -> Self {
        Self {
            vars,
            ops: Vec::new(),
            work: Vec::new(),
            height: 0,
            depth: 0,
        }
    }

    fn run(&mut self, root: &'a ArithmaExpression) -> Result<(), CompileError> {
        self.work.push(Frame::Enter(root));
        let mut guard: usize = 0;
        while let Some(frame) = self.work.pop() {
            guard += 1;
            if guard > ARITHMA_EVALUATE_NODE_CAP || self.ops.len() > ARITHMA_EVALUATE_NODE_CAP {
                return Err(CompileError::TooLarge);
            }
            match frame {
                Frame::Enter(node) => self.enter(node)?,
                Frame::Func(func, n) => self.combine_func(func, n)?,
                Frame::Cond => self.combine_cond(),
            }
        }
        if self.height != 1 {
            return Err(CompileError::Unsupported("unbalanced expression"));
        }
        Ok(())
    }

    fn push_value(&mut self, op: Op) -> Result<(), CompileError> {
        self.ops.push(op);
        self.height += 1;
        self.depth = self.depth.max(self.height);
        if self.height > COMPILED_STACK {
            return Err(CompileError::TooDeep(self.height));
        }
        Ok(())
    }

    fn enter(&mut self, node: &'a ArithmaExpression) -> Result<(), CompileError> {
        match node {
            ArithmaExpression::Number(n) => self.push_value(Op::Const(n.to_f64())),
            ArithmaExpression::Rational(r) => self.push_value(Op::Const(r.to_f64())),
            ArithmaExpression::Constant { cached_value, .. } => match *cached_value {
                Some(v) => self.push_value(Op::Const(v)),
                None => Err(CompileError::UncachedConstant),
            },
            ArithmaExpression::Variable(name) => self.variable(name),
            ArithmaExpression::Function(func, args) => {
                self.work.push(Frame::Func(func, args.len()));
                self.work.extend(args.iter().rev().map(Frame::Enter));
                Ok(())
            }
            ArithmaExpression::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                self.work.push(Frame::Cond);
                self.work.push(Frame::Enter(else_expr));
                self.work.push(Frame::Enter(then_expr));
                self.work.push(Frame::Enter(condition));
                Ok(())
            }
            ArithmaExpression::CachedValue { expr, cached, .. } => match *cached {
                Some(v) => self.push_value(Op::Const(v)),
                None => {
                    self.work.push(Frame::Enter(expr));
                    Ok(())
                }
            },
            ArithmaExpression::FourierOptimized { expr, .. } => {
                self.work.push(Frame::Enter(expr));
                Ok(())
            }
            _ => Err(CompileError::Unsupported("expression variant")),
        }
    }

    fn variable(&mut self, name: &str) -> Result<(), CompileError> {
        match self.vars.iter().position(|v| *v == name) {
            Some(slot) => self.push_value(Op::Var(slot as u16)),
            None => Err(CompileError::Unbound(name.to_string())),
        }
    }

    fn combine_func(&mut self, func: &ArithmaFunction, n: usize) -> Result<(), CompileError> {
        if n == 0 {
            return Err(CompileError::Unsupported("zero-argument function"));
        }
        if n > COMPILED_STACK {
            return Err(CompileError::TooDeep(n));
        }
        let probe = vec![1.0_f64; n];
        if arithma_apply_function(func, &probe).is_err() {
            return Err(CompileError::Unsupported("function or arity"));
        }
        self.ops.push(Op::Call(func.clone(), n as u8));
        self.height = self.height + 1 - n;
        Ok(())
    }

    fn combine_cond(&mut self) {
        self.ops.push(Op::Select);
        self.height -= 2;
    }
}

impl CompiledExpr {
    /// Compile `expr` over the ordered variable list `vars`.
    ///
    /// Fails for anything `evaluate` refuses, for a variable missing from
    /// `vars`, for a repeated name, and for an expression deeper than
    /// [`COMPILED_STACK`] values.
    pub fn new(expr: &ArithmaExpression, vars: &[&str]) -> Result<Self, CompileError> {
        if vars.len() > usize::from(u16::MAX) {
            return Err(CompileError::TooLarge);
        }
        for (i, name) in vars.iter().enumerate() {
            if vars[..i].contains(name) {
                return Err(CompileError::DuplicateName);
            }
        }
        let mut builder = Builder::new(vars);
        builder.run(expr)?;
        Ok(Self {
            ops: builder.ops,
            names: vars.iter().map(|v| (*v).to_string()).collect(),
            depth: builder.depth,
            source: expr.clone(),
        })
    }

    /// Run the ops over `vars`, one value per name in [`vars`](Self::vars) order.
    ///
    /// Allocates nothing when it returns `Ok`. Errors where `evaluate` errors.
    pub fn eval(&self, vars: &[f64]) -> Result<f64, String> {
        if vars.len() != self.names.len() {
            return Err(format!(
                "compiled eval: expected {} values, got {}",
                self.names.len(),
                vars.len()
            ));
        }
        let mut stack = [0.0_f64; COMPILED_STACK];
        let mut sp: usize = 0;
        for op in &self.ops {
            match op {
                Op::Const(v) => push(&mut stack, &mut sp, *v)?,
                Op::Var(slot) => push(&mut stack, &mut sp, vars[usize::from(*slot)])?,
                Op::Call(func, n) => call(&mut stack, &mut sp, func, usize::from(*n))?,
                Op::Select => select(&mut stack, &mut sp)?,
            }
        }
        match (sp, stack.first()) {
            (1, Some(v)) => Ok(*v),
            _ => Err(format!("compiled eval: final stack size {sp} != 1")),
        }
    }

    /// The variable names, in slot order.
    pub fn vars(&self) -> &[String] {
        &self.names
    }

    /// The expression this was compiled from.
    pub fn source(&self) -> &ArithmaExpression {
        &self.source
    }

    /// Number of variables `eval` expects.
    pub fn arity(&self) -> usize {
        self.names.len()
    }

    /// Peak value-stack depth of the compiled ops.
    pub fn depth(&self) -> usize {
        self.depth
    }
}

fn push(stack: &mut [f64; COMPILED_STACK], sp: &mut usize, v: f64) -> Result<(), String> {
    let slot = stack.get_mut(*sp).ok_or("compiled eval: stack overflow")?;
    *slot = v;
    *sp += 1;
    Ok(())
}

fn call(
    stack: &mut [f64; COMPILED_STACK],
    sp: &mut usize,
    func: &ArithmaFunction,
    n: usize,
) -> Result<(), String> {
    let start = sp.checked_sub(n).ok_or("compiled eval: stack underflow")?;
    let out = arithma_apply_function(func, &stack[start..*sp])?;
    stack[start] = out;
    *sp = start + 1;
    Ok(())
}

fn select(stack: &mut [f64; COMPILED_STACK], sp: &mut usize) -> Result<(), String> {
    let start = sp.checked_sub(3).ok_or("compiled eval: conditional underflow")?;
    let (cond, then_v, else_v) = (stack[start], stack[start + 1], stack[start + 2]);
    stack[start] = if cond != 0.0 { then_v } else { else_v };
    *sp = start + 1;
    Ok(())
}
