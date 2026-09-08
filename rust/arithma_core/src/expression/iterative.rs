//====== Arithma/rust/arithma_core/src/expression/iterative.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Iterative simplifier passes
//!
//! Stack-based traversal of `ArithmaExpression` trees. The engine's
//! safety-critical standards forbid recursion (rule 1: avoid recursion); every
//! pass in this module is implemented with an explicit work stack so the call
//! depth stays O(1) regardless of expression depth.
//!
//! Two flavours:
//!
//! - [`ArithmaIterativeSimplifier`] — stateful simplifier with a work queue.
//! - [`simplify_iterative`] — convenience function that owns its simplifier.
//!
//! ## Rules
//!
//! Constant folding, additive and multiplicative identities, absorbing zero,
//! exact division, the power rules, and **like-term collection** --
//! `x + x -> 2x`, `2x + 3x -> 5x`, `x * x -> x^2`, `x^2 * x -> x^3`. Folding
//! runs on `ArithmaInteger`'s unlimited-precision arithmetic and **refuses
//! rather than rounds**: `7 / 2` stays a quotient, and `2 ^ 64` is exact.
//!
//! Comparison of terms is still **syntactic**, but the operands of the
//! commutative operators are sorted into a canonical order before anything
//! compares them, so `2*x + x*2` arrives as `2*x + 2*x` and collects. The
//! order is total, deterministic and idempotent; the last of those is not a
//! nicety, because a sort that reported a change every pass would stop the
//! fixpoint loop converging at all. See [`compare_expressions`].
//!
//! On top of that, the transcendental functions are folded at the arguments
//! where they have an exact value -- `sin(0) = 0`, `cos(0) = 1`,
//! `exp(0) = 1`, `ln(1) = 0` -- and `sin^2(x) + cos^2(x)` reduces to `1`.
//! Every one of those holds for the whole domain. Identities that hold only
//! on part of it are deliberately absent: `sqrt(x^2)` is `|x|`, not `x`.
//!
//! ## Complexity -- this is a contract, not an observation
//!
//! One pass is **O(n)** in the node count, and both benchmark families in
//! `benches/hot_paths.rs` scale linearly. Two earlier designs did not, and
//! both looked correct:
//!
//! 1. Addressing nodes by a path from the root re-walked the spine per node.
//! 2. Cloning operands before deciding whether to rewrite copied whole
//!    subtrees per node -- a no-op pass over 1,000 symbolic nodes cost 36 ms.
//!
//! The current design detaches children, processes them, and reattaches them,
//! and every rule inspects by reference before taking ownership. A no-op pass
//! over those same 1,000 nodes went to 150 us, and to **322 us** once
//! like-term collection landed -- that pass now also runs a borrow-only
//! pre-check per commutative node. Roughly 2x for a real capability, and
//! still linear. Keep the `expression/simplify/no_op_symbolic` benchmarks in
//! place: a regression here is invisible to the test suite and shows up only
//! under load.
//!
//! Canonical ordering was added under the same constraint, and two decisions
//! come straight from it. [`compare_expressions`] decides on the two nodes'
//! own keys before it touches the heap, so the ordinary case -- operands of
//! different kinds -- costs no allocation and no traversal. And the ranking
//! puts compound operands ahead of atoms, which is the shape a
//! left-associative parse already produces, so canonicalising a tree that was
//! written normally is a comparison per node rather than a rewrite of every
//! node in it.

use core::cmp::Ordering;

use crate::expression::{ArithmaExpression, SimplificationConfig};
use crate::function::ArithmaFunction;
use crate::integer::ArithmaInteger;

/// Iterative, stack-based simplifier.
///
/// Owns its own work stack so multiple invocations can reuse the allocation.
/// Implementations must respect `config.max_iterations` and bail rather than
/// loop forever (CLAUDE.md safety rule 2: all loops have fixed bounds).
#[derive(Debug, Default)]
pub struct ArithmaIterativeSimplifier {
    /// Re-usable work stack. Avoids per-call allocations.
    stack: Vec<ArithmaExpression>,
    /// Iterations consumed by the most recent run.
    last_iterations: usize,
    /// Re-usable frame stack for the bottom-up traversal. Held on the
    /// simplifier so a hot loop of calls allocates once, not per pass.
    work: Vec<Frame>,
}

impl ArithmaIterativeSimplifier {
    /// Create a fresh simplifier with empty state.
    pub fn new() -> Self {
        Self {
            stack: Vec::new(),
            last_iterations: 0,
            work: Vec::new(),
        }
    }

    /// Number of iterations the last `simplify` consumed.
    pub fn last_iterations(&self) -> usize {
        self.last_iterations
    }
}

/// Hard ceiling on nodes visited in one pass.
///
/// Safety-critical standard 2 (all loops have fixed bounds). A pathological or
/// maliciously deep expression stops the pass rather than the process; the
/// caller sees the tree unchanged, which is always a safe answer.
pub const MAX_NODES: usize = 65_536;

/// Hard ceiling on tree depth during traversal.
///
/// The traversal is iterative, so depth costs a path vector rather than stack
/// frames -- but it still needs a bound, and 256 is deliberately conservative
/// rather than arbitrary. `ArithmaExpression` nests through `Box`, and the
/// derived `Drop` for a boxed chain **is** recursive: a tree deep enough to
/// trouble this simplifier would overflow the stack when it was freed,
/// wherever it was built. Refusing early keeps the failure legible.
pub const MAX_DEPTH: usize = 1024;

// ---------------------------------------------------------------------------
// Uniform child access
//
// The AST has three shapes of child: a `Vec` (Function), named boxes (Sum,
// Limit, Product, Conditional, CachedValue, FourierOptimized) and none
// (Number, Constant, Variable). These two functions flatten all of that into a
// single index space so the traversal below does not need to know the variant
// list -- adding a variant means touching only these.
// ---------------------------------------------------------------------------

/// Number of direct children of `expr`.
fn child_count(expr: &ArithmaExpression) -> usize {
    match expr {
        ArithmaExpression::Number(_)
        | ArithmaExpression::Constant { .. }
        | ArithmaExpression::Variable(_) => 0,
        ArithmaExpression::Function(_, args) => args.len(),
        ArithmaExpression::Sum { .. } | ArithmaExpression::Product { .. } => 3,
        ArithmaExpression::Limit { .. } => 2,
        ArithmaExpression::Conditional { .. } => 3,
        ArithmaExpression::CachedValue { .. } | ArithmaExpression::FourierOptimized { .. } => 1,
    }
}

/// One entry in the traversal's work stack.
///
/// The fold is destructive: children are detached from their parent, processed,
/// then reattached. That is what makes it O(n) -- the earlier design addressed
/// nodes by a path from the root, which re-walked the spine for every node and
/// came out quadratic (a 200-node no-op pass measured 1.58 ms).
#[derive(Debug)]
enum Frame {
    /// A subtree whose children have not been visited yet.
    Descend(ArithmaExpression),
    /// A node whose children have been detached. When this frame surfaces, the
    /// top `arity` entries of the output stack are its finished children.
    Rebuild(ArithmaExpression, usize),
}

/// Detach every direct child, leaving the node structurally valid but stubbed.
///
/// Boxed children are replaced with a literal zero rather than left dangling,
/// so the node is safe to hold, print or drop between detach and reattach.
fn take_children(expr: &mut ArithmaExpression) -> Vec<ArithmaExpression> {
    let stub = ArithmaExpression::zero;
    match expr {
        ArithmaExpression::Number(_)
        | ArithmaExpression::Constant { .. }
        | ArithmaExpression::Variable(_) => Vec::new(),
        ArithmaExpression::Function(_, args) => core::mem::take(args),
        ArithmaExpression::Sum {
            start,
            end,
            expression,
            ..
        }
        | ArithmaExpression::Product {
            start,
            end,
            expression,
            ..
        } => vec![
            core::mem::replace(start.as_mut(), stub()),
            core::mem::replace(end.as_mut(), stub()),
            core::mem::replace(expression.as_mut(), stub()),
        ],
        ArithmaExpression::Limit {
            approaching,
            expression,
            ..
        } => vec![
            core::mem::replace(approaching.as_mut(), stub()),
            core::mem::replace(expression.as_mut(), stub()),
        ],
        ArithmaExpression::Conditional {
            condition,
            then_expr,
            else_expr,
        } => vec![
            core::mem::replace(condition.as_mut(), stub()),
            core::mem::replace(then_expr.as_mut(), stub()),
            core::mem::replace(else_expr.as_mut(), stub()),
        ],
        ArithmaExpression::CachedValue { expr: inner, .. }
        | ArithmaExpression::FourierOptimized { expr: inner, .. } => {
            vec![core::mem::replace(inner.as_mut(), stub())]
        }
    }
}

/// Reattach children detached by [`take_children`], in the same order.
fn put_children(expr: &mut ArithmaExpression, kids: Vec<ArithmaExpression>) {
    debug_assert!(kids.len() <= MAX_NODES, "implausible child count");
    let mut it = kids.into_iter();
    match expr {
        ArithmaExpression::Number(_)
        | ArithmaExpression::Constant { .. }
        | ArithmaExpression::Variable(_) => {}
        ArithmaExpression::Function(_, args) => {
            *args = it.by_ref().collect();
        }
        ArithmaExpression::Sum {
            start,
            end,
            expression,
            ..
        }
        | ArithmaExpression::Product {
            start,
            end,
            expression,
            ..
        } => {
            for slot in [start, end, expression] {
                if let Some(v) = it.next() {
                    **slot = v;
                }
            }
        }
        ArithmaExpression::Limit {
            approaching,
            expression,
            ..
        } => {
            for slot in [approaching, expression] {
                if let Some(v) = it.next() {
                    **slot = v;
                }
            }
        }
        ArithmaExpression::Conditional {
            condition,
            then_expr,
            else_expr,
        } => {
            for slot in [condition, then_expr, else_expr] {
                if let Some(v) = it.next() {
                    **slot = v;
                }
            }
        }
        ArithmaExpression::CachedValue { expr: inner, .. }
        | ArithmaExpression::FourierOptimized { expr: inner, .. } => {
            if let Some(v) = it.next() {
                **inner = v;
            }
        }
    }
    debug_assert!(
        it.next().is_none(),
        "more children were reattached than the node has slots"
    );
}

/// Whether two function tags denote the same operation.
///
/// `ArithmaFunction` deliberately does not derive `PartialEq` -- several
/// variants carry payloads, including expressions, and the crate leaves the
/// comparison to the consumer. So this is an **explicit allow-list of the
/// payload-free arithmetic and transcendental variants**, plus the three that
/// carry a simple comparable payload.
///
/// Anything not listed returns `false`. That is deliberate and safe: a `false`
/// here means two terms are not collected together, which loses a
/// simplification but never produces a wrong one. Matching on the
/// discriminant alone and defaulting to `true` would be the unsound choice,
/// because a new payload variant would silently start comparing equal.
fn same_function(a: &ArithmaFunction, b: &ArithmaFunction) -> bool {
    use ArithmaFunction as F;
    match (a, b) {
        // Payload-carrying, but the payload is cheap and total to compare.
        (F::Root(x), F::Root(y)) => x == y,
        (F::Pow(x), F::Pow(y)) | (F::LogBase(x), F::LogBase(y)) => {
            x.value == y.value && x.unit == y.unit
        }
        // Payload-free variants. Listed rather than matched by discriminant so
        // that adding a variant to the enum cannot quietly join this set.
        (F::Add, F::Add)
        | (F::Subtract, F::Subtract)
        | (F::Multiply, F::Multiply)
        | (F::Divide, F::Divide)
        | (F::Power, F::Power)
        | (F::Negate, F::Negate)
        | (F::Sin, F::Sin)
        | (F::Cos, F::Cos)
        | (F::Tan, F::Tan)
        | (F::Cot, F::Cot)
        | (F::Sec, F::Sec)
        | (F::Csc, F::Csc)
        | (F::Asin, F::Asin)
        | (F::Acos, F::Acos)
        | (F::Atan, F::Atan)
        | (F::Atan2, F::Atan2)
        | (F::Sinh, F::Sinh)
        | (F::Cosh, F::Cosh)
        | (F::Tanh, F::Tanh)
        | (F::Asinh, F::Asinh)
        | (F::Acosh, F::Acosh)
        | (F::Atanh, F::Atanh)
        | (F::Exp, F::Exp)
        | (F::Ln, F::Ln)
        | (F::Log, F::Log)
        | (F::Log10, F::Log10)
        | (F::Log2, F::Log2)
        | (F::Sqrt, F::Sqrt)
        | (F::Cbrt, F::Cbrt)
        | (F::Gamma, F::Gamma)
        | (F::Beta, F::Beta)
        | (F::Erf, F::Erf)
        | (F::Factorial, F::Factorial)
        | (F::Abs, F::Abs)
        | (F::Sign, F::Sign)
        | (F::Floor, F::Floor)
        | (F::Ceil, F::Ceil)
        | (F::Round, F::Round)
        | (F::Real, F::Real)
        | (F::Imag, F::Imag)
        | (F::Conjugate, F::Conjugate)
        | (F::Arg, F::Arg) => true,
        _ => false,
    }
}

/// Whether two nodes are the same ignoring their children.
fn same_node(a: &ArithmaExpression, b: &ArithmaExpression) -> bool {
    use ArithmaExpression as E;
    match (a, b) {
        (E::Number(x), E::Number(y)) => x.value == y.value && x.unit == y.unit,
        (E::Variable(x), E::Variable(y)) => x == y,
        (E::Constant { symbol: x, .. }, E::Constant { symbol: y, .. }) => x == y,
        (E::Function(f, _), E::Function(g, _)) => same_function(f, g),
        (E::Sum { variable: x, .. }, E::Sum { variable: y, .. })
        | (E::Product { variable: x, .. }, E::Product { variable: y, .. }) => x == y,
        (
            E::Limit {
                variable: x,
                from_right: fx,
                ..
            },
            E::Limit {
                variable: y,
                from_right: fy,
                ..
            },
        ) => x == y && fx == fy,
        (E::Conditional { .. }, E::Conditional { .. }) => true,
        // A cached or Fourier-optimised wrapper compares by what it wraps,
        // which the child walk handles.
        (E::CachedValue { .. }, E::CachedValue { .. })
        | (E::FourierOptimized { .. }, E::FourierOptimized { .. }) => true,
        _ => false,
    }
}

/// Structural equality: same shape, same node data, same children in order.
///
/// Iterative, like everything else here -- an expression deep enough to
/// matter would blow a recursive comparison's stack. Bounded by [`MAX_NODES`]
/// so a pathological input cannot spin; exceeding the bound reports "not
/// equal", which is the conservative answer.
///
/// This is *syntactic*: `x + y` and `y + x` are not equal, and neither are
/// `2*x` and `x*2`. It is not a canonical comparison and does not try to be.
/// What makes collection see through commuted forms is that the operands of
/// the commutative operators are put in canonical order before they get here
/// -- see [`compare_expressions`] -- so both spellings reach this function as
/// the same tree.
fn structural_eq(a: &ArithmaExpression, b: &ArithmaExpression) -> bool {
    let mut stack: Vec<(&ArithmaExpression, &ArithmaExpression)> = vec![(a, b)];
    let mut visited: usize = 0;
    while let Some((x, y)) = stack.pop() {
        visited += 1;
        if visited > MAX_NODES {
            return false;
        }
        if !same_node(x, y) {
            return false;
        }
        let arity = child_count(x);
        if arity != child_count(y) {
            return false;
        }
        for i in 0..arity {
            match (child_at(x, i), child_at(y, i)) {
                (Some(cx), Some(cy)) => stack.push((cx, cy)),
                _ => return false,
            }
        }
    }
    debug_assert!(visited <= MAX_NODES, "node cap not enforced");
    debug_assert!(visited > 0, "comparison visited nothing");
    true
}

// ---------------------------------------------------------------------------
// Canonical operand ordering
//
// Sorting the operands of the commutative operators into one order is what
// removes the commuted-form limit at its source: `x*2` becomes `2*x` on the
// way past, so `structural_eq` -- still syntactic, still cheap -- sees the two
// spellings as the same tree.
//
// Three properties matter, in this order:
//
// 1. **Idempotent.** Sorting a sorted list must report no change. Everything
//    downstream loops until nothing changes; a sort that kept reporting work
//    would spin until the iteration budget ran out and call the result
//    simplified.
// 2. **Total and deterministic.** The comparison is a pure function of the
//    two trees, so the same pair always sorts the same way, whatever else is
//    in the list.
// 3. **Commutative operands only.** Add and Multiply, and nothing else.
//    Reordering `a - b`, `a / b` or `a ^ b` changes the value.
// ---------------------------------------------------------------------------

/// Sort rank of an expression's variant.
///
/// Two of these ranks are load-bearing rather than arbitrary:
///
/// - `Number` sorts first, so a numeric operand leads. That is the form
///   [`split_coefficient`] reads as a coefficient and the form
///   [`ArithmaIterativeSimplifier::finish_commutative`] already writes, so the
///   canonical order agrees with the rest of the module instead of fighting
///   it.
/// - `Variable` sorts last, behind the compound variants. `a + b + c` parses
///   left-associatively as `(a + b) + c`, which puts the compound operand
///   first; ranking compounds ahead of atoms makes that shape already
///   canonical, so ordering an ordinarily-written tree costs one comparison
///   per node instead of rewriting every node in it.
fn kind_rank(expr: &ArithmaExpression) -> u8 {
    match expr {
        ArithmaExpression::Number(_) => 0,
        ArithmaExpression::Constant { .. } => 1,
        ArithmaExpression::Function(..) => 2,
        ArithmaExpression::Sum { .. } => 3,
        ArithmaExpression::Product { .. } => 4,
        ArithmaExpression::Limit { .. } => 5,
        ArithmaExpression::Conditional { .. } => 6,
        ArithmaExpression::CachedValue { .. } => 7,
        ArithmaExpression::FourierOptimized { .. } => 8,
        ArithmaExpression::Variable(_) => 9,
    }
}

/// Rank shared by every operator this module declines to order.
///
/// The counterpart of [`same_function`] returning `false`, and the same
/// allow-list: an operator carrying an expression payload (a bound, an
/// approach point) cannot be separated without walking that payload, which
/// sits outside the child index space this traversal uses. They all share one
/// rank and therefore compare equal, and a stable sort leaves equal operands
/// exactly where they were. Nothing merges on the strength of this
/// comparison -- collection asks `structural_eq` -- so a tie costs a missed
/// reordering and never a wrong answer.
const UNRANKED_FUNCTION: u16 = u16::MAX;

/// Sort rank of an operator.
///
/// Listed variant by variant, in the order the enum declares them, rather
/// than derived from the discriminant: a rank that moved when someone
/// inserted a variant would silently re-sort every stored expression.
fn function_rank(f: &ArithmaFunction) -> u16 {
    use ArithmaFunction as F;
    match f {
        F::Add => 0,
        F::Subtract => 1,
        F::Multiply => 2,
        F::Divide => 3,
        F::Power => 4,
        F::Negate => 5,
        F::Sin => 6,
        F::Cos => 7,
        F::Tan => 8,
        F::Cot => 9,
        F::Sec => 10,
        F::Csc => 11,
        F::Asin => 12,
        F::Acos => 13,
        F::Atan => 14,
        F::Atan2 => 15,
        F::Sinh => 16,
        F::Cosh => 17,
        F::Tanh => 18,
        F::Asinh => 19,
        F::Acosh => 20,
        F::Atanh => 21,
        F::Exp => 22,
        F::Ln => 23,
        F::Log => 24,
        F::Log10 => 25,
        F::Log2 => 26,
        F::LogBase(_) => 27,
        F::Pow(_) => 28,
        F::Sqrt => 29,
        F::Cbrt => 30,
        F::Root(_) => 31,
        F::Gamma => 32,
        F::Beta => 33,
        F::Erf => 34,
        F::Factorial => 35,
        F::Abs => 36,
        F::Sign => 37,
        F::Floor => 38,
        F::Ceil => 39,
        F::Round => 40,
        F::Real => 41,
        F::Imag => 42,
        F::Conjugate => 43,
        F::Arg => 44,
        _ => UNRANKED_FUNCTION,
    }
}

/// Order two operators: by rank, then by whatever payload they carry.
fn compare_functions(a: &ArithmaFunction, b: &ArithmaFunction) -> Ordering {
    use ArithmaFunction as F;
    let by_rank = function_rank(a).cmp(&function_rank(b));
    if by_rank != Ordering::Equal {
        debug_assert!(!same_function(a, b), "distinct ranks for the same operator");
        return by_rank;
    }
    // Equal rank means the same variant, so only the payload is left. These
    // are exactly the three payloads `same_function` compares, which is what
    // keeps "compares equal" and "orders equal" in step.
    let by_payload = match (a, b) {
        (F::Root(x), F::Root(y)) => x.cmp(y),
        (F::Pow(x), F::Pow(y)) | (F::LogBase(x), F::LogBase(y)) => compare_numbers(x, y),
        _ => Ordering::Equal,
    };
    debug_assert!(
        by_payload != Ordering::Equal
            || function_rank(a) == UNRANKED_FUNCTION
            || same_function(a, b),
        "operators that order equal must also compare equal"
    );
    by_payload
}

/// Length of a little-endian magnitude with its leading zero bytes ignored.
///
/// The arithmetic in `crate::integer` trims as it goes, but a magnitude that
/// arrived by deserialisation has not been through it. Trimming here keeps
/// `2` and a zero-padded `2` from ordering apart on their padding.
fn significant_len(magnitude: &[u8]) -> usize {
    let mut len = magnitude.len();
    // Bounded by the slice length; each turn removes one byte.
    while len > 0 && magnitude[len - 1] == 0 {
        len -= 1;
    }
    debug_assert!(
        len <= magnitude.len(),
        "trim reported more bytes than exist"
    );
    debug_assert!(
        len == 0 || magnitude[len - 1] != 0,
        "a leading zero survived the trim"
    );
    len
}

/// Order two little-endian magnitudes by value.
fn compare_magnitude(x: &[u8], y: &[u8]) -> Ordering {
    let xs = significant_len(x);
    let ys = significant_len(y);
    debug_assert!(xs <= x.len(), "left magnitude trimmed past its length");
    debug_assert!(ys <= y.len(), "right magnitude trimmed past its length");
    if xs != ys {
        // More significant bytes means a larger magnitude, both being trimmed.
        return xs.cmp(&ys);
    }
    // Most significant byte first. Bounded: `i` strictly decreases.
    let mut i = xs;
    while i > 0 {
        i -= 1;
        let by_digit = x[i].cmp(&y[i]);
        if by_digit != Ordering::Equal {
            return by_digit;
        }
    }
    Ordering::Equal
}

/// Order two literals.
///
/// Plain integers order by value, sign first, so a canonicalised operand list
/// reads the way a reader expects. Special values -- NaN, infinity, a
/// rational -- have no numeric order to give, so they fall back to the flag
/// word, which is deterministic and is all a sort needs.
fn compare_numbers(a: &ArithmaInteger, b: &ArithmaInteger) -> Ordering {
    debug_assert!(
        !a.value.value.is_empty(),
        "left literal has an empty magnitude"
    );
    debug_assert!(
        !b.value.value.is_empty(),
        "right literal has an empty magnitude"
    );
    // Unit attribution decides first. `5` and `5 m` are different literals,
    // and ordering them as one would sit them next to each other and invite a
    // merge across units.
    let by_unit = a.unit.cmp(&b.unit);
    if by_unit != Ordering::Equal {
        return by_unit;
    }
    let (x, y) = (&a.value, &b.value);
    if !x.is_plain_integer() || !y.is_plain_integer() {
        return x
            .flags
            .cmp(&y.flags)
            .then_with(|| compare_magnitude(&x.value, &y.value));
    }
    match (x.is_negative(), y.is_negative()) {
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        // Both negative: the larger magnitude is the smaller number.
        (true, true) => compare_magnitude(&y.value, &x.value),
        (false, false) => compare_magnitude(&x.value, &y.value),
    }
}

/// Order two nodes on their own data, ignoring their children.
///
/// The key is (kind rank, the data that distinguishes the variant, arity).
/// Arity belongs in the key: without it the pre-order walk in
/// [`compare_expressions`] could run out of one tree while the other still
/// had children, and call two different shapes equal.
fn compare_shallow(a: &ArithmaExpression, b: &ArithmaExpression) -> Ordering {
    use ArithmaExpression as E;
    debug_assert!(
        child_count(a) <= MAX_NODES,
        "left node has an implausible arity"
    );
    debug_assert!(
        child_count(b) <= MAX_NODES,
        "right node has an implausible arity"
    );
    let by_kind = kind_rank(a).cmp(&kind_rank(b));
    if by_kind != Ordering::Equal {
        return by_kind;
    }
    // Equal kind rank means the same variant, so every pair below is
    // like-for-like and the fallthrough only catches the variants that carry
    // no distinguishing data of their own.
    let by_data = match (a, b) {
        (E::Number(x), E::Number(y)) => compare_numbers(x, y),
        (E::Variable(x), E::Variable(y)) => x.cmp(y),
        (E::Constant { symbol: x, .. }, E::Constant { symbol: y, .. }) => x.cmp(y),
        (E::Function(f, _), E::Function(g, _)) => compare_functions(f, g),
        (E::Sum { variable: x, .. }, E::Sum { variable: y, .. })
        | (E::Product { variable: x, .. }, E::Product { variable: y, .. }) => x.cmp(y),
        (
            E::Limit {
                variable: x,
                from_right: fx,
                ..
            },
            E::Limit {
                variable: y,
                from_right: fy,
                ..
            },
        ) => x.cmp(y).then_with(|| fx.cmp(fy)),
        _ => Ordering::Equal,
    };
    if by_data != Ordering::Equal {
        return by_data;
    }
    child_count(a).cmp(&child_count(b))
}

/// Total order over expressions.
///
/// Compares the two trees' pre-order serialisations: each node's own key
/// decides, and only a tie descends into its children. Because arity is part
/// of the key, two trees whose keys all agree have the same shape, so the
/// order is consistent with [`structural_eq`] -- structurally equal
/// expressions always compare `Equal`.
///
/// The converse does not quite hold, on purpose. Operators carrying an
/// expression payload share one rank, and a zero-padded literal orders with
/// its trimmed twin, so two operands can order equal without being equal.
/// That costs a pair left in input order by a stable sort, and nothing more:
/// merging is [`structural_eq`]'s decision, not this one's.
///
/// Iterative and bounded by [`MAX_NODES`], like every other walk here.
/// Exhausting the bound reports `Equal`, which leaves the operands where they
/// are instead of inventing an order for a tree too large to inspect.
fn compare_expressions(a: &ArithmaExpression, b: &ArithmaExpression) -> Ordering {
    // `Vec::new` does not allocate, and the overwhelmingly common case --
    // operands whose root keys already differ -- returns before the first
    // push. Ordering therefore stays off the profile of a pass with nothing
    // to reorder, which is the whole no-op benchmark.
    let mut stack: Vec<(&ArithmaExpression, &ArithmaExpression)> = Vec::new();
    let mut current = (a, b);
    let mut visited: usize = 0;
    loop {
        visited += 1;
        if visited > MAX_NODES {
            return Ordering::Equal;
        }
        let (x, y) = current;
        let shallow = compare_shallow(x, y);
        if shallow != Ordering::Equal {
            return shallow;
        }
        let arity = child_count(x);
        debug_assert_eq!(
            arity,
            child_count(y),
            "arity is part of the key, so equal keys cannot differ in it"
        );
        // Reversed, so child 0 pops first and the walk is pre-order.
        for i in (0..arity).rev() {
            match (child_at(x, i), child_at(y, i)) {
                (Some(cx), Some(cy)) => stack.push((cx, cy)),
                // A disagreement here is a bug in this module, not bad input.
                _ => debug_assert!(false, "child_count/child_at disagree"),
            }
        }
        match stack.pop() {
            Some(next) => current = next,
            None => break,
        }
    }
    debug_assert!(visited <= MAX_NODES, "node cap not enforced");
    Ordering::Equal
}

/// Whether `operands` are already in canonical order.
///
/// Borrow-only, and one comparison per adjacent pair. The commutative
/// rewrites run this before taking ownership of anything, for the same reason
/// the collection pre-checks exist: reordering a list and then reporting "no
/// change" is a lie the fixpoint loop cannot catch.
fn is_canonically_ordered(operands: &[ArithmaExpression]) -> bool {
    debug_assert!(operands.len() <= MAX_NODES, "implausible operand count");
    let mut pairs: usize = 0;
    for window in operands.windows(2) {
        pairs += 1;
        if compare_expressions(&window[0], &window[1]) == Ordering::Greater {
            return false;
        }
    }
    debug_assert_eq!(
        pairs,
        operands.len().saturating_sub(1),
        "the adjacent-pair scan skipped an operand"
    );
    true
}

/// Sort `operands` into canonical order.
///
/// The sort has to be **stable**. Two operands can order equal without being
/// structurally equal, and an unstable sort is free to swap those; swapping
/// them back on the next pass would leave the fixpoint loop oscillating until
/// its budget ran out. `sort_by` is stable, and for an operand list -- which
/// is a handful of entries, not a tree -- it sorts in place without
/// allocating.
fn order_operands(operands: &mut [ArithmaExpression]) {
    debug_assert!(operands.len() <= MAX_NODES, "implausible operand count");
    operands.sort_by(compare_expressions);
    debug_assert!(
        is_canonically_ordered(operands),
        "sorting left the list out of order"
    );
}

/// Split a summand into `(coefficient, base)`.
///
/// `3*x` gives `(3, x)`, `-x` gives `(-1, x)`, and a bare `x` gives `(1, x)`.
/// Only a **leading** numeric factor counts. That is not a limit any more:
/// canonical ordering sorts a literal to the front of a product, so `x*3`
/// reaches this function as `3*x`.
fn split_coefficient(expr: &ArithmaExpression) -> (ArithmaInteger, &ArithmaExpression) {
    let one = ArithmaInteger::from_i64(1);
    match expr {
        ArithmaExpression::Function(ArithmaFunction::Multiply, args) if args.len() == 2 => {
            match as_number(&args[0]) {
                Some(c) if c.unit.is_none() => (c.clone(), &args[1]),
                _ => (one, expr),
            }
        }
        ArithmaExpression::Function(ArithmaFunction::Negate, args) if args.len() == 1 => {
            (ArithmaInteger::from_i64(-1), &args[0])
        }
        _ => (one, expr),
    }
}

/// Split a factor into `(base, exponent)`.
///
/// `x^3` gives `(x, 3)` and a bare `x` gives `(x, 1)`. A non-integer or
/// negative exponent is left alone -- combining `x^-1 * x` would need
/// division rules this simplifier does not have.
fn split_power(expr: &ArithmaExpression) -> (&ArithmaExpression, u32) {
    match expr {
        ArithmaExpression::Function(ArithmaFunction::Power, args) if args.len() == 2 => {
            match as_number(&args[1]).and_then(ArithmaInteger::to_u32) {
                Some(e) if e >= 1 => (&args[0], e),
                _ => (expr, 1),
            }
        }
        _ => (expr, 1),
    }
}

/// If `expr` is a plain `Number`, borrow its integer.
fn as_number(expr: &ArithmaExpression) -> Option<&ArithmaInteger> {
    match expr {
        ArithmaExpression::Number(n) => Some(n),
        _ => None,
    }
}

/// Which half of the Pythagorean identity an operand is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SquaredTrig {
    Sin,
    Cos,
}

/// Recognise `sin(u)^2` or `cos(u)^2`, borrowing `u`.
///
/// Only the exact `Power(f(u), 2)` shape matches, and that is enough: a
/// product written `sin(u)*sin(u)` has already become that power by the time
/// its parent sum is examined, because children are finished before their
/// parent in the same pass.
fn squared_trig(expr: &ArithmaExpression) -> Option<(SquaredTrig, &ArithmaExpression)> {
    debug_assert!(
        child_count(expr) <= MAX_NODES,
        "operand has an implausible arity"
    );
    let (base, exponent) = match expr {
        ArithmaExpression::Function(ArithmaFunction::Power, args) if args.len() == 2 => {
            (&args[0], &args[1])
        }
        _ => return None,
    };
    if as_number(exponent).and_then(ArithmaInteger::to_u32) != Some(2) {
        return None;
    }
    let found = match base {
        ArithmaExpression::Function(ArithmaFunction::Sin, inner) if inner.len() == 1 => {
            Some((SquaredTrig::Sin, &inner[0]))
        }
        ArithmaExpression::Function(ArithmaFunction::Cos, inner) if inner.len() == 1 => {
            Some((SquaredTrig::Cos, &inner[0]))
        }
        _ => None,
    };
    debug_assert!(
        found.is_none() || matches!(expr, ArithmaExpression::Function(ArithmaFunction::Power, _)),
        "only a power can be a squared trigonometric term"
    );
    found
}

/// Find a `sin(u)^2` and a `cos(u)^2` sharing the same `u`.
///
/// Returns their indices in ascending order. The cheap half runs first: one
/// borrow-only scan establishes that both halves are present at all, and only
/// then does the index match run. A sum with no powers in it -- which is
/// nearly every sum -- pays one pattern test per operand and allocates
/// nothing.
fn pythagorean_pair(args: &[ArithmaExpression]) -> Option<(usize, usize)> {
    debug_assert!(args.len() <= MAX_NODES, "implausible operand count");
    let mut has_sin = false;
    let mut has_cos = false;
    for arg in args.iter() {
        match squared_trig(arg) {
            Some((SquaredTrig::Sin, _)) => has_sin = true,
            Some((SquaredTrig::Cos, _)) => has_cos = true,
            None => {}
        }
    }
    if !has_sin || !has_cos {
        return None;
    }
    for (i, left) in args.iter().enumerate() {
        let u = match squared_trig(left) {
            Some((SquaredTrig::Sin, u)) => u,
            _ => continue,
        };
        for (j, right) in args.iter().enumerate() {
            if i == j {
                continue;
            }
            let v = match squared_trig(right) {
                Some((SquaredTrig::Cos, v)) => v,
                _ => continue,
            };
            if structural_eq(u, v) {
                let pair = if i < j { (i, j) } else { (j, i) };
                debug_assert!(pair.0 < pair.1, "the pair indices are not ascending");
                debug_assert!(pair.1 < args.len(), "the pair index is out of range");
                return Some(pair);
            }
        }
    }
    None
}

impl ArithmaIterativeSimplifier {
    /// Rewrite a single node whose children are already in simplest form.
    ///
    /// Returns `true` if the node changed. Every rule here is exact: numeric
    /// folding goes through `ArithmaInteger`'s checked bignum arithmetic, which
    /// refuses rather than rounds, so simplification never loses precision.
    ///
    /// Each rule inspects its operands by reference first and only takes
    /// ownership once it has decided to act. That ordering is load-bearing:
    /// an earlier version cloned every operand up front, which copied whole
    /// subtrees per node and made a no-op pass quadratic (36 ms over 1,000
    /// symbolic nodes, against 0.6 ms now).
    fn rewrite(node: &mut ArithmaExpression, config: &SimplificationConfig) -> bool {
        debug_assert!(
            child_count(node) <= MAX_NODES,
            "node has an implausible arity"
        );
        // Constants collapse to literals only when the caller opted in --
        // otherwise `pi` must stay `pi`, because its cached f64 is an
        // approximation and folding it would silently discard exactness.
        if config.allow_numeric_collapse {
            if let ArithmaExpression::Constant {
                cached_value: Some(v),
                allow_simplification: true,
                ..
            } = node
            {
                let value = *v;
                if value.fract() == 0.0 && value.abs() < 9.007_199_254_740_992e15 {
                    *node = ArithmaExpression::from_i64(value as i64);
                    return true;
                }
            }
        }

        let func = match node {
            ArithmaExpression::Function(f, _) => f.clone(),
            _ => return false,
        };
        match func {
            ArithmaFunction::Add => Self::rewrite_sum(node),
            ArithmaFunction::Subtract => Self::rewrite_difference(node),
            ArithmaFunction::Multiply => Self::rewrite_product(node),
            ArithmaFunction::Divide => Self::rewrite_divide(node),
            ArithmaFunction::Power => Self::rewrite_power(node),
            ArithmaFunction::Negate => Self::rewrite_negate(node),
            _ => Self::rewrite_special_value(node),
        }
    }

    /// The transcendental functions at the arguments where they have an exact
    /// value: `sin(0) = 0`, `cos(0) = 1`, `exp(0) = 1`, `ln(1) = 0`.
    ///
    /// Everything in the table below is unconditional -- true for every
    /// argument the function accepts, with no domain to check first. What is
    /// missing is missing on purpose:
    ///
    /// - `cot(0)` and `csc(0)` are poles, not values, so the trigonometric
    ///   family is here minus those two.
    /// - `acos(0)` is `pi/2`, which this simplifier has no exact literal for.
    /// - `log_b(1)` is `0` only when `b` is a legitimate base; `LogBase`
    ///   carries a caller-supplied one, so it is left alone while the four
    ///   fixed-base logarithms are folded.
    /// - `sqrt(x^2)` is `|x|` and `ln(exp(x))` is `x` only up to a multiple of
    ///   `2*pi*i`. Neither is an identity and neither is here.
    fn rewrite_special_value(node: &mut ArithmaExpression) -> bool {
        use ArithmaFunction as F;
        debug_assert!(
            child_count(node) <= MAX_NODES,
            "node has an implausible arity"
        );
        let value = match &*node {
            ArithmaExpression::Function(func, args) if args.len() == 1 => {
                // A dimensioned or special argument is refused: `sin(0 m)` is
                // not a quantity this rule knows the value of, and neither is
                // `exp(NaN)`.
                match as_number(&args[0]) {
                    Some(n) if n.unit.is_none() && n.value.is_plain_integer() => {
                        let at_zero = n.is_zero();
                        let at_one = n.is_one();
                        match func {
                            F::Sin | F::Tan | F::Sinh | F::Tanh if at_zero => Some(0i64),
                            F::Asin | F::Atan | F::Asinh | F::Atanh if at_zero => Some(0i64),
                            F::Cos | F::Cosh | F::Sec | F::Exp if at_zero => Some(1i64),
                            F::Ln | F::Log | F::Log2 | F::Log10 if at_one => Some(0i64),
                            _ => None,
                        }
                    }
                    _ => None,
                }
            }
            _ => None,
        };
        debug_assert!(
            value.is_none()
                || matches!(node, ArithmaExpression::Function(_, args) if args.len() == 1),
            "only a unary application has a value in this table"
        );
        match value {
            Some(v) => {
                *node = ArithmaExpression::from_i64(v);
                true
            }
            None => false,
        }
    }

    /// Borrow a `Function` node's arguments, or `None` for any other variant.
    fn args_of(node: &ArithmaExpression) -> Option<&[ArithmaExpression]> {
        match node {
            ArithmaExpression::Function(_, args) => Some(args.as_slice()),
            _ => None,
        }
    }

    /// Take ownership of a `Function` node's arguments, leaving it empty.
    ///
    /// Only called once a rule has committed to rewriting, because the node is
    /// left arity-zero until the caller replaces it.
    fn take_args(node: &mut ArithmaExpression) -> Vec<ArithmaExpression> {
        match node {
            ArithmaExpression::Function(_, args) => core::mem::take(args),
            _ => {
                debug_assert!(false, "take_args called on a non-Function node");
                Vec::new()
            }
        }
    }

    /// `a + b + ...` -- exact folding of the numeric operands plus the additive
    /// identity.
    fn rewrite_sum(node: &mut ArithmaExpression) -> bool {
        let args = match Self::args_of(node) {
            Some(a) if a.len() >= 2 => a,
            _ => return false,
        };
        debug_assert!(args.len() <= MAX_NODES, "implausible argument count");
        // Cheap scan first: decide whether there is anything to do before
        // taking ownership of anything.
        let mut numeric = 0usize;
        let mut droppable_zero = false;
        for arg in args.iter() {
            if let Some(n) = as_number(arg) {
                numeric += 1;
                if n.is_zero() {
                    droppable_zero = true;
                }
            }
        }
        let symbolic = args.len() - numeric;
        let will_fold = numeric >= 2 || (droppable_zero && symbolic > 0);
        let will_order = !is_canonically_ordered(args);
        let will_reduce = symbolic >= 2 && pythagorean_pair(args).is_some();
        let will_collect = symbolic >= 2 && Self::would_collect_terms(args);
        if !will_fold && !will_order && !will_reduce && !will_collect {
            return false;
        }

        let mut owned = Self::take_args(node);
        let reduced = Self::apply_pythagorean(&mut owned);
        debug_assert_eq!(
            reduced, will_reduce,
            "the borrow-only pre-check disagreed with the rewrite"
        );
        let (acc, mut rest) = Self::partition(owned, ArithmaInteger::checked_add);
        let merged = Self::collect_like_terms(&mut rest);
        debug_assert!(
            merged || reduced || will_fold || will_order,
            "took ownership without anything to change"
        );
        Self::finish_commutative(node, acc, rest, true)
    }

    /// Replace one `sin^2(u) + cos^2(u)` pair with the literal `1`.
    ///
    /// Unconditional: the identity holds for every `u`, with no domain to
    /// check, which is why it can be applied without knowing anything about
    /// the argument. One pair per call -- a sum holding two of them settles on
    /// the next pass, which the fixpoint loop was already going to run.
    fn apply_pythagorean(operands: &mut Vec<ArithmaExpression>) -> bool {
        debug_assert!(operands.len() <= MAX_NODES, "implausible operand count");
        let (i, j) = match pythagorean_pair(operands) {
            Some(pair) => pair,
            None => return false,
        };
        debug_assert!(i < j, "the pair indices are not ascending");
        let before = operands.len();
        // `j` is the later index, so removing it first leaves `i` valid.
        operands.remove(j);
        operands[i] = ArithmaExpression::from_i64(1);
        debug_assert_eq!(
            operands.len() + 1,
            before,
            "collapsing one pair changed the operand count by something other than one"
        );
        true
    }

    /// `a - b`. Not commutative, so only a trailing zero drops: `0 - x` is
    /// `-x`, and rewriting it to `x` would flip the sign of the result.
    fn rewrite_difference(node: &mut ArithmaExpression) -> bool {
        let args = match Self::args_of(node) {
            Some(a) if a.len() == 2 => a,
            _ => return false,
        };
        debug_assert_eq!(args.len(), 2, "subtraction is binary");
        if let (Some(x), Some(y)) = (as_number(&args[0]), as_number(&args[1])) {
            if let Some(folded) = x.checked_sub(y) {
                *node = ArithmaExpression::Number(folded);
                return true;
            }
        }
        if as_number(&args[1]).map(ArithmaInteger::is_zero) == Some(true) {
            let mut owned = Self::take_args(node);
            debug_assert_eq!(owned.len(), 2, "argument list changed underfoot");
            *node = owned.swap_remove(0);
            return true;
        }
        false
    }

    /// `a * b * ...` -- absorbing zero, the multiplicative identity, and exact
    /// folding of the numeric operands.
    fn rewrite_product(node: &mut ArithmaExpression) -> bool {
        let args = match Self::args_of(node) {
            Some(a) if a.len() >= 2 => a,
            _ => return false,
        };
        debug_assert!(args.len() <= MAX_NODES, "implausible argument count");
        // A literal zero factor makes the whole product zero -- but only when
        // every other factor is a plain value. `0 * (1/y)` is not zero, because
        // `1/y` may be infinite.
        let has_zero = args
            .iter()
            .any(|a| as_number(a).map(ArithmaInteger::is_zero) == Some(true));
        if has_zero && args.iter().all(Self::is_finite_shape) {
            *node = ArithmaExpression::zero();
            return true;
        }

        let mut numeric = 0usize;
        let mut droppable_one = false;
        for arg in args.iter() {
            if let Some(n) = as_number(arg) {
                numeric += 1;
                if n.is_one() {
                    droppable_one = true;
                }
            }
        }
        let symbolic = args.len() - numeric;
        let will_fold = numeric >= 2 || (droppable_one && symbolic > 0);
        let will_order = !is_canonically_ordered(args);
        let will_collect = symbolic >= 2 && Self::would_collect_factors(args);
        if !will_fold && !will_order && !will_collect {
            return false;
        }

        let (acc, mut rest) = Self::partition(Self::take_args(node), ArithmaInteger::checked_mul);
        let merged = Self::collect_like_factors(&mut rest);
        debug_assert!(
            merged || will_fold || will_order,
            "took ownership without anything to change"
        );
        Self::finish_commutative(node, acc, rest, false)
    }

    /// Split owned operands into one folded numeric accumulator and the rest.
    ///
    /// Operands are moved, never cloned. `combine` refusing (mismatched units,
    /// a special flag, an over-cap result) is not an error: the un-combinable
    /// value is simply kept as an ordinary operand.
    fn partition(
        args: Vec<ArithmaExpression>,
        combine: fn(&ArithmaInteger, &ArithmaInteger) -> Option<ArithmaInteger>,
    ) -> (Option<ArithmaInteger>, Vec<ArithmaExpression>) {
        debug_assert!(args.len() <= MAX_NODES, "implausible argument count");
        let mut acc: Option<ArithmaInteger> = None;
        let mut rest: Vec<ArithmaExpression> = Vec::with_capacity(args.len());
        for arg in args {
            match arg {
                ArithmaExpression::Number(n) => match acc.take() {
                    None => acc = Some(n),
                    Some(prev) => match combine(&prev, &n) {
                        Some(folded) => acc = Some(folded),
                        None => {
                            rest.push(ArithmaExpression::Number(prev));
                            acc = Some(n);
                        }
                    },
                },
                other => rest.push(other),
            }
        }
        debug_assert!(acc.is_some() || !rest.is_empty(), "partition lost operands");
        (acc, rest)
    }

    /// Whether [`Self::collect_like_terms`] would merge anything.
    ///
    /// Runs on borrowed operands and clones nothing. The guard below must not
    /// take ownership speculatively: `partition` pulls numeric operands to the
    /// front, so rebuilding a sum that turned out to need no work would
    /// reorder it. Reordering while reporting "no change" is a lie the
    /// fixpoint loop would not catch, and reporting "changed" every pass would
    /// stop it converging at all.
    fn would_collect_terms(args: &[ArithmaExpression]) -> bool {
        debug_assert!(args.len() <= MAX_NODES, "implausible operand count");
        let mut seen: Vec<&ArithmaExpression> = Vec::with_capacity(args.len());
        for term in args.iter() {
            if as_number(term).is_some() {
                continue;
            }
            let (_, base) = split_coefficient(term);
            if seen.iter().any(|b| structural_eq(b, base)) {
                return true;
            }
            seen.push(base);
        }
        debug_assert!(seen.len() <= args.len(), "saw more bases than operands");
        false
    }

    /// Whether [`Self::collect_like_factors`] would merge anything.
    ///
    /// Applies the same exponent cap as the collection itself, so a `true`
    /// here guarantees a merge actually happens.
    fn would_collect_factors(args: &[ArithmaExpression]) -> bool {
        debug_assert!(args.len() <= MAX_NODES, "implausible operand count");
        let mut bases: Vec<&ArithmaExpression> = Vec::with_capacity(args.len());
        let mut exponents: Vec<u32> = Vec::with_capacity(args.len());
        for factor in args.iter() {
            if as_number(factor).is_some() {
                continue;
            }
            let (base, exponent) = split_power(factor);
            match bases.iter().position(|b| structural_eq(b, base)) {
                Some(index) => {
                    if exponents[index].saturating_add(exponent) <= crate::integer::MAX_POW_EXPONENT
                    {
                        return true;
                    }
                }
                None => {
                    bases.push(base);
                    exponents.push(exponent);
                }
            }
        }
        debug_assert_eq!(bases.len(), exponents.len(), "group lists diverged");
        false
    }

    /// Collect like terms in an already-partitioned sum.
    ///
    /// `x + x` becomes `2*x`, `2*x + 3*x` becomes `5*x`, and `x - x` -- once
    /// it has been normalised to `x + (-1)*x` -- becomes `0` and drops out.
    ///
    /// Returns `true` if anything was merged. Comparison is syntactic, so
    /// `x*2 + 2*x` is **not** collected; see [`structural_eq`].
    fn collect_like_terms(rest: &mut Vec<ArithmaExpression>) -> bool {
        debug_assert!(rest.len() <= MAX_NODES, "implausible operand count");
        if rest.len() < 2 {
            return false;
        }
        // Coefficients per distinct base, in first-appearance order. Groups are
        // small (an operand list, not a tree), so a linear scan beats hashing
        // and avoids needing a canonical hash for expressions at all.
        let mut bases: Vec<ArithmaExpression> = Vec::with_capacity(rest.len());
        let mut coefficients: Vec<ArithmaInteger> = Vec::with_capacity(rest.len());
        let mut merged = false;
        for term in rest.iter() {
            let (coefficient, base) = split_coefficient(term);
            match bases.iter().position(|b| structural_eq(b, base)) {
                Some(index) => match coefficients[index].checked_add(&coefficient) {
                    Some(sum) => {
                        coefficients[index] = sum;
                        merged = true;
                    }
                    // Refused (units, overflow): keep the term separate rather
                    // than dropping it.
                    None => {
                        bases.push(base.clone());
                        coefficients.push(coefficient);
                    }
                },
                None => {
                    bases.push(base.clone());
                    coefficients.push(coefficient);
                }
            }
        }
        if !merged {
            return false;
        }
        debug_assert_eq!(bases.len(), coefficients.len(), "group lists diverged");
        debug_assert!(bases.len() < rest.len(), "merging did not reduce the count");
        rest.clear();
        for (base, coefficient) in bases.into_iter().zip(coefficients) {
            if coefficient.is_zero() {
                // `x - x` cancels entirely.
                continue;
            }
            if coefficient.is_one() {
                rest.push(base);
            } else {
                rest.push(ArithmaExpression::mul(
                    ArithmaExpression::Number(coefficient),
                    base,
                ));
            }
        }
        true
    }

    /// Collect equal factors in an already-partitioned product.
    ///
    /// `x * x` becomes `x^2` and `x^2 * x` becomes `x^3`. Exponents are summed
    /// with a checked add against [`crate::integer::MAX_POW_EXPONENT`], so a
    /// crafted product cannot build an unbounded power.
    fn collect_like_factors(rest: &mut Vec<ArithmaExpression>) -> bool {
        debug_assert!(rest.len() <= MAX_NODES, "implausible operand count");
        if rest.len() < 2 {
            return false;
        }
        let mut bases: Vec<ArithmaExpression> = Vec::with_capacity(rest.len());
        let mut exponents: Vec<u32> = Vec::with_capacity(rest.len());
        let mut merged = false;
        for factor in rest.iter() {
            let (base, exponent) = split_power(factor);
            match bases.iter().position(|b| structural_eq(b, base)) {
                Some(index) => {
                    let combined = exponents[index].saturating_add(exponent);
                    if combined > crate::integer::MAX_POW_EXPONENT {
                        // Past the cap: keep them separate rather than build a
                        // power the integer type would refuse to evaluate.
                        bases.push(base.clone());
                        exponents.push(exponent);
                    } else {
                        exponents[index] = combined;
                        merged = true;
                    }
                }
                None => {
                    bases.push(base.clone());
                    exponents.push(exponent);
                }
            }
        }
        if !merged {
            return false;
        }
        debug_assert_eq!(bases.len(), exponents.len(), "group lists diverged");
        debug_assert!(bases.len() < rest.len(), "merging did not reduce the count");
        rest.clear();
        for (base, exponent) in bases.into_iter().zip(exponents) {
            debug_assert!(exponent >= 1, "an exponent below one should not be grouped");
            if exponent == 1 {
                rest.push(base);
            } else {
                rest.push(ArithmaExpression::pow(
                    base,
                    ArithmaExpression::from_u64(u64::from(exponent)),
                ));
            }
        }
        true
    }

    /// Write back the result of a commutative fold.
    ///
    /// `additive` selects which accumulator value is droppable: `0` for a sum,
    /// `1` for a product.
    fn finish_commutative(
        node: &mut ArithmaExpression,
        acc: Option<ArithmaInteger>,
        mut rest: Vec<ArithmaExpression>,
        additive: bool,
    ) -> bool {
        debug_assert!(rest.len() <= MAX_NODES, "implausible operand count");
        // Collection can cancel every operand: `3x + (-3)x` leaves nothing at
        // all, with no numeric accumulator either. The answer is the identity
        // element -- zero for a sum, one for a product -- not an empty node.
        if rest.is_empty() && acc.is_none() {
            *node = if additive {
                ArithmaExpression::zero()
            } else {
                ArithmaExpression::from_i64(1)
            };
            return true;
        }
        let keep = match &acc {
            None => false,
            Some(n) => {
                let is_identity = if additive { n.is_zero() } else { n.is_one() };
                !is_identity || rest.is_empty()
            }
        };
        if keep {
            if let Some(n) = acc {
                // The numeric operand leads, matching the usual written form.
                rest.insert(0, ArithmaExpression::Number(n));
            }
        }
        debug_assert!(!rest.is_empty(), "fold consumed every operand");
        // Write the result in canonical order. `partition` and collection both
        // build their output in first-appearance order, which need not be
        // canonical, and leaving it that way would hand the next pass a
        // reordering to report -- an extra iteration for every fold.
        order_operands(&mut rest);
        *node = if rest.len() == 1 {
            rest.remove(0)
        } else {
            let func = if additive {
                ArithmaFunction::Add
            } else {
                ArithmaFunction::Multiply
            };
            ArithmaExpression::Function(func, rest)
        };
        true
    }

    /// `a / b`. Exact quotients fold; `x / 1` drops; `0 / n` is zero.
    fn rewrite_divide(node: &mut ArithmaExpression) -> bool {
        let args = match Self::args_of(node) {
            Some(a) if a.len() == 2 => a,
            _ => return false,
        };
        debug_assert_eq!(args.len(), 2, "division is binary");
        if as_number(&args[1]).map(ArithmaInteger::is_zero) == Some(true) {
            // Division by zero is for evaluation to report. Rewriting it here
            // would erase the error.
            return false;
        }
        if let (Some(x), Some(y)) = (as_number(&args[0]), as_number(&args[1])) {
            if let Some(folded) = x.checked_div_exact(y) {
                *node = ArithmaExpression::Number(folded);
                return true;
            }
            if x.is_zero() {
                *node = ArithmaExpression::zero();
                return true;
            }
        }
        if as_number(&args[1]).map(ArithmaInteger::is_one) == Some(true) {
            let mut owned = Self::take_args(node);
            debug_assert_eq!(owned.len(), 2, "argument list changed underfoot");
            *node = owned.swap_remove(0);
            return true;
        }
        false
    }

    /// `a ^ b`. `b == 0` gives 1, `b == 1` gives `a`, and small integer powers
    /// of an integer base fold exactly.
    fn rewrite_power(node: &mut ArithmaExpression) -> bool {
        let args = match Self::args_of(node) {
            Some(a) if a.len() == 2 => a,
            _ => return false,
        };
        debug_assert_eq!(args.len(), 2, "exponentiation is binary");
        let exponent = as_number(&args[1]);
        // `x ^ 0` is 1 for every finite x. `0 ^ 0` is left alone: it is a
        // genuine ambiguity, not something to silently pick a side on.
        if exponent.map(ArithmaInteger::is_zero) == Some(true) {
            let base_is_zero = as_number(&args[0]).map(ArithmaInteger::is_zero) == Some(true);
            if !base_is_zero && Self::is_finite_shape(&args[0]) {
                *node = ArithmaExpression::from_i64(1);
                return true;
            }
            return false;
        }
        if exponent.map(ArithmaInteger::is_one) == Some(true) {
            let mut owned = Self::take_args(node);
            debug_assert_eq!(owned.len(), 2, "argument list changed underfoot");
            *node = owned.swap_remove(0);
            return true;
        }
        if let (Some(base), Some(exp)) = (as_number(&args[0]), exponent) {
            if let Some(e) = exp.to_u32() {
                if let Some(folded) = base.checked_pow(e) {
                    *node = ArithmaExpression::Number(folded);
                    return true;
                }
            }
        }
        false
    }

    /// `-a` for a numeric operand. A symbolic operand is left alone.
    fn rewrite_negate(node: &mut ArithmaExpression) -> bool {
        let args = match Self::args_of(node) {
            Some(a) if a.len() == 1 => a,
            _ => return false,
        };
        debug_assert_eq!(args.len(), 1, "negation is unary");
        match as_number(&args[0]).and_then(ArithmaInteger::checked_neg) {
            Some(n) => {
                *node = ArithmaExpression::Number(n);
                true
            }
            None => false,
        }
    }

    /// True when `expr` cannot be infinite or NaN, so absorbing rules like
    /// `0 * x = 0` are sound on it.
    ///
    /// Deliberately conservative: a `Variable` qualifies (an unbound variable
    /// is a real number by the engine's convention) but a general `Function`
    /// does not, because `0 * (1/0)` is not zero.
    fn is_finite_shape(expr: &ArithmaExpression) -> bool {
        match expr {
            ArithmaExpression::Number(n) => n.value.is_plain_integer(),
            ArithmaExpression::Variable(_) => true,
            ArithmaExpression::Constant { .. } => true,
            _ => false,
        }
    }

    /// Check the tree against the size caps before touching it.
    ///
    /// Returns the node count, or `None` if [`MAX_NODES`] or [`MAX_DEPTH`] is
    /// exceeded. This runs on immutable borrows and completes before the
    /// destructive fold begins, which is what lets an over-large tree be
    /// refused **whole** -- a mid-fold abort would leave the caller holding a
    /// half-dismantled expression.
    fn measure(root: &ArithmaExpression) -> Option<usize> {
        let mut frontier: Vec<(&ArithmaExpression, usize)> = vec![(root, 1)];
        let mut nodes: usize = 0;
        while let Some((node, depth)) = frontier.pop() {
            nodes += 1;
            if nodes > MAX_NODES || depth > MAX_DEPTH {
                return None;
            }
            let arity = child_count(node);
            for i in 0..arity {
                match child_at(node, i) {
                    Some(child) => frontier.push((child, depth + 1)),
                    // A disagreement here is a bug in this module, not bad input.
                    None => debug_assert!(false, "child_count/child_at disagree"),
                }
            }
        }
        debug_assert!(nodes <= MAX_NODES, "node cap not enforced");
        debug_assert!(nodes > 0, "measure counted no nodes");
        Some(nodes)
    }

    /// One bottom-up rewrite pass. Returns `true` if anything changed.
    ///
    /// Single traversal, O(n) in the node count: each node is detached once,
    /// rebuilt once and rewritten once. Children are always finished before
    /// their parent is examined, so `(1 + 1) * 3` reaches `6` in this one pass
    /// rather than needing one pass per level.
    fn fold_once(&mut self, root: &mut ArithmaExpression, config: &SimplificationConfig) -> bool {
        self.work.clear();
        self.stack.clear();
        debug_assert!(self.work.is_empty(), "work stack not cleared");
        let taken = core::mem::replace(root, ArithmaExpression::zero());
        self.work.push(Frame::Descend(taken));

        let mut changed = false;
        // Each node produces at most one Descend and one Rebuild frame, so the
        // loop is bounded even if `measure` were somehow bypassed.
        let mut guard: usize = 0;
        let frame_limit = MAX_NODES.saturating_mul(2).saturating_add(2);
        while let Some(frame) = self.work.pop() {
            guard += 1;
            if guard > frame_limit {
                break;
            }
            match frame {
                Frame::Descend(mut node) => {
                    let kids = take_children(&mut node);
                    if kids.is_empty() {
                        if Self::rewrite(&mut node, config) {
                            changed = true;
                        }
                        self.stack.push(node);
                    } else {
                        let arity = kids.len();
                        self.work.push(Frame::Rebuild(node, arity));
                        // Reversed so they pop in source order; the output
                        // stack then holds them in source order too.
                        for kid in kids.into_iter().rev() {
                            self.work.push(Frame::Descend(kid));
                        }
                    }
                }
                Frame::Rebuild(mut shell, arity) => {
                    let split = self.stack.len().saturating_sub(arity);
                    let kids = self.stack.split_off(split);
                    debug_assert_eq!(kids.len(), arity, "output stack underflowed");
                    put_children(&mut shell, kids);
                    if Self::rewrite(&mut shell, config) {
                        changed = true;
                    }
                    self.stack.push(shell);
                }
            }
        }

        debug_assert_eq!(
            self.stack.len(),
            1,
            "a completed traversal must leave exactly one root"
        );
        match self.stack.pop() {
            Some(done) => *root = done,
            // Unreachable given the assertion above, but leaving `root` as the
            // stub zero would silently destroy the caller's expression.
            None => debug_assert!(false, "traversal produced no root"),
        }
        changed
    }

    /// Simplify `expr` in place using the configured policy.
    ///
    /// Runs bottom-up rewrite passes to a fixpoint, bounded by
    /// `config.max_iterations` and by [`MAX_ITERATION_CEILING`]. Returns `true`
    /// if anything changed, so callers can drive their own outer loops; a
    /// second call on the result is guaranteed to return `false`.
    ///
    /// The traversal is iterative throughout (safety-critical standard 1: no
    /// recursion), so depth costs heap rather than stack.
    pub fn simplify(
        &mut self,
        expr: &mut ArithmaExpression,
        config: &SimplificationConfig,
    ) -> bool {
        self.stack.clear();
        self.work.clear();
        self.last_iterations = 0;
        debug_assert!(self.stack.is_empty(), "output stack not cleared");
        let budget = config.max_iterations.min(MAX_ITERATION_CEILING);
        if budget == 0 {
            return false;
        }
        // Checked once: the rules only ever shrink a tree, so a pass that
        // starts inside the caps stays inside them.
        if Self::measure(expr).is_none() {
            return false;
        }
        let mut changed_overall = false;
        for _ in 0..budget {
            self.last_iterations += 1;
            if !self.fold_once(expr, config) {
                break;
            }
            changed_overall = true;
        }
        debug_assert!(
            self.last_iterations <= budget,
            "iteration budget was exceeded"
        );
        changed_overall
    }
}

/// Absolute ceiling on simplification passes, whatever the config asks for.
///
/// Safety-critical standard 2: a caller-supplied `max_iterations` is still
/// caller-supplied, so it is clamped rather than trusted.
pub const MAX_ITERATION_CEILING: usize = 1024;

/// Immutable counterpart of [`child_mut`], used by the collection pass.
fn child_at(expr: &ArithmaExpression, index: usize) -> Option<&ArithmaExpression> {
    debug_assert!(index < MAX_NODES, "child index is implausibly large");
    match expr {
        ArithmaExpression::Number(_)
        | ArithmaExpression::Constant { .. }
        | ArithmaExpression::Variable(_) => None,
        ArithmaExpression::Function(_, args) => args.get(index),
        ArithmaExpression::Sum {
            start,
            end,
            expression,
            ..
        }
        | ArithmaExpression::Product {
            start,
            end,
            expression,
            ..
        } => match index {
            0 => Some(start.as_ref()),
            1 => Some(end.as_ref()),
            2 => Some(expression.as_ref()),
            _ => None,
        },
        ArithmaExpression::Limit {
            approaching,
            expression,
            ..
        } => match index {
            0 => Some(approaching.as_ref()),
            1 => Some(expression.as_ref()),
            _ => None,
        },
        ArithmaExpression::Conditional {
            condition,
            then_expr,
            else_expr,
        } => match index {
            0 => Some(condition.as_ref()),
            1 => Some(then_expr.as_ref()),
            2 => Some(else_expr.as_ref()),
            _ => None,
        },
        ArithmaExpression::CachedValue { expr, .. }
        | ArithmaExpression::FourierOptimized { expr, .. } => {
            if index == 0 {
                Some(expr.as_ref())
            } else {
                None
            }
        }
    }
}

/// Convenience entry point: build a simplifier, run it, and discard it.
pub fn simplify_iterative(expr: &mut ArithmaExpression, config: &SimplificationConfig) -> bool {
    let mut simplifier = ArithmaIterativeSimplifier::new();
    simplifier.simplify(expr, config)
}

// ---------------------------------------------------------------------------
// Backward-compatibility aliases for the pre-rename `Arithmos*` names.
// Retained for one release; downstream (eml-math, eml-spectral, metaphysica,
// periodica) should migrate to the `Arithma*` names above.
// ---------------------------------------------------------------------------
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaIterativeSimplifier`")]
#[allow(unused)]
pub use self::ArithmaIterativeSimplifier as ArithmosIterativeSimplifier;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_simplifier_starts_empty() {
        let s = ArithmaIterativeSimplifier::new();
        assert_eq!(s.last_iterations(), 0);
    }

    #[test]
    fn iterative_pass_returns_false_on_atom() {
        let mut expr = ArithmaExpression::zero();
        let cfg = SimplificationConfig::default();
        assert!(!simplify_iterative(&mut expr, &cfg));
    }

    // ─── Wave-3 specification ──────────────────────────────────────────────
    //
    // `ArithmaIterativeSimplifier::simplify` is currently a no-op: it clears
    // its stack, resets the counter and returns `false` without touching the
    // expression. Nothing above catches that, because the only other test
    // simplifies an *atom*, where returning `false` is the correct answer.
    //
    // These tests were the executable specification for the real
    // implementation, written while `simplify` was still a stub. The stub is
    // gone; they now run as ordinary regression tests and must stay green.

    #[test]
    fn simplify_spec_folds_a_constant_sum() {
        // 1 + 1 must collapse to the single number 2.
        let mut expr = ArithmaExpression::add(
            ArithmaExpression::from_i64(1),
            ArithmaExpression::from_i64(1),
        );
        let cfg = SimplificationConfig::default();

        assert!(
            simplify_iterative(&mut expr, &cfg),
            "simplifying a reducible expression must report that it changed"
        );
        assert_eq!(
            expr.to_f64(),
            Some(2.0),
            "1 + 1 should fold to 2, got {expr:?}"
        );
    }

    #[test]
    fn simplify_spec_reaches_a_fixpoint() {
        // A second pass over an already-simplified expression must report no
        // change, or callers looping until `false` will never terminate.
        let mut expr = ArithmaExpression::add(
            ArithmaExpression::from_i64(2),
            ArithmaExpression::from_i64(3),
        );
        let cfg = SimplificationConfig::default();

        assert!(
            simplify_iterative(&mut expr, &cfg),
            "first pass should change it"
        );
        assert!(
            !simplify_iterative(&mut expr, &cfg),
            "second pass must report no change -- otherwise simplification never converges"
        );
    }

    #[test]
    fn simplify_spec_leaves_a_symbolic_expression_intact() {
        // `x + 1` has no numeric value and must not be mangled into one.
        let mut expr =
            ArithmaExpression::add(ArithmaExpression::var("x"), ArithmaExpression::from_i64(1));
        let cfg = SimplificationConfig::default();

        let _ = simplify_iterative(&mut expr, &cfg);
        assert_eq!(
            expr.to_f64(),
            None,
            "an expression containing a free variable must stay symbolic"
        );
    }
    // ─── rewrite rules ─────────────────────────────────────────────────────

    fn n(v: i64) -> ArithmaExpression {
        ArithmaExpression::from_i64(v)
    }

    fn cfg() -> SimplificationConfig {
        SimplificationConfig::default()
    }

    #[test]
    fn default_config_is_usable() {
        // This was `#[derive(Default)]`, giving max_iterations: 0 -- a
        // simplifier that could never do anything, which is precisely why the
        // no-op implementation went unnoticed.
        let c = SimplificationConfig::default();
        assert!(
            c.max_iterations > 0,
            "a default of zero iterations makes the simplifier a no-op"
        );
        assert!(c.max_iterations <= MAX_ITERATION_CEILING);
    }

    #[test]
    fn a_zero_iteration_budget_changes_nothing() {
        let mut expr = ArithmaExpression::add(n(1), n(1));
        let c = SimplificationConfig {
            max_iterations: 0,
            allow_numeric_collapse: false,
        };
        assert!(!simplify_iterative(&mut expr, &c));
        assert_eq!(expr.to_f64(), Some(2.0), "evaluation is unaffected");
        assert!(
            matches!(expr, ArithmaExpression::Function(..)),
            "the tree itself must be untouched"
        );
    }

    #[test]
    fn nested_folding_completes_in_a_single_pass() {
        // (1 + 1) * 3 -> 6. Reverse pre-order means the inner sum is folded
        // before the product is examined, so one pass suffices.
        let mut expr = ArithmaExpression::mul(ArithmaExpression::add(n(1), n(1)), n(3));
        let mut s = ArithmaIterativeSimplifier::new();
        assert!(s.simplify(&mut expr, &cfg()));
        assert_eq!(expr.to_f64(), Some(6.0));
        assert!(
            matches!(expr, ArithmaExpression::Number(_)),
            "the result must be a literal, not a Function node"
        );
        assert_eq!(
            s.last_iterations(),
            2,
            "one pass to fold, one to confirm the fixpoint"
        );
    }

    #[test]
    fn additive_identity_drops() {
        let mut expr = ArithmaExpression::add(ArithmaExpression::var("x"), n(0));
        assert!(simplify_iterative(&mut expr, &cfg()));
        assert!(
            matches!(&expr, ArithmaExpression::Variable(v) if v == "x"),
            "x + 0 must collapse to x, got {expr:?}"
        );
    }

    #[test]
    fn subtraction_only_drops_a_trailing_zero() {
        let mut right = ArithmaExpression::sub(ArithmaExpression::var("x"), n(0));
        assert!(simplify_iterative(&mut right, &cfg()));
        assert!(matches!(&right, ArithmaExpression::Variable(v) if v == "x"));

        // 0 - x is -x, not x. Getting this wrong flips the sign of the result,
        // so the rule must not be applied commutatively.
        let mut left = ArithmaExpression::sub(n(0), ArithmaExpression::var("x"));
        let _ = simplify_iterative(&mut left, &cfg());
        assert!(
            matches!(left, ArithmaExpression::Function(..)),
            "0 - x must stay a subtraction, got {left:?}"
        );
    }

    #[test]
    fn multiplicative_identity_and_absorbing_zero() {
        let mut one = ArithmaExpression::mul(ArithmaExpression::var("x"), n(1));
        assert!(simplify_iterative(&mut one, &cfg()));
        assert!(matches!(&one, ArithmaExpression::Variable(v) if v == "x"));

        let mut zero = ArithmaExpression::mul(ArithmaExpression::var("x"), n(0));
        assert!(simplify_iterative(&mut zero, &cfg()));
        assert_eq!(zero.to_f64(), Some(0.0), "x * 0 must be 0, got {zero:?}");
    }

    #[test]
    fn absorbing_zero_is_not_applied_through_an_opaque_factor() {
        // 0 * (1/y) is not zero in general -- 1/y may be infinite. The rule is
        // deliberately restricted to operands that cannot be inf or NaN.
        let reciprocal = ArithmaExpression::div(n(1), ArithmaExpression::var("y"));
        let mut expr = ArithmaExpression::mul(n(0), reciprocal);
        let _ = simplify_iterative(&mut expr, &cfg());
        assert!(
            matches!(expr, ArithmaExpression::Function(..)),
            "0 * (1/y) must stay symbolic, got {expr:?}"
        );
    }

    #[test]
    fn division_folds_exactly_or_not_at_all() {
        let mut exact = ArithmaExpression::div(n(12), n(4));
        assert!(simplify_iterative(&mut exact, &cfg()));
        assert_eq!(exact.to_f64(), Some(3.0));

        let mut inexact = ArithmaExpression::div(n(7), n(2));
        let _ = simplify_iterative(&mut inexact, &cfg());
        assert!(
            matches!(inexact, ArithmaExpression::Function(..)),
            "7/2 must stay a quotient rather than round to 3 or 4"
        );

        // Division by zero is an evaluation error, not something to rewrite.
        let mut by_zero = ArithmaExpression::div(n(1), n(0));
        assert!(!simplify_iterative(&mut by_zero, &cfg()));
        assert!(matches!(by_zero, ArithmaExpression::Function(..)));
    }

    #[test]
    fn power_rules() {
        let mut exp_one = ArithmaExpression::pow(ArithmaExpression::var("x"), n(1));
        assert!(simplify_iterative(&mut exp_one, &cfg()));
        assert!(matches!(&exp_one, ArithmaExpression::Variable(v) if v == "x"));

        let mut exp_zero = ArithmaExpression::pow(ArithmaExpression::var("x"), n(0));
        assert!(simplify_iterative(&mut exp_zero, &cfg()));
        assert_eq!(exp_zero.to_f64(), Some(1.0));

        let mut folded = ArithmaExpression::pow(n(2), n(10));
        assert!(simplify_iterative(&mut folded, &cfg()));
        assert_eq!(folded.to_f64(), Some(1024.0));
    }

    #[test]
    fn zero_to_the_zero_is_left_alone() {
        // 0^0 is a genuine ambiguity. Silently picking 1 would be a wrong
        // answer dressed as a simplification.
        let mut expr = ArithmaExpression::pow(n(0), n(0));
        assert!(!simplify_iterative(&mut expr, &cfg()));
        assert!(matches!(expr, ArithmaExpression::Function(..)));
    }

    #[test]
    fn symbolic_subterms_survive_a_partial_fold() {
        // (2 + 3) + x -> 5 + x. The numeric part folds; x is untouched.
        let mut expr = ArithmaExpression::add(
            ArithmaExpression::add(n(2), n(3)),
            ArithmaExpression::var("x"),
        );
        assert!(simplify_iterative(&mut expr, &cfg()));
        assert_eq!(expr.to_f64(), None, "a free variable must remain");
        match &expr {
            ArithmaExpression::Function(_, args) => {
                assert_eq!(args.len(), 2, "expected `5 + x`, got {expr:?}");
                assert_eq!(args[0].to_f64(), Some(5.0), "2 + 3 should have folded");
                assert!(matches!(&args[1], ArithmaExpression::Variable(v) if v == "x"));
            }
            other => panic!("expected a sum, got {other:?}"),
        }
    }

    #[test]
    fn a_chain_at_the_depth_cap_collapses_completely() {
        // The traversal is iterative by mandate (no recursion), so depth costs
        // heap rather than stack and a single pass reduces the whole chain --
        // it is not limited to the top few levels.
        let mut expr = ArithmaExpression::var("x");
        for _ in 0..(MAX_DEPTH - 4) {
            expr = ArithmaExpression::add(expr, n(0));
        }
        let changed = simplify_iterative(&mut expr, &cfg());
        assert!(changed, "a chain of additions of zero should collapse");
        assert!(
            matches!(&expr, ArithmaExpression::Variable(v) if v == "x"),
            "the whole chain should reduce to x"
        );
    }

    #[test]
    fn a_tree_past_the_depth_cap_is_left_untouched_rather_than_truncated() {
        let mut expr = ArithmaExpression::var("x");
        for _ in 0..(MAX_DEPTH + 8) {
            expr = ArithmaExpression::add(expr, n(0));
        }
        // Refusing is the safe answer: returning a partially-rewritten tree
        // would be worse than returning the input.
        let changed = simplify_iterative(&mut expr, &cfg());
        assert!(!changed, "a tree past the depth cap must report no change");
        assert!(matches!(expr, ArithmaExpression::Function(..)));
    }

    #[test]
    fn constants_collapse_only_when_the_caller_opts_in() {
        let two = ArithmaExpression::Constant {
            name: Some("two".to_string()),
            symbol: "two".to_string(),
            cached_value: Some(2.0),
            allow_simplification: true,
            unit: None,
            prefix: None,
        };
        let guarded = SimplificationConfig {
            max_iterations: 32,
            allow_numeric_collapse: false,
        };
        let mut kept = two.clone();
        assert!(!simplify_iterative(&mut kept, &guarded));
        assert!(matches!(kept, ArithmaExpression::Constant { .. }));

        let permissive = SimplificationConfig {
            max_iterations: 32,
            allow_numeric_collapse: true,
        };
        let mut collapsed = two;
        assert!(simplify_iterative(&mut collapsed, &permissive));
        assert_eq!(collapsed.to_f64(), Some(2.0));
    }

    #[test]
    fn the_public_trait_routes_to_the_same_engine() {
        use crate::expression::Simplify;
        let expr = ArithmaExpression::add(n(20), n(22));
        let simplified = expr.simplify(&cfg());
        assert_eq!(simplified.to_f64(), Some(42.0));
        assert!(
            matches!(simplified, ArithmaExpression::Number(_)),
            "Simplify::simplify was a stub returning a clone; it must now fold"
        );

        let mut in_place = ArithmaExpression::add(n(20), n(22));
        assert!(in_place.simplify_in_place(&cfg()));
        assert!(
            !in_place.simplify_in_place(&cfg()),
            "a second call must report no change so caller loops terminate"
        );
    }
    // ─── like-term collection ──────────────────────────────────────────────

    fn latex(expr: &ArithmaExpression) -> String {
        format!("{expr:?}")
    }

    #[test]
    fn like_terms_collect_into_a_coefficient() {
        // x + x -> 2*x
        let mut expr =
            ArithmaExpression::add(ArithmaExpression::var("x"), ArithmaExpression::var("x"));
        assert!(simplify_iterative(&mut expr, &cfg()));
        assert_eq!(expr.to_f64(), None, "the result must stay symbolic");
        match &expr {
            ArithmaExpression::Function(ArithmaFunction::Multiply, args) => {
                assert_eq!(args.len(), 2, "expected 2*x, got {}", latex(&expr));
                assert_eq!(args[0].to_f64(), Some(2.0), "coefficient should be 2");
                assert!(matches!(&args[1], ArithmaExpression::Variable(v) if v == "x"));
            }
            other => panic!("expected a product, got {other:?}"),
        }
    }

    #[test]
    fn existing_coefficients_add() {
        // 2*x + 3*x -> 5*x
        let mut expr = ArithmaExpression::add(
            ArithmaExpression::mul(n(2), ArithmaExpression::var("x")),
            ArithmaExpression::mul(n(3), ArithmaExpression::var("x")),
        );
        assert!(simplify_iterative(&mut expr, &cfg()));
        match &expr {
            ArithmaExpression::Function(ArithmaFunction::Multiply, args) => {
                assert_eq!(args[0].to_f64(), Some(5.0), "got {}", latex(&expr));
            }
            other => panic!("expected 5*x, got {other:?}"),
        }
    }

    #[test]
    fn opposite_terms_cancel_to_zero() {
        // 3*x + (-3)*x -> 0
        let mut expr = ArithmaExpression::add(
            ArithmaExpression::mul(n(3), ArithmaExpression::var("x")),
            ArithmaExpression::mul(n(-3), ArithmaExpression::var("x")),
        );
        assert!(simplify_iterative(&mut expr, &cfg()));
        assert_eq!(
            expr.to_f64(),
            Some(0.0),
            "3x - 3x must vanish, got {expr:?}"
        );
    }

    #[test]
    fn unlike_terms_are_left_alone() {
        // x + y has nothing to collect and must report no change, or the
        // fixpoint loop never converges.
        let mut expr =
            ArithmaExpression::add(ArithmaExpression::var("x"), ArithmaExpression::var("y"));
        assert!(
            !simplify_iterative(&mut expr, &cfg()),
            "x + y is already simplest; got {expr:?}"
        );
    }

    #[test]
    fn a_reordered_sum_reports_that_it_changed() {
        // x + 2 is now reordered to 2 + x -- the canonical order puts the
        // literal first. The property this test has always been about is
        // unchanged: a pass that moves an operand must *say* it changed the
        // tree, and must settle immediately afterwards. A silent reorder is
        // the one failure the fixpoint loop cannot detect.
        let mut expr = ArithmaExpression::add(ArithmaExpression::var("x"), n(2));
        assert!(
            simplify_iterative(&mut expr, &cfg()),
            "reordering x + 2 must be reported"
        );
        match &expr {
            ArithmaExpression::Function(_, args) => {
                assert_eq!(args[0].to_f64(), Some(2.0), "the literal must lead");
                assert!(matches!(&args[1], ArithmaExpression::Variable(v) if v == "x"));
            }
            other => panic!("expected a sum, got {other:?}"),
        }
        assert!(
            !simplify_iterative(&mut expr, &cfg()),
            "an ordered sum must be stable, got {expr:?}"
        );
    }

    #[test]
    fn an_already_canonical_sum_is_left_alone() {
        // The borrow-only pre-check still has to hold: 2 + x is canonical, so
        // nothing may be taken apart and nothing reported.
        let mut expr = ArithmaExpression::add(n(2), ArithmaExpression::var("x"));
        assert!(!simplify_iterative(&mut expr, &cfg()), "got {expr:?}");
    }

    #[test]
    fn collection_reaches_a_fixpoint() {
        // The property the pre-check protects: a second pass must report no
        // change, or every caller looping until `false` spins forever.
        let mut expr =
            ArithmaExpression::add(ArithmaExpression::var("x"), ArithmaExpression::var("x"));
        assert!(simplify_iterative(&mut expr, &cfg()), "first pass collects");
        assert!(
            !simplify_iterative(&mut expr, &cfg()),
            "second pass must be stable, got {expr:?}"
        );
    }

    #[test]
    fn equal_factors_become_a_power() {
        // x * x -> x^2
        let mut expr =
            ArithmaExpression::mul(ArithmaExpression::var("x"), ArithmaExpression::var("x"));
        assert!(simplify_iterative(&mut expr, &cfg()));
        match &expr {
            ArithmaExpression::Function(ArithmaFunction::Power, args) => {
                assert!(matches!(&args[0], ArithmaExpression::Variable(v) if v == "x"));
                assert_eq!(args[1].to_f64(), Some(2.0), "got {}", latex(&expr));
            }
            other => panic!("expected x^2, got {other:?}"),
        }
    }

    #[test]
    fn powers_of_the_same_base_add_their_exponents() {
        // x^2 * x -> x^3
        let mut expr = ArithmaExpression::mul(
            ArithmaExpression::pow(ArithmaExpression::var("x"), n(2)),
            ArithmaExpression::var("x"),
        );
        assert!(simplify_iterative(&mut expr, &cfg()));
        match &expr {
            ArithmaExpression::Function(ArithmaFunction::Power, args) => {
                assert_eq!(args[1].to_f64(), Some(3.0), "got {}", latex(&expr));
            }
            other => panic!("expected x^3, got {other:?}"),
        }
    }

    #[test]
    fn different_bases_do_not_merge() {
        let mut expr =
            ArithmaExpression::mul(ArithmaExpression::var("x"), ArithmaExpression::var("y"));
        assert!(!simplify_iterative(&mut expr, &cfg()));
    }

    #[test]
    fn structural_equality_is_syntactic_but_ordering_gets_there_first() {
        // Comparison is still syntactic -- `structural_eq` has not learned
        // anything about commutativity, and 2*x and x*2 are still two
        // different trees to it.
        let a = ArithmaExpression::mul(n(2), ArithmaExpression::var("x"));
        let b = ArithmaExpression::mul(ArithmaExpression::var("x"), n(2));
        assert!(structural_eq(&a, &a));
        assert!(
            !structural_eq(&a, &b),
            "structural equality compares shape, not value"
        );

        // What changed is that x*2 no longer reaches it in that shape. The
        // limit this test used to pin -- that 2*x + x*2 does not collect --
        // is gone, because canonical ordering normalises both operands first.
        let mut commuted = ArithmaExpression::add(a, b);
        assert!(simplify_iterative(&mut commuted, &cfg()));
        match &commuted {
            ArithmaExpression::Function(ArithmaFunction::Multiply, args) => {
                assert_eq!(
                    args[0].to_f64(),
                    Some(4.0),
                    "2*x + x*2 should collect to 4*x, got {commuted:?}"
                );
                assert!(matches!(&args[1], ArithmaExpression::Variable(v) if v == "x"));
            }
            other => panic!("expected 4*x, got {other:?}"),
        }
        assert!(
            !simplify_iterative(&mut commuted, &cfg()),
            "the collected result must be stable"
        );
    }

    // ─── canonical ordering ────────────────────────────────────────────────

    /// One of each variant this module ranks, plus a few compounds, for the
    /// order-property tests below.
    fn ordering_samples() -> Vec<ArithmaExpression> {
        vec![
            n(-7),
            n(0),
            n(3),
            ArithmaExpression::Constant {
                name: None,
                symbol: "pi".to_string(),
                cached_value: None,
                allow_simplification: false,
                unit: None,
                prefix: None,
            },
            ArithmaExpression::var("a"),
            ArithmaExpression::var("z"),
            ArithmaExpression::sin(ArithmaExpression::var("x")),
            ArithmaExpression::cos(ArithmaExpression::var("x")),
            ArithmaExpression::sin(ArithmaExpression::var("y")),
            ArithmaExpression::mul(n(2), ArithmaExpression::var("x")),
            ArithmaExpression::pow(ArithmaExpression::var("x"), n(2)),
            ArithmaExpression::add(ArithmaExpression::var("x"), ArithmaExpression::var("y")),
        ]
    }

    #[test]
    fn the_order_is_total_and_antisymmetric() {
        let samples = ordering_samples();
        for a in samples.iter() {
            assert_eq!(
                compare_expressions(a, a),
                Ordering::Equal,
                "an expression must order equal to itself: {a:?}"
            );
            for b in samples.iter() {
                let forward = compare_expressions(a, b);
                let backward = compare_expressions(b, a);
                assert_eq!(
                    forward,
                    backward.reverse(),
                    "the order disagrees with itself on {a:?} vs {b:?}"
                );
                // Determinism: a second ask gives the same answer. The whole
                // fixpoint argument rests on this being a pure function.
                assert_eq!(forward, compare_expressions(a, b));
            }
        }
    }

    #[test]
    fn the_order_is_transitive() {
        let samples = ordering_samples();
        for a in samples.iter() {
            for b in samples.iter() {
                for c in samples.iter() {
                    let (ab, bc) = (compare_expressions(a, b), compare_expressions(b, c));
                    if ab == Ordering::Less && bc == Ordering::Less {
                        assert_eq!(
                            compare_expressions(a, c),
                            Ordering::Less,
                            "transitivity fails on {a:?} < {b:?} < {c:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn structurally_equal_expressions_order_equal() {
        // The direction that matters: if collection would merge two operands,
        // the sort must not be able to claim they are different. The converse
        // is allowed to fail and is documented as such.
        for a in ordering_samples() {
            let copy = a.clone();
            assert!(structural_eq(&a, &copy));
            assert_eq!(compare_expressions(&a, &copy), Ordering::Equal);
        }
    }

    #[test]
    fn sorting_is_idempotent() {
        // The single most important property here. Sorting an already-sorted
        // list must not move anything, or `is_canonically_ordered` reports
        // work forever and the fixpoint loop never terminates.
        let mut operands = ordering_samples();
        order_operands(&mut operands);
        assert!(is_canonically_ordered(&operands));
        let once = format!("{operands:?}");
        order_operands(&mut operands);
        assert!(is_canonically_ordered(&operands));
        assert_eq!(once, format!("{operands:?}"), "a second sort moved things");
    }

    #[test]
    fn ordering_ranks_literals_first_and_variables_last() {
        // Both ends of the ranking are load-bearing: a leading literal is what
        // `split_coefficient` reads as a coefficient, and a trailing variable
        // is what leaves `(a + b) + c` -- the shape a left-associative parse
        // produces -- already canonical.
        let compound =
            ArithmaExpression::add(ArithmaExpression::var("a"), ArithmaExpression::var("b"));
        assert_eq!(
            compare_expressions(&n(2), &ArithmaExpression::var("x")),
            Ordering::Less
        );
        assert_eq!(
            compare_expressions(&compound, &ArithmaExpression::var("c")),
            Ordering::Less
        );
        assert!(is_canonically_ordered(&[
            compound,
            ArithmaExpression::var("c")
        ]));
    }

    #[test]
    fn commuted_factors_are_canonicalised() {
        // x*2 -> 2*x. This is the rewrite that makes commuted terms collect.
        let mut expr = ArithmaExpression::mul(ArithmaExpression::var("x"), n(2));
        assert!(simplify_iterative(&mut expr, &cfg()));
        match &expr {
            ArithmaExpression::Function(ArithmaFunction::Multiply, args) => {
                assert_eq!(args[0].to_f64(), Some(2.0), "got {}", latex(&expr));
                assert!(matches!(&args[1], ArithmaExpression::Variable(v) if v == "x"));
            }
            other => panic!("expected 2*x, got {other:?}"),
        }
        assert!(!simplify_iterative(&mut expr, &cfg()), "must be stable");
    }

    #[test]
    fn non_commutative_operands_are_never_reordered() {
        // Subtraction, division and exponentiation all have a literal in the
        // position the canonical order would pull to the front. Moving it
        // changes the value, so none of them may be sorted.
        let cases = [
            ArithmaExpression::sub(ArithmaExpression::var("x"), n(2)),
            ArithmaExpression::div(ArithmaExpression::var("x"), n(2)),
            ArithmaExpression::pow(ArithmaExpression::var("x"), n(2)),
        ];
        for case in cases {
            let mut expr = case.clone();
            let _ = simplify_iterative(&mut expr, &cfg());
            match &expr {
                ArithmaExpression::Function(_, args) => {
                    assert!(
                        matches!(&args[0], ArithmaExpression::Variable(v) if v == "x"),
                        "a non-commutative operator was reordered: {case:?} became {expr:?}"
                    );
                }
                other => panic!("expected the operator to survive, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_reordered_pass_still_reaches_a_fixpoint_on_a_deep_tree() {
        // Ordering runs on every commutative node, so a tree that needs a lot
        // of it is the case where a non-idempotent sort would show up as a
        // pass that never stops reporting work.
        let mut expr = ArithmaExpression::var("x");
        for k in 0..64 {
            expr = ArithmaExpression::mul(expr, n(k + 2));
        }
        let mut simplifier = ArithmaIterativeSimplifier::new();
        assert!(simplifier.simplify(&mut expr, &cfg()));
        assert!(
            simplifier.last_iterations() < cfg().max_iterations,
            "the pass used its whole budget, which means it never converged"
        );
        assert!(!simplify_iterative(&mut expr, &cfg()), "got {expr:?}");
    }

    // ─── transcendental identities ─────────────────────────────────────────

    #[test]
    fn the_transcendentals_fold_where_they_have_an_exact_value() {
        let cases: [(ArithmaExpression, f64); 8] = [
            (ArithmaExpression::sin(n(0)), 0.0),
            (ArithmaExpression::cos(n(0)), 1.0),
            (ArithmaExpression::tan(n(0)), 0.0),
            (ArithmaExpression::exp(n(0)), 1.0),
            (ArithmaExpression::ln(n(1)), 0.0),
            (
                ArithmaExpression::Function(ArithmaFunction::Sinh, vec![n(0)]),
                0.0,
            ),
            (
                ArithmaExpression::Function(ArithmaFunction::Cosh, vec![n(0)]),
                1.0,
            ),
            (
                ArithmaExpression::Function(ArithmaFunction::Log10, vec![n(1)]),
                0.0,
            ),
        ];
        for (case, expected) in cases {
            let mut expr = case.clone();
            assert!(
                simplify_iterative(&mut expr, &cfg()),
                "{case:?} should have folded"
            );
            assert!(
                matches!(expr, ArithmaExpression::Number(_)),
                "{case:?} should have become a literal, got {expr:?}"
            );
            assert_eq!(expr.to_f64(), Some(expected), "wrong value for {case:?}");
            assert!(!simplify_iterative(&mut expr, &cfg()), "must be stable");
        }
    }

    #[test]
    fn the_poles_and_the_partial_identities_are_left_alone() {
        // cot(0) and csc(0) are poles, acos(0) is pi/2 with no exact literal
        // here, sin(1) is irrational, and log_b(1) is 0 only for a legitimate
        // base. None of them may be folded.
        let cases = [
            ArithmaExpression::Function(ArithmaFunction::Cot, vec![n(0)]),
            ArithmaExpression::Function(ArithmaFunction::Csc, vec![n(0)]),
            ArithmaExpression::Function(ArithmaFunction::Acos, vec![n(0)]),
            ArithmaExpression::sin(n(1)),
            ArithmaExpression::Function(
                ArithmaFunction::LogBase(ArithmaInteger::from_i64(1)),
                vec![n(1)],
            ),
        ];
        for case in cases {
            let mut expr = case.clone();
            assert!(
                !simplify_iterative(&mut expr, &cfg()),
                "{case:?} is not an unconditional identity and must not fold"
            );
        }
    }

    #[test]
    fn the_square_root_of_a_square_is_not_the_base() {
        // sqrt(x^2) is |x|. Rewriting it to x is wrong for every negative x,
        // and is the classic wrong simplification, so it stays out.
        let mut expr =
            ArithmaExpression::sqrt(ArithmaExpression::pow(ArithmaExpression::var("x"), n(2)));
        let _ = simplify_iterative(&mut expr, &cfg());
        assert!(
            matches!(&expr, ArithmaExpression::Function(ArithmaFunction::Sqrt, _)),
            "sqrt(x^2) must stay a square root, got {expr:?}"
        );
    }

    #[test]
    fn the_pythagorean_identity_collapses_to_one() {
        // sin^2(x) + cos^2(x) -> 1, for every x, with nothing to check first.
        let mut expr = ArithmaExpression::add(
            ArithmaExpression::pow(ArithmaExpression::sin(ArithmaExpression::var("x")), n(2)),
            ArithmaExpression::pow(ArithmaExpression::cos(ArithmaExpression::var("x")), n(2)),
        );
        assert!(simplify_iterative(&mut expr, &cfg()));
        assert_eq!(expr.to_f64(), Some(1.0), "got {expr:?}");
        assert!(!simplify_iterative(&mut expr, &cfg()), "must be stable");
    }

    #[test]
    fn the_pythagorean_identity_survives_a_commuted_writing() {
        // cos^2(x) + sin^2(x), and the same thing written as products rather
        // than powers. Factor collection turns sin(x)*sin(x) into the power
        // before the sum is examined, so both reach the identity.
        let mut commuted = ArithmaExpression::add(
            ArithmaExpression::pow(ArithmaExpression::cos(ArithmaExpression::var("x")), n(2)),
            ArithmaExpression::pow(ArithmaExpression::sin(ArithmaExpression::var("x")), n(2)),
        );
        assert!(simplify_iterative(&mut commuted, &cfg()));
        assert_eq!(commuted.to_f64(), Some(1.0), "got {commuted:?}");

        let sin = || ArithmaExpression::sin(ArithmaExpression::var("t"));
        let cos = || ArithmaExpression::cos(ArithmaExpression::var("t"));
        let mut as_products = ArithmaExpression::add(
            ArithmaExpression::mul(sin(), sin()),
            ArithmaExpression::mul(cos(), cos()),
        );
        assert!(simplify_iterative(&mut as_products, &cfg()));
        assert_eq!(as_products.to_f64(), Some(1.0), "got {as_products:?}");
    }

    #[test]
    fn the_pythagorean_identity_needs_the_same_argument() {
        // sin^2(x) + cos^2(y) is not 1. The arguments are compared, not
        // assumed.
        let mut expr = ArithmaExpression::add(
            ArithmaExpression::pow(ArithmaExpression::sin(ArithmaExpression::var("x")), n(2)),
            ArithmaExpression::pow(ArithmaExpression::cos(ArithmaExpression::var("y")), n(2)),
        );
        let _ = simplify_iterative(&mut expr, &cfg());
        assert_eq!(
            expr.to_f64(),
            None,
            "sin^2(x) + cos^2(y) must stay symbolic, got {expr:?}"
        );
    }

    #[test]
    fn structural_equality_distinguishes_functions_and_children() {
        let sin_x = ArithmaExpression::sin(ArithmaExpression::var("x"));
        let cos_x = ArithmaExpression::cos(ArithmaExpression::var("x"));
        let sin_y = ArithmaExpression::sin(ArithmaExpression::var("y"));
        assert!(structural_eq(&sin_x, &sin_x.clone()));
        assert!(!structural_eq(&sin_x, &cos_x), "different function tags");
        assert!(!structural_eq(&sin_x, &sin_y), "different arguments");
    }

    #[test]
    fn like_function_terms_collect_too() {
        // sin(x) + sin(x) -> 2*sin(x). Collection is not limited to variables.
        let mut expr = ArithmaExpression::add(
            ArithmaExpression::sin(ArithmaExpression::var("x")),
            ArithmaExpression::sin(ArithmaExpression::var("x")),
        );
        assert!(simplify_iterative(&mut expr, &cfg()));
        match &expr {
            ArithmaExpression::Function(ArithmaFunction::Multiply, args) => {
                assert_eq!(args[0].to_f64(), Some(2.0), "got {}", latex(&expr));
            }
            other => panic!("expected 2*sin(x), got {other:?}"),
        }
    }
}
