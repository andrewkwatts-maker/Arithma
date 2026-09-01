//====== Arithma/rust/arithma_core/src/numerical/interval_analysis.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Interval analysis
//!
//! Interval arithmetic for guaranteed bounds. Used by the root finder to prune
//! search regions and by the simplifier to detect provably-empty branches.

use std::f64::consts::{FRAC_PI_2, PI, TAU};

use crate::expression::ArithmaExpression;
use crate::function::ArithmaFunction;

/// Closed interval `[lo, hi]` over the reals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArithmaInterval {
    pub lo: f64,
    pub hi: f64,
}

impl ArithmaInterval {
    /// Construct an interval. `lo` must be ≤ `hi`.
    pub fn new(lo: f64, hi: f64) -> Self {
        Self { lo, hi }
    }

    /// The whole real line.
    pub fn whole() -> Self {
        Self {
            lo: f64::NEG_INFINITY,
            hi: f64::INFINITY,
        }
    }

    /// The empty interval. Represented with `lo > hi`.
    pub fn empty() -> Self {
        Self {
            lo: f64::INFINITY,
            hi: f64::NEG_INFINITY,
        }
    }

    /// Width `hi - lo`.
    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }

    /// True iff this interval is empty.
    pub fn is_empty(&self) -> bool {
        self.lo > self.hi
    }

    /// True iff `point` is contained in `[lo, hi]`.
    pub fn contains(&self, point: f64) -> bool {
        !self.is_empty() && point >= self.lo && point <= self.hi
    }

    /// A degenerate interval holding the single value `v`.
    pub fn point(v: f64) -> Self {
        Self { lo: v, hi: v }
    }

    /// True iff this interval is a single finite value.
    pub fn is_point(&self) -> bool {
        self.lo.is_finite() && self.lo == self.hi
    }

    /// Smallest interval containing every finite value in `values`.
    ///
    /// NaN entries are ignored; if nothing usable remains the result is
    /// [`ArithmaInterval::whole`] rather than a silently wrong bound.
    pub fn hull(values: &[f64]) -> Self {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        let mut saw_nan = false;
        for &v in values {
            if v.is_nan() {
                saw_nan = true;
                continue;
            }
            lo = lo.min(v);
            hi = hi.max(v);
        }
        if lo > hi || saw_nan {
            return Self::whole();
        }
        Self { lo, hi }
    }

    /// Smallest interval containing both `self` and `other`.
    pub fn union(self, other: Self) -> Self {
        if self.is_empty() {
            return other;
        }
        if other.is_empty() {
            return self;
        }
        Self {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }

    /// Intersection of `self` and `other`; may be empty.
    pub fn intersect(self, other: Self) -> Self {
        Self {
            lo: self.lo.max(other.lo),
            hi: self.hi.min(other.hi),
        }
    }
}

/// Hard cap on visited nodes, mirroring the evaluator's bound (CLAUDE.md safety
/// rule 2: all loops have fixed bounds).
const ARITHMA_INTERVAL_NODE_CAP: usize = 1_048_576;

/// Above this magnitude the argument-reduction used by the periodic functions
/// loses enough precision that we stop claiming a tight bound and widen to the
/// full range instead.
const ARITHMA_INTERVAL_PERIODIC_LIMIT: f64 = 1.0e12;

/// `a * b` under the interval convention that `0 · ∞ = 0` (an interval pinned
/// at zero contributes nothing regardless of the other factor's magnitude).
fn safe_mul(a: f64, b: f64) -> f64 {
    if a == 0.0 || b == 0.0 {
        0.0
    } else {
        a * b
    }
}

/// `a / b` with the same zero convention; `b == 0` is the caller's problem.
fn safe_div(a: f64, b: f64) -> f64 {
    if a == 0.0 && b != 0.0 {
        0.0
    } else {
        a / b
    }
}

fn mul_interval(x: ArithmaInterval, y: ArithmaInterval) -> ArithmaInterval {
    ArithmaInterval::hull(&[
        safe_mul(x.lo, y.lo),
        safe_mul(x.lo, y.hi),
        safe_mul(x.hi, y.lo),
        safe_mul(x.hi, y.hi),
    ])
}

fn div_interval(x: ArithmaInterval, y: ArithmaInterval) -> ArithmaInterval {
    if y.contains(0.0) {
        // The true image is a union of two unbounded rays; the tightest single
        // interval enclosing it is the whole line.
        return ArithmaInterval::whole();
    }
    ArithmaInterval::hull(&[
        safe_div(x.lo, y.lo),
        safe_div(x.lo, y.hi),
        safe_div(x.hi, y.lo),
        safe_div(x.hi, y.hi),
    ])
}

/// True iff some `target + k·2π` (`k ∈ ℤ`) lies inside `iv`.
///
/// Only meaningful for narrow, moderately-sized intervals; callers guard with
/// [`ARITHMA_INTERVAL_PERIODIC_LIMIT`] and a width check first.
fn contains_congruent(iv: ArithmaInterval, target: f64) -> bool {
    let k = ((iv.lo - target) / TAU).ceil();
    let candidate = target + k * TAU;
    candidate >= iv.lo && candidate <= iv.hi
}

/// Image of a bounded periodic function, given its extremum phases.
fn periodic_image(
    iv: ArithmaInterval,
    f: fn(f64) -> f64,
    max_phase: f64,
    min_phase: f64,
) -> ArithmaInterval {
    if !iv.lo.is_finite()
        || !iv.hi.is_finite()
        || iv.width() >= TAU
        || iv.lo.abs() > ARITHMA_INTERVAL_PERIODIC_LIMIT
        || iv.hi.abs() > ARITHMA_INTERVAL_PERIODIC_LIMIT
    {
        return ArithmaInterval::new(-1.0, 1.0);
    }
    let (a, b) = (f(iv.lo), f(iv.hi));
    let mut lo = a.min(b);
    let mut hi = a.max(b);
    if contains_congruent(iv, max_phase) {
        hi = 1.0;
    }
    if contains_congruent(iv, min_phase) {
        lo = -1.0;
    }
    ArithmaInterval::new(lo.max(-1.0), hi.min(1.0))
}

/// Image of a monotonically increasing function over `iv`.
fn increasing(iv: ArithmaInterval, f: fn(f64) -> f64) -> ArithmaInterval {
    ArithmaInterval::hull(&[f(iv.lo), f(iv.hi)])
}

/// Image of a monotonically decreasing function over `iv`.
fn decreasing(iv: ArithmaInterval, f: fn(f64) -> f64) -> ArithmaInterval {
    ArithmaInterval::hull(&[f(iv.hi), f(iv.lo)])
}

/// Restrict `iv` to a function's real domain before mapping it.
///
/// Returns `ArithmaInterval::empty()` when nothing survives — the honest answer
/// for "this expression has no real value anywhere on the input".
fn clip(iv: ArithmaInterval, domain_lo: f64, domain_hi: f64) -> ArithmaInterval {
    iv.intersect(ArithmaInterval::new(domain_lo, domain_hi))
}

/// `iv` raised to an integer power.
fn pow_int(iv: ArithmaInterval, k: i64) -> ArithmaInterval {
    if k == 0 {
        return ArithmaInterval::point(1.0);
    }
    if k < 0 {
        let positive = pow_int(iv, -k);
        return div_interval(ArithmaInterval::point(1.0), positive);
    }
    let e = k as f64;
    if k % 2 == 0 {
        // Even powers fold the negative half onto the positive one, so the
        // interior extremum at x = 0 matters whenever the input straddles it.
        let a = iv.lo.abs();
        let b = iv.hi.abs();
        let outer = a.max(b).powf(e);
        if iv.contains(0.0) {
            ArithmaInterval::new(0.0, outer)
        } else {
            ArithmaInterval::new(a.min(b).powf(e), outer)
        }
    } else {
        // Odd powers are strictly increasing.
        ArithmaInterval::hull(&[iv.lo.powf(e), iv.hi.powf(e)])
    }
}

fn pow_interval(base: ArithmaInterval, exponent: ArithmaInterval) -> ArithmaInterval {
    if exponent.is_point() {
        let e = exponent.lo;
        let rounded = e.round();
        if (e - rounded).abs() < 1e-12 && rounded.abs() <= 1024.0 {
            return pow_int(base, rounded as i64);
        }
        // Fractional exponent: only defined for a non-negative base, so clip
        // the domain first.
        let clipped = clip(base, 0.0, f64::INFINITY);
        if clipped.is_empty() {
            return ArithmaInterval::empty();
        }
        return if e > 0.0 {
            ArithmaInterval::hull(&[clipped.lo.powf(e), clipped.hi.powf(e)])
        } else {
            ArithmaInterval::hull(&[clipped.hi.powf(e), clipped.lo.powf(e)])
        };
    }
    // Non-degenerate exponent. On a strictly positive base, x^y is monotone in
    // each argument separately, so the four corners bound it exactly.
    if base.lo > 0.0 && base.lo.is_finite() && base.hi.is_finite() {
        return ArithmaInterval::hull(&[
            base.lo.powf(exponent.lo),
            base.lo.powf(exponent.hi),
            base.hi.powf(exponent.lo),
            base.hi.powf(exponent.hi),
        ]);
    }
    // Conservative widening: a sign-changing base with a varying exponent has
    // no useful real enclosure.
    ArithmaInterval::whole()
}

/// Apply one `ArithmaFunction` to already-computed argument intervals.
///
/// Returns `Err` for operators this module does not model, so a caller can
/// never mistake "not supported" for "unbounded".
fn apply_interval_function(
    func: &ArithmaFunction,
    args: &[ArithmaInterval],
) -> Result<ArithmaInterval, String> {
    use ArithmaFunction as F;

    // Any empty argument makes the whole application empty.
    if args.iter().any(|a| a.is_empty()) {
        return Ok(ArithmaInterval::empty());
    }
    let first = |n: usize| -> Result<ArithmaInterval, String> {
        args.first()
            .copied()
            .ok_or_else(|| format!("evaluate_interval: {func:?} expects {n} argument(s), got 0"))
    };
    let binary = || -> Result<(ArithmaInterval, ArithmaInterval), String> {
        if args.len() != 2 {
            return Err(format!(
                "evaluate_interval: {func:?} expects 2 arguments, got {}",
                args.len()
            ));
        }
        Ok((args[0], args[1]))
    };

    let out = match func {
        F::Add => {
            let mut acc = ArithmaInterval::point(0.0);
            for a in args {
                acc = ArithmaInterval::new(acc.lo + a.lo, acc.hi + a.hi);
            }
            acc
        }
        F::Subtract => {
            let (x, y) = binary()?;
            ArithmaInterval::new(x.lo - y.hi, x.hi - y.lo)
        }
        F::Multiply => {
            let mut acc = ArithmaInterval::point(1.0);
            for a in args {
                acc = mul_interval(acc, *a);
            }
            acc
        }
        F::Divide => {
            let (x, y) = binary()?;
            div_interval(x, y)
        }
        F::Power => {
            let (x, y) = binary()?;
            pow_interval(x, y)
        }
        F::Negate => {
            let x = first(1)?;
            ArithmaInterval::new(-x.hi, -x.lo)
        }
        F::Sqrt => {
            let x = clip(first(1)?, 0.0, f64::INFINITY);
            if x.is_empty() {
                x
            } else {
                increasing(x, f64::sqrt)
            }
        }
        F::Cbrt => increasing(first(1)?, f64::cbrt),
        F::Exp => increasing(first(1)?, f64::exp),
        F::Ln | F::Log => {
            let x = clip(first(1)?, 0.0, f64::INFINITY);
            if x.is_empty() {
                x
            } else {
                increasing(x, f64::ln)
            }
        }
        F::Log10 => {
            let x = clip(first(1)?, 0.0, f64::INFINITY);
            if x.is_empty() {
                x
            } else {
                increasing(x, f64::log10)
            }
        }
        F::Log2 => {
            let x = clip(first(1)?, 0.0, f64::INFINITY);
            if x.is_empty() {
                x
            } else {
                increasing(x, f64::log2)
            }
        }
        F::Sin => periodic_image(first(1)?, f64::sin, FRAC_PI_2, -FRAC_PI_2),
        F::Cos => periodic_image(first(1)?, f64::cos, 0.0, PI),
        F::Tan => {
            let x = first(1)?;
            // tan blows up at π/2 + kπ. Crossing a pole (or losing precision at
            // huge arguments) means the image is unbounded on both sides.
            if !x.lo.is_finite()
                || !x.hi.is_finite()
                || x.width() >= PI
                || x.lo.abs() > ARITHMA_INTERVAL_PERIODIC_LIMIT
                || x.hi.abs() > ARITHMA_INTERVAL_PERIODIC_LIMIT
                || contains_congruent(x, FRAC_PI_2)
                || contains_congruent(x, -FRAC_PI_2)
            {
                ArithmaInterval::whole()
            } else {
                increasing(x, f64::tan)
            }
        }
        F::Asin => {
            let x = clip(first(1)?, -1.0, 1.0);
            if x.is_empty() {
                x
            } else {
                increasing(x, f64::asin)
            }
        }
        F::Acos => {
            let x = clip(first(1)?, -1.0, 1.0);
            if x.is_empty() {
                x
            } else {
                decreasing(x, f64::acos)
            }
        }
        F::Atan => increasing(first(1)?, f64::atan),
        F::Atan2 => {
            // atan2 is not monotone across the branch cut; a tight enclosure
            // needs quadrant analysis we do not attempt.
            let (y, x) = binary()?;
            if x.lo > 0.0 {
                // Right half-plane: continuous and monotone in each argument.
                ArithmaInterval::hull(&[
                    y.lo.atan2(x.lo),
                    y.lo.atan2(x.hi),
                    y.hi.atan2(x.lo),
                    y.hi.atan2(x.hi),
                ])
            } else {
                ArithmaInterval::new(-PI, PI)
            }
        }
        F::Sinh => increasing(first(1)?, f64::sinh),
        F::Tanh => increasing(first(1)?, f64::tanh),
        F::Cosh => {
            let x = first(1)?;
            let outer = x.lo.cosh().max(x.hi.cosh());
            if x.contains(0.0) {
                ArithmaInterval::new(1.0, outer)
            } else {
                ArithmaInterval::new(x.lo.cosh().min(x.hi.cosh()), outer)
            }
        }
        F::Asinh => increasing(first(1)?, f64::asinh),
        F::Acosh => {
            let x = clip(first(1)?, 1.0, f64::INFINITY);
            if x.is_empty() {
                x
            } else {
                increasing(x, f64::acosh)
            }
        }
        F::Atanh => {
            let x = clip(first(1)?, -1.0, 1.0);
            if x.is_empty() {
                x
            } else {
                increasing(x, f64::atanh)
            }
        }
        F::Abs => {
            let x = first(1)?;
            let outer = x.lo.abs().max(x.hi.abs());
            if x.contains(0.0) {
                ArithmaInterval::new(0.0, outer)
            } else {
                ArithmaInterval::new(x.lo.abs().min(x.hi.abs()), outer)
            }
        }
        F::Sign => {
            let x = first(1)?;
            // `f64::signum` maps ±0.0 to ±1.0, so an input touching zero can
            // land on either branch: widen to the full codomain.
            if x.contains(0.0) {
                ArithmaInterval::new(-1.0, 1.0)
            } else if x.hi < 0.0 {
                ArithmaInterval::point(-1.0)
            } else {
                ArithmaInterval::point(1.0)
            }
        }
        F::Floor => increasing(first(1)?, f64::floor),
        F::Ceil => increasing(first(1)?, f64::ceil),
        F::Round => increasing(first(1)?, f64::round),
        other => return Err(format!("evaluate_interval: unsupported function {other:?}")),
    };
    Ok(out)
}

/// Compute the interval-arithmetic image of `expr` when `var` ranges over
/// `interval`.
///
/// The traversal is iterative (explicit stacks, no recursion) and bounded by
/// [`ARITHMA_INTERVAL_NODE_CAP`], matching `ArithmaExpression::evaluate`.
///
/// # Soundness contract
///
/// The returned interval **always contains the true image** of `expr` over the
/// input — the function may over-estimate, never under-estimate. Two systematic
/// sources of over-estimation are worth knowing about:
///
/// 1. **The dependency problem.** Each variable occurrence is bounded
///    independently, so `x - x` over `[0, 1]` yields `[-1, 1]` rather than
///    `[0, 0]`. This is inherent to naive interval arithmetic and is why the
///    result is a guaranteed enclosure, not the exact range.
/// 2. **Deliberate widening.** Where a tight enclosure would need machinery
///    this module does not carry, it returns the conservative hull instead:
///    - division by an interval straddling zero → [`ArithmaInterval::whole`]
///      (the true image is a union of two unbounded rays);
///    - `tan` over an interval containing a pole, or wider than `π` →
///      [`ArithmaInterval::whole`];
///    - `sin` / `cos` over an interval wider than `2π`, unbounded, or with an
///      endpoint above `1e12` in magnitude (where argument reduction loses the
///      phase) → `[-1, 1]`;
///    - `x^y` with a non-degenerate exponent over a base that can be
///      non-positive → [`ArithmaInterval::whole`];
///    - `sign` over an interval touching zero → `[-1, 1]`;
///    - `atan2` outside the right half-plane → `[-π, π]`.
///
/// Non-monotonic functions that *are* handled tightly account for their
/// interior extrema explicitly: `sin`/`cos` test whether each extremum phase is
/// congruent into the input, and `x^{2k}`, `abs` and `cosh` special-case an
/// input straddling zero so the lower bound comes from the interior minimum
/// rather than from an endpoint.
///
/// # Domain clipping
///
/// Partial functions (`sqrt`, `ln`/`log`/`log10`/`log2`, `asin`, `acos`,
/// `acosh`, `atanh`, fractional powers) are evaluated over the intersection of
/// the input with their real domain. `sqrt([-1, 4])` is therefore `[0, 2]`, and
/// an input entirely outside the domain yields [`ArithmaInterval::empty`]. No
/// real value is produced there, so nothing is lost from the image.
///
/// # Errors
///
/// Returns `Err` for a free variable other than `var`, a constant with no
/// cached value, an operator this module does not model, or a node cap
/// overrun — never a silently wrong bound.
pub fn evaluate_interval(
    expr: &ArithmaExpression,
    var: &str,
    interval: ArithmaInterval,
) -> Result<ArithmaInterval, String> {
    enum Frame<'a> {
        Enter(&'a ArithmaExpression),
        CombineFunc(&'a ArithmaFunction, usize),
        CombineCond,
    }

    let mut work: Vec<Frame> = Vec::with_capacity(32);
    let mut values: Vec<ArithmaInterval> = Vec::with_capacity(32);
    work.push(Frame::Enter(expr));
    let mut guard: usize = 0;

    while let Some(frame) = work.pop() {
        guard += 1;
        if guard > ARITHMA_INTERVAL_NODE_CAP {
            return Err("evaluate_interval: node cap exceeded".into());
        }
        match frame {
            Frame::Enter(node) => match node {
                ArithmaExpression::Number(n) => values.push(ArithmaInterval::point(n.to_f64())),
                ArithmaExpression::Constant {
                    cached_value,
                    symbol,
                    ..
                } => match *cached_value {
                    Some(v) => values.push(ArithmaInterval::point(v)),
                    None => {
                        return Err(format!(
                            "evaluate_interval: constant '{symbol}' has no cached value"
                        ))
                    }
                },
                ArithmaExpression::Variable(name) => {
                    if name == var {
                        values.push(interval);
                    } else {
                        return Err(format!("evaluate_interval: unbound variable '{name}'"));
                    }
                }
                ArithmaExpression::Function(func, args) => {
                    work.push(Frame::CombineFunc(func, args.len()));
                    for a in args.iter().rev() {
                        work.push(Frame::Enter(a));
                    }
                }
                ArithmaExpression::Conditional {
                    condition,
                    then_expr,
                    else_expr,
                } => {
                    work.push(Frame::CombineCond);
                    work.push(Frame::Enter(else_expr));
                    work.push(Frame::Enter(then_expr));
                    work.push(Frame::Enter(condition));
                }
                ArithmaExpression::CachedValue { expr, .. } => work.push(Frame::Enter(expr)),
                ArithmaExpression::FourierOptimized { expr, .. } => work.push(Frame::Enter(expr)),
                _ => return Err("evaluate_interval: unsupported expression variant".into()),
            },
            Frame::CombineFunc(func, n) => {
                if values.len() < n {
                    return Err("evaluate_interval: value stack underflow".into());
                }
                let start = values.len() - n;
                let arg_intervals: Vec<ArithmaInterval> = values.drain(start..).collect();
                values.push(apply_interval_function(func, &arg_intervals)?);
            }
            Frame::CombineCond => {
                if values.len() < 3 {
                    return Err("evaluate_interval: conditional underflow".into());
                }
                let else_iv = values.pop().unwrap();
                let then_iv = values.pop().unwrap();
                let cond_iv = values.pop().unwrap();
                // A condition interval pinned at exactly zero always takes the
                // else branch; one that excludes zero always takes the then
                // branch. Anything else could go either way, so union them.
                let out = if cond_iv.is_point() && cond_iv.lo == 0.0 {
                    else_iv
                } else if !cond_iv.contains(0.0) {
                    then_iv
                } else {
                    then_iv.union(else_iv)
                };
                values.push(out);
            }
        }
    }

    if values.len() != 1 {
        return Err(format!(
            "evaluate_interval: final stack size {} != 1",
            values.len()
        ));
    }
    Ok(values[0])
}

// ---------------------------------------------------------------------------
// Backward-compatibility aliases for the pre-rename `Arithmos*` names.
// Retained for one release; downstream (eml-math, eml-spectral, metaphysica,
// periodica) should migrate to the `Arithma*` names above.
// ---------------------------------------------------------------------------
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaInterval`")]
#[allow(unused)]
pub use self::ArithmaInterval as ArithmosInterval;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::{ArithmaBindings, Evaluable};

    fn x() -> ArithmaExpression {
        ArithmaExpression::var("x")
    }

    fn image(expr: &ArithmaExpression, lo: f64, hi: f64) -> ArithmaInterval {
        evaluate_interval(expr, "x", ArithmaInterval::new(lo, hi)).unwrap()
    }

    /// Densely sample `expr` over `[lo, hi]` and assert every real value it
    /// takes lies inside `iv`. This is the soundness property that matters:
    /// the enclosure may be loose, but it must never miss a value.
    fn assert_encloses(expr: &ArithmaExpression, lo: f64, hi: f64, iv: ArithmaInterval) {
        const SAMPLES: usize = 2001;
        let mut bindings = ArithmaBindings::new();
        for k in 0..SAMPLES {
            let t = lo + (hi - lo) * k as f64 / (SAMPLES - 1) as f64;
            bindings.insert("x".to_string(), t);
            if let Ok(v) = expr.evaluate(&bindings) {
                if v.is_finite() {
                    assert!(
                        iv.contains(v),
                        "f({t}) = {v} escapes the enclosure [{}, {}]",
                        iv.lo,
                        iv.hi
                    );
                }
            }
        }
    }

    #[test]
    fn unit_interval_contains_zero_five() {
        let i = ArithmaInterval::new(0.0, 1.0);
        assert!(i.contains(0.5));
    }

    #[test]
    fn empty_interval_contains_nothing() {
        let i = ArithmaInterval::empty();
        assert!(i.is_empty());
    }

    // ---- monotonic cases must be tight -----------------------------------

    #[test]
    fn affine_map_is_exact() {
        // 2x + 1 over [0, 1] → exactly [1, 3].
        let expr = ArithmaExpression::add(
            ArithmaExpression::mul(ArithmaExpression::from_i64(2), x()),
            ArithmaExpression::from_i64(1),
        );
        let iv = image(&expr, 0.0, 1.0);
        assert!((iv.lo - 1.0).abs() < 1e-12, "lo = {}", iv.lo);
        assert!((iv.hi - 3.0).abs() < 1e-12, "hi = {}", iv.hi);
    }

    #[test]
    fn exp_is_tight_on_a_monotonic_range() {
        let iv = image(&ArithmaExpression::exp(x()), 0.0, 1.0);
        assert!((iv.lo - 1.0).abs() < 1e-12, "lo = {}", iv.lo);
        assert!(
            (iv.hi - std::f64::consts::E).abs() < 1e-12,
            "hi = {}",
            iv.hi
        );
    }

    #[test]
    fn sin_on_a_monotonic_quarter_period_is_tight() {
        let iv = image(&ArithmaExpression::sin(x()), 0.0, FRAC_PI_2);
        assert!(iv.lo.abs() < 1e-12, "lo = {}", iv.lo);
        assert!((iv.hi - 1.0).abs() < 1e-12, "hi = {}", iv.hi);
    }

    #[test]
    fn subtraction_reverses_the_second_operand() {
        // 5 - x over [1, 2] → [3, 4].
        let expr = ArithmaExpression::sub(ArithmaExpression::from_i64(5), x());
        let iv = image(&expr, 1.0, 2.0);
        assert!((iv.lo - 3.0).abs() < 1e-12);
        assert!((iv.hi - 4.0).abs() < 1e-12);
    }

    // ---- non-monotonic: interior extrema must be picked up ---------------

    #[test]
    fn sin_over_a_full_period_covers_the_whole_unit_range() {
        let iv = image(&ArithmaExpression::sin(x()), 0.0, TAU);
        assert!(iv.contains(-1.0) && iv.contains(1.0), "{iv:?}");
        // …and does not over-claim beyond the codomain.
        assert!(
            (iv.lo + 1.0).abs() < 1e-12 && (iv.hi - 1.0).abs() < 1e-12,
            "{iv:?}"
        );
    }

    #[test]
    fn cos_over_a_full_period_covers_the_whole_unit_range() {
        let iv = image(&ArithmaExpression::cos(x()), -PI, PI);
        assert!(iv.contains(-1.0) && iv.contains(1.0), "{iv:?}");
        assert!(
            (iv.lo + 1.0).abs() < 1e-12 && (iv.hi - 1.0).abs() < 1e-12,
            "{iv:?}"
        );
    }

    #[test]
    fn sin_picks_up_only_the_extrema_it_actually_spans() {
        // [π/4, 3π/4] contains the maximum at π/2 but no minimum.
        let iv = image(&ArithmaExpression::sin(x()), PI / 4.0, 3.0 * PI / 4.0);
        assert!((iv.hi - 1.0).abs() < 1e-12, "hi = {}", iv.hi);
        let sqrt_half = std::f64::consts::FRAC_1_SQRT_2;
        assert!((iv.lo - sqrt_half).abs() < 1e-12, "lo = {}", iv.lo);
        assert_encloses(&ArithmaExpression::sin(x()), PI / 4.0, 3.0 * PI / 4.0, iv);
    }

    #[test]
    fn even_power_straddling_zero_bottoms_out_at_zero() {
        // x² over [-2, 1] → [0, 4], not [1, 4].
        let expr = ArithmaExpression::pow(x(), ArithmaExpression::from_i64(2));
        let iv = image(&expr, -2.0, 1.0);
        assert!(iv.lo.abs() < 1e-12, "lo = {}", iv.lo);
        assert!((iv.hi - 4.0).abs() < 1e-12, "hi = {}", iv.hi);
        assert_encloses(&expr, -2.0, 1.0, iv);
    }

    #[test]
    fn even_power_away_from_zero_keeps_both_endpoints() {
        let expr = ArithmaExpression::pow(x(), ArithmaExpression::from_i64(2));
        let iv = image(&expr, 2.0, 3.0);
        assert!((iv.lo - 4.0).abs() < 1e-12, "lo = {}", iv.lo);
        assert!((iv.hi - 9.0).abs() < 1e-12, "hi = {}", iv.hi);
    }

    #[test]
    fn odd_power_stays_monotonic() {
        let expr = ArithmaExpression::pow(x(), ArithmaExpression::from_i64(3));
        let iv = image(&expr, -2.0, 1.0);
        assert!((iv.lo + 8.0).abs() < 1e-12, "lo = {}", iv.lo);
        assert!((iv.hi - 1.0).abs() < 1e-12, "hi = {}", iv.hi);
    }

    #[test]
    fn abs_and_cosh_bottom_out_at_their_interior_minimum() {
        let abs_expr = ArithmaExpression::func(ArithmaFunction::Abs, vec![x()]);
        let iv = image(&abs_expr, -3.0, 1.0);
        assert!(iv.lo.abs() < 1e-12 && (iv.hi - 3.0).abs() < 1e-12, "{iv:?}");

        let cosh_expr = ArithmaExpression::func(ArithmaFunction::Cosh, vec![x()]);
        let iv = image(&cosh_expr, -1.0, 0.5);
        assert!((iv.lo - 1.0).abs() < 1e-12, "lo = {}", iv.lo);
        assert!((iv.hi - 1.0_f64.cosh()).abs() < 1e-12, "hi = {}", iv.hi);
    }

    // ---- conservative widening -------------------------------------------

    #[test]
    fn division_by_an_interval_straddling_zero_widens_to_the_whole_line() {
        let expr = ArithmaExpression::div(ArithmaExpression::from_i64(1), x());
        let iv = image(&expr, -1.0, 1.0);
        assert_eq!(iv.lo, f64::NEG_INFINITY);
        assert_eq!(iv.hi, f64::INFINITY);
    }

    #[test]
    fn division_away_from_zero_stays_tight() {
        let expr = ArithmaExpression::div(ArithmaExpression::from_i64(1), x());
        let iv = image(&expr, 2.0, 4.0);
        assert!((iv.lo - 0.25).abs() < 1e-12, "lo = {}", iv.lo);
        assert!((iv.hi - 0.5).abs() < 1e-12, "hi = {}", iv.hi);
    }

    #[test]
    fn tan_across_a_pole_widens_but_stays_tight_between_poles() {
        let tan_expr = ArithmaExpression::tan(x());
        let across = image(&tan_expr, 1.0, 2.0);
        assert_eq!(across.lo, f64::NEG_INFINITY);
        assert_eq!(across.hi, f64::INFINITY);

        let between = image(&tan_expr, -0.5, 0.5);
        assert!((between.lo + 0.5_f64.tan()).abs() < 1e-12, "{between:?}");
        assert!((between.hi - 0.5_f64.tan()).abs() < 1e-12, "{between:?}");
    }

    #[test]
    fn huge_arguments_widen_the_periodic_functions() {
        let iv = image(&ArithmaExpression::sin(x()), 1e15, 1e15 + 0.1);
        assert_eq!(iv, ArithmaInterval::new(-1.0, 1.0));
    }

    #[test]
    fn dependency_problem_is_a_widening_not_a_wrong_answer() {
        // x - x is identically 0, but naive interval arithmetic gives [-1, 1].
        let expr = ArithmaExpression::sub(x(), x());
        let iv = image(&expr, 0.0, 1.0);
        assert!(iv.contains(0.0));
        assert!(iv.width() >= 2.0 - 1e-12, "expected widening, got {iv:?}");
    }

    // ---- domain clipping --------------------------------------------------

    #[test]
    fn sqrt_clips_to_its_real_domain() {
        let iv = image(&ArithmaExpression::sqrt(x()), -1.0, 4.0);
        assert!(iv.lo.abs() < 1e-12, "lo = {}", iv.lo);
        assert!((iv.hi - 2.0).abs() < 1e-12, "hi = {}", iv.hi);
    }

    #[test]
    fn sqrt_entirely_below_zero_is_empty() {
        let iv = image(&ArithmaExpression::sqrt(x()), -4.0, -1.0);
        assert!(iv.is_empty(), "{iv:?}");
    }

    #[test]
    fn ln_clips_at_zero_and_reports_an_unbounded_lower_end() {
        let iv = image(&ArithmaExpression::ln(x()), -1.0, std::f64::consts::E);
        assert_eq!(iv.lo, f64::NEG_INFINITY);
        assert!((iv.hi - 1.0).abs() < 1e-12, "hi = {}", iv.hi);
    }

    #[test]
    fn asin_clips_to_minus_one_plus_one() {
        let iv = image(
            &ArithmaExpression::func(ArithmaFunction::Asin, vec![x()]),
            -5.0,
            5.0,
        );
        assert!((iv.lo + FRAC_PI_2).abs() < 1e-12, "lo = {}", iv.lo);
        assert!((iv.hi - FRAC_PI_2).abs() < 1e-12, "hi = {}", iv.hi);
    }

    // ---- composite soundness ---------------------------------------------

    #[test]
    fn composite_expression_encloses_every_sampled_value() {
        // x·sin(x) + x² over [-3, 3] — non-monotonic, mixed operators.
        let expr = ArithmaExpression::add(
            ArithmaExpression::mul(x(), ArithmaExpression::sin(x())),
            ArithmaExpression::pow(x(), ArithmaExpression::from_i64(2)),
        );
        let iv = image(&expr, -3.0, 3.0);
        assert!(
            !iv.is_empty() && iv.lo.is_finite() && iv.hi.is_finite(),
            "{iv:?}"
        );
        assert_encloses(&expr, -3.0, 3.0, iv);
    }

    #[test]
    fn nested_transcendentals_enclose_every_sampled_value() {
        // exp(cos(x)) / (2 + sin(x)) over [0, 2π].
        let expr = ArithmaExpression::div(
            ArithmaExpression::exp(ArithmaExpression::cos(x())),
            ArithmaExpression::add(ArithmaExpression::from_i64(2), ArithmaExpression::sin(x())),
        );
        let iv = image(&expr, 0.0, TAU);
        assert!(iv.lo.is_finite() && iv.hi.is_finite(), "{iv:?}");
        assert_encloses(&expr, 0.0, TAU, iv);
    }

    // ---- errors -----------------------------------------------------------

    #[test]
    fn a_foreign_variable_is_an_error() {
        let err = evaluate_interval(
            &ArithmaExpression::var("y"),
            "x",
            ArithmaInterval::new(0.0, 1.0),
        )
        .unwrap_err();
        assert!(err.contains('y'), "unexpected error: {err}");
    }

    #[test]
    fn an_unmodelled_operator_is_an_error_not_a_silent_bound() {
        let expr = ArithmaExpression::func(ArithmaFunction::Gamma, vec![x()]);
        assert!(evaluate_interval(&expr, "x", ArithmaInterval::new(1.0, 2.0)).is_err());
    }
}
