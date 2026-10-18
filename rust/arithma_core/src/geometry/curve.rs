//====== Arithma/rust/arithma_core/src/geometry/curve.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Curve
//!
//! N-dimensional symbolic curves: closed-form Bézier (Bernstein basis),
//! Catmull–Rom segments (uniform, centripetal and chordal, via the
//! Barry–Goldman construction) and piecewise Catmull–Rom splines through a
//! point sequence. Every curve is built as an [`ArithmaExpression`] tree, so
//! it stays symbolic until evaluated.

use crate::expression::ArithmaExpression;
use crate::function::ArithmaFunction;

/// Largest number of Bézier control points [`bezier`] will accept.
pub const ARITHMA_BEZIER_MAX_CONTROL_POINTS: usize = 32;

/// Why a curve could not be built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArithmaCurveError {
    /// No control/through points were supplied.
    EmptyControlPoints,
    /// More control points than [`ARITHMA_BEZIER_MAX_CONTROL_POINTS`] were supplied.
    TooManyControlPoints { got: usize, limit: usize },
    /// Two points did not carry the same number of axes.
    DimensionMismatch { expected: usize, got: usize },
    /// A piecewise curve needs at least two through-points.
    TooFewPoints { got: usize },
}

impl core::fmt::Display for ArithmaCurveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyControlPoints => write!(f, "no control points were supplied"),
            Self::TooManyControlPoints { got, limit } => {
                write!(f, "{got} control points exceeds the limit of {limit}")
            }
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected} axes, got {got}")
            }
            Self::TooFewPoints { got } => {
                write!(f, "a piecewise curve needs at least 2 points, got {got}")
            }
        }
    }
}

impl std::error::Error for ArithmaCurveError {}

/// Which knot-spacing convention a Catmull–Rom segment uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CatmullRomKind {
    /// Knots spaced by a constant step of 1, regardless of control-point distance.
    Uniform,
    /// Knots spaced by the square root of the Euclidean distance between controls.
    Centripetal,
    /// Knots spaced by the Euclidean distance between controls.
    Chordal,
}

/// Checks that every point has the same, nonzero number of axes and returns it.
fn validate_points<'a>(
    points: impl IntoIterator<Item = &'a [ArithmaExpression]>,
) -> Result<usize, ArithmaCurveError> {
    let mut dim: Option<usize> = None;
    for p in points {
        match dim {
            None => dim = Some(p.len()),
            Some(expected) if expected != p.len() => {
                return Err(ArithmaCurveError::DimensionMismatch {
                    expected,
                    got: p.len(),
                });
            }
            Some(_) => {}
        }
    }
    match dim {
        Some(0) | None => Err(ArithmaCurveError::EmptyControlPoints),
        Some(d) => Ok(d),
    }
}

/// Exact binomial coefficient `C(n, k)` via the multiplicative formula.
fn binomial(n: u32, k: u32) -> i64 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut result: i64 = 1;
    for i in 0..k {
        result = result * (n - i) as i64 / (i + 1) as i64;
    }
    result
}

/// Closed-form N-dimensional Bézier curve from `control` points as a Bernstein
/// polynomial per axis; `degree = control.len() - 1`.
pub fn bezier(
    control: &[Vec<ArithmaExpression>],
    parameter: &str,
) -> Result<Vec<ArithmaExpression>, ArithmaCurveError> {
    if control.len() > ARITHMA_BEZIER_MAX_CONTROL_POINTS {
        return Err(ArithmaCurveError::TooManyControlPoints {
            got: control.len(),
            limit: ARITHMA_BEZIER_MAX_CONTROL_POINTS,
        });
    }
    let dim = validate_points(control.iter().map(|v| v.as_slice()))?;
    let degree = control.len() - 1;
    let t = ArithmaExpression::var(parameter);
    let one_minus_t = ArithmaExpression::sub(ArithmaExpression::from_i64(1), t.clone());
    let mut axes = vec![ArithmaExpression::zero(); dim];
    for (i, point) in control.iter().enumerate() {
        let coeff = ArithmaExpression::from_i64(binomial(degree as u32, i as u32));
        let t_pow = ArithmaExpression::pow(t.clone(), ArithmaExpression::from_i64(i as i64));
        let u_pow = ArithmaExpression::pow(
            one_minus_t.clone(),
            ArithmaExpression::from_i64((degree - i) as i64),
        );
        let basis = ArithmaExpression::mul(ArithmaExpression::mul(coeff, t_pow), u_pow);
        for (axis, coord) in axes.iter_mut().zip(point.iter()) {
            *axis = ArithmaExpression::add(
                axis.clone(),
                ArithmaExpression::mul(basis.clone(), coord.clone()),
            );
        }
    }
    Ok(axes)
}

/// Euclidean distance between two same-dimension points, built symbolically.
fn euclidean_distance(a: &[ArithmaExpression], b: &[ArithmaExpression]) -> ArithmaExpression {
    let mut sum_sq = ArithmaExpression::zero();
    for (ai, bi) in a.iter().zip(b.iter()) {
        let diff = ArithmaExpression::sub(ai.clone(), bi.clone());
        sum_sq = ArithmaExpression::add(sum_sq, ArithmaExpression::mul(diff.clone(), diff));
    }
    ArithmaExpression::sqrt(sum_sq)
}

/// Knot spacing between two consecutive controls for a given [`CatmullRomKind`].
fn knot_step(a: &[ArithmaExpression], b: &[ArithmaExpression], kind: CatmullRomKind) -> ArithmaExpression {
    match kind {
        CatmullRomKind::Uniform => ArithmaExpression::from_i64(1),
        CatmullRomKind::Centripetal => ArithmaExpression::sqrt(euclidean_distance(a, b)),
        CatmullRomKind::Chordal => euclidean_distance(a, b),
    }
}

/// Linear blend of `p_lo`/`p_hi` at `t` between knots `t_lo`/`t_hi` (one step of
/// the Barry–Goldman recursion), elementwise across axes.
fn blend(
    t: &ArithmaExpression,
    t_lo: &ArithmaExpression,
    t_hi: &ArithmaExpression,
    p_lo: &[ArithmaExpression],
    p_hi: &[ArithmaExpression],
) -> Vec<ArithmaExpression> {
    let denom = ArithmaExpression::sub(t_hi.clone(), t_lo.clone());
    let w_hi = ArithmaExpression::sub(t.clone(), t_lo.clone());
    let w_lo = ArithmaExpression::sub(t_hi.clone(), t.clone());
    p_lo
        .iter()
        .zip(p_hi.iter())
        .map(|(lo, hi)| {
            let term_lo = ArithmaExpression::mul(w_lo.clone(), lo.clone());
            let term_hi = ArithmaExpression::mul(w_hi.clone(), hi.clone());
            ArithmaExpression::div(ArithmaExpression::add(term_lo, term_hi), denom.clone())
        })
        .collect()
}

/// One Catmull–Rom segment from `p1` (`parameter` = 0) to `p2` (`parameter` =
/// 1), built via the Barry–Goldman recursion so it covers uniform, centripetal
/// and chordal spacing with one formula.
pub fn catmull_rom(
    p0: &[ArithmaExpression],
    p1: &[ArithmaExpression],
    p2: &[ArithmaExpression],
    p3: &[ArithmaExpression],
    kind: CatmullRomKind,
    parameter: &str,
) -> Result<Vec<ArithmaExpression>, ArithmaCurveError> {
    validate_points([p0, p1, p2, p3])?;
    let t = ArithmaExpression::var(parameter);
    let d01 = knot_step(p0, p1, kind);
    let d12 = knot_step(p1, p2, kind);
    let d23 = knot_step(p2, p3, kind);
    let t0 = ArithmaExpression::zero();
    let t1 = d01;
    let t2 = ArithmaExpression::add(t1.clone(), d12.clone());
    let t3 = ArithmaExpression::add(t2.clone(), d23);
    let t_global = ArithmaExpression::add(t1.clone(), ArithmaExpression::mul(t, d12));
    let a1 = blend(&t_global, &t0, &t1, p0, p1);
    let a2 = blend(&t_global, &t1, &t2, p1, p2);
    let a3 = blend(&t_global, &t2, &t3, p2, p3);
    let b1 = blend(&t_global, &t0, &t2, &a1, &a2);
    let b2 = blend(&t_global, &t1, &t3, &a2, &a3);
    Ok(blend(&t_global, &t1, &t2, &b1, &b2))
}

/// Reflects `near` through itself away from `far`: `2 * near - far`. Used to
/// synthesize the virtual control point before the first, or after the last,
/// through-point of a piecewise Catmull–Rom spline.
fn reflect(far: &[ArithmaExpression], near: &[ArithmaExpression]) -> Vec<ArithmaExpression> {
    near.iter()
        .zip(far.iter())
        .map(|(n, f)| {
            ArithmaExpression::sub(
                ArithmaExpression::mul(ArithmaExpression::from_i64(2), n.clone()),
                f.clone(),
            )
        })
        .collect()
}

/// `1` when `x` evaluates to a non-negative number, `0` otherwise. Stands in
/// for a "greater-or-equal" comparison, which Arithma has no dedicated
/// operator for.
fn step_ge_zero(x: ArithmaExpression) -> ArithmaExpression {
    let sign = ArithmaExpression::func(ArithmaFunction::Sign, vec![x]);
    let abs_sign = ArithmaExpression::func(ArithmaFunction::Abs, vec![sign.clone()]);
    ArithmaExpression::div(
        ArithmaExpression::add(sign, abs_sign),
        ArithmaExpression::from_i64(2),
    )
}

/// Replaces every occurrence of the variable named `var` in `expr` with
/// `var - by`, so a segment built for a local parameter in `[0, 1]` can be
/// embedded at offset `by` along a shared global parameter.
fn shift_variable(expr: &ArithmaExpression, var: &str, by: i64) -> ArithmaExpression {
    match expr {
        ArithmaExpression::Variable(name) if name == var => ArithmaExpression::sub(
            ArithmaExpression::var(var),
            ArithmaExpression::from_i64(by),
        ),
        ArithmaExpression::Function(f, args) => ArithmaExpression::func(
            f.clone(),
            args.iter().map(|a| shift_variable(a, var, by)).collect(),
        ),
        other => other.clone(),
    }
}

/// A curve stitched together from consecutive segments, each valid over one
/// unit of a shared global parameter.
#[derive(Debug, Clone)]
pub struct PiecewiseCurve {
    /// Segment `k`'s per-axis expression, written in a local parameter over `[0, 1]`.
    pub segments: Vec<Vec<ArithmaExpression>>,
}

impl PiecewiseCurve {
    /// Builds a piecewise Catmull–Rom spline through every point in `points`,
    /// reflecting the second/penultimate point to synthesize the end tangents.
    pub fn catmull_rom_through(
        points: &[Vec<ArithmaExpression>],
        kind: CatmullRomKind,
        parameter: &str,
    ) -> Result<Self, ArithmaCurveError> {
        if points.len() < 2 {
            return Err(ArithmaCurveError::TooFewPoints { got: points.len() });
        }
        validate_points(points.iter().map(|v| v.as_slice()))?;
        let n = points.len();
        let mut segments = Vec::with_capacity(n - 1);
        for i in 0..n - 1 {
            let p0 = if i == 0 {
                reflect(&points[1], &points[0])
            } else {
                points[i - 1].clone()
            };
            let p3 = if i + 2 >= n {
                reflect(&points[n - 2], &points[n - 1])
            } else {
                points[i + 2].clone()
            };
            segments.push(catmull_rom(&p0, &points[i], &points[i + 1], &p3, kind, parameter)?);
        }
        Ok(Self { segments })
    }

    /// The whole curve as one expression per axis, selecting segment `k` for
    /// `parameter` in `[k, k + 1)`, with the last segment closed at
    /// `parameter` = segment count.
    pub fn expression(&self, parameter: &str) -> Result<Vec<ArithmaExpression>, ArithmaCurveError> {
        if self.segments.is_empty() {
            return Err(ArithmaCurveError::EmptyControlPoints);
        }
        let dim = validate_points(self.segments.iter().map(|v| v.as_slice()))?;
        let seg_count = self.segments.len();
        Ok((0..dim)
            .map(|axis| self.axis_expression(axis, seg_count, parameter))
            .collect())
    }

    /// Builds the nested `Conditional` for one axis across all segments.
    fn axis_expression(&self, axis: usize, seg_count: usize, parameter: &str) -> ArithmaExpression {
        let mut expr = shift_variable(
            &self.segments[seg_count - 1][axis],
            parameter,
            (seg_count - 1) as i64,
        );
        for k in (0..seg_count - 1).rev() {
            let seg_expr = shift_variable(&self.segments[k][axis], parameter, k as i64);
            let boundary = ArithmaExpression::from_i64((k + 1) as i64);
            let condition = step_ge_zero(ArithmaExpression::sub(
                boundary,
                ArithmaExpression::var(parameter),
            ));
            expr = ArithmaExpression::Conditional {
                condition: Box::new(condition),
                then_expr: Box::new(seg_expr),
                else_expr: Box::new(expr),
            };
        }
        expr
    }
}
