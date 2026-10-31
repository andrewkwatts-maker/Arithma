//! # NURBS
//!
//! Non-uniform rational B-spline curves in 3D. Every span has a closed-form
//! symbolic expression ([`NurbsCurve::span_expression`], Cox-de Boor over the
//! span's own knots) so derivatives and lowering stay inside Arithma. The
//! numeric [`NurbsCurve::evaluate`] is de Boor's algorithm, the fast path that
//! touches only the `degree + 1` points that can contribute; it agrees with the
//! symbolic form to round-off (tested). [`NurbsCurve::tangent`] differentiates
//! the symbolic span expression, so it is exact rather than a finite difference.
//!
//! Analytic stays analytic: [`NurbsCurve::circular_arc`] is a rational
//! quadratic with weight `cos(half)`, an exact circle.

use crate::expression::{ArithmaBindings, ArithmaExpression, Differentiable, Evaluable};

/// Why a NURBS curve could not be built or used.
#[derive(Debug, Clone, PartialEq)]
pub enum ArithmaNurbsError {
    /// Control points, weights or knots do not describe a curve.
    InvalidCurve(String),
    /// Tessellation without a finite positive tolerance.
    InvalidTolerance(f64),
    /// Arithma failed to differentiate or evaluate a span expression.
    Expression(String),
}

impl std::fmt::Display for ArithmaNurbsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidCurve(m) => write!(f, "invalid curve: {m}"),
            Self::InvalidTolerance(t) => write!(
                f,
                "tessellation needs a finite positive tolerance, got {t}; this is the one place a curve stops being exact, so the tolerance is stated rather than defaulted"
            ),
            Self::Expression(m) => write!(f, "nurbs expression: {m}"),
        }
    }
}

impl std::error::Error for ArithmaNurbsError {}

/// A NURBS result.
pub type NurbsResult<T> = Result<T, ArithmaNurbsError>;

/// Homogeneous point `(w*x, w*y, w*z, w)`.
type Hom = [f64; 4];

/// A non-uniform rational B-spline curve in 3D.
#[derive(Clone, Debug, PartialEq)]
pub struct NurbsCurve {
    degree: usize,
    points: Vec<Hom>,
    knots: Vec<f64>,
}

fn invalid<T>(message: String) -> NurbsResult<T> {
    Err(ArithmaNurbsError::InvalidCurve(message))
}

fn expr_error(reason: String) -> ArithmaNurbsError {
    ArithmaNurbsError::Expression(reason)
}

fn check_weights(weights: &[f64]) -> NurbsResult<()> {
    match weights.iter().position(|w| !(w.is_finite() && *w > 0.0)) {
        Some(i) => invalid(format!(
            "weight {i} is {}; weights must be finite and positive or the rational divide has a pole",
            weights[i]
        )),
        None => Ok(()),
    }
}

fn check_shape(degree: usize, points: usize, weights: usize, knots: &[f64]) -> NurbsResult<()> {
    if degree == 0 {
        return invalid("degree 0 is a set of disconnected points, not a curve".into());
    }
    if points != weights {
        return invalid(format!("{points} control points but {weights} weights"));
    }
    if points <= degree {
        return invalid(format!(
            "a degree-{degree} curve needs at least {} control points, got {points}",
            degree + 1
        ));
    }
    let expected = points + degree + 1;
    if knots.len() != expected {
        return invalid(format!(
            "a degree-{degree} curve over {points} points needs {expected} knots, got {}",
            knots.len()
        ));
    }
    if knots.windows(2).any(|w| w[1] < w[0]) {
        return invalid("the knot vector must be non-decreasing".into());
    }
    Ok(())
}

fn lit(v: f64) -> ArithmaExpression {
    ArithmaExpression::from_f64(v)
}

impl NurbsCurve {
    /// Build a curve from Cartesian points, weights and a knot vector.
    ///
    /// # Errors
    ///
    /// [`ArithmaNurbsError::InvalidCurve`] if degree, counts, knot order or
    /// weights are unusable.
    pub fn new(degree: usize, points: Vec<[f64; 3]>, weights: Vec<f64>, knots: Vec<f64>) -> NurbsResult<Self> {
        check_shape(degree, points.len(), weights.len(), &knots)?;
        check_weights(&weights)?;
        let points = points
            .iter()
            .zip(&weights)
            .map(|(p, w)| [p[0] * w, p[1] * w, p[2] * w, *w])
            .collect();
        Ok(Self { degree, points, knots })
    }

    /// A curve with evenly spaced interior knots, clamped at both ends.
    ///
    /// # Errors
    ///
    /// As [`NurbsCurve::new`].
    pub fn clamped(degree: usize, points: Vec<[f64; 3]>, weights: Vec<f64>) -> NurbsResult<Self> {
        let n = points.len();
        if n <= degree {
            return invalid(format!(
                "a degree-{degree} curve needs at least {} control points, got {n}",
                degree + 1
            ));
        }
        let interior = n - degree - 1;
        let mut knots = vec![0.0; degree + 1];
        knots.extend((1..=interior).map(|i| i as f64 / (interior + 1) as f64));
        knots.extend(std::iter::repeat_n(1.0, degree + 1));
        Self::new(degree, points, weights, knots)
    }

    /// A polyline as a degree-1 curve.
    ///
    /// # Errors
    ///
    /// As [`NurbsCurve::clamped`].
    pub fn polyline(points: Vec<[f64; 3]>) -> NurbsResult<Self> {
        let n = points.len();
        Self::clamped(1, points, vec![1.0; n])
    }

    /// An exact circular arc in the XY plane as a rational quadratic; `sweep`
    /// is in radians and must lie in `(0, pi)`.
    ///
    /// # Errors
    ///
    /// [`ArithmaNurbsError::InvalidCurve`] for a bad radius or sweep.
    pub fn circular_arc(centre: [f64; 3], radius: f64, sweep: f64) -> NurbsResult<Self> {
        if !(radius.is_finite() && radius > 0.0) {
            return invalid(format!("arc radius must be finite and positive, got {radius}"));
        }
        if !(sweep.is_finite() && sweep > 0.0 && sweep < std::f64::consts::PI) {
            return invalid(format!(
                "a single rational quadratic span covers less than half a turn; got a sweep of {sweep} radians"
            ));
        }
        // direct-float: arc construction constants, lowered once at build time
        let half = sweep / 2.0;
        let w = half.cos();
        let at = |x: f64, y: f64| [centre[0] + x, centre[1] + y, centre[2]];
        let start = at(radius, 0.0);
        let mid = at(radius * half.cos() / w.max(1e-12), radius * half.sin() / w.max(1e-12));
        let end = at(radius * sweep.cos(), radius * sweep.sin());
        Self::new(2, vec![start, mid, end], vec![1.0, w, 1.0], vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    }

    /// The curve's degree.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// How many control points.
    pub fn control_count(&self) -> usize {
        self.points.len()
    }

    /// The control points, back in Cartesian coordinates.
    pub fn control_points(&self) -> Vec<[f64; 3]> {
        self.points.iter().map(|p| [p[0] / p[3], p[1] / p[3], p[2] / p[3]]).collect()
    }

    /// The weights.
    pub fn weights(&self) -> Vec<f64> {
        self.points.iter().map(|p| p[3]).collect()
    }

    /// The knot vector.
    pub fn knots(&self) -> &[f64] {
        &self.knots
    }

    /// The parameter range the curve is defined over.
    pub fn domain(&self) -> (f64, f64) {
        (self.knots[self.degree], self.knots[self.points.len()])
    }

    /// The knot span containing `t` (the last non-empty span at the domain end).
    pub fn span(&self, t: f64) -> usize {
        let n = self.points.len() - 1;
        if t >= self.knots[n + 1] {
            return n;
        }
        let (mut lo, mut hi) = (self.degree, n + 1);
        let mut mid = (lo + hi) / 2;
        while t < self.knots[mid] || t >= self.knots[mid + 1] {
            if t < self.knots[mid] {
                hi = mid;
            } else {
                lo = mid;
            }
            mid = (lo + hi) / 2;
        }
        mid
    }

    /// The point at `t`, clamped to the domain, by de Boor's algorithm.
    pub fn evaluate(&self, t: f64) -> [f64; 3] {
        let (lo, hi) = self.domain();
        let t = t.clamp(lo, hi);
        let k = self.span(t);
        let p = self.degree;
        // direct-float: de Boor fast path, agrees with `span_expression`
        let mut d: Vec<Hom> = (0..=p).map(|j| self.points[j + k - p]).collect();
        for r in 1..=p {
            for j in (r..=p).rev() {
                let i = j + k - p;
                let denom = self.knots[i + p + 1 - r] - self.knots[i];
                let alpha = if denom.abs() < f64::EPSILON { 0.0 } else { (t - self.knots[i]) / denom };
                for c in 0..4 {
                    d[j][c] = d[j - 1][c] * (1.0 - alpha) + d[j][c] * alpha;
                }
            }
        }
        let h = d[p];
        [h[0] / h[3], h[1] / h[3], h[2] / h[3]]
    }

    /// One Cox-de Boor term `num / (hi - lo) * basis`; a zero-width knot
    /// interval contributes nothing.
    fn cox_term(basis: &ArithmaExpression, num: ArithmaExpression, width: f64) -> Option<ArithmaExpression> {
        if width.abs() < f64::EPSILON {
            return None;
        }
        Some(ArithmaExpression::mul(ArithmaExpression::div(num, lit(width)), basis.clone()))
    }

    /// Symbolic nonzero basis functions of span `k`, `N_{k-p..=k, p}`, in
    /// index order, as expressions in `parameter`.
    fn span_basis(&self, k: usize, parameter: &str) -> Vec<ArithmaExpression> {
        let t = ArithmaExpression::var(parameter);
        let u = &self.knots;
        let mut cur = vec![lit(1.0)];
        for q in 1..=self.degree {
            let mut next: Vec<ArithmaExpression> = Vec::with_capacity(q + 1);
            for j in 0..=q {
                let i = k + j - q;
                let mut sum = ArithmaExpression::zero();
                if j >= 1 {
                    let num = ArithmaExpression::sub(t.clone(), lit(u[i]));
                    if let Some(term) = Self::cox_term(&cur[j - 1], num, u[i + q] - u[i]) {
                        sum = ArithmaExpression::add(sum, term);
                    }
                }
                if j < q {
                    let num = ArithmaExpression::sub(lit(u[i + q + 1]), t.clone());
                    if let Some(term) = Self::cox_term(&cur[j], num, u[i + q + 1] - u[i + 1]) {
                        sum = ArithmaExpression::add(sum, term);
                    }
                }
                next.push(sum);
            }
            cur = next;
        }
        cur
    }

    /// The per-axis rational expression `[x, y, z](parameter)` valid on span
    /// `k` (`knots[k] <= t < knots[k+1]`).
    pub fn span_expression(&self, k: usize, parameter: &str) -> Vec<ArithmaExpression> {
        let basis = self.span_basis(k, parameter);
        let p = self.degree;
        let mut axes = vec![ArithmaExpression::zero(); 4];
        for (j, n) in basis.iter().enumerate() {
            let point = &self.points[j + k - p];
            for (axis, coord) in axes.iter_mut().zip(point.iter()) {
                *axis = ArithmaExpression::add(axis.clone(), ArithmaExpression::mul(lit(*coord), n.clone()));
            }
        }
        let w = axes.pop().unwrap_or_else(ArithmaExpression::zero);
        axes.into_iter().map(|a| ArithmaExpression::div(a, w.clone())).collect()
    }

    /// The tangent direction at `t`, unit length, from the exact derivative of
    /// the span's symbolic expression (zero vector if degenerate).
    ///
    /// # Errors
    ///
    /// [`ArithmaNurbsError::Expression`] if Arithma cannot differentiate or
    /// evaluate the span expression.
    pub fn tangent(&self, t: f64) -> NurbsResult<[f64; 3]> {
        let (lo, hi) = self.domain();
        let t = t.clamp(lo, hi);
        let mut bindings = ArithmaBindings::new();
        bindings.insert("t".to_string(), t);
        let mut d = [0.0; 3];
        for (slot, axis) in d.iter_mut().zip(self.span_expression(self.span(t), "t")) {
            let derivative = axis.differentiate("t").map_err(expr_error)?;
            *slot = derivative.evaluate(&bindings).map_err(expr_error)?;
        }
        // direct-float: lowering at evaluation
        let norm = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        Ok(if norm > 0.0 { [d[0] / norm, d[1] / norm, d[2] / norm] } else { [0.0; 3] })
    }

    /// Sample the curve into a polyline at a stated tolerance. This is the only
    /// place a curve stops being exact; the segment count is derived from the
    /// control polygon length against the tolerance.
    ///
    /// # Errors
    ///
    /// [`ArithmaNurbsError::InvalidTolerance`] unless `tolerance` is finite and
    /// positive.
    pub fn to_polyline(&self, tolerance: f64) -> NurbsResult<Vec<[f64; 3]>> {
        if !(tolerance.is_finite() && tolerance > 0.0) {
            return Err(ArithmaNurbsError::InvalidTolerance(tolerance));
        }
        let control = self.control_points();
        // direct-float: tessellation sizing heuristic
        let length: f64 = control
            .windows(2)
            .map(|w| {
                let d = [w[1][0] - w[0][0], w[1][1] - w[0][1], w[1][2] - w[0][2]];
                (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
            })
            .sum();
        let segments = ((length / tolerance).sqrt().ceil() as usize).clamp(2, 4096);
        let (lo, hi) = self.domain();
        Ok((0..=segments).map(|i| self.evaluate(lo + (hi - lo) * i as f64 / segments as f64)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    }

    fn quad_curve() -> NurbsCurve {
        let pts = vec![[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [3.0, 2.0, 1.0], [4.0, 0.0, 0.0], [5.0, 1.0, 2.0]];
        NurbsCurve::clamped(2, pts, vec![1.0, 2.0, 0.5, 1.5, 1.0]).unwrap()
    }

    #[test]
    fn a_clamped_curve_starts_and_ends_at_its_outer_control_points() {
        let pts = vec![[0.0; 3], [1.0, 2.0, 0.0], [3.0, 2.0, 0.0], [4.0, 0.0, 0.0]];
        let c = NurbsCurve::clamped(3, pts.clone(), vec![1.0; 4]).unwrap();
        assert!(dist(c.evaluate(0.0), pts[0]) < 1e-9);
        assert!(dist(c.evaluate(1.0), pts[3]) < 1e-9);
    }

    #[test]
    fn a_degree_one_curve_is_the_straight_line_between_its_points() {
        let l = NurbsCurve::polyline(vec![[0.0; 3], [2.0, 0.0, 0.0]]).unwrap();
        assert!(dist(l.evaluate(0.5), [1.0, 0.0, 0.0]) < 1e-12);
        assert!(dist(l.tangent(0.5).unwrap(), [1.0, 0.0, 0.0]) < 1e-9);
    }

    #[test]
    fn a_rational_quadratic_is_an_exact_circle() {
        let centre = [1.0, -2.0, 0.0];
        let arc = NurbsCurve::circular_arc(centre, 3.0, std::f64::consts::FRAC_PI_2).unwrap();
        for i in 0..=20 {
            let p = arc.evaluate(i as f64 / 20.0);
            assert!((dist(p, centre) - 3.0).abs() < 1e-12);
            assert!(p[2].abs() < 1e-12);
        }
    }

    #[test]
    fn an_arc_of_half_a_turn_or_more_is_refused() {
        assert!(NurbsCurve::circular_arc([0.0; 3], 1.0, std::f64::consts::PI).is_err());
        assert!(NurbsCurve::circular_arc([0.0; 3], 1.0, 4.0).is_err());
        assert!(NurbsCurve::circular_arc([0.0; 3], 0.0, 1.0).is_err());
        assert!(NurbsCurve::circular_arc([0.0; 3], 1.0, 0.0).is_err());
    }

    #[test]
    fn a_wrong_knot_count_is_refused_rather_than_silently_changing_the_degree() {
        let pts = vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert!(NurbsCurve::new(2, pts.clone(), vec![1.0; 3], vec![0.0; 5]).is_err());
        assert!(NurbsCurve::new(2, pts.clone(), vec![1.0; 3], vec![0.0; 6]).is_ok());
        assert!(NurbsCurve::new(2, pts, vec![1.0; 3], vec![0.0, 0.0, 0.0, 1.0, 0.5, 1.0]).is_err());
    }

    #[test]
    fn a_non_positive_weight_is_refused_because_it_is_a_pole() {
        let pts = vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
            assert!(NurbsCurve::new(2, pts.clone(), vec![1.0, bad, 1.0], knots).is_err());
        }
    }

    #[test]
    fn too_few_control_points_for_the_degree_is_refused() {
        assert!(NurbsCurve::clamped(3, vec![[0.0; 3], [1.0, 0.0, 0.0]], vec![1.0; 2]).is_err());
        assert!(NurbsCurve::clamped(0, vec![[0.0; 3], [1.0, 0.0, 0.0]], vec![1.0; 2]).is_err());
        assert!(NurbsCurve::polyline(vec![[0.0; 3]]).is_err());
    }

    #[test]
    fn the_end_of_the_domain_evaluates_and_outside_clamps() {
        let c = NurbsCurve::clamped(
            2,
            vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
            vec![1.0; 5],
        )
        .unwrap();
        let end = c.evaluate(1.0);
        assert!(dist(end, [1.0; 3]) < 1e-9);
        assert_eq!(c.evaluate(5.0), end);
        assert_eq!(c.evaluate(-5.0), c.evaluate(0.0));
    }

    #[test]
    fn tessellation_requires_a_stated_tolerance() {
        let arc = NurbsCurve::circular_arc([0.0; 3], 1.0, 1.5).unwrap();
        assert!(arc.to_polyline(0.0).is_err());
        assert!(arc.to_polyline(-1.0).is_err());
        assert!(arc.to_polyline(f64::NAN).is_err());
        let coarse = arc.to_polyline(0.1).unwrap();
        let fine = arc.to_polyline(1e-4).unwrap();
        assert!(fine.len() > coarse.len());
        for p in &fine {
            assert!((dist(*p, [0.0; 3]) - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn control_points_and_weights_survive_the_homogeneous_round_trip() {
        let pts = vec![[0.0; 3], [1.0, 2.0, 3.0], [0.0, 1.0, 0.0]];
        let w = vec![1.0, 0.5, 2.0];
        let knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let c = NurbsCurve::new(2, pts.clone(), w.clone(), knots).unwrap();
        for (a, b) in c.control_points().iter().zip(&pts) {
            assert!(dist(*a, *b) < 1e-12);
        }
        assert_eq!(c.weights(), w);
        assert_eq!((c.degree(), c.control_count(), c.knots().len()), (2, 3, 6));
        assert_eq!(c.domain(), (0.0, 1.0));
    }

    #[test]
    fn the_symbolic_span_expression_matches_de_boor() {
        let c = quad_curve();
        for i in 0..40 {
            let t = i as f64 / 40.0 + 0.0123;
            let k = c.span(t);
            let mut b = ArithmaBindings::new();
            b.insert("t".to_string(), t);
            let sym: Vec<f64> = c.span_expression(k, "t").iter().map(|e| e.evaluate(&b).unwrap()).collect();
            let num = c.evaluate(t);
            assert!(dist([sym[0], sym[1], sym[2]], num) < 1e-9, "t={t}: {sym:?} vs {num:?}");
        }
    }

    #[test]
    fn the_exact_tangent_matches_a_central_difference() {
        let c = quad_curve();
        let t = 0.37;
        let h = 1e-6;
        let (a, b) = (c.evaluate(t - h), c.evaluate(t + h));
        let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let want = [d[0] / n, d[1] / n, d[2] / n];
        assert!(dist(c.tangent(t).unwrap(), want) < 1e-5);
    }
}
