//====== Arithma/rust/arithma_core/src/numerical/critical_points.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Critical points
//!
//! Local maxima, minima, saddle points and inflection points found via the
//! first- and second-derivative tests. The analysis pipeline:
//!
//! 1. `find_stationary_points` — solve `f'(x) = 0` over `[lo, hi]`
//! 2. `analyze_point`         — classify each root via `f''` and `f'''`
//! 3. `find_extrema`          — split classified stationary points into
//!    maxima and minima
//! 4. `find_inflection_points`— solve `f''(x) = 0` and verify `f'''(x) ≠ 0`
//! 5. `analyze_intervals`     — combine stationary + inflection + monotonic
//!    + concavity into a single `ArithmaFunctionAnalysis`
//!
//! Wave 2 establishes the full *type surface* so downstream code can compile
//! against the Arithma API today. Wave 3 wires the bodies once the supporting
//! infrastructure (`crate::calculus::differentiation`, `crate::numerical::root_finding`,
//! `crate::numerical::interval_analysis`) lands its real implementations.

use std::collections::HashMap;

use crate::calculus::differentiation::differentiate;
use crate::expression::{ArithmaBindings, ArithmaExpression, Evaluable};
use crate::numerical::root_finding::{find_root_bisection, ArithmaRootFindingConfig};

/// Evaluate `expr` with `var` bound to `x`, rejecting non-finite results.
fn eval_at(expr: &ArithmaExpression, var: &str, x: f64) -> Result<f64, String> {
    let mut bindings: ArithmaBindings = HashMap::with_capacity(1);
    bindings.insert(var.to_string(), x);
    let y = expr.evaluate(&bindings)?;
    if y.is_finite() {
        Ok(y)
    } else {
        Err(format!("f({x}) evaluated to a non-finite value: {y}"))
    }
}

// ─── Defaults (data-driven thresholds per CLAUDE.md §6) ──────────────────
//
// These mirror the engine-side `PTCriticalPointsConfig::default()` literals
// in `pt-arithmos/src/math/numerical/pt_critical_points.rs`. Lifting them to
// named constants here avoids the magic-number anti-pattern.

const ARITHMA_DEFAULT_CONVERGENCE_THRESHOLD: f64 = 1.0e-10;
const ARITHMA_DEFAULT_SECOND_DERIVATIVE_THRESHOLD: f64 = 1.0e-8;
const ARITHMA_DEFAULT_NUMERICAL_TOLERANCE: f64 = 1.0e-12;
const ARITHMA_DEFAULT_MAX_SEARCH_ITERATIONS: usize = 100;

/// Classification of a critical point. Matches the calculus-textbook taxonomy
/// (max / min / saddle / inflection) with a fifth `Inconclusive` slot for
/// points where the derivative tests fail (e.g. `f''(x) = 0` and `f'''(x) = 0`,
/// requiring higher-order analysis or numerical neighbour-comparison).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArithmaCriticalPointKind {
    /// `f'(x*)=0` and `f''(x*)<0` (or higher-order test confirms a maximum).
    Maximum,
    /// `f'(x*)=0` and `f''(x*)>0`.
    Minimum,
    /// `f'(x*)=0` and the function changes from increasing to decreasing or
    /// vice-versa without producing a local extremum.
    Saddle,
    /// `f''(x*)=0` and `f'''(x*) ≠ 0`. A change of concavity.
    Inflection,
    /// Tests inconclusive — derivatives vanish past the order we evaluated.
    Inconclusive,
}

/// Full classification record for one critical point. Carries the location,
/// the function value at that location, the derivative samples used to
/// classify it, and the resulting kind.
#[derive(Debug, Clone)]
pub struct ArithmaCriticalPoint {
    /// Location along the variable axis.
    pub x: f64,
    /// Function value `f(x)` at this point.
    pub y: f64,
    /// Classification kind.
    pub kind: ArithmaCriticalPointKind,
    /// Sampled `f'(x)` (approximately zero for stationary points; populated
    /// when the analyser was able to evaluate the derivative).
    pub first_derivative: Option<f64>,
    /// Sampled `f''(x)` — sign drives the second-derivative test.
    pub second_derivative: Option<f64>,
    /// Sampled `f'''(x)` — used to confirm inflection points.
    pub third_derivative: Option<f64>,
}

/// Tunable thresholds for the critical-point analyser. Defaults follow the
/// existing engine values so the Wave 3 wiring can swap implementations
/// without re-tuning callers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArithmaCriticalPointsConfig {
    /// `|f'(x)|` below this is considered "stationary".
    pub convergence_threshold: f64,
    /// `|f''(x)|` below this is treated as zero by the second-derivative test.
    pub second_derivative_threshold: f64,
    /// General floating-point slack; pairs with `convergence_threshold` to
    /// decide when a higher-order test is required.
    pub numerical_tolerance: f64,
    /// Hard cap on root-finding iterations to satisfy the bounded-loop
    /// safety-critical rule.
    pub max_search_iterations: usize,
}

impl Default for ArithmaCriticalPointsConfig {
    fn default() -> Self {
        Self {
            convergence_threshold: ARITHMA_DEFAULT_CONVERGENCE_THRESHOLD,
            second_derivative_threshold: ARITHMA_DEFAULT_SECOND_DERIVATIVE_THRESHOLD,
            numerical_tolerance: ARITHMA_DEFAULT_NUMERICAL_TOLERANCE,
            max_search_iterations: ARITHMA_DEFAULT_MAX_SEARCH_ITERATIONS,
        }
    }
}

/// Half-open interval over which to search for critical points. Kept as a
/// dedicated type rather than `(f64, f64)` so the API's intent is explicit
/// and so we can attach validation methods.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArithmaSearchRange {
    pub lo: f64,
    pub hi: f64,
}

impl ArithmaSearchRange {
    /// Construct a search range with `lo <= hi`.
    pub fn new(lo: f64, hi: f64) -> Self {
        Self { lo, hi }
    }

    /// Return the width `hi - lo`.
    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }

    /// True iff the range is well-formed (non-empty, non-NaN, finite-or-±∞).
    pub fn is_valid(&self) -> bool {
        !self.lo.is_nan() && !self.hi.is_nan() && self.lo <= self.hi
    }
}

/// A monotonic sub-interval `[lo, hi]` annotated with the function's slope sign.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArithmaMonotonicInterval {
    pub lo: f64,
    pub hi: f64,
    pub increasing: bool,
}

/// A concavity sub-interval `[lo, hi]` annotated with whether `f''` is positive.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArithmaConcavityInterval {
    pub lo: f64,
    pub hi: f64,
    pub concave_up: bool,
}

/// Aggregate analysis result: every classified critical point, every
/// inflection point, plus the monotonic and concavity decompositions of the
/// search range.
#[derive(Debug, Clone)]
pub struct ArithmaFunctionAnalysis {
    pub stationary_points: Vec<ArithmaCriticalPoint>,
    pub inflection_points: Vec<ArithmaCriticalPoint>,
    pub range: ArithmaSearchRange,
    pub monotonic_intervals: Vec<ArithmaMonotonicInterval>,
    pub concavity_intervals: Vec<ArithmaConcavityInterval>,
}

/// Analyser bundling the configuration with the critical-point routines.
/// Builder-pattern shape mirrors the engine `PTCriticalPoints` so call-sites
/// translate one-to-one when Wave 3 lands.
#[derive(Debug, Clone, Default)]
pub struct ArithmaCriticalPoints {
    config: ArithmaCriticalPointsConfig,
}

impl ArithmaCriticalPoints {
    /// Construct with default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct with custom configuration.
    pub fn with_config(config: ArithmaCriticalPointsConfig) -> Self {
        Self { config }
    }

    /// Read-only access to the active configuration.
    pub fn config(&self) -> &ArithmaCriticalPointsConfig {
        &self.config
    }

    /// Replace the active configuration.
    pub fn set_config(&mut self, config: ArithmaCriticalPointsConfig) {
        self.config = config;
    }

    /// Solve `f'(x) = 0` over `range` and classify each root.
    ///
    /// Roots are located by scanning the range for sign changes in `f'` and
    /// bisecting each bracket. The scan resolution is `max_search_iterations`
    /// sub-intervals, so two stationary points closer together than
    /// `range.width() / max_search_iterations` may be missed — raise the
    /// iteration budget for densely-oscillating functions. Scanning rather
    /// than root-polishing from arbitrary seeds is what makes the result
    /// deterministic and the coverage explicit.
    pub fn find_stationary_points(
        &self,
        expr: &ArithmaExpression,
        var: &str,
        range: ArithmaSearchRange,
    ) -> Result<Vec<ArithmaCriticalPoint>, String> {
        if !range.is_valid() {
            return Err(format!("invalid search range [{}, {}]", range.lo, range.hi));
        }
        let first = differentiate(expr, var)?;
        let roots = self.scan_for_roots(&first, var, range)?;

        let mut out = Vec::with_capacity(roots.len());
        for x in roots {
            out.push(self.analyze_point(expr, var, x)?);
        }
        Ok(out)
    }

    /// Solve `f''(x) = 0` over `range` and keep the candidates where the
    /// concavity genuinely changes.
    ///
    /// A root of `f''` is only an inflection point if the concavity actually
    /// flips there; `f'''(x) ≠ 0` is the standard sufficient condition, and
    /// where `f'''` also vanishes we fall back to comparing the sign of `f''`
    /// either side. That second check is what stops `x⁴` at the origin being
    /// reported as an inflection.
    pub fn find_inflection_points(
        &self,
        expr: &ArithmaExpression,
        var: &str,
        range: ArithmaSearchRange,
    ) -> Result<Vec<ArithmaCriticalPoint>, String> {
        if !range.is_valid() {
            return Err(format!("invalid search range [{}, {}]", range.lo, range.hi));
        }
        let first = differentiate(expr, var)?;
        let second = differentiate(&first, var)?;
        let third = differentiate(&second, var)?;
        let roots = self.scan_for_roots(&second, var, range)?;

        let step = self.probe_step(range);
        let mut out = Vec::new();
        for x in roots {
            let f3 = eval_at(&third, var, x).ok();
            let confirmed = match f3 {
                Some(v) if v.abs() > self.config.second_derivative_threshold => true,
                _ => {
                    // f''' vanished too — check the sign of f'' either side.
                    let l = eval_at(&second, var, x - step).unwrap_or(0.0);
                    let r = eval_at(&second, var, x + step).unwrap_or(0.0);
                    l * r < 0.0
                }
            };
            if confirmed {
                out.push(ArithmaCriticalPoint {
                    x,
                    y: eval_at(expr, var, x)?,
                    kind: ArithmaCriticalPointKind::Inflection,
                    first_derivative: eval_at(&first, var, x).ok(),
                    second_derivative: eval_at(&second, var, x).ok(),
                    third_derivative: f3,
                });
            }
        }
        Ok(out)
    }

    /// Classify one specific point via the derivative tests.
    pub fn classify_point(
        &self,
        expr: &ArithmaExpression,
        var: &str,
        point: f64,
    ) -> Result<ArithmaCriticalPointKind, String> {
        Ok(self.analyze_point(expr, var, point)?.kind)
    }

    /// Full analysis for a single point: location, value, derivative samples,
    /// and kind.
    ///
    /// Classification follows the textbook order: if `f'` is non-zero the point
    /// is not stationary, so the only thing it can be is an inflection (when
    /// `f''` vanishes and concavity flips). For stationary points the sign of
    /// `f''` decides maximum vs minimum; when `f''` is within
    /// `second_derivative_threshold` of zero we use `f'''` to separate a saddle
    /// from an inconclusive higher-order case.
    pub fn analyze_point(
        &self,
        expr: &ArithmaExpression,
        var: &str,
        point: f64,
    ) -> Result<ArithmaCriticalPoint, String> {
        let first = differentiate(expr, var)?;
        let second = differentiate(&first, var)?;
        let third = differentiate(&second, var)?;

        let y = eval_at(expr, var, point)?;
        let f1 = eval_at(&first, var, point)?;
        let f2 = eval_at(&second, var, point).ok();
        let f3 = eval_at(&third, var, point).ok();

        let stationary = f1.abs() <= self.config.convergence_threshold;
        let kind = if !stationary {
            // Not stationary: the only classification available is inflection.
            let step = self.probe_step_around(point);
            match f2 {
                Some(v) if v.abs() <= self.config.second_derivative_threshold => {
                    let l = eval_at(&second, var, point - step).unwrap_or(0.0);
                    let r = eval_at(&second, var, point + step).unwrap_or(0.0);
                    if l * r < 0.0 {
                        ArithmaCriticalPointKind::Inflection
                    } else {
                        ArithmaCriticalPointKind::Inconclusive
                    }
                }
                _ => ArithmaCriticalPointKind::Inconclusive,
            }
        } else {
            match f2 {
                Some(v) if v > self.config.second_derivative_threshold => {
                    ArithmaCriticalPointKind::Minimum
                }
                Some(v) if v < -self.config.second_derivative_threshold => {
                    ArithmaCriticalPointKind::Maximum
                }
                // f'' ≈ 0: a non-zero f''' means the concavity flips through a
                // stationary point, i.e. a saddle (e.g. x³ at the origin).
                _ => match f3 {
                    Some(v) if v.abs() > self.config.second_derivative_threshold => {
                        ArithmaCriticalPointKind::Saddle
                    }
                    _ => ArithmaCriticalPointKind::Inconclusive,
                },
            }
        };

        Ok(ArithmaCriticalPoint {
            x: point,
            y,
            kind,
            first_derivative: Some(f1),
            second_derivative: f2,
            third_derivative: f3,
        })
    }

    /// Split classified stationary points into `(maxima, minima)`.
    ///
    /// Saddles and inconclusive points are deliberately in neither list.
    pub fn find_extrema(
        &self,
        expr: &ArithmaExpression,
        var: &str,
        range: ArithmaSearchRange,
    ) -> Result<(Vec<ArithmaCriticalPoint>, Vec<ArithmaCriticalPoint>), String> {
        let points = self.find_stationary_points(expr, var, range)?;
        let mut maxima = Vec::new();
        let mut minima = Vec::new();
        for p in points {
            match p.kind {
                ArithmaCriticalPointKind::Maximum => maxima.push(p),
                ArithmaCriticalPointKind::Minimum => minima.push(p),
                _ => {}
            }
        }
        Ok((maxima, minima))
    }

    /// Combined report: stationary points, inflection points, and the
    /// monotonic / concavity decompositions of the range.
    ///
    /// The decompositions are cut at the stationary and inflection points
    /// respectively, and each resulting sub-interval is labelled by sampling
    /// the relevant derivative at its midpoint.
    pub fn analyze_intervals(
        &self,
        expr: &ArithmaExpression,
        var: &str,
        range: ArithmaSearchRange,
    ) -> Result<ArithmaFunctionAnalysis, String> {
        let stationary_points = self.find_stationary_points(expr, var, range)?;
        let inflection_points = self.find_inflection_points(expr, var, range)?;

        let first = differentiate(expr, var)?;
        let second = differentiate(&first, var)?;

        let monotonic_intervals =
            self.split_intervals(&first, var, range, &stationary_points, |lo, hi, sign| {
                ArithmaMonotonicInterval {
                    lo,
                    hi,
                    increasing: sign > 0.0,
                }
            })?;
        let concavity_intervals =
            self.split_intervals(&second, var, range, &inflection_points, |lo, hi, sign| {
                ArithmaConcavityInterval {
                    lo,
                    hi,
                    concave_up: sign > 0.0,
                }
            })?;

        Ok(ArithmaFunctionAnalysis {
            stationary_points,
            inflection_points,
            range,
            monotonic_intervals,
            concavity_intervals,
        })
    }

    // ── internals ─────────────────────────────────────────────────────────

    /// Uniform scan width used when probing either side of a point.
    fn probe_step(&self, range: ArithmaSearchRange) -> f64 {
        let w = range.width().abs();
        if w.is_finite() && w > 0.0 {
            (w / self.config.max_search_iterations.max(1) as f64).max(1e-9)
        } else {
            1e-6
        }
    }

    fn probe_step_around(&self, point: f64) -> f64 {
        (point.abs().max(1.0) * 1e-6).max(1e-9)
    }

    /// Scan `range` for sign changes in `f` and bisect each bracket.
    /// Also captures points where `f` is already ~0 at a sample.
    fn scan_for_roots(
        &self,
        f: &ArithmaExpression,
        var: &str,
        range: ArithmaSearchRange,
    ) -> Result<Vec<f64>, String> {
        let steps = self.config.max_search_iterations.max(1);
        let width = range.width();
        if !width.is_finite() {
            return Err("cannot scan an unbounded range".to_string());
        }
        let dx = width / steps as f64;

        let cfg = ArithmaRootFindingConfig {
            tol: self.config.convergence_threshold,
            max_iterations: self.config.max_search_iterations,
        };

        let mut roots: Vec<f64> = Vec::new();
        let mut samples = 0_usize;
        let mut zero_samples = 0_usize;
        // Slightly wider than one scan step, so two hits on consecutive samples
        // collapse to one root despite floating-point drift in `lo + dx*i`.
        let dedup_tol = (dx.abs() * 1.5).max(self.config.numerical_tolerance);
        let push_unique = |roots: &mut Vec<f64>, x: f64| {
            if !roots.iter().any(|r: &f64| (r - x).abs() <= dedup_tol) {
                roots.push(x);
            }
        };

        let mut prev_x = range.lo;
        let mut prev_y = eval_at(f, var, prev_x)?;
        samples += 1;
        if prev_y.abs() <= self.config.convergence_threshold {
            zero_samples += 1;
            push_unique(&mut roots, prev_x);
        }

        for i in 1..=steps {
            let x = if i == steps {
                range.hi
            } else {
                range.lo + dx * i as f64
            };
            let y = match eval_at(f, var, x) {
                Ok(v) => v,
                // A pole or domain gap inside the range is not fatal — skip the
                // sample and continue scanning the rest of the interval.
                Err(_) => {
                    prev_x = x;
                    continue;
                }
            };

            samples += 1;
            if y.abs() <= self.config.convergence_threshold {
                zero_samples += 1;
                push_unique(&mut roots, x);
            } else if prev_y * y < 0.0 {
                if let Ok(r) = find_root_bisection(f, var, prev_x, x, &cfg) {
                    push_unique(&mut roots, r.root);
                }
            }
            prev_x = x;
            prev_y = y;
        }

        // `f` vanishing at *every* sample means it is identically zero on the
        // range (a constant function's derivative, say). Such a function has no
        // *isolated* stationary points, so enumerating one per scan sample
        // would be noise dressed up as a result. Report none.
        if samples > 2 && zero_samples == samples {
            return Ok(Vec::new());
        }

        roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Ok(roots)
    }

    /// Cut `range` at each of `cuts` and label every sub-interval by the sign
    /// of `f` at its midpoint.
    fn split_intervals<T>(
        &self,
        f: &ArithmaExpression,
        var: &str,
        range: ArithmaSearchRange,
        cuts: &[ArithmaCriticalPoint],
        make: impl Fn(f64, f64, f64) -> T,
    ) -> Result<Vec<T>, String> {
        let mut bounds: Vec<f64> = vec![range.lo];
        bounds.extend(
            cuts.iter()
                .map(|p| p.x)
                .filter(|x| *x > range.lo && *x < range.hi),
        );
        bounds.push(range.hi);
        bounds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut out = Vec::new();
        for w in bounds.windows(2) {
            let (lo, hi) = (w[0], w[1]);
            if hi <= lo {
                continue;
            }
            let mid = lo + (hi - lo) * 0.5;
            // An unevaluable midpoint means we cannot label the sub-interval;
            // omitting it is honest, inventing a sign is not.
            if let Ok(sign) = eval_at(f, var, mid) {
                out.push(make(lo, hi, sign));
            }
        }
        Ok(out)
    }
}

/// Convenience free-function: stationary points of `expr` over `[lo, hi]`
/// using the default configuration.
pub fn find_critical_points(
    expr: &ArithmaExpression,
    var: &str,
    lo: f64,
    hi: f64,
) -> Result<Vec<ArithmaCriticalPoint>, String> {
    ArithmaCriticalPoints::new().find_stationary_points(expr, var, ArithmaSearchRange::new(lo, hi))
}

// ---------------------------------------------------------------------------
// Backward-compatibility aliases for the pre-rename `Arithmos*` names.
// Retained for one release; downstream (eml-math, eml-spectral, metaphysica,
// periodica) should migrate to the `Arithma*` names above.
// ---------------------------------------------------------------------------
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaConcavityInterval`")]
#[allow(unused)]
pub use self::ArithmaConcavityInterval as ArithmosConcavityInterval;
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaCriticalPoint`")]
#[allow(unused)]
pub use self::ArithmaCriticalPoint as ArithmosCriticalPoint;
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaCriticalPointKind`")]
#[allow(unused)]
pub use self::ArithmaCriticalPointKind as ArithmosCriticalPointKind;
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaCriticalPoints`")]
#[allow(unused)]
pub use self::ArithmaCriticalPoints as ArithmosCriticalPoints;
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaCriticalPointsConfig`")]
#[allow(unused)]
pub use self::ArithmaCriticalPointsConfig as ArithmosCriticalPointsConfig;
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaFunctionAnalysis`")]
#[allow(unused)]
pub use self::ArithmaFunctionAnalysis as ArithmosFunctionAnalysis;
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaMonotonicInterval`")]
#[allow(unused)]
pub use self::ArithmaMonotonicInterval as ArithmosMonotonicInterval;
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaSearchRange`")]
#[allow(unused)]
pub use self::ArithmaSearchRange as ArithmosSearchRange;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds_are_distinct() {
        assert_ne!(
            ArithmaCriticalPointKind::Maximum,
            ArithmaCriticalPointKind::Minimum
        );
        assert_ne!(
            ArithmaCriticalPointKind::Saddle,
            ArithmaCriticalPointKind::Inflection
        );
        assert_ne!(
            ArithmaCriticalPointKind::Inflection,
            ArithmaCriticalPointKind::Inconclusive
        );
    }

    #[test]
    fn default_config_uses_named_constants() {
        let cfg = ArithmaCriticalPointsConfig::default();
        assert_eq!(
            cfg.convergence_threshold,
            ARITHMA_DEFAULT_CONVERGENCE_THRESHOLD
        );
        assert_eq!(
            cfg.second_derivative_threshold,
            ARITHMA_DEFAULT_SECOND_DERIVATIVE_THRESHOLD
        );
        assert_eq!(cfg.numerical_tolerance, ARITHMA_DEFAULT_NUMERICAL_TOLERANCE);
        assert_eq!(
            cfg.max_search_iterations,
            ARITHMA_DEFAULT_MAX_SEARCH_ITERATIONS
        );
    }

    #[test]
    fn custom_config_round_trips() {
        let cfg = ArithmaCriticalPointsConfig {
            convergence_threshold: 1.0e-15,
            second_derivative_threshold: 1.0e-12,
            numerical_tolerance: 1.0e-15,
            max_search_iterations: 200,
        };
        let mut analyser = ArithmaCriticalPoints::with_config(cfg);
        assert_eq!(analyser.config().max_search_iterations, 200);
        analyser.set_config(ArithmaCriticalPointsConfig::default());
        assert_eq!(
            analyser.config().max_search_iterations,
            ARITHMA_DEFAULT_MAX_SEARCH_ITERATIONS
        );
    }

    #[test]
    fn search_range_validates_lo_le_hi() {
        let valid = ArithmaSearchRange::new(-2.0, 2.0);
        assert!(valid.is_valid());
        assert_eq!(valid.width(), 4.0);

        let inverted = ArithmaSearchRange::new(1.0, 0.0);
        assert!(!inverted.is_valid());
    }

    #[test]
    fn search_range_rejects_nan() {
        let nan_lo = ArithmaSearchRange::new(f64::NAN, 1.0);
        assert!(!nan_lo.is_valid());
        let nan_hi = ArithmaSearchRange::new(0.0, f64::NAN);
        assert!(!nan_hi.is_valid());
    }

    #[test]
    fn search_range_supports_infinite_bounds() {
        let whole = ArithmaSearchRange::new(f64::NEG_INFINITY, f64::INFINITY);
        assert!(whole.is_valid());
        assert!(whole.width().is_infinite());
    }

    #[test]
    fn monotonic_and_concavity_intervals_round_trip() {
        let m = ArithmaMonotonicInterval {
            lo: 0.0,
            hi: 1.0,
            increasing: true,
        };
        let c = ArithmaConcavityInterval {
            lo: 0.0,
            hi: 1.0,
            concave_up: false,
        };
        assert!(m.increasing);
        assert!(!c.concave_up);
    }

    #[test]
    fn critical_point_struct_carries_derivative_samples() {
        let cp = ArithmaCriticalPoint {
            x: 0.0,
            y: 0.0,
            kind: ArithmaCriticalPointKind::Minimum,
            first_derivative: Some(0.0),
            second_derivative: Some(2.0),
            third_derivative: None,
        };
        assert_eq!(cp.kind, ArithmaCriticalPointKind::Minimum);
        assert_eq!(cp.first_derivative, Some(0.0));
        assert_eq!(cp.second_derivative, Some(2.0));
        assert_eq!(cp.third_derivative, None);
    }

    #[test]
    fn function_analysis_aggregates_all_data() {
        let analysis = ArithmaFunctionAnalysis {
            stationary_points: Vec::new(),
            inflection_points: Vec::new(),
            range: ArithmaSearchRange::new(-1.0, 1.0),
            monotonic_intervals: Vec::new(),
            concavity_intervals: Vec::new(),
        };
        assert_eq!(analysis.range.width(), 2.0);
        assert!(analysis.stationary_points.is_empty());
        assert!(analysis.inflection_points.is_empty());
    }

    #[test]
    fn analyser_builder_pattern_compiles() {
        // Smoke test: prove the builder pattern is reachable from external
        // call-sites without touching any unimplemented stub.
        let analyser = ArithmaCriticalPoints::new();
        assert_eq!(
            analyser.config().convergence_threshold,
            ARITHMA_DEFAULT_CONVERGENCE_THRESHOLD
        );
    }

    // ── analysis over real functions ──────────────────────────────────────

    fn x() -> ArithmaExpression {
        ArithmaExpression::var("x")
    }

    fn pow(n: u32) -> ArithmaExpression {
        let mut acc = x();
        for _ in 1..n {
            acc = ArithmaExpression::mul(acc, x());
        }
        acc
    }

    /// x² — a single minimum at the origin.
    fn parabola() -> ArithmaExpression {
        pow(2)
    }

    /// x³ - 3x — maximum at -1, minimum at +1, inflection at 0.
    fn cubic() -> ArithmaExpression {
        ArithmaExpression::sub(
            pow(3),
            ArithmaExpression::mul(ArithmaExpression::from_i64(3), x()),
        )
    }

    #[test]
    fn parabola_has_one_minimum_at_the_origin() {
        let a = ArithmaCriticalPoints::new();
        let pts = a
            .find_stationary_points(&parabola(), "x", ArithmaSearchRange::new(-3.0, 3.0))
            .unwrap();
        assert_eq!(pts.len(), 1, "got {pts:?}");
        assert!(pts[0].x.abs() < 1e-6, "x = {}", pts[0].x);
        assert_eq!(pts[0].kind, ArithmaCriticalPointKind::Minimum);
    }

    #[test]
    fn cubic_max_and_min_are_found_and_classified() {
        let a = ArithmaCriticalPoints::new();
        let (maxima, minima) = a
            .find_extrema(&cubic(), "x", ArithmaSearchRange::new(-3.0, 3.0))
            .unwrap();
        assert_eq!(maxima.len(), 1, "maxima: {maxima:?}");
        assert_eq!(minima.len(), 1, "minima: {minima:?}");
        assert!((maxima[0].x + 1.0).abs() < 1e-5, "max at {}", maxima[0].x);
        assert!((minima[0].x - 1.0).abs() < 1e-5, "min at {}", minima[0].x);
        // f(-1) = 2, f(1) = -2
        assert!((maxima[0].y - 2.0).abs() < 1e-4);
        assert!((minima[0].y + 2.0).abs() < 1e-4);
    }

    #[test]
    fn cubic_has_an_inflection_at_the_origin() {
        let a = ArithmaCriticalPoints::new();
        let pts = a
            .find_inflection_points(&cubic(), "x", ArithmaSearchRange::new(-3.0, 3.0))
            .unwrap();
        assert_eq!(pts.len(), 1, "got {pts:?}");
        assert!(pts[0].x.abs() < 1e-6);
        assert_eq!(pts[0].kind, ArithmaCriticalPointKind::Inflection);
    }

    #[test]
    fn quartic_origin_is_not_an_inflection() {
        // x⁴ has f''(0) = 0 but concavity does NOT change — the classic false
        // positive if you only test f'' == 0.
        let a = ArithmaCriticalPoints::new();
        let pts = a
            .find_inflection_points(&pow(4), "x", ArithmaSearchRange::new(-2.0, 2.0))
            .unwrap();
        assert!(pts.is_empty(), "x^4 should have no inflection, got {pts:?}");
    }

    #[test]
    fn x_cubed_origin_is_a_saddle_not_an_extremum() {
        let a = ArithmaCriticalPoints::new();
        let kind = a.classify_point(&pow(3), "x", 0.0).unwrap();
        assert_eq!(kind, ArithmaCriticalPointKind::Saddle);
    }

    #[test]
    fn analyze_point_reports_the_derivative_samples() {
        let a = ArithmaCriticalPoints::new();
        let p = a.analyze_point(&parabola(), "x", 0.0).unwrap();
        assert_eq!(p.kind, ArithmaCriticalPointKind::Minimum);
        assert!(p.first_derivative.unwrap().abs() < 1e-9);
        assert!(
            (p.second_derivative.unwrap() - 2.0).abs() < 1e-9,
            "f'' should be 2"
        );
    }

    #[test]
    fn monotonic_intervals_split_at_the_stationary_points() {
        let a = ArithmaCriticalPoints::new();
        let an = a
            .analyze_intervals(&cubic(), "x", ArithmaSearchRange::new(-3.0, 3.0))
            .unwrap();
        // Two stationary points cut [-3, 3] into three monotonic runs:
        // increasing, decreasing, increasing.
        assert_eq!(
            an.monotonic_intervals.len(),
            3,
            "{:?}",
            an.monotonic_intervals
        );
        assert!(an.monotonic_intervals[0].increasing);
        assert!(!an.monotonic_intervals[1].increasing);
        assert!(an.monotonic_intervals[2].increasing);
        // One inflection cuts concavity into two runs: concave down then up.
        assert_eq!(
            an.concavity_intervals.len(),
            2,
            "{:?}",
            an.concavity_intervals
        );
        assert!(!an.concavity_intervals[0].concave_up);
        assert!(an.concavity_intervals[1].concave_up);
    }

    #[test]
    fn free_function_matches_the_analyser() {
        let viaf = find_critical_points(&cubic(), "x", -3.0, 3.0).unwrap();
        let via_analyser = ArithmaCriticalPoints::new()
            .find_stationary_points(&cubic(), "x", ArithmaSearchRange::new(-3.0, 3.0))
            .unwrap();
        assert_eq!(viaf.len(), via_analyser.len());
    }

    #[test]
    fn invalid_range_is_rejected() {
        let a = ArithmaCriticalPoints::new();
        let bad = ArithmaSearchRange::new(f64::NAN, 1.0);
        assert!(a.find_stationary_points(&parabola(), "x", bad).is_err());
    }

    #[test]
    fn constant_function_has_no_isolated_stationary_points() {
        // f = 5 has f' ≡ 0 everywhere. Every point is stationary, so none is
        // *isolated* — reporting one per scan sample would be noise. The
        // identically-zero guard must collapse this to an empty result.
        let a = ArithmaCriticalPoints::new();
        let pts = a
            .find_stationary_points(
                &ArithmaExpression::from_i64(5),
                "x",
                ArithmaSearchRange::new(-1.0, 1.0),
            )
            .unwrap();
        assert!(pts.is_empty(), "expected none, got {}", pts.len());
    }

    #[test]
    fn a_genuine_root_is_not_swallowed_by_the_degenerate_guard() {
        // Guards against the identically-zero check over-reaching: x² still
        // reports its single minimum even though f' is zero at one sample.
        let a = ArithmaCriticalPoints::new();
        let pts = a
            .find_stationary_points(&parabola(), "x", ArithmaSearchRange::new(-1.0, 1.0))
            .unwrap();
        assert_eq!(pts.len(), 1);
    }
}
