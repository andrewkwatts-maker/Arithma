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
//! 1. `find_stationary_points` â€” solve `f'(x) = 0` over `[lo, hi]`
//! 2. `analyze_point`         â€” classify each root via `f''` and `f'''`
//! 3. `find_extrema`          â€” split classified stationary points into
//!                              maxima and minima
//! 4. `find_inflection_points`â€” solve `f''(x) = 0` and verify `f'''(x) â‰  0`
//! 5. `analyze_intervals`     â€” combine stationary + inflection + monotonic
//!                              + concavity into a single `ArithmosFunctionAnalysis`
//!
//! Wave 2 establishes the full *type surface* so downstream code can compile
//! against the Arithmos API today. Wave 3 wires the bodies once the supporting
//! infrastructure (`crate::calculus::differentiation`, `crate::numerical::root_finding`,
//! `crate::numerical::interval_analysis`) lands its real implementations.

use crate::expression::ArithmosExpression;

// â”€â”€â”€ Defaults (data-driven thresholds per CLAUDE.md Â§6) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// These mirror the engine-side `PTCriticalPointsConfig::default()` literals
// in `pt-arithmos/src/math/numerical/pt_critical_points.rs`. Lifting them to
// named constants here avoids the magic-number anti-pattern.

const ARITHMOS_DEFAULT_CONVERGENCE_THRESHOLD: f64 = 1.0e-10;
const ARITHMOS_DEFAULT_SECOND_DERIVATIVE_THRESHOLD: f64 = 1.0e-8;
const ARITHMOS_DEFAULT_NUMERICAL_TOLERANCE: f64 = 1.0e-12;
const ARITHMOS_DEFAULT_MAX_SEARCH_ITERATIONS: usize = 100;

/// Classification of a critical point. Matches the calculus-textbook taxonomy
/// (max / min / saddle / inflection) with a fifth `Inconclusive` slot for
/// points where the derivative tests fail (e.g. `f''(x) = 0` and `f'''(x) = 0`,
/// requiring higher-order analysis or numerical neighbour-comparison).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArithmosCriticalPointKind {
    /// `f'(x*)=0` and `f''(x*)<0` (or higher-order test confirms a maximum).
    Maximum,
    /// `f'(x*)=0` and `f''(x*)>0`.
    Minimum,
    /// `f'(x*)=0` and the function changes from increasing to decreasing or
    /// vice-versa without producing a local extremum.
    Saddle,
    /// `f''(x*)=0` and `f'''(x*) â‰  0`. A change of concavity.
    Inflection,
    /// Tests inconclusive â€” derivatives vanish past the order we evaluated.
    Inconclusive,
}

/// Full classification record for one critical point. Carries the location,
/// the function value at that location, the derivative samples used to
/// classify it, and the resulting kind.
#[derive(Debug, Clone)]
pub struct ArithmosCriticalPoint {
    /// Location along the variable axis.
    pub x: f64,
    /// Function value `f(x)` at this point.
    pub y: f64,
    /// Classification kind.
    pub kind: ArithmosCriticalPointKind,
    /// Sampled `f'(x)` (approximately zero for stationary points; populated
    /// when the analyser was able to evaluate the derivative).
    pub first_derivative: Option<f64>,
    /// Sampled `f''(x)` â€” sign drives the second-derivative test.
    pub second_derivative: Option<f64>,
    /// Sampled `f'''(x)` â€” used to confirm inflection points.
    pub third_derivative: Option<f64>,
}

/// Tunable thresholds for the critical-point analyser. Defaults follow the
/// existing engine values so the Wave 3 wiring can swap implementations
/// without re-tuning callers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArithmosCriticalPointsConfig {
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

impl Default for ArithmosCriticalPointsConfig {
    fn default() -> Self {
        Self {
            convergence_threshold: ARITHMOS_DEFAULT_CONVERGENCE_THRESHOLD,
            second_derivative_threshold: ARITHMOS_DEFAULT_SECOND_DERIVATIVE_THRESHOLD,
            numerical_tolerance: ARITHMOS_DEFAULT_NUMERICAL_TOLERANCE,
            max_search_iterations: ARITHMOS_DEFAULT_MAX_SEARCH_ITERATIONS,
        }
    }
}

/// Half-open interval over which to search for critical points. Kept as a
/// dedicated type rather than `(f64, f64)` so the API's intent is explicit
/// and so we can attach validation methods.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArithmosSearchRange {
    pub lo: f64,
    pub hi: f64,
}

impl ArithmosSearchRange {
    /// Construct a search range with `lo <= hi`.
    pub fn new(lo: f64, hi: f64) -> Self {
        Self { lo, hi }
    }

    /// Return the width `hi - lo`.
    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }

    /// True iff the range is well-formed (non-empty, non-NaN, finite-or-Â±âˆž).
    pub fn is_valid(&self) -> bool {
        self.lo.is_nan() == false
            && self.hi.is_nan() == false
            && self.lo <= self.hi
    }
}

/// A monotonic sub-interval `[lo, hi]` annotated with the function's slope sign.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArithmosMonotonicInterval {
    pub lo: f64,
    pub hi: f64,
    pub increasing: bool,
}

/// A concavity sub-interval `[lo, hi]` annotated with whether `f''` is positive.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArithmosConcavityInterval {
    pub lo: f64,
    pub hi: f64,
    pub concave_up: bool,
}

/// Aggregate analysis result: every classified critical point, every
/// inflection point, plus the monotonic and concavity decompositions of the
/// search range.
#[derive(Debug, Clone)]
pub struct ArithmosFunctionAnalysis {
    pub stationary_points: Vec<ArithmosCriticalPoint>,
    pub inflection_points: Vec<ArithmosCriticalPoint>,
    pub range: ArithmosSearchRange,
    pub monotonic_intervals: Vec<ArithmosMonotonicInterval>,
    pub concavity_intervals: Vec<ArithmosConcavityInterval>,
}

/// Analyser bundling the configuration with the critical-point routines.
/// Builder-pattern shape mirrors the engine `PTCriticalPoints` so call-sites
/// translate one-to-one when Wave 3 lands.
#[derive(Debug, Clone, Default)]
pub struct ArithmosCriticalPoints {
    config: ArithmosCriticalPointsConfig,
}

impl ArithmosCriticalPoints {
    /// Construct with default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct with custom configuration.
    pub fn with_config(config: ArithmosCriticalPointsConfig) -> Self {
        Self { config }
    }

    /// Read-only access to the active configuration.
    pub fn config(&self) -> &ArithmosCriticalPointsConfig {
        &self.config
    }

    /// Replace the active configuration.
    pub fn set_config(&mut self, config: ArithmosCriticalPointsConfig) {
        self.config = config;
    }

    /// Solve `f'(x) = 0` over `range` and classify each root. Wave-3 stub.
    pub fn find_stationary_points(
        &self,
        _expr: &ArithmosExpression,
        _var: &str,
        _range: ArithmosSearchRange,
    ) -> Result<Vec<ArithmosCriticalPoint>, String> {
        unimplemented!("find_stationary_points â€” populated in Wave 3")
    }

    /// Solve `f''(x) = 0` over `range` and verify each candidate via `f'''`.
    /// Wave-3 stub.
    pub fn find_inflection_points(
        &self,
        _expr: &ArithmosExpression,
        _var: &str,
        _range: ArithmosSearchRange,
    ) -> Result<Vec<ArithmosCriticalPoint>, String> {
        unimplemented!("find_inflection_points â€” populated in Wave 3")
    }

    /// Classify one specific point. Wave-3 stub.
    pub fn classify_point(
        &self,
        _expr: &ArithmosExpression,
        _var: &str,
        _point: f64,
    ) -> Result<ArithmosCriticalPointKind, String> {
        unimplemented!("classify_point â€” populated in Wave 3")
    }

    /// Full analysis for a single point: location, value, derivative samples,
    /// and kind. Wave-3 stub.
    pub fn analyze_point(
        &self,
        _expr: &ArithmosExpression,
        _var: &str,
        _point: f64,
    ) -> Result<ArithmosCriticalPoint, String> {
        unimplemented!("analyze_point â€” populated in Wave 3")
    }

    /// Split classified stationary points into `(maxima, minima)`. Wave-3 stub.
    pub fn find_extrema(
        &self,
        _expr: &ArithmosExpression,
        _var: &str,
        _range: ArithmosSearchRange,
    ) -> Result<(Vec<ArithmosCriticalPoint>, Vec<ArithmosCriticalPoint>), String> {
        unimplemented!("find_extrema â€” populated in Wave 3")
    }

    /// Combined report across the search range. Wave-3 stub.
    pub fn analyze_intervals(
        &self,
        _expr: &ArithmosExpression,
        _var: &str,
        _range: ArithmosSearchRange,
    ) -> Result<ArithmosFunctionAnalysis, String> {
        unimplemented!("analyze_intervals â€” populated in Wave 3")
    }
}

/// Convenience free-function. Equivalent to
/// `ArithmosCriticalPoints::new().find_stationary_points(...)` but with the
/// older return shape preserved. Wave-3 stub.
pub fn find_critical_points(
    _expr: &ArithmosExpression,
    _var: &str,
    _lo: f64,
    _hi: f64,
) -> Result<Vec<ArithmosCriticalPoint>, String> {
    unimplemented!("find_critical_points â€” populated in Wave 3")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds_are_distinct() {
        assert_ne!(
            ArithmosCriticalPointKind::Maximum,
            ArithmosCriticalPointKind::Minimum
        );
        assert_ne!(
            ArithmosCriticalPointKind::Saddle,
            ArithmosCriticalPointKind::Inflection
        );
        assert_ne!(
            ArithmosCriticalPointKind::Inflection,
            ArithmosCriticalPointKind::Inconclusive
        );
    }

    #[test]
    fn default_config_uses_named_constants() {
        let cfg = ArithmosCriticalPointsConfig::default();
        assert_eq!(cfg.convergence_threshold, ARITHMOS_DEFAULT_CONVERGENCE_THRESHOLD);
        assert_eq!(cfg.second_derivative_threshold, ARITHMOS_DEFAULT_SECOND_DERIVATIVE_THRESHOLD);
        assert_eq!(cfg.numerical_tolerance, ARITHMOS_DEFAULT_NUMERICAL_TOLERANCE);
        assert_eq!(cfg.max_search_iterations, ARITHMOS_DEFAULT_MAX_SEARCH_ITERATIONS);
    }

    #[test]
    fn custom_config_round_trips() {
        let cfg = ArithmosCriticalPointsConfig {
            convergence_threshold: 1.0e-15,
            second_derivative_threshold: 1.0e-12,
            numerical_tolerance: 1.0e-15,
            max_search_iterations: 200,
        };
        let mut analyser = ArithmosCriticalPoints::with_config(cfg);
        assert_eq!(analyser.config().max_search_iterations, 200);
        analyser.set_config(ArithmosCriticalPointsConfig::default());
        assert_eq!(
            analyser.config().max_search_iterations,
            ARITHMOS_DEFAULT_MAX_SEARCH_ITERATIONS
        );
    }

    #[test]
    fn search_range_validates_lo_le_hi() {
        let valid = ArithmosSearchRange::new(-2.0, 2.0);
        assert!(valid.is_valid());
        assert_eq!(valid.width(), 4.0);

        let inverted = ArithmosSearchRange::new(1.0, 0.0);
        assert!(!inverted.is_valid());
    }

    #[test]
    fn search_range_rejects_nan() {
        let nan_lo = ArithmosSearchRange::new(f64::NAN, 1.0);
        assert!(!nan_lo.is_valid());
        let nan_hi = ArithmosSearchRange::new(0.0, f64::NAN);
        assert!(!nan_hi.is_valid());
    }

    #[test]
    fn search_range_supports_infinite_bounds() {
        let whole = ArithmosSearchRange::new(f64::NEG_INFINITY, f64::INFINITY);
        assert!(whole.is_valid());
        assert!(whole.width().is_infinite());
    }

    #[test]
    fn monotonic_and_concavity_intervals_round_trip() {
        let m = ArithmosMonotonicInterval { lo: 0.0, hi: 1.0, increasing: true };
        let c = ArithmosConcavityInterval { lo: 0.0, hi: 1.0, concave_up: false };
        assert!(m.increasing);
        assert!(!c.concave_up);
    }

    #[test]
    fn critical_point_struct_carries_derivative_samples() {
        let cp = ArithmosCriticalPoint {
            x: 0.0,
            y: 0.0,
            kind: ArithmosCriticalPointKind::Minimum,
            first_derivative: Some(0.0),
            second_derivative: Some(2.0),
            third_derivative: None,
        };
        assert_eq!(cp.kind, ArithmosCriticalPointKind::Minimum);
        assert_eq!(cp.first_derivative, Some(0.0));
        assert_eq!(cp.second_derivative, Some(2.0));
        assert_eq!(cp.third_derivative, None);
    }

    #[test]
    fn function_analysis_aggregates_all_data() {
        let analysis = ArithmosFunctionAnalysis {
            stationary_points: Vec::new(),
            inflection_points: Vec::new(),
            range: ArithmosSearchRange::new(-1.0, 1.0),
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
        let analyser = ArithmosCriticalPoints::new();
        assert_eq!(
            analyser.config().convergence_threshold,
            ARITHMOS_DEFAULT_CONVERGENCE_THRESHOLD
        );
    }
}

