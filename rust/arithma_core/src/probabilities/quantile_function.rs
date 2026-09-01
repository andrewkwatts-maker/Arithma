//====== Arithma/rust/arithma_core/src/probabilities/quantile_function.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Quantile function
//!
//! Inverse-CDF lookups, implemented generically by bisection so that any
//! [`ArithmaDistribution`] gets a quantile function for free.

use crate::probabilities::ArithmaDistribution;

/// Doublings allowed while searching for a bracket. `2^1100` is already past
/// `f64::MAX`, so this can never be the binding constraint in practice — it
/// only stops a pathological CDF from looping forever.
const MAX_BRACKET_STEPS: u32 = 1100;

/// Bisection steps. Each halves the bracket, so 200 is far more than enough to
/// reach `f64` resolution from any bracket that fits in an `f64`.
const MAX_BISECT_STEPS: u32 = 200;

/// Absolute width at which the bracket is considered converged, scaled by the
/// magnitude of the answer.
const BISECT_TOL: f64 = 1e-12;

/// Helper for evaluating inverse CDFs.
pub struct ArithmaQuantileFunction;

impl ArithmaQuantileFunction {
    /// Inverse CDF `Q(p) = inf{ x : F(x) ≥ p }`, found by bisection.
    ///
    /// # Algorithm
    ///
    /// 1. Expand a bracket outward from `[-1, 1]`, doubling each bound until
    ///    `F(lo) < p ≤ F(hi)`. Capped at [`MAX_BRACKET_STEPS`].
    /// 2. Bisect, maintaining the invariant `F(lo) < p ≤ F(hi)`, for at most
    ///    [`MAX_BISECT_STEPS`] steps or until the bracket is narrower than
    ///    `BISECT_TOL · (1 + |hi|)`.
    /// 3. Return `hi` — the smallest bracket endpoint known to satisfy
    ///    `F(x) ≥ p`, matching the `inf{…}` definition.
    ///
    /// Bisection only needs `F` to be non-decreasing, so this works for
    /// discrete distributions too. For those, `F` is a step function and the
    /// result converges to the jump point *from above*: the 0.5-quantile of
    /// `Binomial(5, 0.5)` comes back as `2` plus at most `BISECT_TOL`, so round
    /// the result if an exact integer is wanted.
    ///
    /// # Errors
    ///
    /// - `p` NaN, `p ≤ 0` or `p ≥ 1`. The extreme quantiles are typically
    ///   `±∞` (unbounded support) or the support endpoint (bounded), which are
    ///   not usefully expressible through a single bisection; callers wanting
    ///   them should special-case the distribution.
    /// - the bracket search fails to straddle `p` (a CDF that is not a proper
    ///   distribution function).
    /// - any propagated error from `dist.cdf`.
    pub fn inverse_cdf(dist: &dyn ArithmaDistribution, p: f64) -> Result<f64, String> {
        if p.is_nan() {
            return Err("ArithmaQuantileFunction::inverse_cdf: p must not be NaN".to_string());
        }
        if p <= 0.0 || p >= 1.0 {
            return Err(format!(
                "ArithmaQuantileFunction::inverse_cdf: p must lie strictly in (0, 1), got {p}"
            ));
        }

        // --- 1. bracket -----------------------------------------------------
        let mut lo = -1.0_f64;
        let mut steps = 0;
        while dist.cdf(lo)? >= p {
            lo *= 2.0;
            steps += 1;
            if steps > MAX_BRACKET_STEPS || !lo.is_finite() {
                return Err(format!(
                    "ArithmaQuantileFunction::inverse_cdf: could not bracket p = {p} from below"
                ));
            }
        }

        let mut hi = 1.0_f64;
        steps = 0;
        while dist.cdf(hi)? < p {
            hi *= 2.0;
            steps += 1;
            if steps > MAX_BRACKET_STEPS || !hi.is_finite() {
                return Err(format!(
                    "ArithmaQuantileFunction::inverse_cdf: could not bracket p = {p} from above"
                ));
            }
        }

        // `lo <= -1 < 1 <= hi` by construction, so the bracket is well ordered.

        // --- 2. bisect ------------------------------------------------------
        // Invariant: F(lo) < p <= F(hi).
        for _ in 0..MAX_BISECT_STEPS {
            if hi - lo <= BISECT_TOL * (1.0 + hi.abs()) {
                break;
            }
            let mid = lo + 0.5 * (hi - lo);
            if mid <= lo || mid >= hi {
                break; // adjacent floats: cannot narrow further
            }
            if dist.cdf(mid)? >= p {
                hi = mid;
            } else {
                lo = mid;
            }
        }

        // --- 3. report the smallest x known to satisfy F(x) >= p ------------
        Ok(hi)
    }
}

// ---------------------------------------------------------------------------
// Backward-compatibility aliases for the pre-rename `Arithmos*` names.
// Retained for one release; downstream (eml-math, eml-spectral, metaphysica,
// periodica) should migrate to the `Arithma*` names above.
// ---------------------------------------------------------------------------
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaQuantileFunction`")]
#[allow(unused)]
pub use self::ArithmaQuantileFunction as ArithmosQuantileFunction;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probabilities::{ArithmaBernoulli, ArithmaBinomial, ArithmaNormal};

    fn close(got: f64, want: f64, tol: f64, what: &str) {
        assert!(
            (got - want).abs() <= tol,
            "{what}: got {got}, want {want} (tol {tol})"
        );
    }

    #[test]
    fn standard_normal_quantiles_match_known_values() {
        let n = ArithmaNormal::standard();
        close(
            ArithmaQuantileFunction::inverse_cdf(&n, 0.5).unwrap(),
            0.0,
            1e-10,
            "z(0.5)",
        );
        // The two-sided 95% critical value.
        close(
            ArithmaQuantileFunction::inverse_cdf(&n, 0.975).unwrap(),
            1.959_963_984_540_054,
            1e-10,
            "z(0.975)",
        );
        close(
            ArithmaQuantileFunction::inverse_cdf(&n, 0.025).unwrap(),
            -1.959_963_984_540_054,
            1e-10,
            "z(0.025)",
        );
        // The one-sided 95% critical value.
        close(
            ArithmaQuantileFunction::inverse_cdf(&n, 0.95).unwrap(),
            1.644_853_626_951_472_2,
            1e-10,
            "z(0.95)",
        );
        close(
            ArithmaQuantileFunction::inverse_cdf(&n, 0.841_344_746_068_542_9).unwrap(),
            1.0,
            1e-10,
            "z(Phi(1))",
        );
    }

    #[test]
    fn shifted_normal_quantiles_shift_and_scale() {
        let n = ArithmaNormal::new(10.0, 2.0);
        close(
            ArithmaQuantileFunction::inverse_cdf(&n, 0.5).unwrap(),
            10.0,
            1e-9,
            "Q(0.5)",
        );
        close(
            ArithmaQuantileFunction::inverse_cdf(&n, 0.975).unwrap(),
            10.0 + 2.0 * 1.959_963_984_540_054,
            1e-9,
            "Q(0.975)",
        );
    }

    #[test]
    fn quantile_round_trips_through_the_cdf() {
        let n = ArithmaNormal::new(-3.0, 0.5);
        for i in 1..20 {
            let p = f64::from(i) / 20.0;
            let x = ArithmaQuantileFunction::inverse_cdf(&n, p).unwrap();
            close(n.cdf(x).unwrap(), p, 1e-10, &format!("F(Q({p}))"));
        }
    }

    #[test]
    fn discrete_quantiles_land_on_the_jump_points() {
        // Binomial(5, 0.5): F(2) = 0.5 exactly, so Q(0.5) = 2.
        let b = ArithmaBinomial::new(5, 0.5);
        let q = ArithmaQuantileFunction::inverse_cdf(&b, 0.5).unwrap();
        close(q, 2.0, 1e-9, "binom Q(0.5)");
        assert!(b.cdf(q).unwrap() >= 0.5, "Q must satisfy F(Q) >= p");
        // F(0) = 0.03125, F(1) = 0.1875 => Q(0.1) = 1.
        close(
            ArithmaQuantileFunction::inverse_cdf(&b, 0.1).unwrap(),
            1.0,
            1e-9,
            "binom Q(0.1)",
        );

        // Bernoulli(0.3): F(0) = 0.7, so Q(0.5) = 0 and Q(0.9) = 1.
        let bern = ArithmaBernoulli::new(0.3);
        close(
            ArithmaQuantileFunction::inverse_cdf(&bern, 0.5).unwrap(),
            0.0,
            1e-9,
            "bern Q(0.5)",
        );
        close(
            ArithmaQuantileFunction::inverse_cdf(&bern, 0.9).unwrap(),
            1.0,
            1e-9,
            "bern Q(0.9)",
        );
    }

    #[test]
    fn quantiles_are_monotone_in_p() {
        let n = ArithmaNormal::standard();
        let mut prev = f64::NEG_INFINITY;
        for i in 1..100 {
            let x = ArithmaQuantileFunction::inverse_cdf(&n, f64::from(i) / 100.0).unwrap();
            assert!(x > prev, "quantile not increasing at p = {i}/100");
            prev = x;
        }
    }

    #[test]
    fn out_of_range_p_is_rejected() {
        let n = ArithmaNormal::standard();
        // Edge cases: the closed endpoints and beyond.
        assert!(ArithmaQuantileFunction::inverse_cdf(&n, 0.0).is_err());
        assert!(ArithmaQuantileFunction::inverse_cdf(&n, 1.0).is_err());
        assert!(ArithmaQuantileFunction::inverse_cdf(&n, -0.5).is_err());
        assert!(ArithmaQuantileFunction::inverse_cdf(&n, 1.5).is_err());
        assert!(ArithmaQuantileFunction::inverse_cdf(&n, f64::NAN).is_err());
        // A CDF that itself errors propagates rather than panicking.
        let bad = ArithmaNormal::new(0.0, 0.0);
        assert!(ArithmaQuantileFunction::inverse_cdf(&bad, 0.5).is_err());
    }
}
