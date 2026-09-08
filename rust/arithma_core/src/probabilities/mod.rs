//====== Arithma/rust/arithma_core/src/probabilities/mod.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Probabilities
//!
//! Probability distributions, quantile functions, statistical moments and
//! samplers. Mirrors `pt_arithmos::math::probabilities`. The core abstraction
//! is the [`ArithmaDistribution`] trait — every concrete distribution
//! (Normal, Binomial, Bernoulli, …) implements it so downstream code can be
//! generic over the kind.
//!
//! ## Submodules
//!
//! - [`bernoulli`], [`binomial`], [`normal`] — concrete distributions.
//! - [`distribution_factory`] — JSON-driven distribution constructor.
//! - [`quantile_function`] — inverse-CDF helpers.
//! - [`confidence_interval`] — interval estimation.
//! - [`statistical_moment`] — variance / skew / kurtosis.
//! - [`statistical_sampler`] — generic sampling driver.

pub mod bernoulli;
pub mod binomial;
pub mod confidence_interval;
pub mod distribution_factory;
pub mod normal;
pub mod quantile_function;
pub mod statistical_moment;
pub mod statistical_sampler;

pub use bernoulli::ArithmaBernoulli;
pub use binomial::ArithmaBinomial;
pub use confidence_interval::ArithmaConfidenceInterval;
pub use distribution_factory::ArithmaDistributionFactory;
pub use normal::ArithmaNormal;
pub use quantile_function::ArithmaQuantileFunction;
pub use statistical_moment::ArithmaStatisticalMoment;
pub use statistical_sampler::ArithmaStatisticalSampler;

/// Natural log of the gamma function, by the Lanczos approximation (g = 7,
/// n = 9). Relative error is under 1e-13 across the range this crate uses.
///
/// Added because [`crate::probabilities::binomial::ln_binomial_coefficient`]
/// otherwise had to sum `min(k, n-k)` terms, which is an unbounded loop over
/// `u64` arguments. With this, the large-argument case is O(1).
///
/// Only defined for `x > 0`; returns `f64::NAN` otherwise, matching libm's
/// `lgamma` for the poles at zero and the negative integers.
pub fn ln_gamma(x: f64) -> f64 {
    if !x.is_finite() || x <= 0.0 {
        return f64::NAN;
    }
    // Lanczos coefficients for g = 7, n = 9.
    const G: f64 = 7.0;
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_5e-7,
    ];
    // The reflection formula is not needed: x > 0 here by the guard above.
    let z = x - 1.0;
    let mut series = COEFFS[0];
    for (i, c) in COEFFS.iter().enumerate().skip(1) {
        series += c / (z + i as f64);
    }
    debug_assert!(series.is_finite(), "Lanczos series diverged");
    let t = z + G + 0.5;
    debug_assert!(t > 0.0, "Lanczos t must be positive for x > 0");
    0.5 * (2.0 * core::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + series.ln()
}

/// The common contract every distribution must implement.
///
/// Methods are intentionally `f64`-typed — distributions own the numeric
/// approximation; the caller is expected to maintain symbolic structure at a
/// higher level via `ArithmaExpression`. Implementations MUST NOT panic;
/// invalid inputs return `Err`.
pub trait ArithmaDistribution {
    /// Probability density function (continuous) or probability mass function
    /// (discrete) at `x`.
    fn pdf(&self, x: f64) -> Result<f64, String>;

    /// Cumulative distribution function `P(X ≤ x)`.
    fn cdf(&self, x: f64) -> Result<f64, String>;

    /// Mean / expected value.
    fn mean(&self) -> Result<f64, String>;

    /// Variance.
    fn variance(&self) -> Result<f64, String>;
}

// ---------------------------------------------------------------------------
// Backward-compatibility aliases for the pre-rename `Arithmos*` names.
// Retained for one release; downstream (eml-math, eml-spectral, metaphysica,
// periodica) should migrate to the `Arithma*` names above.
// ---------------------------------------------------------------------------
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaDistribution`")]
#[allow(unused)]
pub use self::ArithmaDistribution as ArithmosDistribution;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distribution_re_exports_resolve() {
        let _: Option<ArithmaBernoulli> = None;
        let _: Option<ArithmaBinomial> = None;
        let _: Option<ArithmaNormal> = None;
    }
}
