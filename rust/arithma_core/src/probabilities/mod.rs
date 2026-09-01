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
