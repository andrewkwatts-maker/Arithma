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
//! is the [`ArithmosDistribution`] trait â€” every concrete distribution
//! (Normal, Binomial, Bernoulli, â€¦) implements it so downstream code can be
//! generic over the kind.
//!
//! ## Submodules
//!
//! - [`bernoulli`], [`binomial`], [`normal`] â€” concrete distributions.
//! - [`distribution_factory`] â€” JSON-driven distribution constructor.
//! - [`quantile_function`] â€” inverse-CDF helpers.
//! - [`confidence_interval`] â€” interval estimation.
//! - [`statistical_moment`] â€” variance / skew / kurtosis.
//! - [`statistical_sampler`] â€” generic sampling driver.

pub mod bernoulli;
pub mod binomial;
pub mod confidence_interval;
pub mod distribution_factory;
pub mod normal;
pub mod quantile_function;
pub mod statistical_moment;
pub mod statistical_sampler;

pub use bernoulli::ArithmosBernoulli;
pub use binomial::ArithmosBinomial;
pub use confidence_interval::ArithmosConfidenceInterval;
pub use distribution_factory::ArithmosDistributionFactory;
pub use normal::ArithmosNormal;
pub use quantile_function::ArithmosQuantileFunction;
pub use statistical_moment::ArithmosStatisticalMoment;
pub use statistical_sampler::ArithmosStatisticalSampler;

/// The common contract every distribution must implement.
///
/// Methods are intentionally `f64`-typed â€” distributions own the numeric
/// approximation; the caller is expected to maintain symbolic structure at a
/// higher level via `ArithmosExpression`. Implementations MUST NOT panic;
/// invalid inputs return `Err`.
pub trait ArithmosDistribution {
    /// Probability density function (continuous) or probability mass function
    /// (discrete) at `x`.
    fn pdf(&self, x: f64) -> Result<f64, String>;

    /// Cumulative distribution function `P(X â‰¤ x)`.
    fn cdf(&self, x: f64) -> Result<f64, String>;

    /// Mean / expected value.
    fn mean(&self) -> Result<f64, String>;

    /// Variance.
    fn variance(&self) -> Result<f64, String>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distribution_re_exports_resolve() {
        let _: Option<ArithmosBernoulli> = None;
        let _: Option<ArithmosBinomial> = None;
        let _: Option<ArithmosNormal> = None;
    }
}

