//====== Arithma/rust/arithma_core/src/probabilities/distribution_factory.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Distribution factory
//!
//! JSON-driven construction of distributions, used by the engine's data-driven
//! initialisation pattern (CLAUDE.md §6).

use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::probabilities::{ArithmaBernoulli, ArithmaBinomial, ArithmaDistribution, ArithmaNormal};

/// Distribution kind as it appears in JSON configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum ArithmaDistributionSpec {
    /// `Normal(mean, std_dev)`.
    Normal { mean: f64, std_dev: f64 },
    /// `Binomial(n, p)`.
    Binomial { n: u64, p: f64 },
    /// `Bernoulli(p)`.
    Bernoulli { p: f64 },
}

/// Factory that turns specs into trait objects.
pub struct ArithmaDistributionFactory;

impl ArithmaDistributionFactory {
    /// Construct a distribution from a spec. Returns a trait object so callers
    /// don't have to know the concrete type at compile time.
    ///
    /// Parameters are **not** validated here — the signature is infallible, and
    /// the distributions themselves reject bad parameters at the point of use
    /// (`pdf`/`cdf` return `Err` for, say, `std_dev = 0` or `p = 1.5`). This
    /// keeps hot-reload from panicking on a malformed config file; the error
    /// surfaces on first evaluation instead.
    pub fn create(spec: &ArithmaDistributionSpec) -> Arc<dyn ArithmaDistribution + Send + Sync> {
        match *spec {
            ArithmaDistributionSpec::Normal { mean, std_dev } => {
                Arc::new(ArithmaNormal::new(mean, std_dev))
            }
            ArithmaDistributionSpec::Binomial { n, p } => Arc::new(ArithmaBinomial::new(n, p)),
            ArithmaDistributionSpec::Bernoulli { p } => Arc::new(ArithmaBernoulli::new(p)),
        }
    }

    /// Construct a distribution from a JSON string. Convenience for hot-reload.
    pub fn from_json(json: &str) -> Result<ArithmaDistributionSpec, String> {
        serde_json::from_str(json).map_err(|e| format!("Failed to parse spec: {e}"))
    }
}

// ---------------------------------------------------------------------------
// Backward-compatibility aliases for the pre-rename `Arithmos*` names.
// Retained for one release; downstream (eml-math, eml-spectral, metaphysica,
// periodica) should migrate to the `Arithma*` names above.
// ---------------------------------------------------------------------------
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaDistributionFactory`")]
#[allow(unused)]
pub use self::ArithmaDistributionFactory as ArithmosDistributionFactory;
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaDistributionSpec`")]
#[allow(unused)]
pub use self::ArithmaDistributionSpec as ArithmosDistributionSpec;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_spec_round_trips_through_json() {
        let spec = ArithmaDistributionSpec::Normal {
            mean: 0.0,
            std_dev: 1.0,
        };
        let json = serde_json::to_string(&spec).unwrap();
        let back = ArithmaDistributionFactory::from_json(&json).unwrap();
        assert!(matches!(back, ArithmaDistributionSpec::Normal { .. }));
    }

    #[test]
    fn create_builds_a_working_normal() {
        let d = ArithmaDistributionFactory::create(&ArithmaDistributionSpec::Normal {
            mean: 0.0,
            std_dev: 1.0,
        });
        assert_eq!(d.mean().unwrap(), 0.0);
        assert_eq!(d.variance().unwrap(), 1.0);
        assert_eq!(d.pdf(0.0).unwrap(), 0.398_942_280_401_432_7);
        assert_eq!(d.cdf(0.0).unwrap(), 0.5);
    }

    #[test]
    fn create_builds_a_working_binomial() {
        let d =
            ArithmaDistributionFactory::create(&ArithmaDistributionSpec::Binomial { n: 5, p: 0.5 });
        assert_eq!(d.mean().unwrap(), 2.5);
        assert_eq!(d.variance().unwrap(), 1.25);
        assert_eq!(d.pdf(2.0).unwrap(), 0.312_5);
        assert_eq!(d.cdf(2.0).unwrap(), 0.5);
    }

    #[test]
    fn create_builds_a_working_bernoulli() {
        let d = ArithmaDistributionFactory::create(&ArithmaDistributionSpec::Bernoulli { p: 0.25 });
        assert_eq!(d.mean().unwrap(), 0.25);
        assert_eq!(d.variance().unwrap(), 0.187_5);
        assert_eq!(d.pdf(1.0).unwrap(), 0.25);
        assert_eq!(d.cdf(0.0).unwrap(), 0.75);
    }

    #[test]
    fn create_goes_end_to_end_from_json() {
        let d = ArithmaDistributionFactory::create(
            &ArithmaDistributionFactory::from_json(r#"{"kind":"binomial","n":10,"p":0.3}"#)
                .unwrap(),
        );
        assert!((d.mean().unwrap() - 3.0).abs() < 1e-15);
        assert!((d.cdf(3.0).unwrap() - 0.649_610_718_4).abs() < 1e-14);
    }

    #[test]
    fn malformed_json_errors_and_bad_parameters_defer_to_evaluation() {
        // Edge case: an unknown kind is a parse error, not a panic.
        assert!(ArithmaDistributionFactory::from_json(r#"{"kind":"cauchy","x0":0.0}"#).is_err());
        assert!(ArithmaDistributionFactory::from_json("not json at all").is_err());

        // A spec with impossible parameters still constructs; the error shows
        // up when the distribution is evaluated.
        let bad = ArithmaDistributionFactory::create(&ArithmaDistributionSpec::Normal {
            mean: 0.0,
            std_dev: 0.0,
        });
        assert!(bad.pdf(0.0).is_err());

        let bad_p =
            ArithmaDistributionFactory::create(&ArithmaDistributionSpec::Bernoulli { p: 2.0 });
        assert!(bad_p.cdf(0.0).is_err());
    }
}
