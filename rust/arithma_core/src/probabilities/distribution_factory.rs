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

use crate::probabilities::ArithmaDistribution;

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
    pub fn create(_spec: &ArithmaDistributionSpec) -> Arc<dyn ArithmaDistribution + Send + Sync> {
        unimplemented!("ArithmaDistributionFactory::create — populated in Wave 3")
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
}
