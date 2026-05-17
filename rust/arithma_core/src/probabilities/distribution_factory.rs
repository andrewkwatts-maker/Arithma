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
//! initialisation pattern (CLAUDE.md Â§6).

use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::probabilities::ArithmosDistribution;

/// Distribution kind as it appears in JSON configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum ArithmosDistributionSpec {
    /// `Normal(mean, std_dev)`.
    Normal { mean: f64, std_dev: f64 },
    /// `Binomial(n, p)`.
    Binomial { n: u64, p: f64 },
    /// `Bernoulli(p)`.
    Bernoulli { p: f64 },
}

/// Factory that turns specs into trait objects.
pub struct ArithmosDistributionFactory;

impl ArithmosDistributionFactory {
    /// Construct a distribution from a spec. Returns a trait object so callers
    /// don't have to know the concrete type at compile time.
    pub fn create(_spec: &ArithmosDistributionSpec) -> Arc<dyn ArithmosDistribution + Send + Sync> {
        unimplemented!("ArithmosDistributionFactory::create â€” populated in Wave 3")
    }

    /// Construct a distribution from a JSON string. Convenience for hot-reload.
    pub fn from_json(json: &str) -> Result<ArithmosDistributionSpec, String> {
        serde_json::from_str(json).map_err(|e| format!("Failed to parse spec: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_spec_round_trips_through_json() {
        let spec = ArithmosDistributionSpec::Normal {
            mean: 0.0,
            std_dev: 1.0,
        };
        let json = serde_json::to_string(&spec).unwrap();
        let back = ArithmosDistributionFactory::from_json(&json).unwrap();
        assert!(matches!(back, ArithmosDistributionSpec::Normal { .. }));
    }
}

