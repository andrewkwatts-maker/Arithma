//! `TermSum`: a sum `Σ c_k · term_k` whose coefficients are named symbols.
//!
//! Coefficients and inputs are listed separately so a caller (a brush) can
//! change coefficients and nothing else.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use super::{ArithmaBindings, ArithmaExpression, Evaluable};
use crate::integer::ArithmaInteger;

/// Why a `TermSum` could not be built or evaluated.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TermSumError {
    /// Two terms share one coefficient name.
    #[error("coefficient `{0}` is used by more than one term")]
    RepeatedCoefficient(String),
    /// A coefficient name also appears as a symbol inside a term.
    #[error("coefficient `{0}` also appears inside a term")]
    ShadowedCoefficient(String),
    /// A coefficient had no value at evaluation.
    #[error("coefficient `{0}` is not bound")]
    UnboundCoefficient(String),
    /// The underlying expression failed to evaluate.
    #[error("evaluation failed: {0}")]
    Evaluation(String),
}

/// `Σ c_k · term_k` with each `c_k` a named symbol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TermSum {
    terms: Vec<(String, ArithmaExpression)>,
}

impl TermSum {
    /// Build from `(coefficient name, term)` pairs.
    ///
    /// Fails if a coefficient name repeats or appears inside any term.
    pub fn new(terms: Vec<(String, ArithmaExpression)>) -> Result<Self, TermSumError> {
        let mut seen = BTreeSet::new();
        for (name, _) in &terms {
            if !seen.insert(name.clone()) {
                return Err(TermSumError::RepeatedCoefficient(name.clone()));
            }
        }
        for (_, term) in &terms {
            if let Some(hit) = term.free_symbols().into_iter().find(|s| seen.contains(s)) {
                return Err(TermSumError::ShadowedCoefficient(hit));
            }
        }
        Ok(Self { terms })
    }

    /// `c0 + c1·x + … + cn·x^n`, coefficients named `{prefix}{k}`.
    pub fn polynomial(input: &str, degree: usize, prefix: &str) -> Result<Self, TermSumError> {
        let terms = (0..=degree)
            .map(|k| (format!("{prefix}{k}"), power_term(input, k)))
            .collect();
        Self::new(terms)
    }

    /// The `(coefficient name, term)` pairs in order.
    pub fn terms(&self) -> &[(String, ArithmaExpression)] {
        &self.terms
    }

    /// Coefficient names in term order.
    pub fn coefficients(&self) -> Vec<String> {
        self.terms.iter().map(|(c, _)| c.clone()).collect()
    }

    /// Free symbols of the terms (the inputs), sorted.
    pub fn inputs(&self) -> Vec<String> {
        let mut all = BTreeSet::new();
        for (_, term) in &self.terms {
            all.extend(term.free_symbols());
        }
        all.into_iter().collect()
    }

    /// `Σ var(c_k) · term_k` as one expression (zero when empty).
    pub fn to_expression(&self) -> ArithmaExpression {
        let mut products = self
            .terms
            .iter()
            .map(|(c, t)| ArithmaExpression::mul(ArithmaExpression::var(c), t.clone()));
        let Some(first) = products.next() else {
            return ArithmaExpression::num(ArithmaInteger::from_i64(0));
        };
        products.fold(first, ArithmaExpression::add)
    }

    /// Evaluate with separate input and coefficient bindings.
    ///
    /// Fails naming the first coefficient with no value.
    pub fn evaluate(
        &self,
        inputs: &ArithmaBindings,
        coefficients: &ArithmaBindings,
    ) -> Result<f64, TermSumError> {
        if let Some((name, _)) = self.terms.iter().find(|(c, _)| !coefficients.contains_key(c)) {
            return Err(TermSumError::UnboundCoefficient(name.clone()));
        }
        let mut merged = inputs.clone();
        merged.extend(coefficients.iter().map(|(k, v)| (k.clone(), *v)));
        self.to_expression()
            .evaluate(&merged)
            .map_err(TermSumError::Evaluation)
    }
}

/// `x^k`, with `k = 0` as 1 and `k = 1` as `x`.
fn power_term(input: &str, k: usize) -> ArithmaExpression {
    let int = |n: i64| ArithmaExpression::num(ArithmaInteger::from_i64(n));
    match k {
        0 => int(1),
        1 => ArithmaExpression::var(input),
        _ => ArithmaExpression::pow(ArithmaExpression::var(input), int(k as i64)),
    }
}
