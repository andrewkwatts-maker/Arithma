//====== Arithma/rust/arithma_core/src/variable.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Variable
//!
//! `ArithmaVariable` — a named symbol with optional bound value, optional unit,
//! and optional documentation. Variables are resolved through the global
//! constants registry by [`crate::constants::lookup_symbol`].

use serde::{Deserialize, Serialize};

use crate::expression::ArithmaExpression;

/// The runtime value attached to a variable. Either a numeric literal or a
/// symbolic expression — the second form is what enables `x = 2π` style
/// derived variables that retain symbolic structure for further simplification.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum ArithmaVariableValue {
    /// Bound to a literal f64. Used for fast numeric evaluation paths.
    Float(f64),
    /// Bound to a symbolic expression. The expression is evaluated lazily
    /// each time the variable is referenced so simplification can flow through.
    Symbolic(Box<ArithmaExpression>),
    /// Unbound — referencing the variable in an evaluator returns
    /// `Err("unbound variable")`.
    #[default]
    Unbound,
}

/// A named variable with an optional bound value and optional unit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArithmaVariable {
    /// Variable name (the textual symbol used in expressions).
    pub name: String,
    /// Current binding.
    pub value: ArithmaVariableValue,
    /// Optional unit string (e.g. "m/s").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    /// Optional human-readable description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl ArithmaVariable {
    /// Create a new unbound variable with just a name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: ArithmaVariableValue::Unbound,
            unit: None,
            description: None,
        }
    }

    /// Bind the variable to an f64.
    pub fn with_float(mut self, value: f64) -> Self {
        self.value = ArithmaVariableValue::Float(value);
        self
    }

    /// Bind the variable to a symbolic expression.
    pub fn with_symbolic(mut self, expr: ArithmaExpression) -> Self {
        self.value = ArithmaVariableValue::Symbolic(Box::new(expr));
        self
    }

    /// Set the unit string.
    pub fn with_unit(mut self, unit: impl Into<String>) -> Self {
        self.unit = Some(unit.into());
        self
    }

    /// Set the description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Return whether the variable is unbound.
    pub fn is_unbound(&self) -> bool {
        matches!(self.value, ArithmaVariableValue::Unbound)
    }
}

// ---------------------------------------------------------------------------
// Backward-compatibility aliases for the pre-rename `Arithmos*` names.
// Retained for one release; downstream (eml-math, eml-spectral, metaphysica,
// periodica) should migrate to the `Arithma*` names above.
// ---------------------------------------------------------------------------
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaVariable`")]
#[allow(unused)]
pub use self::ArithmaVariable as ArithmosVariable;
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaVariableValue`")]
#[allow(unused)]
pub use self::ArithmaVariableValue as ArithmosVariableValue;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_variable_is_unbound() {
        let v = ArithmaVariable::new("x");
        assert_eq!(v.name, "x");
        assert!(v.is_unbound());
    }

    #[test]
    fn with_float_binds() {
        let v = ArithmaVariable::new("g").with_float(9.81);
        assert!(!v.is_unbound());
        assert!(matches!(v.value, ArithmaVariableValue::Float(_)));
    }

    #[test]
    fn unit_round_trip() {
        let v = ArithmaVariable::new("v").with_unit("m/s");
        assert_eq!(v.unit.as_deref(), Some("m/s"));
    }
}
