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
//! `ArithmosVariable` â€” a named symbol with optional bound value, optional unit,
//! and optional documentation. Variables are resolved through the global
//! constants registry by [`crate::constants::lookup_symbol`].

use serde::{Deserialize, Serialize};

use crate::expression::ArithmosExpression;

/// The runtime value attached to a variable. Either a numeric literal or a
/// symbolic expression â€” the second form is what enables `x = 2Ï€` style
/// derived variables that retain symbolic structure for further simplification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArithmosVariableValue {
    /// Bound to a literal f64. Used for fast numeric evaluation paths.
    Float(f64),
    /// Bound to a symbolic expression. The expression is evaluated lazily
    /// each time the variable is referenced so simplification can flow through.
    Symbolic(Box<ArithmosExpression>),
    /// Unbound â€” referencing the variable in an evaluator returns
    /// `Err("unbound variable")`.
    Unbound,
}

impl Default for ArithmosVariableValue {
    fn default() -> Self {
        Self::Unbound
    }
}

/// A named variable with an optional bound value and optional unit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArithmosVariable {
    /// Variable name (the textual symbol used in expressions).
    pub name: String,
    /// Current binding.
    pub value: ArithmosVariableValue,
    /// Optional unit string (e.g. "m/s").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    /// Optional human-readable description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl ArithmosVariable {
    /// Create a new unbound variable with just a name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: ArithmosVariableValue::Unbound,
            unit: None,
            description: None,
        }
    }

    /// Bind the variable to an f64.
    pub fn with_float(mut self, value: f64) -> Self {
        self.value = ArithmosVariableValue::Float(value);
        self
    }

    /// Bind the variable to a symbolic expression.
    pub fn with_symbolic(mut self, expr: ArithmosExpression) -> Self {
        self.value = ArithmosVariableValue::Symbolic(Box::new(expr));
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
        matches!(self.value, ArithmosVariableValue::Unbound)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_variable_is_unbound() {
        let v = ArithmosVariable::new("x");
        assert_eq!(v.name, "x");
        assert!(v.is_unbound());
    }

    #[test]
    fn with_float_binds() {
        let v = ArithmosVariable::new("g").with_float(9.81);
        assert!(!v.is_unbound());
        assert!(matches!(v.value, ArithmosVariableValue::Float(_)));
    }

    #[test]
    fn unit_round_trip() {
        let v = ArithmosVariable::new("v").with_unit("m/s");
        assert_eq!(v.unit.as_deref(), Some("m/s"));
    }
}

