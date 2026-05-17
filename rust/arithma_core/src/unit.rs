//====== Arithma/rust/arithma_core/src/unit.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! Unit-of-measure types for dimensional analysis. Wave-2 placeholder.

use serde::{Deserialize, Serialize};

/// A unit of measure (e.g. "meter", "kilogram"). Composite units like
/// "mÂ·sâ»Â¹" are stored as a `Vec<(ArithmosUnit, i32)>` exponent list elsewhere;
/// this struct represents one base or derived unit.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ArithmosUnit {
    /// SI symbol (e.g. "m", "kg").
    pub symbol: String,
    /// Long-form name (e.g. "meter").
    pub name: String,
}

impl ArithmosUnit {
    pub fn new(symbol: impl Into<String>, name: impl Into<String>) -> Self {
        Self { symbol: symbol.into(), name: name.into() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_constructs_unit() {
        let u = ArithmosUnit::new("m", "meter");
        assert_eq!(u.symbol, "m");
        assert_eq!(u.name, "meter");
    }
}

