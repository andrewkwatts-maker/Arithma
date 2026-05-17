//====== Arithma/rust/arithma_core/src/lookup/math_hash.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! General math hash-lookup table for expensive transcendentals.
//!
//! ## Migration note
//!
//! The eight `*_ids` sub-modules below are migrated from
//! `pt-arithmos/src/math/pt_math_hash_lookup.rs::math_ids`. Each ID is a
//! stable hash slot keyed by *name*; downstream consumers always reference
//! the constants symbolically (`algebraic_ids::ZERO_PLUS_X`) rather than by
//! numeric value, so the named constants are the migration contract.
//!
//! In the original source all IDs lived in a single flat namespace, which
//! produced numeric collisions across categories (e.g. `LN_E = 5000` and
//! `LIMIT_SIN_X_OVER_X = 5000`). The collisions are inert because callers
//! match on the constant name, not the integer â€” but we split into
//! per-category sub-modules so a future hash-into-map use site doesn't
//! accidentally lose one. The integer values themselves are preserved
//! verbatim so the equation-ID texture (plan Â§C) keeps writer/reader
//! agreement.
//!
//! The fast-path simplifier functions (`fast_algebra::*`, `fast_integrals::*`,
//! â€¦) that returned `PTExpression` results stay in pt-arithmos for now;
//! they migrate here once `ArithmosExpression` graduates from stub.

// â”€â”€â”€ Algebraic simplifications (2000-series) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Algebraic-identity hash slots. e.g. `ZERO_PLUS_X = 2000` for the rewrite
/// rule `0 + x â†’ x`. The fast-path simplifier consults these to short-circuit
/// trivial expressions before invoking the full simplifier.
pub mod algebraic_ids {
    pub const ZERO_PLUS_X: u32 = 2000;       // 0 + x = x
    pub const X_PLUS_ZERO: u32 = 2001;       // x + 0 = x
    pub const ZERO_TIMES_X: u32 = 2002;      // 0 * x = 0
    pub const X_TIMES_ZERO: u32 = 2003;      // x * 0 = 0
    pub const ONE_TIMES_X: u32 = 2004;       // 1 * x = x
    pub const X_TIMES_ONE: u32 = 2005;       // x * 1 = x
    pub const X_MINUS_X: u32 = 2006;         // x - x = 0
    pub const X_DIVIDED_BY_X: u32 = 2007;    // x / x = 1
    pub const X_DIVIDED_BY_ONE: u32 = 2008;  // x / 1 = x
    pub const ZERO_DIVIDED_BY_X: u32 = 2009; // 0 / x = 0
    pub const X_POW_ZERO: u32 = 2010;        // x^0 = 1
    pub const X_POW_ONE: u32 = 2011;         // x^1 = x
    pub const ONE_POW_X: u32 = 2012;         // 1^x = 1
}

// â”€â”€â”€ Standard integrals (3000-series) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

pub mod integral_ids {
    pub const INTEGRAL_X: u32 = 3000;             // âˆ«x dx = xÂ²/2
    pub const INTEGRAL_X_SQUARED: u32 = 3001;     // âˆ«xÂ² dx = xÂ³/3
    pub const INTEGRAL_ONE_OVER_X: u32 = 3002;    // âˆ«1/x dx = ln(x)
    pub const INTEGRAL_E_POW_X: u32 = 3003;       // âˆ«e^x dx = e^x
    pub const INTEGRAL_SIN_X: u32 = 3004;         // âˆ«sin(x) dx = -cos(x)
    pub const INTEGRAL_COS_X: u32 = 3005;         // âˆ«cos(x) dx = sin(x)
    pub const INTEGRAL_TAN_X: u32 = 3006;         // âˆ«tan(x) dx = -ln(|cos(x)|)
    pub const INTEGRAL_SEC_SQUARED_X: u32 = 3007; // âˆ«secÂ²(x) dx = tan(x)
    pub const INTEGRAL_ONE: u32 = 3008;           // âˆ«1 dx = x
    pub const INTEGRAL_SQRT_X: u32 = 3009;        // âˆ«âˆšx dx = (2/3)x^(3/2)
}

// â”€â”€â”€ Derivatives (4000-series) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

pub mod derivative_ids {
    pub const DERIVATIVE_X: u32 = 4000;          // d/dx[x] = 1
    pub const DERIVATIVE_X_SQUARED: u32 = 4001;  // d/dx[xÂ²] = 2x
    pub const DERIVATIVE_SIN_X: u32 = 4002;      // d/dx[sin(x)] = cos(x)
    pub const DERIVATIVE_COS_X: u32 = 4003;      // d/dx[cos(x)] = -sin(x)
    pub const DERIVATIVE_TAN_X: u32 = 4004;      // d/dx[tan(x)] = secÂ²(x)
    pub const DERIVATIVE_E_POW_X: u32 = 4005;    // d/dx[e^x] = e^x
    pub const DERIVATIVE_LN_X: u32 = 4006;       // d/dx[ln(x)] = 1/x
    pub const DERIVATIVE_CONSTANT: u32 = 4007;   // d/dx[c] = 0
}

// â”€â”€â”€ Standard limits (5000-series) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

pub mod limit_ids {
    pub const LIMIT_SIN_X_OVER_X: u32 = 5000;             // lim[xâ†’0] sin(x)/x = 1
    pub const LIMIT_TAN_X_OVER_X: u32 = 5001;             // lim[xâ†’0] tan(x)/x = 1
    pub const LIMIT_EXP_MINUS_ONE_OVER_X: u32 = 5002;     // lim[xâ†’0] (e^x-1)/x = 1
    pub const LIMIT_LN_ONE_PLUS_X_OVER_X: u32 = 5003;     // lim[xâ†’0] ln(1+x)/x = 1
    pub const LIMIT_ONE_MINUS_COS_X_OVER_X_SQ: u32 = 5004;// lim[xâ†’0] (1-cos(x))/xÂ² = 1/2
    pub const LIMIT_COMPOUND_INTEREST: u32 = 5005;        // lim[nâ†’âˆž] (1+1/n)^n = e
}

// â”€â”€â”€ Standard summations (6000-series) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

pub mod sum_ids {
    pub const SUM_LINEAR: u32 = 6000;             // Î£[k=1..n] k = n(n+1)/2
    pub const SUM_SQUARES: u32 = 6001;            // Î£[k=1..n] kÂ² = n(n+1)(2n+1)/6
    pub const SUM_CUBES: u32 = 6002;              // Î£[k=1..n] kÂ³ = (n(n+1)/2)Â²
    pub const SUM_GEOMETRIC: u32 = 6003;          // Î£[k=0..n] ar^k = a(1-r^(n+1))/(1-r)
    pub const SUM_INFINITE_GEOMETRIC: u32 = 6004; // Î£[k=0..âˆž] ar^k = a/(1-r) when |r|<1
    pub const SUM_POWERS_OF_TWO: u32 = 6005;      // Î£[k=0..n] 2^k = 2^(n+1) - 1
    pub const SUM_HARMONIC: u32 = 6006;           // Î£[k=1..n] 1/k â‰ˆ ln(n) + Î³
    pub const SUM_RECIPROCAL_SQUARES: u32 = 6007; // Î£[k=1..âˆž] 1/kÂ² = Ï€Â²/6 (Basel)
}

// â”€â”€â”€ Logarithmic simplifications (collides with limit_ids in legacy) â”€â”€â”€â”€â”€â”€â”€
//
// These reuse the 5000-series in pt-arithmos' flat namespace; preserved here
// in the same numeric range so the equation-ID texture writer in
// pt-arithmos and the reader in arithmos_core stay in lock-step. Use the
// constant names, not the numeric values, when matching.
pub mod logarithm_ids {
    pub const LN_E: u32 = 5000;          // ln(e) = 1
    pub const LN_ONE: u32 = 5001;        // ln(1) = 0
    pub const LOG_SAME_BASE: u32 = 5002; // log_b(b) = 1
}

// â”€â”€â”€ Exponential simplifications (collides with sum_ids in legacy) â”€â”€â”€â”€â”€â”€â”€â”€â”€

pub mod exponential_ids {
    pub const E_POW_ZERO: u32 = 6000; // e^0 = 1
    pub const E_POW_ONE: u32 = 6001;  // e^1 = e
    pub const E_POW_LN_X: u32 = 6002; // e^(ln(x)) = x
}

// â”€â”€â”€ Square-root simplifications (7000-series) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

pub mod sqrt_ids {
    pub const SQRT_ZERO: u32 = 7000;       // âˆš0 = 0
    pub const SQRT_ONE: u32 = 7001;        // âˆš1 = 1
    pub const SQRT_FOUR: u32 = 7002;       // âˆš4 = 2
    pub const SQRT_NINE: u32 = 7003;       // âˆš9 = 3
    pub const SQRT_X_SQUARED: u32 = 7004;  // âˆš(xÂ²) = |x|
}

// â”€â”€â”€ Category tagging â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Category an `*_ids` constant belongs to. Useful for the equation-ID
/// texture writer when it needs to disambiguate same-numbered IDs across
/// the legacy flat namespace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MathIdKind {
    Algebraic,
    Integral,
    Derivative,
    Limit,
    Sum,
    Logarithm,
    Exponential,
    Sqrt,
}

/// Derive a category tag from a sub-module's range. Algebraic / integral /
/// derivative / sqrt are unique by range; the 5000/6000 series are
/// ambiguous (Limit vs Logarithm; Sum vs Exponential) and require an
/// out-of-band hint â€” here we default to the *first* category that owns the
/// range. Callers that need the disambiguation should use the `*_ids`
/// constants directly rather than this function.
pub fn classify(id: u32) -> Option<MathIdKind> {
    match id {
        2000..=2099 => Some(MathIdKind::Algebraic),
        3000..=3099 => Some(MathIdKind::Integral),
        4000..=4099 => Some(MathIdKind::Derivative),
        5000..=5099 => Some(MathIdKind::Limit),       // shadows Logarithm at 5000-5002
        6000..=6099 => Some(MathIdKind::Sum),         // shadows Exponential at 6000-6002
        7000..=7099 => Some(MathIdKind::Sqrt),
        _ => None,
    }
}

/// Look up an exact symbolic form for `exp(x)` if `x` matches a known
/// special value. Returns `None` otherwise. Wave-2 returns the f64 value;
/// Wave-3 will return an `ArithmosExpression` once that AST graduates.
pub fn lookup_exp(x: f64) -> Option<f64> {
    if x == 0.0 { Some(1.0) }
    else if x == 1.0 { Some(std::f64::consts::E) }
    else if x.is_nan() { None }
    else { None }
}

/// Look up an exact symbolic form for `ln(x)` if `x` matches a known
/// special value.
pub fn lookup_ln(x: f64) -> Option<f64> {
    if x == 1.0 { Some(0.0) }
    else if x == std::f64::consts::E { Some(1.0) }
    else { None }
}

/// Look up an exact symbolic form for `sqrt(x)` if `x` matches a known
/// special value.
pub fn lookup_sqrt(x: f64) -> Option<f64> {
    if x == 0.0 { Some(0.0) }
    else if x == 1.0 { Some(1.0) }
    else if x == 4.0 { Some(2.0) }
    else if x == 9.0 { Some(3.0) }
    else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_algebraic_range() {
        assert_eq!(classify(algebraic_ids::ZERO_PLUS_X), Some(MathIdKind::Algebraic));
        assert_eq!(classify(algebraic_ids::ONE_POW_X), Some(MathIdKind::Algebraic));
    }

    #[test]
    fn classify_integral_range() {
        assert_eq!(classify(integral_ids::INTEGRAL_X), Some(MathIdKind::Integral));
        assert_eq!(classify(integral_ids::INTEGRAL_SQRT_X), Some(MathIdKind::Integral));
    }

    #[test]
    fn classify_derivative_range() {
        assert_eq!(classify(derivative_ids::DERIVATIVE_X), Some(MathIdKind::Derivative));
        assert_eq!(classify(derivative_ids::DERIVATIVE_CONSTANT), Some(MathIdKind::Derivative));
    }

    #[test]
    fn classify_returns_none_outside_ranges() {
        assert_eq!(classify(0), None);
        assert_eq!(classify(999), None);
        assert_eq!(classify(8000), None);
        assert_eq!(classify(u32::MAX), None);
    }

    #[test]
    fn legacy_collisions_preserved() {
        // The migration contract: same numeric values as pt-arithmos'
        // flat namespace, so the equation-ID texture stays compatible.
        assert_eq!(limit_ids::LIMIT_SIN_X_OVER_X, logarithm_ids::LN_E);
        assert_eq!(sum_ids::SUM_LINEAR, exponential_ids::E_POW_ZERO);
    }

    #[test]
    fn algebraic_ids_are_distinct_within_category() {
        let ids = [
            algebraic_ids::ZERO_PLUS_X, algebraic_ids::X_PLUS_ZERO,
            algebraic_ids::ZERO_TIMES_X, algebraic_ids::X_TIMES_ZERO,
            algebraic_ids::ONE_TIMES_X, algebraic_ids::X_TIMES_ONE,
            algebraic_ids::X_MINUS_X, algebraic_ids::X_DIVIDED_BY_X,
            algebraic_ids::X_DIVIDED_BY_ONE, algebraic_ids::ZERO_DIVIDED_BY_X,
            algebraic_ids::X_POW_ZERO, algebraic_ids::X_POW_ONE,
            algebraic_ids::ONE_POW_X,
        ];
        let mut seen = std::collections::HashSet::new();
        for id in ids { assert!(seen.insert(id)); }
    }

    #[test]
    fn lookup_exp_canonical() {
        assert_eq!(lookup_exp(0.0), Some(1.0));
        assert!((lookup_exp(1.0).unwrap() - std::f64::consts::E).abs() < 1e-15);
        assert_eq!(lookup_exp(0.5), None);
        assert_eq!(lookup_exp(f64::NAN), None);
    }

    #[test]
    fn lookup_ln_canonical() {
        assert_eq!(lookup_ln(1.0), Some(0.0));
        assert_eq!(lookup_ln(std::f64::consts::E), Some(1.0));
        assert_eq!(lookup_ln(0.5), None);
    }

    #[test]
    fn lookup_sqrt_canonical() {
        assert_eq!(lookup_sqrt(0.0), Some(0.0));
        assert_eq!(lookup_sqrt(1.0), Some(1.0));
        assert_eq!(lookup_sqrt(4.0), Some(2.0));
        assert_eq!(lookup_sqrt(9.0), Some(3.0));
        assert_eq!(lookup_sqrt(2.0), None);
    }

    #[test]
    fn id_ranges_match_pt_arithmos_contract() {
        // Numeric values are the migration contract â€” do not change without
        // updating pt-arithmos' lookup table in lock-step.
        assert_eq!(algebraic_ids::ZERO_PLUS_X, 2000);
        assert_eq!(integral_ids::INTEGRAL_X, 3000);
        assert_eq!(derivative_ids::DERIVATIVE_X, 4000);
        assert_eq!(limit_ids::LIMIT_SIN_X_OVER_X, 5000);
        assert_eq!(sum_ids::SUM_LINEAR, 6000);
        assert_eq!(sqrt_ids::SQRT_ZERO, 7000);
    }
}

