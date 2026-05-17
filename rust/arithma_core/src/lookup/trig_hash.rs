//====== Arithma/rust/arithma_core/src/lookup/trig_hash.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! Hash-based canonical-angle lookup for trig functions.
//!
//! ## Migration note
//!
//! The `angle_ids` block below was migrated from
//! `pt-arithmos/src/math/pt_trig_hash_lookup.rs::trig_ids` (Wave 3, plan
//! Â§F.8 step 1). The IDs are the contract â€” same numeric values, same
//! semantics â€” so the engine and any other downstream consumer can route
//! a canonical-angle lookup through either Arithmos (here) or pt-arithmos
//! (legacy) and get the same answer.
//!
//! The `fast_trig::*` lookups that produced `ArithmosExpression` results live
//! in pt-arithmos until `ArithmosExpression` and `ArithmosInteger` graduate
//! from stub status. Once they do, those functions migrate here too.

/// Compile-time constant IDs for ultra-fast trigonometric evaluation. Each
/// ID names a canonical angle in the unit circle. The actual `f64` value of
/// the angle is NOT what's stored â€” these are stable hash slots so multiple
/// expressions referring to the same angle (e.g. `pi/4`, `45 deg in rad`,
/// `cached(0.7853981633974483)`) collapse onto the same lookup key.
pub mod angle_ids {
    /// Hash slot for the angle `0` (zero radians).
    pub const ZERO: u32 = 1000;

    /// Hash slot for `Ï€/6` (30Â°).
    pub const PI_OVER_6: u32 = 1001;
    /// Hash slot for `Ï€/4` (45Â°).
    pub const PI_OVER_4: u32 = 1002;
    /// Hash slot for `Ï€/3` (60Â°).
    pub const PI_OVER_3: u32 = 1003;
    /// Hash slot for `Ï€/2` (90Â°).
    pub const PI_OVER_2: u32 = 1004;
    /// Hash slot for `2Ï€/3` (120Â°).
    pub const TWO_PI_OVER_3: u32 = 1005;
    /// Hash slot for `3Ï€/4` (135Â°).
    pub const THREE_PI_OVER_4: u32 = 1006;
    /// Hash slot for `5Ï€/6` (150Â°).
    pub const FIVE_PI_OVER_6: u32 = 1007;
    /// Hash slot for `Ï€` (180Â°).
    pub const PI: u32 = 1008;
    /// Hash slot for `3Ï€/2` (270Â°).
    pub const THREE_PI_OVER_2: u32 = 1009;
    /// Hash slot for `2Ï€` (360Â°).
    pub const TWO_PI: u32 = 1010;
}

/// Returns the human-readable name of an angle ID, or `None` if the ID is
/// not registered. Useful for debug output and equation-ID texture
/// inspection (plan Â§C).
pub fn angle_name(id: u32) -> Option<&'static str> {
    match id {
        angle_ids::ZERO => Some("0"),
        angle_ids::PI_OVER_6 => Some("Ï€/6"),
        angle_ids::PI_OVER_4 => Some("Ï€/4"),
        angle_ids::PI_OVER_3 => Some("Ï€/3"),
        angle_ids::PI_OVER_2 => Some("Ï€/2"),
        angle_ids::TWO_PI_OVER_3 => Some("2Ï€/3"),
        angle_ids::THREE_PI_OVER_4 => Some("3Ï€/4"),
        angle_ids::FIVE_PI_OVER_6 => Some("5Ï€/6"),
        angle_ids::PI => Some("Ï€"),
        angle_ids::THREE_PI_OVER_2 => Some("3Ï€/2"),
        angle_ids::TWO_PI => Some("2Ï€"),
        _ => None,
    }
}

/// Returns the f64 radian value of an angle ID, or `None` if the ID is not
/// registered. The radians are derived from `std::f64::consts::PI` so they
/// match the standard library's interpretation exactly.
pub fn angle_radians(id: u32) -> Option<f64> {
    use std::f64::consts::PI;
    match id {
        angle_ids::ZERO => Some(0.0),
        angle_ids::PI_OVER_6 => Some(PI / 6.0),
        angle_ids::PI_OVER_4 => Some(PI / 4.0),
        angle_ids::PI_OVER_3 => Some(PI / 3.0),
        angle_ids::PI_OVER_2 => Some(PI / 2.0),
        angle_ids::TWO_PI_OVER_3 => Some(2.0 * PI / 3.0),
        angle_ids::THREE_PI_OVER_4 => Some(3.0 * PI / 4.0),
        angle_ids::FIVE_PI_OVER_6 => Some(5.0 * PI / 6.0),
        angle_ids::PI => Some(PI),
        angle_ids::THREE_PI_OVER_2 => Some(3.0 * PI / 2.0),
        angle_ids::TWO_PI => Some(2.0 * PI),
        _ => None,
    }
}

/// Look up an exact symbolic value of `sin(angle_id)` for a canonical-angle
/// ID. Wave-2 returns a numeric f64 (the underlying value); Wave 3 will
/// switch this to return `ArithmosExpression` once that AST graduates from
/// stub.
pub fn lookup_sin(angle_id: u32) -> Option<f64> {
    angle_radians(angle_id).map(f64::sin)
}

/// Look up an exact symbolic value of `cos(angle_id)` for a canonical-angle
/// ID. See [`lookup_sin`] for migration notes.
pub fn lookup_cos(angle_id: u32) -> Option<f64> {
    angle_radians(angle_id).map(f64::cos)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_distinct() {
        let ids = [
            angle_ids::ZERO,
            angle_ids::PI_OVER_6,
            angle_ids::PI_OVER_4,
            angle_ids::PI_OVER_3,
            angle_ids::PI_OVER_2,
            angle_ids::TWO_PI_OVER_3,
            angle_ids::THREE_PI_OVER_4,
            angle_ids::FIVE_PI_OVER_6,
            angle_ids::PI,
            angle_ids::THREE_PI_OVER_2,
            angle_ids::TWO_PI,
        ];
        let mut seen = std::collections::HashSet::new();
        for id in ids {
            assert!(seen.insert(id), "duplicate angle id {id}");
        }
    }

    #[test]
    fn ids_match_pt_arithmos_contract() {
        // The numeric values are the migration contract with pt-arithmos â€”
        // do not change them without updating the legacy lookup tables too.
        assert_eq!(angle_ids::ZERO, 1000);
        assert_eq!(angle_ids::PI_OVER_6, 1001);
        assert_eq!(angle_ids::PI_OVER_4, 1002);
        assert_eq!(angle_ids::PI_OVER_3, 1003);
        assert_eq!(angle_ids::PI_OVER_2, 1004);
        assert_eq!(angle_ids::PI, 1008);
        assert_eq!(angle_ids::TWO_PI, 1010);
    }

    #[test]
    fn angle_name_round_trips() {
        assert_eq!(angle_name(angle_ids::ZERO), Some("0"));
        assert_eq!(angle_name(angle_ids::PI), Some("Ï€"));
        assert_eq!(angle_name(angle_ids::TWO_PI), Some("2Ï€"));
        assert_eq!(angle_name(0), None);
        assert_eq!(angle_name(9999), None);
    }

    #[test]
    fn angle_radians_match_std_consts() {
        use std::f64::consts::PI;
        assert_eq!(angle_radians(angle_ids::ZERO), Some(0.0));
        assert_eq!(angle_radians(angle_ids::PI_OVER_2), Some(PI / 2.0));
        assert_eq!(angle_radians(angle_ids::PI), Some(PI));
        assert_eq!(angle_radians(angle_ids::TWO_PI), Some(2.0 * PI));
        assert_eq!(angle_radians(0), None);
    }

    #[test]
    fn lookup_sin_canonical_values() {
        let eps = 1e-12;
        assert!((lookup_sin(angle_ids::ZERO).unwrap() - 0.0).abs() < eps);
        assert!((lookup_sin(angle_ids::PI_OVER_2).unwrap() - 1.0).abs() < eps);
        assert!((lookup_sin(angle_ids::PI).unwrap() - 0.0).abs() < eps);
        assert!((lookup_sin(angle_ids::THREE_PI_OVER_2).unwrap() + 1.0).abs() < eps);
        assert!(lookup_sin(0).is_none());
    }

    #[test]
    fn lookup_cos_canonical_values() {
        let eps = 1e-12;
        assert!((lookup_cos(angle_ids::ZERO).unwrap() - 1.0).abs() < eps);
        assert!((lookup_cos(angle_ids::PI_OVER_2).unwrap() - 0.0).abs() < eps);
        assert!((lookup_cos(angle_ids::PI).unwrap() + 1.0).abs() < eps);
        assert!((lookup_cos(angle_ids::TWO_PI).unwrap() - 1.0).abs() < eps);
        assert!(lookup_cos(0).is_none());
    }

    #[test]
    fn lookup_sin_pythagorean_identity() {
        // sinÂ² + cosÂ² = 1 for every canonical angle.
        let eps = 1e-12;
        for id in [
            angle_ids::ZERO, angle_ids::PI_OVER_6, angle_ids::PI_OVER_4,
            angle_ids::PI_OVER_3, angle_ids::PI_OVER_2, angle_ids::TWO_PI_OVER_3,
            angle_ids::THREE_PI_OVER_4, angle_ids::FIVE_PI_OVER_6, angle_ids::PI,
            angle_ids::THREE_PI_OVER_2, angle_ids::TWO_PI,
        ] {
            let s = lookup_sin(id).unwrap();
            let c = lookup_cos(id).unwrap();
            assert!((s * s + c * c - 1.0).abs() < eps, "id={id}");
        }
    }
}

