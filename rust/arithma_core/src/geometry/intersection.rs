//====== Arithma/rust/arithma_core/src/geometry/intersection.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Intersection
//!
//! Intersection routines between geometry primitives. All results carry
//! symbolic parameters so they survive through the simplifier and the Fourier
//! pipeline.

use crate::expression::{ArithmaBindings, ArithmaExpression, Evaluable};
use crate::geometry::line::ArithmaLine;
use crate::geometry::plane::ArithmaPlane;
use crate::geometry::sphere::ArithmaSphere;
use crate::geometry::vector::ArithmaVector;

/// Outcome of an intersection test. Variants cover both empty and one-or-more-
/// hit cases.
#[derive(Debug, Clone)]
pub enum ArithmaIntersectionResult {
    /// No intersection.
    None,
    /// Single hit at the given point.
    Point(ArithmaVector),
    /// Two hits — typical for ray/sphere.
    TwoPoints(ArithmaVector, ArithmaVector),
    /// Continuous overlap (e.g. line lies in plane).
    Continuous,
}

/// Tolerance used when a branch decision genuinely requires a numeric answer
/// (parallel tests, discriminant sign). Kept tight because the inputs to these
/// routines are exact literals far more often than measured floats.
const ARITHMA_GEOMETRY_EPSILON: f64 = 1.0e-12;

/// Classification of a scalar expression for the purpose of picking an
/// intersection branch.
///
/// See [`classify`] for the convention: only expressions that reduce to a
/// number against an *empty* binding map are classified; anything that stays
/// symbolic is [`Sign::Unknown`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Sign {
    /// Reduces to a value within [`ARITHMA_GEOMETRY_EPSILON`] of zero.
    Zero,
    /// Reduces to a value greater than `+ARITHMA_GEOMETRY_EPSILON`.
    Positive,
    /// Reduces to a value less than `-ARITHMA_GEOMETRY_EPSILON`.
    Negative,
    /// Does not reduce to a finite number with no bindings — it is symbolic,
    /// contains free variables, or evaluation failed. Callers treat this as the
    /// *generic* (non-degenerate) case.
    Unknown,
}

/// Classify a scalar expression by evaluating it against an empty binding map.
///
/// This is the single place where these routines make a numeric decision. Every
/// degenerate branch (line parallel to a plane, tangent sphere hit, parallel
/// planes) is a discrete choice of return variant and therefore cannot be
/// expressed structurally in `ArithmaIntersectionResult`, so the deciding scalar
/// is evaluated here. An expression that still contains free variables — or that
/// evaluation rejects for any other reason — yields [`Sign::Unknown`] and the
/// caller takes the general-position branch, which keeps the symbolic answer
/// intact instead of collapsing it to `None`.
fn classify(expr: &ArithmaExpression) -> Sign {
    match expr.evaluate(&ArithmaBindings::new()) {
        Ok(v) if v.is_finite() => {
            if v.abs() <= ARITHMA_GEOMETRY_EPSILON {
                Sign::Zero
            } else if v > 0.0 {
                Sign::Positive
            } else {
                Sign::Negative
            }
        }
        _ => Sign::Unknown,
    }
}

/// Static collection of intersection routines.
pub struct ArithmaIntersection;

impl ArithmaIntersection {
    /// Line vs plane intersection.
    ///
    /// Solves `normal · (origin + t·direction) = offset` for the ray parameter
    ///
    /// ```text
    /// t = (offset - normal · origin) / (normal · direction)
    /// ```
    ///
    /// and returns [`ArithmaIntersectionResult::Point`] at `line.at(t)`.
    ///
    /// Degenerate cases follow the [`classify`] convention:
    ///
    /// - denominator zero, numerator zero — the line lies *in* the plane, so
    ///   every `t` is a hit: [`ArithmaIntersectionResult::Continuous`].
    /// - denominator zero, numerator non-zero — the line is parallel to but off
    ///   the plane: [`ArithmaIntersectionResult::None`].
    /// - denominator not numerically decidable (symbolic) — the general case is
    ///   assumed and a symbolic `Point` is returned.
    pub fn line_plane(line: &ArithmaLine, plane: &ArithmaPlane) -> ArithmaIntersectionResult {
        let denominator = plane.normal.dot(&line.direction);
        let numerator =
            ArithmaExpression::sub(plane.offset.clone(), plane.normal.dot(&line.origin));

        if classify(&denominator) == Sign::Zero {
            return if classify(&numerator) == Sign::Zero {
                ArithmaIntersectionResult::Continuous
            } else {
                ArithmaIntersectionResult::None
            };
        }

        let t = ArithmaExpression::div(numerator, denominator);
        ArithmaIntersectionResult::Point(line.at(t))
    }

    /// Line vs sphere intersection.
    ///
    /// Substituting `P(t) = origin + t·direction` into `|P - centre|² = r²`
    /// gives the quadratic `a t² + b t + c = 0` with
    ///
    /// ```text
    /// oc = origin - centre
    /// a  = direction · direction
    /// b  = 2 (oc · direction)
    /// c  = oc · oc - r²
    /// ```
    ///
    /// The discriminant `b² - 4ac` selects the branch, using the [`classify`]
    /// convention:
    ///
    /// - negative — the line misses: [`ArithmaIntersectionResult::None`].
    /// - zero — tangent, one hit at `t = -b / 2a`:
    ///   [`ArithmaIntersectionResult::Point`].
    /// - positive *or* symbolic — two hits at `t = (-b ∓ √disc) / 2a`:
    ///   [`ArithmaIntersectionResult::TwoPoints`], ordered by increasing `t`.
    ///   Treating the symbolic case as two points is what keeps a fully
    ///   symbolic query from being silently reported as a miss; the caller can
    ///   inspect the discriminant themselves via the returned coordinates.
    ///
    /// A direction whose squared length is numerically zero is not a line at
    /// all, so it returns [`ArithmaIntersectionResult::None`].
    pub fn line_sphere(line: &ArithmaLine, sphere: &ArithmaSphere) -> ArithmaIntersectionResult {
        let oc = line.origin.sub_vec(&sphere.centre);
        let a = line.direction.magnitude_squared();
        if classify(&a) == Sign::Zero {
            return ArithmaIntersectionResult::None;
        }

        let b = ArithmaExpression::mul(ArithmaExpression::from_i64(2), oc.dot(&line.direction));
        let c = ArithmaExpression::sub(oc.magnitude_squared(), sphere.radius_squared());

        let discriminant = ArithmaExpression::sub(
            ArithmaExpression::mul(b.clone(), b.clone()),
            ArithmaExpression::mul(
                ArithmaExpression::mul(ArithmaExpression::from_i64(4), a.clone()),
                c,
            ),
        );

        let two_a = ArithmaExpression::mul(ArithmaExpression::from_i64(2), a);
        let neg_b = ArithmaExpression::neg(b);

        match classify(&discriminant) {
            Sign::Negative => ArithmaIntersectionResult::None,
            Sign::Zero => {
                let t = ArithmaExpression::div(neg_b, two_a);
                ArithmaIntersectionResult::Point(line.at(t))
            }
            Sign::Positive | Sign::Unknown => {
                let root = ArithmaExpression::sqrt(discriminant);
                let t_near = ArithmaExpression::div(
                    ArithmaExpression::sub(neg_b.clone(), root.clone()),
                    two_a.clone(),
                );
                let t_far = ArithmaExpression::div(ArithmaExpression::add(neg_b, root), two_a);
                ArithmaIntersectionResult::TwoPoints(line.at(t_near), line.at(t_far))
            }
        }
    }

    /// Plane vs plane intersection, as a line.
    ///
    /// The direction is `n₁ × n₂`. A point on the line is found by solving in
    /// the `{n₁, n₂}` basis:
    ///
    /// ```text
    /// det = (n₁·n₁)(n₂·n₂) - (n₁·n₂)²      [ = |n₁ × n₂|² ]
    /// c₁  = (d₁ (n₂·n₂) - d₂ (n₁·n₂)) / det
    /// c₂  = (d₂ (n₁·n₁) - d₁ (n₁·n₂)) / det
    /// p₀  = c₁ n₁ + c₂ n₂
    /// ```
    ///
    /// Returns `None` when `det` classifies as [`Sign::Zero`] — the normals are
    /// parallel (or one is degenerate), so the planes are either coincident or
    /// disjoint and no unique line exists. A `det` that stays symbolic is
    /// treated as non-degenerate and a symbolic line is returned.
    pub fn plane_plane(a: &ArithmaPlane, b: &ArithmaPlane) -> Option<ArithmaLine> {
        let n1 = &a.normal;
        let n2 = &b.normal;

        let n1n1 = n1.magnitude_squared();
        let n2n2 = n2.magnitude_squared();
        let n1n2 = n1.dot(n2);

        let det = ArithmaExpression::sub(
            ArithmaExpression::mul(n1n1.clone(), n2n2.clone()),
            ArithmaExpression::mul(n1n2.clone(), n1n2.clone()),
        );
        if classify(&det) == Sign::Zero {
            return None;
        }

        let c1 = ArithmaExpression::div(
            ArithmaExpression::sub(
                ArithmaExpression::mul(a.offset.clone(), n2n2),
                ArithmaExpression::mul(b.offset.clone(), n1n2.clone()),
            ),
            det.clone(),
        );
        let c2 = ArithmaExpression::div(
            ArithmaExpression::sub(
                ArithmaExpression::mul(b.offset.clone(), n1n1),
                ArithmaExpression::mul(a.offset.clone(), n1n2),
            ),
            det,
        );

        let origin = n1.scale(&c1).add_vec(&n2.scale(&c2));
        Some(ArithmaLine::new(origin, n1.cross(n2)))
    }

    /// Closest-point parameter `t` on a line for a target point.
    ///
    /// Projection of `point - origin` onto `direction`, normalised by
    /// `|direction|²`:
    ///
    /// ```text
    /// t = ((point - origin) · direction) / (direction · direction)
    /// ```
    ///
    /// Feeding the result back through [`ArithmaLine::at`] gives the foot of
    /// the perpendicular.
    ///
    /// Degenerate case: a zero-length direction leaves a division by zero in
    /// the expression. It is built structurally rather than rejected here, so
    /// symbolic directions are never dropped; the failure surfaces as an `Err`
    /// from [`crate::expression::Evaluable`] when the caller evaluates.
    pub fn closest_point_param(line: &ArithmaLine, point: &ArithmaVector) -> ArithmaExpression {
        let offset = point.sub_vec(&line.origin);
        ArithmaExpression::div(
            offset.dot(&line.direction),
            line.direction.magnitude_squared(),
        )
    }
}

// ---------------------------------------------------------------------------
// Backward-compatibility aliases for the pre-rename `Arithmos*` names.
// Retained for one release; downstream (eml-math, eml-spectral, metaphysica,
// periodica) should migrate to the `Arithma*` names above.
// ---------------------------------------------------------------------------
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaIntersection`")]
#[allow(unused)]
pub use self::ArithmaIntersection as ArithmosIntersection;
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaIntersectionResult`")]
#[allow(unused)]
pub use self::ArithmaIntersectionResult as ArithmosIntersectionResult;

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: i64, y: i64, z: i64) -> ArithmaVector {
        ArithmaVector::new(
            ArithmaExpression::from_i64(x),
            ArithmaExpression::from_i64(y),
            ArithmaExpression::from_i64(z),
        )
    }

    fn num(e: &ArithmaExpression) -> f64 {
        e.evaluate(&ArithmaBindings::new())
            .expect("expression should evaluate numerically")
    }

    /// Assert a vector's components equal the given floats.
    fn assert_vec(got: &ArithmaVector, x: f64, y: f64, z: f64) {
        let (gx, gy, gz) = (num(&got.x), num(&got.y), num(&got.z));
        assert!(
            (gx - x).abs() < 1e-9 && (gy - y).abs() < 1e-9 && (gz - z).abs() < 1e-9,
            "expected ({x}, {y}, {z}), got ({gx}, {gy}, {gz})"
        );
    }

    #[test]
    fn intersection_none_variant_constructs() {
        let r = ArithmaIntersectionResult::None;
        assert!(matches!(r, ArithmaIntersectionResult::None));
    }

    // ----- line_plane -----

    #[test]
    fn line_plane_hits_at_the_expected_point() {
        // Line down the z axis from (0,0,10); plane z = 0.
        let line = ArithmaLine::new(v(0, 0, 10), v(0, 0, -1));
        let plane = ArithmaPlane::new(v(0, 0, 1), ArithmaExpression::zero());
        match ArithmaIntersection::line_plane(&line, &plane) {
            ArithmaIntersectionResult::Point(p) => assert_vec(&p, 0.0, 0.0, 0.0),
            other => panic!("expected Point, got {other:?}"),
        }
    }

    #[test]
    fn line_plane_hits_an_offset_plane_off_axis() {
        // Line (1,2,0) + t(0,0,2); plane z = 6.
        let line = ArithmaLine::new(v(1, 2, 0), v(0, 0, 2));
        let plane = ArithmaPlane::new(v(0, 0, 1), ArithmaExpression::from_i64(6));
        match ArithmaIntersection::line_plane(&line, &plane) {
            ArithmaIntersectionResult::Point(p) => assert_vec(&p, 1.0, 2.0, 6.0),
            other => panic!("expected Point, got {other:?}"),
        }
    }

    #[test]
    fn line_plane_oblique_hit() {
        // Line (0,0,0) + t(1,1,1); plane x + y + z = 3 -> t = 1 -> (1,1,1).
        let line = ArithmaLine::new(v(0, 0, 0), v(1, 1, 1));
        let plane = ArithmaPlane::new(v(1, 1, 1), ArithmaExpression::from_i64(3));
        match ArithmaIntersection::line_plane(&line, &plane) {
            ArithmaIntersectionResult::Point(p) => assert_vec(&p, 1.0, 1.0, 1.0),
            other => panic!("expected Point, got {other:?}"),
        }
    }

    #[test]
    fn line_plane_parallel_and_off_the_plane_misses() {
        // Line at z = 5 travelling along x; plane z = 0.
        let line = ArithmaLine::new(v(0, 0, 5), v(1, 0, 0));
        let plane = ArithmaPlane::new(v(0, 0, 1), ArithmaExpression::zero());
        assert!(matches!(
            ArithmaIntersection::line_plane(&line, &plane),
            ArithmaIntersectionResult::None
        ));
    }

    #[test]
    fn line_lying_in_the_plane_is_continuous() {
        let line = ArithmaLine::new(v(3, 4, 0), v(1, 0, 0));
        let plane = ArithmaPlane::new(v(0, 0, 1), ArithmaExpression::zero());
        assert!(matches!(
            ArithmaIntersection::line_plane(&line, &plane),
            ArithmaIntersectionResult::Continuous
        ));
    }

    #[test]
    fn line_plane_with_symbolic_direction_takes_the_generic_branch() {
        let line = ArithmaLine::new(
            v(0, 0, 0),
            ArithmaVector::new(
                ArithmaExpression::zero(),
                ArithmaExpression::zero(),
                ArithmaExpression::var("dz"),
            ),
        );
        let plane = ArithmaPlane::new(v(0, 0, 1), ArithmaExpression::from_i64(6));
        match ArithmaIntersection::line_plane(&line, &plane) {
            ArithmaIntersectionResult::Point(p) => {
                let mut bindings = ArithmaBindings::new();
                bindings.insert("dz".to_string(), 3.0);
                // t = 6/3 = 2, z = 2 * 3 = 6.
                let got = p.z.evaluate(&bindings).expect("bound");
                assert!((got - 6.0).abs() < 1e-9, "got {got}");
            }
            other => panic!("expected symbolic Point, got {other:?}"),
        }
    }

    // ----- line_sphere -----

    #[test]
    fn line_sphere_two_hits_through_the_centre() {
        // Unit sphere at the origin; line along x from (-5,0,0).
        let line = ArithmaLine::new(v(-5, 0, 0), v(1, 0, 0));
        let sphere = ArithmaSphere::new(v(0, 0, 0), ArithmaExpression::from_i64(1));
        match ArithmaIntersection::line_sphere(&line, &sphere) {
            ArithmaIntersectionResult::TwoPoints(near, far) => {
                assert_vec(&near, -1.0, 0.0, 0.0);
                assert_vec(&far, 1.0, 0.0, 0.0);
            }
            other => panic!("expected TwoPoints, got {other:?}"),
        }
    }

    #[test]
    fn line_sphere_two_hits_on_an_offset_sphere() {
        // Sphere radius 3 centred at (10,0,0); line along x from the origin.
        let line = ArithmaLine::new(v(0, 0, 0), v(1, 0, 0));
        let sphere = ArithmaSphere::new(v(10, 0, 0), ArithmaExpression::from_i64(3));
        match ArithmaIntersection::line_sphere(&line, &sphere) {
            ArithmaIntersectionResult::TwoPoints(near, far) => {
                assert_vec(&near, 7.0, 0.0, 0.0);
                assert_vec(&far, 13.0, 0.0, 0.0);
            }
            other => panic!("expected TwoPoints, got {other:?}"),
        }
    }

    #[test]
    fn line_sphere_tangent_gives_a_single_point() {
        // Unit sphere at origin; line along x at y = 1 -> touches (0,1,0).
        let line = ArithmaLine::new(v(-5, 1, 0), v(1, 0, 0));
        let sphere = ArithmaSphere::new(v(0, 0, 0), ArithmaExpression::from_i64(1));
        match ArithmaIntersection::line_sphere(&line, &sphere) {
            ArithmaIntersectionResult::Point(p) => assert_vec(&p, 0.0, 1.0, 0.0),
            other => panic!("expected Point, got {other:?}"),
        }
    }

    #[test]
    fn line_sphere_miss_returns_none() {
        // Line at y = 5 vs unit sphere at the origin.
        let line = ArithmaLine::new(v(-5, 5, 0), v(1, 0, 0));
        let sphere = ArithmaSphere::new(v(0, 0, 0), ArithmaExpression::from_i64(1));
        assert!(matches!(
            ArithmaIntersection::line_sphere(&line, &sphere),
            ArithmaIntersectionResult::None
        ));
    }

    #[test]
    fn line_sphere_with_zero_direction_returns_none() {
        let line = ArithmaLine::new(v(0, 0, 0), ArithmaVector::zero());
        let sphere = ArithmaSphere::new(v(0, 0, 0), ArithmaExpression::from_i64(1));
        assert!(matches!(
            ArithmaIntersection::line_sphere(&line, &sphere),
            ArithmaIntersectionResult::None
        ));
    }

    #[test]
    fn line_sphere_hits_satisfy_the_sphere_equation() {
        // Non-axis-aligned case: direction (1,2,3) is deliberately not unit.
        let line = ArithmaLine::new(v(-4, -9, -14), v(1, 2, 3));
        let sphere = ArithmaSphere::new(v(1, 1, 1), ArithmaExpression::from_i64(4));
        match ArithmaIntersection::line_sphere(&line, &sphere) {
            ArithmaIntersectionResult::TwoPoints(p, q) => {
                for hit in [&p, &q] {
                    let d = hit.sub_vec(&sphere.centre).magnitude_squared();
                    assert!((num(&d) - 16.0).abs() < 1e-6, "|p-c|^2 = {}", num(&d));
                }
                // Near hit must come first along the direction.
                assert!(num(&p.x) < num(&q.x));
            }
            other => panic!("expected TwoPoints, got {other:?}"),
        }
    }

    // ----- plane_plane -----

    #[test]
    fn plane_plane_of_xz_and_yz_is_the_z_axis() {
        let a = ArithmaPlane::new(v(1, 0, 0), ArithmaExpression::zero());
        let b = ArithmaPlane::new(v(0, 1, 0), ArithmaExpression::zero());
        let line = ArithmaIntersection::plane_plane(&a, &b).expect("planes are not parallel");
        assert_vec(&line.origin, 0.0, 0.0, 0.0);
        assert_vec(&line.direction, 0.0, 0.0, 1.0);
    }

    #[test]
    fn plane_plane_with_offsets_produces_a_point_on_both_planes() {
        // x = 2 and y = 3 meet along the vertical line through (2,3,0).
        let a = ArithmaPlane::new(v(1, 0, 0), ArithmaExpression::from_i64(2));
        let b = ArithmaPlane::new(v(0, 1, 0), ArithmaExpression::from_i64(3));
        let line = ArithmaIntersection::plane_plane(&a, &b).expect("not parallel");
        assert_vec(&line.origin, 2.0, 3.0, 0.0);
        assert_vec(&line.direction, 0.0, 0.0, 1.0);
    }

    #[test]
    fn plane_plane_oblique_origin_lies_on_both_planes() {
        // x = 1 and x + y = 2.
        let a = ArithmaPlane::new(v(1, 0, 0), ArithmaExpression::from_i64(1));
        let b = ArithmaPlane::new(v(1, 1, 0), ArithmaExpression::from_i64(2));
        let line = ArithmaIntersection::plane_plane(&a, &b).expect("not parallel");
        assert_vec(&line.origin, 1.0, 1.0, 0.0);
        // Direction (1,0,0) x (1,1,0) = (0,0,1).
        assert_vec(&line.direction, 0.0, 0.0, 1.0);
        // And the whole line stays on both planes.
        let far = line.at(ArithmaExpression::from_i64(7));
        assert!((num(&a.normal.dot(&far)) - 1.0).abs() < 1e-9);
        assert!((num(&b.normal.dot(&far)) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn plane_plane_parallel_returns_none() {
        let a = ArithmaPlane::new(v(0, 0, 1), ArithmaExpression::zero());
        let b = ArithmaPlane::new(v(0, 0, 2), ArithmaExpression::from_i64(5));
        assert!(ArithmaIntersection::plane_plane(&a, &b).is_none());
    }

    #[test]
    fn plane_plane_coincident_returns_none() {
        let a = ArithmaPlane::new(v(0, 0, 1), ArithmaExpression::from_i64(4));
        let b = ArithmaPlane::new(v(0, 0, 1), ArithmaExpression::from_i64(4));
        assert!(ArithmaIntersection::plane_plane(&a, &b).is_none());
    }

    // ----- closest_point_param -----

    #[test]
    fn closest_point_param_on_the_x_axis() {
        let line = ArithmaLine::new(v(0, 0, 0), v(1, 0, 0));
        let t = ArithmaIntersection::closest_point_param(&line, &v(7, 3, 0));
        assert!((num(&t) - 7.0).abs() < 1e-12);
    }

    #[test]
    fn closest_point_param_normalises_by_squared_length() {
        // Direction is length 2, so the parameter is half the projected length.
        let line = ArithmaLine::new(v(0, 0, 0), v(2, 0, 0));
        let t = ArithmaIntersection::closest_point_param(&line, &v(7, 3, 0));
        assert!((num(&t) - 3.5).abs() < 1e-12, "got {}", num(&t));
    }

    #[test]
    fn closest_point_param_accounts_for_the_origin() {
        let line = ArithmaLine::new(v(1, 1, 1), v(0, 1, 0));
        let t = ArithmaIntersection::closest_point_param(&line, &v(9, 4, -3));
        // (point - origin) = (8, 3, -4); dot with (0,1,0) = 3; /1 = 3.
        assert!((num(&t) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn closest_point_is_perpendicular_to_the_direction() {
        let line = ArithmaLine::new(v(1, 2, 3), v(1, 2, 2));
        let point = v(4, 0, -1);
        let t = ArithmaIntersection::closest_point_param(&line, &point);
        let foot = line.at(t);
        let residual = point.sub_vec(&foot).dot(&line.direction);
        assert!(num(&residual).abs() < 1e-6, "got {}", num(&residual));
    }

    #[test]
    fn closest_point_param_with_zero_direction_is_not_evaluable() {
        let line = ArithmaLine::new(v(0, 0, 0), ArithmaVector::zero());
        let t = ArithmaIntersection::closest_point_param(&line, &v(1, 2, 3));
        assert!(t.evaluate(&ArithmaBindings::new()).is_err());
    }

    // ----- classification convention -----

    #[test]
    fn classify_follows_the_documented_convention() {
        assert_eq!(classify(&ArithmaExpression::zero()), Sign::Zero);
        assert_eq!(classify(&ArithmaExpression::from_i64(3)), Sign::Positive);
        assert_eq!(classify(&ArithmaExpression::from_i64(-3)), Sign::Negative);
        assert_eq!(classify(&ArithmaExpression::var("q")), Sign::Unknown);
    }
}
