//====== Arithma/rust/arithma_core/tests/curve_expressions.rs ======//
//! Acceptance tests for `arithma_core::geometry::curve`: N-dimensional
//! Bézier, Catmull–Rom (uniform, centripetal, chordal) and piecewise curves.

use arithma_core::expression::{ArithmaBindings, ArithmaExpression, Differentiable, Evaluable};
use arithma_core::geometry::curve::{bezier, catmull_rom, ArithmaCurveError, CatmullRomKind, PiecewiseCurve};

/// Builds a point `Vec<ArithmaExpression>` from integer coordinates.
fn pt(coords: &[i64]) -> Vec<ArithmaExpression> {
    coords.iter().map(|&c| ArithmaExpression::from_i64(c)).collect()
}

/// Evaluates every axis of `exprs` with `parameter` bound to `t`.
fn eval_at(exprs: &[ArithmaExpression], parameter: &str, t: f64) -> Vec<f64> {
    let mut bindings = ArithmaBindings::new();
    bindings.insert(parameter.to_string(), t);
    exprs
        .iter()
        .map(|e| e.evaluate(&bindings).expect("curve expression should evaluate"))
        .collect()
}

/// Reference de Casteljau evaluator over plain `f64` points, used as an
/// independent oracle for the closed-form Bernstein construction.
fn de_casteljau(control: &[Vec<f64>], t: f64) -> Vec<f64> {
    let mut pts: Vec<Vec<f64>> = control.to_vec();
    let n = pts.len();
    for r in 1..n {
        for i in 0..n - r {
            for axis in 0..pts[i].len() {
                pts[i][axis] = (1.0 - t) * pts[i][axis] + t * pts[i + 1][axis];
            }
        }
    }
    pts[0].clone()
}

fn assert_close(got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want.iter()) {
        assert!((g - w).abs() < tol, "got {g}, want {w}");
    }
}

#[test]
fn bezier_endpoints_equal_first_and_last_control_points() {
    let control = vec![pt(&[0, 0, 0]), pt(&[1, 5, -2]), pt(&[4, -3, 1]), pt(&[6, 2, 7])];
    let curve = bezier(&control, "t").expect("bezier should build");
    assert_close(&eval_at(&curve, "t", 0.0), &[0.0, 0.0, 0.0], 1e-12);
    assert_close(&eval_at(&curve, "t", 1.0), &[6.0, 2.0, 7.0], 1e-12);
}

#[test]
fn bezier_matches_de_casteljau_2d() {
    let control_i = vec![pt(&[0, 0]), pt(&[2, 8]), pt(&[5, -4]), pt(&[9, 3])];
    let control_f: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0],
        vec![2.0, 8.0],
        vec![5.0, -4.0],
        vec![9.0, 3.0],
    ];
    let curve = bezier(&control_i, "t").expect("bezier should build");
    for i in 0..=50 {
        let t = i as f64 / 50.0;
        let got = eval_at(&curve, "t", t);
        let want = de_casteljau(&control_f, t);
        assert_close(&got, &want, 1e-9);
    }
}

#[test]
fn bezier_matches_de_casteljau_3d() {
    let control_i = vec![
        pt(&[0, 0, 1]),
        pt(&[1, 4, -2]),
        pt(&[3, -2, 5]),
        pt(&[6, 1, 0]),
        pt(&[8, -3, 2]),
    ];
    let control_f: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0, 1.0],
        vec![1.0, 4.0, -2.0],
        vec![3.0, -2.0, 5.0],
        vec![6.0, 1.0, 0.0],
        vec![8.0, -3.0, 2.0],
    ];
    let curve = bezier(&control_i, "t").expect("bezier should build");
    for i in 0..=50 {
        let t = i as f64 / 50.0;
        let got = eval_at(&curve, "t", t);
        let want = de_casteljau(&control_f, t);
        assert_close(&got, &want, 1e-9);
    }
}

#[test]
fn catmull_rom_segment_passes_through_p1_and_p2_for_every_kind() {
    let p0 = pt(&[-1, -2]);
    let p1 = pt(&[0, 0]);
    let p2 = pt(&[3, 1]);
    let p3 = pt(&[5, 4]);
    for kind in [
        CatmullRomKind::Uniform,
        CatmullRomKind::Centripetal,
        CatmullRomKind::Chordal,
    ] {
        let seg = catmull_rom(&p0, &p1, &p2, &p3, kind, "t").expect("segment should build");
        assert_close(&eval_at(&seg, "t", 0.0), &[0.0, 0.0], 1e-9);
        assert_close(&eval_at(&seg, "t", 1.0), &[3.0, 1.0], 1e-9);
    }
}

#[test]
fn piecewise_curve_passes_through_every_point() {
    let points = vec![
        pt(&[0, 0]),
        pt(&[1, 2]),
        pt(&[3, 3]),
        pt(&[4, 1]),
        pt(&[6, 0]),
    ];
    let piecewise =
        PiecewiseCurve::catmull_rom_through(&points, CatmullRomKind::Uniform, "t").unwrap();
    let curve = piecewise.expression("t").expect("expression should build");
    for (k, point) in points.iter().enumerate() {
        let want: Vec<f64> = point
            .iter()
            .map(|e| e.evaluate(&ArithmaBindings::new()).unwrap())
            .collect();
        let got = eval_at(&curve, "t", k as f64);
        assert_close(&got, &want, 1e-9);
    }
}

#[test]
fn piecewise_curve_is_c1_at_interior_joints_for_uniform_kind() {
    let points = vec![
        pt(&[0, 0]),
        pt(&[1, 2]),
        pt(&[3, 3]),
        pt(&[4, 1]),
        pt(&[6, 0]),
    ];
    let piecewise =
        PiecewiseCurve::catmull_rom_through(&points, CatmullRomKind::Uniform, "t").unwrap();
    let mut bindings_end = ArithmaBindings::new();
    bindings_end.insert("t".to_string(), 1.0);
    let mut bindings_start = ArithmaBindings::new();
    bindings_start.insert("t".to_string(), 0.0);

    for k in 0..piecewise.segments.len() - 1 {
        for axis in 0..piecewise.segments[k].len() {
            let left = piecewise.segments[k][axis]
                .differentiate("t")
                .expect("should differentiate")
                .evaluate(&bindings_end)
                .expect("should evaluate");
            let right = piecewise.segments[k + 1][axis]
                .differentiate("t")
                .expect("should differentiate")
                .evaluate(&bindings_start)
                .expect("should evaluate");
            assert!((left - right).abs() < 1e-9, "joint {k} axis {axis}: {left} vs {right}");
        }
    }
}

#[test]
fn dimension_mismatch_is_an_error() {
    let control = vec![pt(&[0, 0]), vec![ArithmaExpression::from_i64(1)]];
    match bezier(&control, "t") {
        Err(ArithmaCurveError::DimensionMismatch { expected: 2, got: 1 }) => {}
        other => panic!("expected a dimension mismatch error, got {other:?}"),
    }

    let p0 = pt(&[0, 0]);
    let p1 = pt(&[1, 1]);
    let p2 = pt(&[2, 2]);
    let p3 = vec![ArithmaExpression::from_i64(3)];
    assert!(catmull_rom(&p0, &p1, &p2, &p3, CatmullRomKind::Uniform, "t").is_err());
}
