//====== Arithma/rust/arithma_core/benches/hot_paths.rs ======//
//! Criterion benchmarks for the paths a symbolic engine spends its time in.
//!
//! These exist so "fully accelerated" is a measurement rather than a claim.
//! They establish a baseline against which the release-profile settings
//! (LTO, one codegen unit) can be judged, and they catch a regression before
//! a user does.
//!
//! Run with `cargo bench`. The `bench` profile inherits `release` with thin
//! LTO, so numbers are representative of a shipped build without paying full
//! cross-crate LTO link time on every run.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use arithma_core::expression::iterative::simplify_iterative;
use arithma_core::expression::ArithmaExpression as E;
use arithma_core::expression::SimplificationConfig;
use arithma_core::integer::ArithmaInteger as I;

/// A left-leaning sum `((((1 + 1) + 2) + 3) ... )` of the requested depth.
/// Deep and narrow, which is what stresses recursive traversal.
fn nested_sum(depth: i64) -> E {
    let mut e = E::from_i64(1);
    for k in 1..=depth {
        e = E::add(e, E::from_i64(k));
    }
    e
}

fn bench_build(c: &mut Criterion) {
    let mut g = c.benchmark_group("expression/build");
    for depth in [8i64, 64, 512] {
        g.bench_function(format!("nested_sum/{depth}"), |b| {
            b.iter(|| black_box(nested_sum(black_box(depth))))
        });
    }
    g.finish();
}

fn bench_numeric_collapse(c: &mut Criterion) {
    // `to_f64` walks the whole tree, so this measures traversal cost -- the
    // hot path for any consumer that evaluates an expression per frame.
    let mut g = c.benchmark_group("expression/to_f64");
    for depth in [8i64, 64, 512] {
        let expr = nested_sum(depth);
        g.bench_function(format!("nested_sum/{depth}"), |b| {
            b.iter(|| black_box(black_box(&expr).to_f64()))
        });
    }
    g.finish();
}

fn bench_integer(c: &mut Criterion) {
    let mut g = c.benchmark_group("integer");
    g.bench_function("from_i64", |b| {
        b.iter(|| black_box(I::from_i64(black_box(987_654_321))))
    });
    let a = I::from_i64(987_654_321);
    g.bench_function("to_f64", |b| b.iter(|| black_box(black_box(&a).to_f64())));
    g.finish();
}

fn bench_atoms(c: &mut Criterion) {
    let mut g = c.benchmark_group("expression/atoms");
    g.bench_function("from_i64", |b| {
        b.iter(|| black_box(E::from_i64(black_box(42))))
    });
    g.bench_function("var", |b| b.iter(|| black_box(E::var(black_box("x")))));
    g.finish();
}

/// A sum of `count` distinct free variables. Nothing here folds, so this
/// measures the cost of a pass that finds no work -- the common case for an
/// already-simplified tree, and the one a caller pays on every fixpoint check.
fn symbolic_sum(count: usize) -> E {
    let mut e = E::var("x0");
    for k in 1..count {
        e = E::add(e, E::var(&format!("x{k}")));
    }
    e
}

fn bench_simplify(c: &mut Criterion) {
    let cfg = SimplificationConfig::default();

    let mut g = c.benchmark_group("expression/simplify");
    // Folding a fully numeric tree: the whole thing collapses to one literal.
    for depth in [8i64, 64, 200, 1000] {
        let tree = nested_sum(depth);
        g.bench_function(format!("fold_numeric/{depth}"), |b| {
            b.iter_batched(
                || tree.clone(),
                |mut e| black_box(simplify_iterative(&mut e, black_box(&cfg))),
                criterion::BatchSize::SmallInput,
            )
        });
    }
    // No-op pass over a purely symbolic tree. This is the cost of *asking*.
    for count in [8usize, 64, 200, 1000] {
        let tree = symbolic_sum(count);
        g.bench_function(format!("no_op_symbolic/{count}"), |b| {
            b.iter_batched(
                || tree.clone(),
                |mut e| black_box(simplify_iterative(&mut e, black_box(&cfg))),
                criterion::BatchSize::SmallInput,
            )
        });
    }
    g.finish();
}

fn bench_integer_arithmetic(c: &mut Criterion) {
    // Exact bignum arithmetic is what constant folding rests on, so its cost
    // is the floor for every fold.
    let small_a = I::from_i64(123_456);
    let small_b = I::from_i64(654_321);
    let big_a = I::from_i64(2).checked_pow(256).expect("2^256");
    let big_b = I::from_i64(3).checked_pow(160).expect("3^160");

    let mut g = c.benchmark_group("integer/arithmetic");
    g.bench_function("add/small", |b| {
        b.iter(|| black_box(black_box(&small_a).checked_add(black_box(&small_b))))
    });
    g.bench_function("add/256bit", |b| {
        b.iter(|| black_box(black_box(&big_a).checked_add(black_box(&big_b))))
    });
    g.bench_function("mul/small", |b| {
        b.iter(|| black_box(black_box(&small_a).checked_mul(black_box(&small_b))))
    });
    g.bench_function("mul/256bit", |b| {
        b.iter(|| black_box(black_box(&big_a).checked_mul(black_box(&big_b))))
    });
    g.bench_function("div_exact/256bit", |b| {
        b.iter(|| black_box(black_box(&big_a).checked_div_exact(black_box(&big_b))))
    });
    g.bench_function("pow/2^64", |b| {
        b.iter(|| black_box(I::from_i64(2).checked_pow(black_box(64))))
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_build,
    bench_numeric_collapse,
    bench_integer,
    bench_atoms,
    bench_simplify,
    bench_integer_arithmetic
);
criterion_main!(benches);
