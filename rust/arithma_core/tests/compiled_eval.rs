//! `CompiledExpr` matches `evaluate` bit for bit and allocates nothing on `Ok`.
//! Run single-threaded so the counted windows are clean:
//! `cargo test --release -p arithma_core --test compiled_eval -- --test-threads=1`.

#[path = "support/counting.rs"]
mod counting;

use arithma_core::expression::compiled::{CompileError, CompiledExpr};
use arithma_core::expression::term_sum::TermSum;
use arithma_core::expression::{ArithmaBindings, ArithmaExpression as E, Evaluable};
use arithma_core::function::ArithmaFunction as F;
use arithma_core::integer::ArithmaInteger;
use arithma_core::rational::Rational;
use counting::Counting;

#[global_allocator]
static ALLOC: Counting = Counting::new();

const POINTS: usize = 10_000;

const VARS: [&str; 18] = [
    "p", "kappa", "bcap", "smax", "gap", "b0", "d", "dl", "rho", "rc", "dr", "x", "y", "z", "c0",
    "c1", "c2", "c3",
];

const SECOND_ORDER_VARS: [&str; 5] = ["p", "rho", "rc", "dr", "dl"];

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A value in `[-4, 4)`, with exact zeros (both signs) and small integers mixed in.
    fn value(&mut self) -> f64 {
        let r = self.next_u64();
        match r % 12 {
            0 => 0.0,
            1 => -0.0,
            2 => ((r >> 8) % 7) as f64 - 3.0,
            _ => (r >> 11) as f64 / 9_007_199_254_740_992.0 * 8.0 - 4.0, // direct-float: test input generator, not a formula
        }
    }
}

fn points(seed: u64) -> Vec<[f64; VARS.len()]> {
    let mut rng = Rng(seed);
    (0..POINTS)
        .map(|_| {
            let mut p = [0.0; VARS.len()];
            p.iter_mut().for_each(|v| *v = rng.value());
            p
        })
        .collect()
}

fn num(v: f64) -> E {
    E::constant("", None, Some(v), false)
}

fn var(name: &str) -> E {
    E::var(name)
}

fn int(n: i64) -> E {
    E::num(ArithmaInteger::from_i64(n))
}

fn inv_sq(x: E) -> E {
    E::div(num(1.0), E::mul(x.clone(), x))
}

fn inv_pow(x: E, n: u32) -> E {
    let prod = (1..n).fold(x.clone(), |acc, _| E::mul(acc, x.clone()));
    E::div(num(1.0), prod)
}

fn bound_expr() -> E {
    let near = inv_sq(var("dl"));
    let drift = E::add(
        E::mul(E::mul(var("p"), var("kappa")), var("bcap")),
        E::mul(var("smax"), var("kappa")),
    );
    let spread = E::mul(
        E::mul(var("smax"), var("b0")),
        E::sub(near.clone(), inv_sq(var("d"))),
    );
    let clamp = E::mul(E::mul(var("gap"), var("b0")), inv_sq(var("d")));
    E::add(E::add(E::mul(drift, near), spread), clamp)
}

fn second_order_expr() -> E {
    let taylor = E::mul(
        E::mul(num(19.0), E::mul(var("p"), E::mul(var("rho"), var("rho")))),
        inv_pow(var("dr"), 4),
    );
    let drift = E::mul(
        E::mul(num(6.0), E::mul(var("p"), var("rc"))),
        inv_pow(var("dl"), 3),
    );
    E::add(taylor, drift)
}

fn four_args(f: F) -> E {
    E::func(f, vec![var("x"), var("y"), var("z"), var("d")])
}

fn conditional() -> E {
    E::Conditional {
        condition: Box::new(var("x")),
        then_expr: Box::new(E::sin(var("y"))),
        else_expr: Box::new(E::div(var("z"), var("d"))),
    }
}

fn rational_mix() -> E {
    let third = Rational::new(ArithmaInteger::from_i64(1), ArithmaInteger::from_i64(3));
    let third = E::Rational(third.expect("1/3 is a valid rational"));
    E::add(E::mul(third, var("x")), int(7))
}

fn fourier_wrapper() -> E {
    E::FourierOptimized {
        expr: Box::new(E::mul(E::sin(var("x")), var("y"))),
        transform: None,
        variable: None,
        range: None,
    }
}

fn cached_value(cached: Option<f64>) -> E {
    E::CachedValue {
        expr: Box::new(E::mul(var("x"), var("y"))),
        cached,
        dirty: false,
    }
}

fn unary_functions() -> Vec<F> {
    vec![
        F::Negate,
        F::Sqrt,
        F::Cbrt,
        F::Exp,
        F::Ln,
        F::Log10,
        F::Log2,
        F::Sin,
        F::Cos,
        F::Tan,
        F::Asin,
        F::Acos,
        F::Atan,
        F::Sinh,
        F::Cosh,
        F::Tanh,
        F::Abs,
        F::Sign,
        F::Floor,
        F::Ceil,
        F::Round,
    ]
}

fn binary_functions() -> Vec<F> {
    vec![
        F::Add,
        F::Subtract,
        F::Multiply,
        F::Divide,
        F::Power,
        F::Atan2,
        F::Min,
        F::Max,
    ]
}

fn cases() -> Vec<(String, E)> {
    let mut all = vec![
        ("bound_expr".to_string(), bound_expr()),
        ("second_order_expr".to_string(), second_order_expr()),
        ("add4".to_string(), four_args(F::Add)),
        ("multiply4".to_string(), four_args(F::Multiply)),
        ("conditional".to_string(), conditional()),
        ("rational".to_string(), rational_mix()),
        ("cached_constant".to_string(), E::mul(num(3.25), var("x"))),
        ("fourier".to_string(), fourier_wrapper()),
        ("cached_some".to_string(), cached_value(Some(2.5))),
        ("cached_none".to_string(), cached_value(None)),
        (
            "term_sum".to_string(),
            TermSum::polynomial("x", 3, "c")
                .expect("polynomial builds")
                .to_expression(),
        ),
    ];
    for f in unary_functions() {
        all.push((format!("{f:?}"), E::func(f, vec![var("x")])));
    }
    for f in binary_functions() {
        all.push((format!("{f:?}2"), E::func(f, vec![var("x"), var("y")])));
    }
    all
}

/// The points where `expr` evaluates `Ok`: error paths may allocate their message.
fn ok_points(expr: &E, pts: Vec<[f64; VARS.len()]>) -> Vec<[f64; VARS.len()]> {
    let kept: Vec<_> = pts
        .into_iter()
        .filter(|p| expr.evaluate(&bindings(p)).is_ok())
        .collect();
    assert!(kept.len() > POINTS / 2, "too few Ok points: {}", kept.len());
    kept
}

fn bindings(point: &[f64; VARS.len()]) -> ArithmaBindings {
    VARS.iter()
        .zip(point)
        .map(|(k, v)| ((*k).to_string(), *v))
        .collect()
}

fn assert_same(name: &str, point: &[f64; VARS.len()], reference: &Result<f64, String>, got: &Result<f64, String>) {
    match (reference, got) {
        (Ok(v), Ok(w)) => assert_eq!(
            v.to_bits(),
            w.to_bits(),
            "{name} differs at {point:?}: evaluate {v:?}, eval {w:?}"
        ),
        (Err(_), Err(_)) => {}
        _ => panic!("{name} disagrees on Ok/Err at {point:?}: {reference:?} vs {got:?}"),
    }
}

#[test]
fn matches_evaluate_bitwise() {
    let pts = points(0x9E37_79B9_7F4A_7C15);
    for (name, expr) in cases() {
        let compiled = CompiledExpr::new(&expr, &VARS).unwrap_or_else(|e| panic!("{name}: {e}"));
        for point in &pts {
            let reference = expr.evaluate(&bindings(point));
            assert_same(&name, point, &reference, &compiled.eval(point));
        }
    }
}

#[test]
fn eval_allocates_nothing() {
    let expr = second_order_expr();
    let pts = ok_points(&expr, points(0xD1B5_4A32_D192_ED03));
    let compiled = CompiledExpr::new(&expr, &VARS).expect("compiles");
    let mut acc = 0.0_f64;
    let before = Counting::thread_count();
    for point in &pts {
        if let Ok(v) = compiled.eval(point) {
            acc += v; // direct-float: test accumulator keeping the calls observable
        }
    }
    let after = Counting::thread_count();
    std::hint::black_box(acc);
    assert_eq!(after - before, 0, "eval allocated");
}

#[test]
fn evaluate_allocs_two_per_call() {
    let expr = second_order_expr();
    let pts = ok_points(&expr, points(0x2545_F491_4F6C_DD1D));
    let mut map = bindings(&pts[0]);
    let mut acc = 0.0_f64;
    let before = Counting::thread_count();
    for point in &pts {
        for (name, v) in VARS.iter().zip(point) {
            if let Some(slot) = map.get_mut(*name) {
                *slot = *v;
            }
        }
        if let Ok(v) = expr.evaluate(&map) {
            acc += v; // direct-float: test accumulator keeping the calls observable
        }
    }
    let after = Counting::thread_count();
    std::hint::black_box(acc);
    assert_eq!(after - before, 2 * pts.len() as u64, "evaluate allocations per call");
}

#[test]
fn unsupported_is_refused() {
    let x = || var("x");
    let sum = E::Sum {
        variable: "k".into(),
        start: Box::new(int(0)),
        end: Box::new(int(3)),
        expression: Box::new(x()),
    };
    let refused = |e: &E| CompiledExpr::new(e, &["x"]).unwrap_err();
    assert!(matches!(refused(&sum), CompileError::Unsupported(_)));
    assert!(matches!(
        refused(&E::func(F::Add, vec![])),
        CompileError::Unsupported(_)
    ));
    assert!(matches!(
        refused(&E::func(F::Gamma, vec![x()])),
        CompileError::Unsupported(_)
    ));
    assert!(matches!(
        refused(&E::func(F::Subtract, vec![x(), x(), x()])),
        CompileError::Unsupported(_)
    ));
    assert_eq!(
        refused(&E::constant("c", None, None, false)),
        CompileError::UncachedConstant
    );
    assert_eq!(refused(&var("q")), CompileError::Unbound("q".into()));
    assert_eq!(
        CompiledExpr::new(&x(), &["x", "x"]).unwrap_err(),
        CompileError::DuplicateName
    );
}

#[test]
fn too_deep_and_too_large_are_refused() {
    let mut chain = int(1);
    for _ in 0..70 {
        chain = E::add(int(1), chain);
    }
    assert!(matches!(
        CompiledExpr::new(&chain, &[]).unwrap_err(),
        CompileError::TooDeep(_)
    ));
    let names: Vec<String> = (0..=usize::from(u16::MAX)).map(|i| format!("v{i}")).collect();
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    assert_eq!(
        CompiledExpr::new(&int(1), &refs).unwrap_err(),
        CompileError::TooLarge
    );
}

#[test]
fn wrong_value_count_is_an_error() {
    let compiled = CompiledExpr::new(&E::add(var("x"), var("y")), &["x", "y"]).expect("compiles");
    assert!(compiled.eval(&[1.0]).is_err());
    assert!(compiled.eval(&[1.0, 2.0, 3.0]).is_err());
    assert_eq!(compiled.eval(&[1.0, 2.0]), Ok(3.0));
    assert_eq!(compiled.arity(), 2);
    assert_eq!(compiled.vars(), ["x", "y"]);
    assert!(compiled.depth() >= 2);
}

#[test]
fn compiled_expr_is_send_sync_and_keeps_source() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<CompiledExpr>();
    let expr = second_order_expr();
    let compiled = CompiledExpr::new(&expr, &SECOND_ORDER_VARS).expect("compiles");
    assert_eq!(
        format!("{:?}", compiled.source()),
        format!("{expr:?}"),
        "source() is the compiled expression"
    );
}
