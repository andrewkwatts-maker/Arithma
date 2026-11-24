use arithma_core::expression::{ArithmaExpression, Emit, EmitTarget};
use arithma_core::parse::parse_expression;

const HOSTS: [EmitTarget; 3] = [EmitTarget::Rust, EmitTarget::C, EmitTarget::Python];
const NEW_TARGETS: [EmitTarget; 4] = [
    EmitTarget::Rust,
    EmitTarget::C,
    EmitTarget::Python,
    EmitTarget::Wgsl,
];

const FUNCTION_NAMES: [(&str, usize); 25] = [
    ("abs", 1), ("acos", 1), ("asin", 1), ("atan", 1), ("atan2", 2), ("cbrt", 1),
    ("ceil", 1), ("cos", 1), ("cosh", 1), ("exp", 1), ("floor", 1), ("ln", 1),
    ("log", 1), ("log10", 1), ("log2", 1), ("max", 2), ("min", 2), ("pow", 2),
    ("round", 1), ("sign", 1), ("sin", 1), ("sinh", 1), ("sqrt", 1), ("tan", 1),
    ("tanh", 1),
];

fn emit(src: &str, target: EmitTarget) -> Result<String, String> {
    parse_expression(src).unwrap().emit(target)
}

fn refused(name: &str, target: EmitTarget) -> bool {
    match target {
        EmitTarget::Rust => name == "log",
        EmitTarget::C => matches!(name, "log" | "sign"),
        EmitTarget::Python => matches!(name, "log" | "sign" | "round"),
        EmitTarget::Wgsl => matches!(name, "log" | "log10" | "round" | "sign"),
        _ => false,
    }
}

#[test]
fn rust_calls_use_the_f64_path() {
    assert_eq!(
        emit("sin(x) + pow(y, 2)", EmitTarget::Rust).unwrap(),
        "f64::sin(x) + f64::powf(y, 2.0)"
    );
}

#[test]
fn rust_sign_is_signum() {
    assert_eq!(emit("sign(x)", EmitTarget::Rust).unwrap(), "f64::signum(x)");
}

#[test]
fn c_uses_math_h_spellings() {
    assert_eq!(
        emit("abs(x) + ln(y) + min(a, b)", EmitTarget::C).unwrap(),
        "fabs(x) + log(y) + fmin(a, b)"
    );
}

#[test]
fn python_uses_math_and_double_star() {
    assert_eq!(
        emit("x^2 + sqrt(y)", EmitTarget::Python).unwrap(),
        "x ** 2.0 + math.sqrt(y)"
    );
}

#[test]
fn grouping_survives_on_every_host() {
    let cases = [
        ("-x^2", "-f64::powf(x, 2.0)", "-pow(x, 2.0)", "-x ** 2.0"),
        ("(-x)^2", "f64::powf(-x, 2.0)", "pow(-x, 2.0)", "(-x) ** 2.0"),
        (
            "2^3^2",
            "f64::powf(2.0, f64::powf(3.0, 2.0))",
            "pow(2.0, pow(3.0, 2.0))",
            "2.0 ** (3.0 ** 2.0)",
        ),
        ("a - (b - c)", "a - (b - c)", "a - (b - c)", "a - (b - c)"),
        ("a / (b * c)", "a / (b * c)", "a / (b * c)", "a / (b * c)"),
    ];
    for (src, rust, c, python) in cases {
        assert_eq!(emit(src, EmitTarget::Rust).unwrap(), rust, "{src}");
        assert_eq!(emit(src, EmitTarget::C).unwrap(), c, "{src}");
        assert_eq!(emit(src, EmitTarget::Python).unwrap(), python, "{src}");
    }
}

#[test]
fn python_negative_literal_base_is_bracketed() {
    assert_eq!(emit("(-2)^2", EmitTarget::Python).unwrap(), "(-2.0) ** 2.0");
}

#[test]
fn numbers_carry_a_decimal_point() {
    for target in HOSTS {
        assert_eq!(emit("1/3", target).unwrap(), "1.0 / 3.0", "{target:?}");
    }
}

#[test]
fn cbrt_is_a_call_on_every_host() {
    assert_eq!(emit("cbrt(x)", EmitTarget::Rust).unwrap(), "f64::cbrt(x)");
    assert_eq!(emit("cbrt(x)", EmitTarget::C).unwrap(), "cbrt(x)");
    assert_eq!(emit("cbrt(x)", EmitTarget::Python).unwrap(), "math.cbrt(x)");
}

#[test]
fn every_parsed_function_has_a_spelling_or_a_named_refusal() {
    for target in NEW_TARGETS {
        for (name, arity) in FUNCTION_NAMES {
            let src = if arity == 1 {
                format!("{name}(x)")
            } else {
                format!("{name}(x, y)")
            };
            let out = emit(&src, target);
            if refused(name, target) {
                let err = out.unwrap_err();
                assert!(err.contains(name) && err.contains(&format!("{target:?}")), "{err}");
            } else {
                assert!(!out.unwrap().is_empty(), "{name} on {target:?}");
            }
        }
    }
}

#[test]
fn constants_inline_or_are_refused() {
    let with_value = ArithmaExpression::constant("pi", None, Some(3.5), false);
    let without = ArithmaExpression::constant("pi", None, None, false);
    for target in NEW_TARGETS {
        assert_eq!(with_value.emit(target).unwrap(), "3.5", "{target:?}");
        let err = without.emit(target).unwrap_err();
        assert!(err.contains("pi"), "{err}");
    }
}

#[test]
fn wgsl_spellings_match_builtins() {
    assert_eq!(
        emit("ln(x) + atan2(y, x) + cbrt(z)", EmitTarget::Wgsl).unwrap(),
        "log(x) + atan2(y, x) + (sign(z) * pow(abs(z), 1.0/3.0))"
    );
}

#[test]
fn wgsl_refuses_by_name() {
    for name in ["log10", "round", "sign", "log"] {
        let err = emit(&format!("{name}(x)"), EmitTarget::Wgsl).unwrap_err();
        assert!(err.contains(name) && err.contains("Wgsl"), "{err}");
    }
    let err = emit("1e300", EmitTarget::Wgsl).unwrap_err();
    assert!(err.contains("1e300") && err.contains("Wgsl"), "{err}");
}

#[test]
fn existing_targets_are_unchanged() {
    let cases = [
        (
            "sin(x)+pow(y,2)",
            ["sin(x) + y^2", "sin(x) + pow(y, 2.0)", "sin(x) + pow(y, 2.0)", "x sin y 2 ^ +"],
        ),
        (
            "atan2(y,x)+log10(x)",
            [
                "atan2(y, x) + log10(x)",
                "atan2(y, x) + log10(x)",
                "atan2(y, x) + log10(x)",
                "y x atan2 x log10 +",
            ],
        ),
        ("-x^2", ["-x^2", "-pow(x, 2.0)", "-pow(x, 2.0)", "x 2 ^ neg"]),
    ];
    let targets = [
        EmitTarget::Text,
        EmitTarget::Glsl,
        EmitTarget::Hlsl,
        EmitTarget::EmlRpn,
    ];
    for (src, expected) in cases {
        for (target, want) in targets.into_iter().zip(expected) {
            assert_eq!(emit(src, target).unwrap(), want, "{src} on {target:?}");
        }
    }
}
