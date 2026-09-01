//====== Arithma/rust/arithma_core/src/expression/emit.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! Source emission for [`ArithmaExpression`].
//!
//! Implements [`Emit`](super::Emit) across every [`EmitTarget`](super::EmitTarget):
//! plain text, LaTeX, MathML, GLSL, HLSL and RPN.
//!
//! The walk is **iterative**, matching the convention the differentiator
//! established — a deeply nested expression must not blow the stack just
//! because someone asked for its LaTeX.
//!
//! Precedence handling: each node reports the precedence of the operator it
//! produced, and a child is parenthesised only when its precedence is lower
//! than the context requires. That keeps `a*(b+c)` bracketed and `a+b+c` clean.
//!
//! ## Known duplication
//!
//! `pyfacade::expression_to_latex` predates this module and still backs the
//! Python `to_latex()`. Its output is pinned by tests there, so it is left in
//! place rather than switched over blindly; folding it into this emitter is a
//! follow-up. Until then, LaTeX changes must be made in both places.

use super::{ArithmaExpression, Emit, EmitTarget};
use crate::function::ArithmaFunction;

/// Operator precedence. Higher binds tighter.
const PREC_ADD: u8 = 1;
const PREC_MUL: u8 = 2;
const PREC_UNARY: u8 = 3;
const PREC_POW: u8 = 4;
const PREC_ATOM: u8 = 5;

/// A rendered fragment plus the precedence of its outermost operator.
struct Frag {
    text: String,
    prec: u8,
}

impl Frag {
    fn atom(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            prec: PREC_ATOM,
        }
    }
    /// Wrap in parentheses if this fragment binds looser than `needed`.
    fn paren_if(&self, needed: u8) -> String {
        if self.prec < needed {
            format!("({})", self.text)
        } else {
            self.text.clone()
        }
    }
}

/// Traversal step for the iterative walk.
enum Step<'a> {
    Enter(&'a ArithmaExpression),
    Combine(&'a ArithmaExpression),
}

impl Emit for ArithmaExpression {
    fn emit(&self, target: EmitTarget) -> Result<String, String> {
        // Post-order walk: children are rendered and pushed onto `out`, then
        // the parent pops them and assembles its own fragment.
        let mut work: Vec<Step> = vec![Step::Enter(self)];
        let mut out: Vec<Frag> = Vec::new();

        while let Some(step) = work.pop() {
            match step {
                Step::Enter(node) => match node {
                    ArithmaExpression::Function(_, args) => {
                        work.push(Step::Combine(node));
                        // Push in reverse so children pop left-to-right.
                        for a in args.iter().rev() {
                            work.push(Step::Enter(a));
                        }
                    }
                    leaf => out.push(emit_leaf(leaf, target)?),
                },
                Step::Combine(node) => {
                    if let ArithmaExpression::Function(f, args) = node {
                        let at = out.len() - args.len();
                        let kids: Vec<Frag> = out.split_off(at);
                        out.push(emit_call(f, &kids, target)?);
                    }
                }
            }
        }

        out.pop()
            .map(|f| f.text)
            .ok_or_else(|| "emit produced no output".to_string())
    }
}

/// Render a leaf node.
fn emit_leaf(node: &ArithmaExpression, target: EmitTarget) -> Result<Frag, String> {
    Ok(match node {
        ArithmaExpression::Number(i) => {
            let v = i.to_f64();
            Frag::atom(fmt_number(v, target))
        }
        ArithmaExpression::Variable(name) => match target {
            EmitTarget::MathMl => Frag::atom(format!("<mi>{name}</mi>")),
            _ => Frag::atom(name.clone()),
        },
        ArithmaExpression::Constant {
            symbol,
            cached_value,
            ..
        } => match target {
            // Shader targets have no symbol table — inline the value or fail
            // loudly rather than emitting an undefined identifier.
            EmitTarget::Glsl | EmitTarget::Hlsl => match cached_value {
                Some(v) => Frag::atom(fmt_number(*v, target)),
                None => {
                    return Err(format!(
                        "constant `{symbol}` has no cached value; cannot inline it into a shader"
                    ))
                }
            },
            EmitTarget::MathMl => Frag::atom(format!("<mi>{symbol}</mi>")),
            EmitTarget::EmlRpn => Frag::atom(symbol.clone()),
            EmitTarget::Latex => Frag::atom(latex_symbol(symbol)),
            EmitTarget::Text => Frag::atom(symbol.clone()),
        },
        other => {
            return Err(format!(
                "emit does not support this node yet: {}",
                node_kind(other)
            ))
        }
    })
}

/// Render a function application from already-rendered children.
fn emit_call(f: &ArithmaFunction, kids: &[Frag], target: EmitTarget) -> Result<Frag, String> {
    // Binary infix operators.
    let infix = |sym: &str, prec: u8| -> Frag {
        let l = kids[0].paren_if(prec);
        let r = kids[1].paren_if(prec + 1);
        Frag {
            text: format!("{l} {sym} {r}"),
            prec,
        }
    };

    Ok(match (f, kids.len()) {
        (ArithmaFunction::Add, 2) => match target {
            EmitTarget::EmlRpn => rpn(kids, "+"),
            EmitTarget::MathMl => mathml_infix(kids, "+"),
            _ => infix("+", PREC_ADD),
        },
        (ArithmaFunction::Subtract, 2) => match target {
            EmitTarget::EmlRpn => rpn(kids, "-"),
            EmitTarget::MathMl => mathml_infix(kids, "-"),
            _ => infix("-", PREC_ADD),
        },
        (ArithmaFunction::Multiply, 2) => match target {
            EmitTarget::EmlRpn => rpn(kids, "*"),
            EmitTarget::MathMl => mathml_infix(kids, "&#x22C5;"),
            // LaTeX renders multiplication by juxtaposition.
            EmitTarget::Latex => Frag {
                text: format!(
                    "{} {}",
                    kids[0].paren_if(PREC_MUL),
                    kids[1].paren_if(PREC_MUL + 1)
                ),
                prec: PREC_MUL,
            },
            _ => infix("*", PREC_MUL),
        },
        (ArithmaFunction::Divide, 2) => match target {
            EmitTarget::EmlRpn => rpn(kids, "/"),
            EmitTarget::Latex => Frag {
                text: format!("\\frac{{{}}}{{{}}}", kids[0].text, kids[1].text),
                prec: PREC_ATOM,
            },
            EmitTarget::MathMl => Frag {
                text: format!("<mfrac>{}{}</mfrac>", kids[0].text, kids[1].text),
                prec: PREC_ATOM,
            },
            _ => infix("/", PREC_MUL),
        },
        (ArithmaFunction::Negate, 1) => match target {
            EmitTarget::EmlRpn => Frag {
                text: format!("{} neg", kids[0].text),
                prec: PREC_ATOM,
            },
            EmitTarget::MathMl => Frag {
                text: format!("<mo>-</mo>{}", kids[0].text),
                prec: PREC_UNARY,
            },
            _ => Frag {
                text: format!("-{}", kids[0].paren_if(PREC_UNARY)),
                prec: PREC_UNARY,
            },
        },
        (ArithmaFunction::Power, 2) => power_frag(&kids[0], &kids[1], target),
        (ArithmaFunction::Pow(n), 1) => {
            let e = Frag::atom(fmt_number(n.to_f64(), target));
            power_frag(&kids[0], &e, target)
        }
        (ArithmaFunction::Sqrt, 1) => match target {
            EmitTarget::Latex => Frag {
                text: format!("\\sqrt{{{}}}", kids[0].text),
                prec: PREC_ATOM,
            },
            EmitTarget::MathMl => Frag {
                text: format!("<msqrt>{}</msqrt>", kids[0].text),
                prec: PREC_ATOM,
            },
            EmitTarget::EmlRpn => Frag {
                text: format!("{} sqrt", kids[0].text),
                prec: PREC_ATOM,
            },
            _ => Frag::atom(format!("sqrt({})", kids[0].text)),
        },
        // Everything else is a named call: name(arg, ...).
        (other, _) => {
            let name = call_name(other, target)?;
            match target {
                EmitTarget::EmlRpn => {
                    let args = kids
                        .iter()
                        .map(|k| k.text.as_str())
                        .collect::<Vec<_>>()
                        .join(" ");
                    Frag {
                        text: format!("{args} {name}"),
                        prec: PREC_ATOM,
                    }
                }
                EmitTarget::Latex => Frag {
                    text: format!(
                        "\\{}{{{}}}",
                        name,
                        kids.iter()
                            .map(|k| k.text.clone())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                    prec: PREC_ATOM,
                },
                EmitTarget::MathMl => Frag {
                    text: format!(
                        "<mi>{}</mi><mo>&#x2061;</mo><mrow><mo>(</mo>{}<mo>)</mo></mrow>",
                        name,
                        kids.iter()
                            .map(|k| k.text.clone())
                            .collect::<Vec<_>>()
                            .join("")
                    ),
                    prec: PREC_ATOM,
                },
                _ => Frag::atom(format!(
                    "{}({})",
                    name,
                    kids.iter()
                        .map(|k| k.text.clone())
                        .collect::<Vec<_>>()
                        .join(", ")
                )),
            }
        }
    })
}

fn power_frag(base: &Frag, exp: &Frag, target: EmitTarget) -> Frag {
    match target {
        EmitTarget::Latex => Frag {
            text: format!("{}^{{{}}}", base.paren_if(PREC_POW + 1), exp.text),
            prec: PREC_POW,
        },
        EmitTarget::MathMl => Frag {
            text: format!("<msup>{}{}</msup>", base.text, exp.text),
            prec: PREC_ATOM,
        },
        EmitTarget::EmlRpn => Frag {
            text: format!("{} {} ^", base.text, exp.text),
            prec: PREC_ATOM,
        },
        // GLSL and HLSL have no `^` operator for floats — it is bitwise xor.
        // Emitting `pow()` is the difference between a shader that compiles
        // and one that silently does the wrong thing.
        EmitTarget::Glsl | EmitTarget::Hlsl => {
            Frag::atom(format!("pow({}, {})", base.text, exp.text))
        }
        EmitTarget::Text => Frag {
            text: format!(
                "{}^{}",
                base.paren_if(PREC_POW + 1),
                exp.paren_if(PREC_POW + 1)
            ),
            prec: PREC_POW,
        },
    }
}

fn rpn(kids: &[Frag], op: &str) -> Frag {
    Frag {
        text: format!("{} {} {op}", kids[0].text, kids[1].text),
        prec: PREC_ATOM,
    }
}

fn mathml_infix(kids: &[Frag], op: &str) -> Frag {
    Frag {
        text: format!("{}<mo>{op}</mo>{}", kids[0].text, kids[1].text),
        prec: PREC_ADD,
    }
}

/// Function name for the target dialect.
fn call_name(f: &ArithmaFunction, target: EmitTarget) -> Result<String, String> {
    let name = match f {
        ArithmaFunction::Sin => "sin",
        ArithmaFunction::Cos => "cos",
        ArithmaFunction::Tan => "tan",
        ArithmaFunction::Exp => "exp",
        ArithmaFunction::Ln => "ln",
        ArithmaFunction::Log10 => "log10",
        ArithmaFunction::Log2 => "log2",
        ArithmaFunction::Abs => "abs",
        ArithmaFunction::Floor => "floor",
        ArithmaFunction::Ceil => "ceil",
        ArithmaFunction::Min => "min",
        ArithmaFunction::Max => "max",
        other => {
            return Err(format!(
                "no {target:?} spelling for `{other:?}`; add it to call_name"
            ))
        }
    };

    // Shader dialects diverge from the maths spelling in a couple of places.
    Ok(match (target, name) {
        (EmitTarget::Glsl, "ln") => "log".to_string(),
        (EmitTarget::Hlsl, "ln") => "log".to_string(),
        (EmitTarget::Hlsl, "log10") => "log10".to_string(),
        _ => name.to_string(),
    })
}

/// LaTeX spelling for a constant symbol.
fn latex_symbol(symbol: &str) -> String {
    match symbol {
        "π" | "pi" => "\\pi".to_string(),
        "τ" | "tau" => "\\tau".to_string(),
        "γ" | "gamma" => "\\gamma".to_string(),
        "φ" | "phi" => "\\phi".to_string(),
        "∞" => "\\infty".to_string(),
        other => other.to_string(),
    }
}

/// Format a number for the target.
fn fmt_number(v: f64, target: EmitTarget) -> String {
    match target {
        // GLSL/HLSL float literals need a decimal point, or `2` is an int and
        // `2 / 3` becomes integer division.
        EmitTarget::Glsl | EmitTarget::Hlsl => {
            if v.fract() == 0.0 && v.is_finite() {
                format!("{v:.1}")
            } else {
                format!("{v}")
            }
        }
        EmitTarget::MathMl => format!("<mn>{}</mn>", trim_number(v)),
        _ => trim_number(v),
    }
}

fn trim_number(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

fn node_kind(e: &ArithmaExpression) -> &'static str {
    match e {
        ArithmaExpression::Number(_) => "Number",
        ArithmaExpression::Variable(_) => "Variable",
        ArithmaExpression::Constant { .. } => "Constant",
        ArithmaExpression::Function(_, _) => "Function",
        ArithmaExpression::Sum { .. } => "Sum",
        ArithmaExpression::Product { .. } => "Product",
        ArithmaExpression::Limit { .. } => "Limit",
        ArithmaExpression::Conditional { .. } => "Conditional",
        ArithmaExpression::CachedValue { .. } => "CachedValue",
        ArithmaExpression::FourierOptimized { .. } => "FourierOptimized",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn x() -> ArithmaExpression {
        ArithmaExpression::var("x")
    }
    fn y() -> ArithmaExpression {
        ArithmaExpression::var("y")
    }
    fn n(v: i64) -> ArithmaExpression {
        ArithmaExpression::from_i64(v)
    }

    #[test]
    fn text_of_a_sum() {
        let e = ArithmaExpression::add(x(), n(2));
        assert_eq!(e.emit(EmitTarget::Text).unwrap(), "x + 2");
    }

    #[test]
    fn precedence_parenthesises_only_where_needed() {
        // (x + y) * 2 needs brackets; x + y*2 does not.
        let needs = ArithmaExpression::mul(ArithmaExpression::add(x(), y()), n(2));
        assert_eq!(needs.emit(EmitTarget::Text).unwrap(), "(x + y) * 2");

        let clean = ArithmaExpression::add(x(), ArithmaExpression::mul(y(), n(2)));
        assert_eq!(clean.emit(EmitTarget::Text).unwrap(), "x + y * 2");
    }

    #[test]
    fn subtraction_right_operand_is_bracketed() {
        // x - (y - 2) must keep its brackets or the value changes.
        let e = ArithmaExpression::sub(x(), ArithmaExpression::sub(y(), n(2)));
        assert_eq!(e.emit(EmitTarget::Text).unwrap(), "x - (y - 2)");
    }

    #[test]
    fn latex_uses_frac_and_juxtaposition() {
        let e = ArithmaExpression::div(x(), n(2));
        assert_eq!(e.emit(EmitTarget::Latex).unwrap(), "\\frac{x}{2}");

        let m = ArithmaExpression::mul(n(2), x());
        assert_eq!(m.emit(EmitTarget::Latex).unwrap(), "2 x");
    }

    #[test]
    fn latex_named_function_and_sqrt() {
        assert_eq!(
            ArithmaExpression::sin(x()).emit(EmitTarget::Latex).unwrap(),
            "\\sin{x}"
        );
        assert_eq!(
            ArithmaExpression::sqrt(x())
                .emit(EmitTarget::Latex)
                .unwrap(),
            "\\sqrt{x}"
        );
    }

    #[test]
    fn glsl_emits_float_literals_not_ints() {
        // `2` would be an int in GLSL and `x / 2` would truncate.
        let e = ArithmaExpression::div(x(), n(2));
        assert_eq!(e.emit(EmitTarget::Glsl).unwrap(), "x / 2.0");
    }

    #[test]
    fn glsl_uses_pow_not_caret() {
        // `^` is bitwise xor in GLSL — emitting it would compile and be wrong.
        let e = ArithmaExpression::pow(x(), n(3));
        let out = e.emit(EmitTarget::Glsl).unwrap();
        assert!(out.starts_with("pow("), "got {out}");
        assert!(!out.contains('^'), "got {out}");
    }

    #[test]
    fn glsl_renames_ln_to_log() {
        let e = ArithmaExpression::ln(x());
        assert_eq!(e.emit(EmitTarget::Glsl).unwrap(), "log(x)");
        // Text keeps the maths spelling.
        assert_eq!(e.emit(EmitTarget::Text).unwrap(), "ln(x)");
    }

    #[test]
    fn rpn_is_postfix_and_needs_no_brackets() {
        // (x + y) * 2  ->  x y + 2 *
        let e = ArithmaExpression::mul(ArithmaExpression::add(x(), y()), n(2));
        assert_eq!(e.emit(EmitTarget::EmlRpn).unwrap(), "x y + 2 *");
    }

    #[test]
    fn mathml_wraps_leaves_in_tags() {
        let e = ArithmaExpression::add(x(), n(2));
        let out = e.emit(EmitTarget::MathMl).unwrap();
        assert!(out.contains("<mi>x</mi>"), "got {out}");
        assert!(out.contains("<mn>2</mn>"), "got {out}");
        assert!(out.contains("<mo>+</mo>"), "got {out}");
    }

    #[test]
    fn deep_nesting_does_not_overflow_the_stack() {
        // The whole reason the walk is iterative.
        //
        // The expression is deliberately leaked with `mem::forget`. Dropping a
        // 20,000-deep `ArithmaExpression` overflows the stack on its own,
        // because the derived `Drop` recurses through the boxed children — a
        // pre-existing property of the AST that has nothing to do with `emit`.
        // Leaking isolates this test to the thing it is meant to prove.
        // See `deep_nesting_cannot_be_dropped` below, which pins the real limit.
        let mut e = x();
        for _ in 0..20_000 {
            e = ArithmaExpression::add(e, n(1));
        }
        let out = e.emit(EmitTarget::Text).unwrap();
        assert!(out.starts_with("x + 1"), "unexpected head: {}", &out[..12]);
        std::mem::forget(e);
    }

    #[test]
    fn moderately_deep_nesting_round_trips_including_drop() {
        // A depth the AST can both emit *and* drop, so this one owns its value
        // normally. If `ArithmaExpression` ever gains an iterative `Drop`, raise
        // this and delete the `mem::forget` above.
        let mut e = x();
        for _ in 0..500 {
            e = ArithmaExpression::add(e, n(1));
        }
        let out = e.emit(EmitTarget::Text).unwrap();
        assert!(out.starts_with("x + 1"));
        assert_eq!(out.matches("+ 1").count(), 500);
    }

    #[test]
    fn shader_target_refuses_a_constant_without_a_value() {
        // Emitting a bare `π` into GLSL would reference an undefined symbol.
        let c = ArithmaExpression::constant("π", Some("Pi"), None, false);
        assert!(c.emit(EmitTarget::Glsl).is_err());
        // With a cached value it inlines fine.
        let c2 = ArithmaExpression::constant("π", Some("Pi"), Some(std::f64::consts::PI), false);
        assert!(c2.emit(EmitTarget::Glsl).unwrap().starts_with("3.14"));
    }

    #[test]
    fn unsupported_operator_reports_which_one() {
        let e = ArithmaExpression::func(ArithmaFunction::Gamma, vec![x()]);
        let err = e.emit(EmitTarget::Text).unwrap_err();
        assert!(err.contains("Gamma"), "unexpected error: {err}");
    }
}
