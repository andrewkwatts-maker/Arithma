//====== Arithma/rust/arithma_core/src/expression/emit_names.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! Function spellings for the host and WGSL emit targets.
//!
//! A spelling is honest when it matches the CPU oracle
//! (`arithma_apply_function`) on finite inputs, to the target's float precision
//! for WGSL. A function with no honest spelling is refused by name, so source
//! text never carries a call that means something else on the other side.

use super::EmitTarget;
use crate::function::ArithmaFunction;

/// Rust, C and Python spelling of the parsed function `name` (as `call_name`
/// writes it), or an error naming the function and the target.
///
/// Rust writes `f64::<name>(x)`, C the `math.h` name, Python `math.<name>(x)`.
/// `sign` is `f64::signum` on Rust, which matches the oracle (sign(0) = 1) and
/// has no C or Python equivalent. `round` is refused on Python only (half to
/// even there; half away from zero on the oracle, Rust and C). One-argument
/// `log` has no fixed base and is refused everywhere.
pub(super) fn host_name(name: &str, target: EmitTarget) -> Result<String, String> {
    let spelled = match (target, name) {
        (_, "log") => None,
        (EmitTarget::Rust, "sign") => Some("f64::signum".to_string()),
        (EmitTarget::C | EmitTarget::Python, "sign") => None,
        (EmitTarget::Python, "round") => None,
        (EmitTarget::Rust, other) => Some(format!("f64::{other}")),
        (EmitTarget::C, other) => Some(c_name(other).to_string()),
        (EmitTarget::Python, "abs" | "min" | "max") => Some(name.to_string()),
        (EmitTarget::Python, "ln") => Some("math.log".to_string()),
        (EmitTarget::Python, other) => Some(format!("math.{other}")),
        _ => return Err(format!("`{name}` is not a host call for {target:?}")),
    };
    spelled.ok_or_else(|| refusal(name, target))
}

/// WGSL spelling of the parsed function `name`, or an error naming the
/// function and `Wgsl`.
///
/// `ln` is `log`. `log10` has no built-in, `round` is half to even and `sign`
/// is 0 at 0 (the oracle's is 1), so those three and one-argument `log` are
/// refused. `cbrt` is written by `emit_call` from `pow`, `sign` and `abs`.
pub(super) fn wgsl_name(name: &str) -> Result<String, String> {
    match name {
        "log10" | "round" | "sign" | "log" => Err(refusal(name, EmitTarget::Wgsl)),
        "ln" => Ok("log".to_string()),
        other => Ok(other.to_string()),
    }
}

/// Refuse a literal the target cannot hold: any non-finite value on the host
/// targets, and additionally anything beyond `f32::MAX` on WGSL. Targets
/// outside the new four accept every value, as before.
pub(super) fn check_literal(v: f64, target: EmitTarget) -> Result<(), String> {
    let holds = match target {
        EmitTarget::Rust | EmitTarget::C | EmitTarget::Python => v.is_finite(),
        EmitTarget::Wgsl => v.is_finite() && v.abs() <= f64::from(f32::MAX),
        _ => true,
    };
    if holds {
        Ok(())
    } else {
        Err(format!("the value {v:e} cannot be written as a {target:?} literal"))
    }
}

/// `math.h` name for a parsed function name.
fn c_name(name: &str) -> &str {
    match name {
        "ln" => "log",
        "abs" => "fabs",
        "min" => "fmin",
        "max" => "fmax",
        other => other,
    }
}

/// Error for a function the target cannot spell honestly.
fn refusal(name: &str, target: EmitTarget) -> String {
    format!("`{name}` has no honest {target:?} spelling")
}

/// Maths spelling of a parsed function, before any target-specific change.
pub(super) fn maths_name(f: &ArithmaFunction) -> Option<&'static str> {
    Some(match f {
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
        // The inverse, hyperbolic and rounding families. These were missing,
        // so every one of them PARSED and then could not be written back --
        // `parse::FUNCTIONS` accepted 22 names and this table spelled 12. The
        // symmetry test in `parse` now fails if the two ever drift again.
        ArithmaFunction::Asin => "asin",
        ArithmaFunction::Acos => "acos",
        ArithmaFunction::Atan => "atan",
        ArithmaFunction::Atan2 => "atan2",
        ArithmaFunction::Sinh => "sinh",
        ArithmaFunction::Cosh => "cosh",
        ArithmaFunction::Tanh => "tanh",
        ArithmaFunction::Asinh => "asinh",
        ArithmaFunction::Acosh => "acosh",
        ArithmaFunction::Atanh => "atanh",
        ArithmaFunction::Cbrt => "cbrt",
        ArithmaFunction::Log => "log",
        ArithmaFunction::Round => "round",
        ArithmaFunction::Sign => "sign",
        _ => return None,
    })
}
