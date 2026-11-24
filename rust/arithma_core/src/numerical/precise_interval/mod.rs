//====== Arithma/rust/arithma_core/src/numerical/precise_interval/mod.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Precise interval evaluation
//!
//! [`evaluate_interval`](crate::numerical::interval_analysis::evaluate_interval)
//! narrows the image of a *one-variable* expression over an input interval,
//! but has no way to evaluate a fully-bound (closed-form constant)
//! expression to a requested number of bits: every leaf collapses to a
//! point `f64`, so the result is exactly the `f64` evaluation, no tighter.
//! This module closes that gap with [`evaluate_precise`], which evaluates
//! any supported expression over exact rational bindings to a caller-chosen
//! precision, bounded end to end by exact `ArithmaInteger` arithmetic — no
//! `f64` is read anywhere in the numeric path.
//!
//! ## Rational type
//!
//! [`Rational`] lives in [`crate::rational`] (re-exported here); every
//! arithmetic operation is exact. The "directed outward rounding" this
//! module's contract calls for is realised differently for each bounded
//! operation, never by truncating a rational:
//! - `sqrt` brackets the root with a Newton iterate and its AM-GM-derived
//!   partner (`y`, `x/y`), both *exact* rationals that bracket the root with
//!   no rounding step at all;
//! - `exp`/`ln`/`sin`/`cos`/`atan` sum a Taylor series to a partial sum `S`
//!   plus a rigorous tail bound `T` (itself an exact rational), returning
//!   `[S - T, S + T]` — the bound is added outward, never trimmed back in.
//!
//! Every one of these brackets is a sound enclosure of the true value by
//! construction, so composing them (via the same corner-hull rules
//! [`interval_analysis`](crate::numerical::interval_analysis) uses, just
//! over exact rationals instead of `f64`) never under-claims precision.
//!
//! ## The `pi` constant
//!
//! The only `Constant` this module recognises is `pi`: a `symbol` of
//! exactly `"pi"` or `"π"`. There is no other convention to match (nothing
//! else in the codebase evaluates `pi` symbolically), so this file's own
//! tests are the only contract for that spelling.
//!
//! ## Supported nodes
//!
//! `Number`, a bound `Variable`, the `pi` `Constant`, and `Function` nodes
//! for `+ − × ÷`, `pow` (small integer exponent only), `sqrt`, `exp`, `ln`,
//! `sin`, `cos`. Everything else — `Sum`, `Limit`, `Product`, `Conditional`,
//! `CachedValue`, `FourierOptimized`, and any `ArithmaFunction` not listed
//! above — is a named error, never a silent approximation.

mod evaluate;
mod interval;
mod rounding;
mod series;
mod sqrt;
#[cfg(test)]
mod tests;

pub use crate::rational::Rational;
pub use evaluate::evaluate_precise;
pub use interval::PreciseInterval;
