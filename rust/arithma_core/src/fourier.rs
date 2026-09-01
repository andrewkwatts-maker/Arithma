//====== Arithma/rust/arithma_core/src/fourier.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Fourier
//!
//! Configuration plus the Fourier-transform pipeline. The engine uses this for
//! material-property baking (Phase 6, §E-Periodica) where every periodica
//! property is expanded into a Fourier coefficient texture for fast per-ray
//! evaluation in shaders.

use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use crate::expression::{ArithmaBindings, ArithmaExpression, Evaluable};

/// Window function used by the discrete transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArithmaFourierWindow {
    Rectangular,
    Hann,
    Hamming,
    Blackman,
    BlackmanHarris,
    Gaussian,
}

impl ArithmaFourierWindow {
    /// Window weight at normalised position `t ∈ [0, 1]`, where `t = 0` is the
    /// low end of the configured range and `t = 1` the high end (the
    /// "symmetric" — as opposed to "periodic" — window convention).
    ///
    /// `t` is clamped, so out-of-range inputs cannot produce a nonsensical
    /// weight.
    pub fn weight(self, t: f64) -> f64 {
        let t = if t.is_nan() { 0.0 } else { t.clamp(0.0, 1.0) };
        let a1 = 2.0 * PI * t;
        let a2 = 4.0 * PI * t;
        let a3 = 6.0 * PI * t;
        match self {
            Self::Rectangular => 1.0,
            Self::Hann => 0.5 - 0.5 * a1.cos(),
            Self::Hamming => 0.54 - 0.46 * a1.cos(),
            Self::Blackman => 0.42 - 0.5 * a1.cos() + 0.08 * a2.cos(),
            Self::BlackmanHarris => {
                0.35875 - 0.48829 * a1.cos() + 0.14128 * a2.cos() - 0.01168 * a3.cos()
            }
            Self::Gaussian => {
                // σ = 0.4 of the half-width, the usual DSP default.
                const SIGMA: f64 = 0.4;
                let u = (t - 0.5) / (SIGMA * 0.5);
                (-0.5 * u * u).exp()
            }
        }
    }
}

/// Per-pipeline configuration: sample count, range, accuracy target and window.
///
/// Mirrors `pt_arithmos::PTFourierConfig`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArithmaFourierConfig {
    /// Number of samples in the discrete transform.
    pub sample_count: usize,
    /// Domain range over which the transform is computed.
    pub range: (f64, f64),
    /// Number of harmonics retained in the truncated series.
    pub harmonics: usize,
    /// Target reconstruction accuracy (RMS error). Drives adaptive harmonic
    /// addition / dropout.
    pub accuracy: f64,
    /// Window function applied before the transform.
    pub window: ArithmaFourierWindow,
}

impl Default for ArithmaFourierConfig {
    fn default() -> Self {
        Self {
            sample_count: 1024,
            range: (-std::f64::consts::PI, std::f64::consts::PI),
            harmonics: 32,
            accuracy: 1e-6,
            window: ArithmaFourierWindow::Hann,
        }
    }
}

/// The result of running the Fourier pipeline. Carries the cosine and sine
/// coefficient arrays alongside the originating config so re-evaluation is
/// fully deterministic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArithmaFourierTransform {
    /// Configuration used to compute this transform.
    pub config: ArithmaFourierConfig,
    /// Cosine (real) coefficients.
    pub cos_coeffs: Vec<f64>,
    /// Sine (imaginary) coefficients.
    pub sin_coeffs: Vec<f64>,
    /// Constant DC offset.
    pub dc: f64,
}

impl ArithmaFourierTransform {
    /// Empty transform for a given config.
    pub fn empty(config: ArithmaFourierConfig) -> Self {
        let h = config.harmonics;
        Self {
            config,
            cos_coeffs: vec![0.0; h],
            sin_coeffs: vec![0.0; h],
            dc: 0.0,
        }
    }

    /// Angular frequency of the fundamental, `ω = 2π / (hi - lo)`.
    ///
    /// Returns `None` when the configured range is degenerate or non-finite.
    fn fundamental(&self) -> Option<f64> {
        let (lo, hi) = self.config.range;
        let length = hi - lo;
        if !length.is_finite() || length <= 0.0 {
            return None;
        }
        Some(2.0 * PI / length)
    }

    /// Reconstruct the value at `x` using the truncated Fourier series.
    ///
    /// Evaluates
    ///
    /// ```text
    /// f(x) ≈ dc + Σ_{n=1..H} [ a_n·cos(n·ω·x) + b_n·sin(n·ω·x) ],  ω = 2π/(hi-lo)
    /// ```
    ///
    /// Note the basis uses the *absolute* coordinate `x`, not an offset from
    /// `range.0`. That is legitimate for any range of length `L` because
    /// `cos(nωx)` / `sin(nωx)` have period `L/n`, so they remain orthogonal on
    /// every interval of length `L` — and it makes the coefficients match the
    /// textbook series (for the default range `[-π, π]`, `sin(x)` has
    /// `b_1 = 1`).
    ///
    /// Degenerate configs (zero-width or non-finite range) reconstruct to the
    /// DC term alone rather than producing NaN.
    pub fn evaluate(&self, x: f64) -> f64 {
        let Some(omega) = self.fundamental() else {
            return self.dc;
        };
        let mut acc = self.dc;
        // Bounded by the shorter coefficient array, so a mismatched pair of
        // vectors cannot index out of range.
        for (index, (&a, &b)) in self
            .cos_coeffs
            .iter()
            .zip(self.sin_coeffs.iter())
            .enumerate()
        {
            let angle = (index as f64 + 1.0) * omega * x;
            acc += a * angle.cos() + b * angle.sin();
        }
        acc
    }

    /// Root-mean-square error of the reconstruction against `expr`, sampled
    /// uniformly across the configured range.
    ///
    /// Useful for checking whether the `accuracy` target was actually met.
    pub fn rms_error(&self, expr: &ArithmaExpression, var: &str) -> Result<f64, String> {
        let (lo, hi) = self.config.range;
        let length = hi - lo;
        if !length.is_finite() || length <= 0.0 {
            return Err("rms_error: degenerate range".into());
        }
        let samples = self
            .config
            .sample_count
            .clamp(2, ARITHMA_FOURIER_MAX_SAMPLES);
        let mut bindings = ArithmaBindings::new();
        bindings.insert(var.to_string(), lo);
        let mut acc = 0.0;
        // Fixed bound: `samples` is clamped above.
        for k in 0..samples {
            let x = lo + length * (k as f64 + 0.5) / samples as f64;
            match bindings.get_mut(var) {
                Some(slot) => *slot = x,
                None => return Err("rms_error: binding slot missing".into()),
            }
            let truth = expr.evaluate(&bindings)?;
            let approx = self.evaluate(x);
            let d = truth - approx;
            acc += d * d;
        }
        Ok((acc / samples as f64).sqrt())
    }
}

/// Hard cap on `ArithmaFourierConfig::sample_count` (CLAUDE.md safety rule 2:
/// every loop has a fixed bound).
pub const ARITHMA_FOURIER_MAX_SAMPLES: usize = 1 << 20;
/// Hard cap on `ArithmaFourierConfig::harmonics`.
pub const ARITHMA_FOURIER_MAX_HARMONICS: usize = 1 << 12;
/// Hard cap on the `sample_count × harmonics` quadrature work budget.
const ARITHMA_FOURIER_MAX_WORK: usize = 1 << 26;

/// Composite Simpson quadrature over `values`, which must hold `n + 1` samples
/// of the integrand at uniform spacing `step` with `n` even.
///
/// Returns `0.0` for degenerate input rather than panicking.
fn simpson(values: &[f64], step: f64) -> f64 {
    if values.len() < 3 || (values.len() - 1) % 2 != 0 {
        return 0.0;
    }
    let last = values.len() - 1;
    let mut sum = values[0] + values[last];
    for (k, v) in values.iter().enumerate().take(last).skip(1) {
        sum += if k % 2 == 1 { 4.0 } else { 2.0 } * v;
    }
    sum * step / 3.0
}

/// Compute the real Fourier series of `expr` with respect to `var` over the
/// configured range.
///
/// # Method
///
/// The expression is sampled on a uniform grid of `sample_count` intervals
/// (rounded up to the next even number so composite Simpson applies) spanning
/// `config.range = [lo, hi]`. With `L = hi - lo` and `ω = 2π/L`, each
/// coefficient is a quadrature of the windowed integrand:
///
/// ```text
/// g   = (1/L) ∫ w(x) dx                       (window coherent gain)
/// dc  = (1/(L·g)) ∫ f(x)·w(x) dx
/// a_n = (2/(L·g)) ∫ f(x)·w(x)·cos(nωx) dx
/// b_n = (2/(L·g)) ∫ f(x)·w(x)·sin(nωx) dx
/// ```
///
/// Integration is **composite Simpson's rule** — `O(h⁴)` error on smooth
/// integrands, so the default 1024 samples resolve a sinusoid to roughly
/// `1e-13`.
///
/// Discontinuous input is the one case where the quadrature order does not
/// help: a jump inside a Simpson panel degrades the whole rule to `O(h)`, and a
/// jump sitting exactly *on* a node still leaves an `O(h)` residue because the
/// single sampled value cannot be both one-sided limit at once. Expect roughly
/// `h/π` absolute error per coefficient for a unit square wave (~2e-4 at 4096
/// samples over a `2π` range). Raise `sample_count` if that matters.
///
/// Dividing by the coherent gain `g` normalises the window away from the
/// amplitude scale, so a constant signal reproduces its own value as `dc` and a
/// pure sinusoid at a bin centre keeps unit amplitude under *any* window. A
/// rectangular window has `g = 1` and is therefore a pure no-op.
///
/// # Accuracy / harmonic dropout
///
/// `config.accuracy` is a target RMS reconstruction error. After the transform,
/// harmonic pairs whose amplitude `√(a_n² + b_n²)` falls below
/// `accuracy · √(2/H)` are zeroed. Because a harmonic of amplitude `A`
/// contributes `A/√2` in RMS, dropping up to `H` such harmonics keeps the total
/// discarded RMS energy at or below `accuracy`. Pass `accuracy = 0.0` to keep
/// the raw coefficients.
///
/// # Errors
///
/// Returns `Err` for a degenerate or non-finite range, a `sample_count` below 2
/// or above [`ARITHMA_FOURIER_MAX_SAMPLES`], a `harmonics` count of 0 or above
/// [`ARITHMA_FOURIER_MAX_HARMONICS`], a work budget overrun, a window with no
/// coherent gain, or any sample where `expr` fails to evaluate to a finite f64.
pub fn fourier_transform(
    expr: &ArithmaExpression,
    var: &str,
    config: &ArithmaFourierConfig,
) -> Result<ArithmaFourierTransform, String> {
    let (lo, hi) = config.range;
    if !lo.is_finite() || !hi.is_finite() {
        return Err(format!("fourier_transform: non-finite range ({lo}, {hi})"));
    }
    let length = hi - lo;
    if length <= 0.0 {
        return Err(format!(
            "fourier_transform: range ({lo}, {hi}) must satisfy lo < hi"
        ));
    }
    if config.sample_count < 2 {
        return Err("fourier_transform: sample_count must be at least 2".into());
    }
    if config.sample_count > ARITHMA_FOURIER_MAX_SAMPLES {
        return Err(format!(
            "fourier_transform: sample_count {} exceeds cap {ARITHMA_FOURIER_MAX_SAMPLES}",
            config.sample_count
        ));
    }
    if config.harmonics == 0 {
        return Err("fourier_transform: harmonics must be at least 1".into());
    }
    if config.harmonics > ARITHMA_FOURIER_MAX_HARMONICS {
        return Err(format!(
            "fourier_transform: harmonics {} exceeds cap {ARITHMA_FOURIER_MAX_HARMONICS}",
            config.harmonics
        ));
    }

    // Simpson needs an even number of intervals.
    let intervals = config.sample_count + (config.sample_count % 2);
    if intervals.saturating_mul(config.harmonics) > ARITHMA_FOURIER_MAX_WORK {
        return Err(format!(
            "fourier_transform: sample_count × harmonics exceeds the {ARITHMA_FOURIER_MAX_WORK} work budget"
        ));
    }

    let step = length / intervals as f64;
    let omega = 2.0 * PI / length;

    // ---- sample the expression and the window once ------------------------
    let mut bindings = ArithmaBindings::new();
    bindings.insert(var.to_string(), lo);
    let mut abscissae = Vec::with_capacity(intervals + 1);
    let mut windowed = Vec::with_capacity(intervals + 1);
    let mut window_only = Vec::with_capacity(intervals + 1);
    for k in 0..=intervals {
        // Snap the final abscissa to `hi` so rounding cannot walk off the range.
        let x = if k == intervals {
            hi
        } else {
            lo + step * k as f64
        };
        match bindings.get_mut(var) {
            Some(slot) => *slot = x,
            None => return Err("fourier_transform: binding slot missing".into()),
        }
        let value = expr.evaluate(&bindings)?;
        if !value.is_finite() {
            return Err(format!(
                "fourier_transform: expression is not finite at {var} = {x}"
            ));
        }
        let w = config.window.weight(k as f64 / intervals as f64);
        abscissae.push(x);
        window_only.push(w);
        windowed.push(value * w);
    }

    // ---- window coherent gain --------------------------------------------
    let gain = simpson(&window_only, step) / length;
    if !gain.is_finite() || gain.abs() < 1e-12 {
        return Err(format!(
            "fourier_transform: window {:?} has no usable coherent gain ({gain})",
            config.window
        ));
    }

    let dc = simpson(&windowed, step) / (length * gain);
    let scale = 2.0 / (length * gain);

    // ---- per-harmonic quadrature -----------------------------------------
    let mut cos_coeffs = Vec::with_capacity(config.harmonics);
    let mut sin_coeffs = Vec::with_capacity(config.harmonics);
    for harmonic in 1..=config.harmonics {
        let freq = harmonic as f64 * omega;
        let mut cos_sum = 0.0;
        let mut sin_sum = 0.0;
        for (k, (&fw, &x)) in windowed.iter().zip(abscissae.iter()).enumerate() {
            let quad_weight = if k == 0 || k == intervals {
                1.0
            } else if k % 2 == 1 {
                4.0
            } else {
                2.0
            };
            let angle = freq * x;
            cos_sum += quad_weight * fw * angle.cos();
            sin_sum += quad_weight * fw * angle.sin();
        }
        cos_coeffs.push(cos_sum * step / 3.0 * scale);
        sin_coeffs.push(sin_sum * step / 3.0 * scale);
    }

    // ---- adaptive dropout against the accuracy target ---------------------
    if config.accuracy.is_finite() && config.accuracy > 0.0 {
        let threshold = config.accuracy * (2.0 / config.harmonics as f64).sqrt();
        for (a, b) in cos_coeffs.iter_mut().zip(sin_coeffs.iter_mut()) {
            if (*a * *a + *b * *b).sqrt() < threshold {
                *a = 0.0;
                *b = 0.0;
            }
        }
    }

    Ok(ArithmaFourierTransform {
        config: config.clone(),
        cos_coeffs,
        sin_coeffs,
        dc,
    })
}

// ---------------------------------------------------------------------------
// Backward-compatibility aliases for the pre-rename `Arithmos*` names.
// Retained for one release; downstream (eml-math, eml-spectral, metaphysica,
// periodica) should migrate to the `Arithma*` names above.
// ---------------------------------------------------------------------------
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaFourierConfig`")]
#[allow(unused)]
pub use self::ArithmaFourierConfig as ArithmosFourierConfig;
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaFourierTransform`")]
#[allow(unused)]
pub use self::ArithmaFourierTransform as ArithmosFourierTransform;
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaFourierWindow`")]
#[allow(unused)]
pub use self::ArithmaFourierWindow as ArithmosFourierWindow;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::function::ArithmaFunction;

    /// Raw (no-dropout, no-window) config over `[-π, π]`.
    fn raw_config(samples: usize, harmonics: usize) -> ArithmaFourierConfig {
        ArithmaFourierConfig {
            sample_count: samples,
            range: (-PI, PI),
            harmonics,
            accuracy: 0.0,
            window: ArithmaFourierWindow::Rectangular,
        }
    }

    fn sin_x() -> ArithmaExpression {
        ArithmaExpression::sin(ArithmaExpression::var("x"))
    }

    /// `sgn(sin x)` — the unit square wave on `[-π, π]`.
    fn square_wave() -> ArithmaExpression {
        ArithmaExpression::func(ArithmaFunction::Sign, vec![sin_x()])
    }

    #[test]
    fn default_config_is_sensible() {
        let cfg = ArithmaFourierConfig::default();
        assert!(cfg.sample_count > 0);
        assert!(cfg.harmonics > 0);
        assert!(cfg.accuracy > 0.0);
    }

    #[test]
    fn empty_transform_has_correct_size() {
        let cfg = ArithmaFourierConfig::default();
        let transform = ArithmaFourierTransform::empty(cfg.clone());
        assert_eq!(transform.cos_coeffs.len(), cfg.harmonics);
        assert_eq!(transform.sin_coeffs.len(), cfg.harmonics);
    }

    // ---- window weights ---------------------------------------------------

    #[test]
    fn window_weights_match_their_definitions() {
        assert_eq!(ArithmaFourierWindow::Rectangular.weight(0.37), 1.0);
        // Hann is 0 at both ends and 1 in the middle.
        assert!(ArithmaFourierWindow::Hann.weight(0.0).abs() < 1e-12);
        assert!((ArithmaFourierWindow::Hann.weight(0.5) - 1.0).abs() < 1e-12);
        assert!(ArithmaFourierWindow::Hann.weight(1.0).abs() < 1e-12);
        // Hamming has the classic 0.08 pedestal.
        assert!((ArithmaFourierWindow::Hamming.weight(0.0) - 0.08).abs() < 1e-12);
        // Gaussian peaks at 1 in the centre and decays symmetrically.
        assert!((ArithmaFourierWindow::Gaussian.weight(0.5) - 1.0).abs() < 1e-12);
        let g = ArithmaFourierWindow::Gaussian;
        assert!((g.weight(0.25) - g.weight(0.75)).abs() < 1e-12);
        assert!(g.weight(0.0) < g.weight(0.25));
    }

    // ---- coefficient accuracy --------------------------------------------

    #[test]
    fn pure_sine_gives_unit_first_harmonic_and_nothing_else() {
        let t = fourier_transform(&sin_x(), "x", &raw_config(1024, 8)).unwrap();
        assert!(t.dc.abs() < 1e-10, "dc = {}", t.dc);
        assert!(
            (t.sin_coeffs[0] - 1.0).abs() < 1e-9,
            "b1 = {}",
            t.sin_coeffs[0]
        );
        for (n, &b) in t.sin_coeffs.iter().enumerate().skip(1) {
            assert!(b.abs() < 1e-9, "b{} = {b}", n + 1);
        }
        for (n, &a) in t.cos_coeffs.iter().enumerate() {
            assert!(a.abs() < 1e-9, "a{} = {a}", n + 1);
        }
    }

    #[test]
    fn cos_of_two_x_lands_on_the_second_cosine_harmonic() {
        // cos(2x)
        let expr = ArithmaExpression::cos(ArithmaExpression::mul(
            ArithmaExpression::from_i64(2),
            ArithmaExpression::var("x"),
        ));
        let t = fourier_transform(&expr, "x", &raw_config(1024, 6)).unwrap();
        assert!(
            (t.cos_coeffs[1] - 1.0).abs() < 1e-9,
            "a2 = {}",
            t.cos_coeffs[1]
        );
        assert!(t.cos_coeffs[0].abs() < 1e-9);
        assert!(t.cos_coeffs[2].abs() < 1e-9);
        assert!(t.dc.abs() < 1e-10);
        for &b in &t.sin_coeffs {
            assert!(b.abs() < 1e-9, "sine leakage {b}");
        }
    }

    #[test]
    fn constant_offset_lands_entirely_in_dc() {
        // 3 + sin(x)
        let expr = ArithmaExpression::add(ArithmaExpression::from_i64(3), sin_x());
        let t = fourier_transform(&expr, "x", &raw_config(512, 4)).unwrap();
        assert!((t.dc - 3.0).abs() < 1e-10, "dc = {}", t.dc);
        assert!((t.sin_coeffs[0] - 1.0).abs() < 1e-9);
        assert!(t.cos_coeffs[0].abs() < 1e-9);
    }

    #[test]
    fn square_wave_odd_harmonics_follow_four_over_n_pi() {
        // 4096 intervals: the jumps at x = 0, ±π land on Simpson panel
        // boundaries, so each smooth half is integrated at full order.
        let t = fourier_transform(&square_wave(), "x", &raw_config(4096, 9)).unwrap();
        // `sgn` at the jump (x = 0) can only report one of the two one-sided
        // limits, which leaves an O(h) ≈ 2e-4 residue in the even-symmetric
        // terms. The sine coefficients are immune because sin(n·0) = 0.
        assert!(t.dc.abs() < 1e-3, "dc = {}", t.dc);
        for n in [1usize, 3, 5, 7, 9] {
            let expected = 4.0 / (n as f64 * PI);
            let got = t.sin_coeffs[n - 1];
            assert!(
                (got - expected).abs() < 1e-3 * expected,
                "b{n} = {got}, expected {expected}"
            );
        }
        for n in [2usize, 4, 6, 8] {
            assert!(
                t.sin_coeffs[n - 1].abs() < 1e-6,
                "even harmonic b{n} = {}",
                t.sin_coeffs[n - 1]
            );
        }
        // Odd function ⇒ no cosine content.
        for (n, &a) in t.cos_coeffs.iter().enumerate() {
            assert!(a.abs() < 1e-3, "a{} = {a}", n + 1);
        }
    }

    // ---- reconstruction ---------------------------------------------------

    #[test]
    fn evaluate_reconstructs_a_sine_wave() {
        let t = fourier_transform(&sin_x(), "x", &raw_config(1024, 8)).unwrap();
        for k in 0..17 {
            let x = -PI + 2.0 * PI * k as f64 / 16.0;
            assert!(
                (t.evaluate(x) - x.sin()).abs() < 1e-8,
                "reconstruction at {x}: {} vs {}",
                t.evaluate(x),
                x.sin()
            );
            assert!((t.evaluate(x) - x.sin()).abs() < 1e-8);
        }
        assert!(t.rms_error(&sin_x(), "x").unwrap() < 1e-8);
    }

    #[test]
    fn evaluate_reconstructs_square_wave_plateaus_with_gibbs_overshoot() {
        let t = fourier_transform(&square_wave(), "x", &raw_config(4096, 15)).unwrap();
        // Mid-plateau values sit near ±1 (Gibbs ripple is a few percent here).
        assert!(
            (t.evaluate(PI / 2.0) - 1.0).abs() < 0.12,
            "at +π/2: {}",
            t.evaluate(PI / 2.0)
        );
        assert!(
            (t.evaluate(-PI / 2.0) + 1.0).abs() < 0.12,
            "at -π/2: {}",
            t.evaluate(-PI / 2.0)
        );
        // The jump itself reconstructs to the midpoint of the two limits (up to
        // the O(h) sampling residue at the discontinuity — see above).
        assert!(t.evaluate(0.0).abs() < 1e-2, "at 0: {}", t.evaluate(0.0));
        // Gibbs overshoot: the first ripple past the jump — at roughly
        // x = π/(M+1) for an M-harmonic partial sum — exceeds 1 by ~18%.
        let overshoot = t.evaluate(PI / 16.0);
        assert!(
            overshoot > 1.1 && overshoot < 1.25,
            "expected ~1.18 Gibbs overshoot, got {overshoot}"
        );
    }

    #[test]
    fn evaluate_is_periodic_with_the_configured_range() {
        let t = fourier_transform(&sin_x(), "x", &raw_config(512, 4)).unwrap();
        let x = 0.7;
        assert!((t.evaluate(x) - t.evaluate(x + 2.0 * PI)).abs() < 1e-9);
    }

    // ---- windowing --------------------------------------------------------

    #[test]
    fn coherent_gain_normalisation_preserves_dc_under_every_window() {
        let expr = ArithmaExpression::from_i64(5);
        for window in [
            ArithmaFourierWindow::Rectangular,
            ArithmaFourierWindow::Hann,
            ArithmaFourierWindow::Hamming,
            ArithmaFourierWindow::Blackman,
            ArithmaFourierWindow::BlackmanHarris,
            ArithmaFourierWindow::Gaussian,
        ] {
            let cfg = ArithmaFourierConfig {
                window,
                ..raw_config(512, 4)
            };
            let t = fourier_transform(&expr, "x", &cfg).unwrap();
            assert!(
                (t.dc - 5.0).abs() < 1e-9,
                "{window:?} gave dc = {} (expected 5)",
                t.dc
            );
        }
    }

    #[test]
    fn hann_window_smears_a_pure_sine_into_neighbouring_harmonics() {
        let rect = fourier_transform(&sin_x(), "x", &raw_config(1024, 4)).unwrap();
        let cfg = ArithmaFourierConfig {
            window: ArithmaFourierWindow::Hann,
            ..raw_config(1024, 4)
        };
        let hann = fourier_transform(&sin_x(), "x", &cfg).unwrap();
        // Rectangular is exact; Hann leaks measurably into the 2nd harmonic.
        assert!(rect.sin_coeffs[1].abs() < 1e-9);
        assert!(
            hann.sin_coeffs[1].abs() > 0.1,
            "expected Hann leakage, got {}",
            hann.sin_coeffs[1]
        );
        // The fundamental still keeps roughly unit amplitude.
        assert!((hann.sin_coeffs[0] - 1.0).abs() < 0.5);
    }

    // ---- accuracy-driven dropout -----------------------------------------

    #[test]
    fn accuracy_target_zeroes_negligible_harmonics() {
        let cfg = ArithmaFourierConfig {
            accuracy: 1e-3,
            ..raw_config(1024, 8)
        };
        let t = fourier_transform(&sin_x(), "x", &cfg).unwrap();
        assert!((t.sin_coeffs[0] - 1.0).abs() < 1e-9);
        // Dropout works on (a_n, b_n) *pairs*: harmonic 1 survives because its
        // sine part is 1.0, so only its cosine rounding noise remains.
        assert!(t.cos_coeffs[0].abs() < 1e-9);
        for (n, &b) in t.sin_coeffs.iter().enumerate().skip(1) {
            assert_eq!(b, 0.0, "b{} should have been dropped", n + 1);
        }
        for (n, &a) in t.cos_coeffs.iter().enumerate().skip(1) {
            assert_eq!(a, 0.0, "a{} should have been dropped", n + 1);
        }
        // Dropping only sub-threshold harmonics keeps the promised RMS bound.
        assert!(t.rms_error(&sin_x(), "x").unwrap() <= 1e-3);
    }

    #[test]
    fn accuracy_zero_keeps_raw_coefficients() {
        let t = fourier_transform(&sin_x(), "x", &raw_config(1024, 8)).unwrap();
        // Rounding noise is retained rather than snapped to exactly zero.
        assert!(
            t.sin_coeffs[1..].iter().any(|b| *b != 0.0) || t.cos_coeffs.iter().any(|a| *a != 0.0)
        );
    }

    // ---- validation -------------------------------------------------------

    #[test]
    fn rejects_degenerate_and_oversized_configs() {
        let bad_range = ArithmaFourierConfig {
            range: (1.0, 1.0),
            ..raw_config(64, 4)
        };
        assert!(fourier_transform(&sin_x(), "x", &bad_range).is_err());

        let no_samples = raw_config(1, 4);
        assert!(fourier_transform(&sin_x(), "x", &no_samples).is_err());

        let no_harmonics = raw_config(64, 0);
        assert!(fourier_transform(&sin_x(), "x", &no_harmonics).is_err());

        let too_many = raw_config(64, ARITHMA_FOURIER_MAX_HARMONICS + 1);
        assert!(fourier_transform(&sin_x(), "x", &too_many).is_err());

        let nan_range = ArithmaFourierConfig {
            range: (f64::NAN, 1.0),
            ..raw_config(64, 4)
        };
        assert!(fourier_transform(&sin_x(), "x", &nan_range).is_err());
    }

    #[test]
    fn unbound_variable_is_an_error_not_a_zero_transform() {
        let expr = ArithmaExpression::var("y");
        let err = fourier_transform(&expr, "x", &raw_config(64, 4)).unwrap_err();
        assert!(err.contains('y'), "unexpected error: {err}");
    }

    #[test]
    fn non_finite_samples_are_rejected() {
        // 1/x over a range straddling zero hits a division by zero.
        let expr =
            ArithmaExpression::div(ArithmaExpression::from_i64(1), ArithmaExpression::var("x"));
        assert!(fourier_transform(&expr, "x", &raw_config(64, 4)).is_err());
    }

    #[test]
    fn degenerate_transform_evaluates_to_dc_without_nan() {
        let mut t = ArithmaFourierTransform::empty(ArithmaFourierConfig {
            range: (0.0, 0.0),
            ..raw_config(64, 4)
        });
        t.dc = 2.5;
        assert_eq!(t.evaluate(1.234), 2.5);
    }
}
