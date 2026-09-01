//====== Arithma/rust/arithma_core/src/probabilities/statistical_moment.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # Statistical moment
//!
//! Moments of distributions and samples — variance, skewness, kurtosis.
//!
//! ## Convention (read this before comparing against another library)
//!
//! Every function here returns the **sample** (unbiased-estimator) statistic,
//! not the population one, and kurtosis is reported in **excess** form
//! (normal ⇒ 0, i.e. already `− 3`). Concretely:
//!
//! | Function     | Formula                                   | Matches |
//! |--------------|-------------------------------------------|---------|
//! | [`ArithmaStatisticalMoment::variance`] | `s² = Σ(xᵢ-x̄)²/(n-1)` (Bessel) | NumPy `var(ddof=1)`, Excel `VAR.S`, R `var` |
//! | [`ArithmaStatisticalMoment::skewness`] | `G₁ = n/((n-1)(n-2)) · Σ((xᵢ-x̄)/s)³` | Excel `SKEW`, SAS/SPSS, `scipy.stats.skew(bias=False)` |
//! | [`ArithmaStatisticalMoment::kurtosis`] | `G₂ = n(n+1)/((n-1)(n-2)(n-3)) · Σ((xᵢ-x̄)/s)⁴ − 3(n-1)²/((n-2)(n-3))` | Excel `KURT`, SAS/SPSS, `scipy.stats.kurtosis(fisher=True, bias=False)` |
//!
//! These deliberately do **not** match NumPy's / `scipy`'s *default* moment
//! helpers, which report the biased population forms
//! `g₁ = m₃/m₂^{3/2}` and `g₂ = m₄/m₂² − 3`. If you need those, divide out:
//! `g₁ = G₁ · (n-1)^{3/2}/n` … it is not a scale factor away for kurtosis, so
//! pick the convention deliberately.
//!
//! Because they are unbiased estimators, each has a minimum sample size:
//! mean needs `n ≥ 1`, variance `n ≥ 2`, skewness `n ≥ 3`, kurtosis `n ≥ 4`.
//! Below that the estimator is undefined and an `Err` is returned.

/// Static helper for moment calculations on f64 datasets.
pub struct ArithmaStatisticalMoment;

impl ArithmaStatisticalMoment {
    /// Reject datasets containing non-finite values, which would silently
    /// poison every downstream moment with `NaN`.
    fn check_finite(data: &[f64], who: &str) -> Result<(), String> {
        if let Some(pos) = data.iter().position(|v| !v.is_finite()) {
            return Err(format!(
                "ArithmaStatisticalMoment::{who}: data[{pos}] is not finite ({})",
                data[pos]
            ));
        }
        Ok(())
    }

    /// Arithmetic mean `x̄ = Σxᵢ / n`.
    ///
    /// Requires a non-empty slice; an empty slice is an `Err`, not `NaN`.
    pub fn mean(data: &[f64]) -> Result<f64, String> {
        if data.is_empty() {
            return Err("ArithmaStatisticalMoment::mean: data must not be empty".to_string());
        }
        Self::check_finite(data, "mean")?;
        Ok(data.iter().sum::<f64>() / data.len() as f64)
    }

    /// **Sample** variance `s² = Σ(xᵢ - x̄)² / (n - 1)` — Bessel-corrected, so
    /// it is the unbiased estimator of the population variance and matches
    /// Excel `VAR.S` / `numpy.var(ddof=1)` / R `var`.
    ///
    /// Requires `n ≥ 2`.
    ///
    /// Note this is the *sample* convention, whereas the `variance()` method on
    /// the distribution types is the exact *population* variance of that
    /// distribution — the two answer different questions and are expected to
    /// differ by the `n/(n-1)` factor when a sample is drawn from a
    /// distribution.
    pub fn variance(data: &[f64]) -> Result<f64, String> {
        if data.len() < 2 {
            return Err(format!(
                "ArithmaStatisticalMoment::variance: needs at least 2 points, got {}",
                data.len()
            ));
        }
        Self::check_finite(data, "variance")?;
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let ss: f64 = data.iter().map(|v| (v - mean) * (v - mean)).sum();
        Ok(ss / (n - 1.0))
    }

    /// **Sample** skewness — the adjusted Fisher–Pearson standardised third
    /// moment `G₁`:
    ///
    /// ```text
    /// G1 = n / ((n-1)(n-2)) * Σ ((xᵢ - x̄) / s)³ ,   s = Bessel-corrected SD
    /// ```
    ///
    /// This is the unbiased-under-normality estimator used by Excel `SKEW`,
    /// SAS, SPSS and `scipy.stats.skew(bias=False)` — *not* the biased
    /// population form `m₃/m₂^{3/2}`.
    ///
    /// Requires `n ≥ 3` and a non-zero spread; a constant dataset has no
    /// defined skewness and returns `Err`.
    pub fn skewness(data: &[f64]) -> Result<f64, String> {
        if data.len() < 3 {
            return Err(format!(
                "ArithmaStatisticalMoment::skewness: needs at least 3 points, got {}",
                data.len()
            ));
        }
        Self::check_finite(data, "skewness")?;
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let ss: f64 = data.iter().map(|v| (v - mean) * (v - mean)).sum();
        let s = (ss / (n - 1.0)).sqrt();
        if s <= 0.0 {
            return Err(
                "ArithmaStatisticalMoment::skewness: zero variance — skewness is undefined"
                    .to_string(),
            );
        }
        let sum_cubed: f64 = data
            .iter()
            .map(|v| {
                let z = (v - mean) / s;
                z * z * z
            })
            .sum();
        Ok(n / ((n - 1.0) * (n - 2.0)) * sum_cubed)
    }

    /// **Sample excess** kurtosis — the standardised fourth moment `G₂`, with
    /// the `−3` already applied so a normal distribution scores `0`:
    ///
    /// ```text
    /// G2 = n(n+1) / ((n-1)(n-2)(n-3)) * Σ ((xᵢ - x̄)/s)⁴
    ///      - 3(n-1)² / ((n-2)(n-3))
    /// ```
    ///
    /// This is *excess*, not raw: subtract nothing further, and add `3` if a
    /// raw fourth standardised moment is wanted. It matches Excel `KURT`, SAS,
    /// SPSS and `scipy.stats.kurtosis(fisher=True, bias=False)`.
    ///
    /// Requires `n ≥ 4` and a non-zero spread.
    pub fn kurtosis(data: &[f64]) -> Result<f64, String> {
        if data.len() < 4 {
            return Err(format!(
                "ArithmaStatisticalMoment::kurtosis: needs at least 4 points, got {}",
                data.len()
            ));
        }
        Self::check_finite(data, "kurtosis")?;
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let ss: f64 = data.iter().map(|v| (v - mean) * (v - mean)).sum();
        let s = (ss / (n - 1.0)).sqrt();
        if s <= 0.0 {
            return Err(
                "ArithmaStatisticalMoment::kurtosis: zero variance — kurtosis is undefined"
                    .to_string(),
            );
        }
        let sum_fourth: f64 = data
            .iter()
            .map(|v| {
                let z = (v - mean) / s;
                z * z * z * z
            })
            .sum();
        let lead = n * (n + 1.0) / ((n - 1.0) * (n - 2.0) * (n - 3.0));
        let correction = 3.0 * (n - 1.0) * (n - 1.0) / ((n - 2.0) * (n - 3.0));
        Ok(lead * sum_fourth - correction)
    }
}

// ---------------------------------------------------------------------------
// Backward-compatibility aliases for the pre-rename `Arithmos*` names.
// Retained for one release; downstream (eml-math, eml-spectral, metaphysica,
// periodica) should migrate to the `Arithma*` names above.
// ---------------------------------------------------------------------------
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaStatisticalMoment`")]
#[allow(unused)]
pub use self::ArithmaStatisticalMoment as ArithmosStatisticalMoment;

#[cfg(test)]
mod tests {
    use super::*;

    /// Textbook dataset: mean 5, population variance 4, sample variance 32/7.
    const CLASSIC: [f64; 8] = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];

    fn close(got: f64, want: f64, tol: f64, what: &str) {
        assert!(
            (got - want).abs() <= tol,
            "{what}: got {got}, want {want} (tol {tol})"
        );
    }

    #[test]
    fn mean_matches_known_dataset() {
        assert_eq!(ArithmaStatisticalMoment::mean(&CLASSIC).unwrap(), 5.0);
        assert_eq!(ArithmaStatisticalMoment::mean(&[42.0]).unwrap(), 42.0);
        close(
            ArithmaStatisticalMoment::mean(&[1.0, 2.0, 4.0]).unwrap(),
            7.0 / 3.0,
            1e-15,
            "mean([1,2,4])",
        );
    }

    #[test]
    fn variance_is_bessel_corrected() {
        // Sample variance = 32/7, NOT the population variance of 4.
        close(
            ArithmaStatisticalMoment::variance(&CLASSIC).unwrap(),
            32.0 / 7.0,
            1e-14,
            "variance(CLASSIC)",
        );
        // [1..5]: Sum of squared deviations 10, /(n-1)=4 => 2.5.
        close(
            ArithmaStatisticalMoment::variance(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap(),
            2.5,
            1e-15,
            "variance([1..5])",
        );
        // A constant dataset has zero spread, not an error.
        assert_eq!(
            ArithmaStatisticalMoment::variance(&[3.0, 3.0, 3.0]).unwrap(),
            0.0
        );
    }

    #[test]
    fn skewness_uses_the_adjusted_fisher_pearson_g1() {
        // Excel SKEW(0,0,0,1) = 2 exactly.
        close(
            ArithmaStatisticalMoment::skewness(&[0.0, 0.0, 0.0, 1.0]).unwrap(),
            2.0,
            1e-12,
            "G1([0,0,0,1])",
        );
        // Mirror image flips the sign.
        close(
            ArithmaStatisticalMoment::skewness(&[1.0, 1.0, 1.0, 0.0]).unwrap(),
            -2.0,
            1e-12,
            "G1([1,1,1,0])",
        );
        // Symmetric data is unskewed.
        close(
            ArithmaStatisticalMoment::skewness(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap(),
            0.0,
            1e-14,
            "G1([1..5])",
        );
        // CLASSIC by hand: n=8, x̄=5, Σd²=32 so s²=32/7; Σd³=42; and
        // n/((n-1)(n-2)) = 8/42, so G1 = (8/42)·42/s³ = 8/(32/7)^{3/2}
        //                              = 7√14/32 = 0.8184875533568...
        close(
            ArithmaStatisticalMoment::skewness(&CLASSIC).unwrap(),
            7.0 * 14.0_f64.sqrt() / 32.0,
            1e-12,
            "G1(CLASSIC)",
        );
    }

    #[test]
    fn kurtosis_is_sample_excess_g2() {
        // Excel KURT(1,2,3,4,5) = -1.2 (excess: a normal would score 0).
        close(
            ArithmaStatisticalMoment::kurtosis(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap(),
            -1.2,
            1e-12,
            "G2([1..5])",
        );
        // Excel KURT(0,0,0,1) = 4.
        close(
            ArithmaStatisticalMoment::kurtosis(&[0.0, 0.0, 0.0, 1.0]).unwrap(),
            4.0,
            1e-12,
            "G2([0,0,0,1])",
        );
        // CLASSIC by hand: Σd⁴ = 356 and s⁴ = (32/7)² = 1024/49, so
        // Σ(d/s)⁴ = 356·49/1024 = 17444/1024. Then
        // G2 = (8·9/(7·6·5))·17444/1024 − 3·49/(6·5)
        //    = 5.840625 − 4.9 = 0.940625 exactly.
        close(
            ArithmaStatisticalMoment::kurtosis(&CLASSIC).unwrap(),
            0.940_625,
            1e-12,
            "G2(CLASSIC)",
        );
    }

    #[test]
    fn undersized_and_empty_inputs_are_rejected() {
        // Empty slice.
        assert!(ArithmaStatisticalMoment::mean(&[]).is_err());
        assert!(ArithmaStatisticalMoment::variance(&[]).is_err());
        assert!(ArithmaStatisticalMoment::skewness(&[]).is_err());
        assert!(ArithmaStatisticalMoment::kurtosis(&[]).is_err());

        // One below each estimator's minimum sample size.
        assert!(ArithmaStatisticalMoment::variance(&[1.0]).is_err());
        assert!(ArithmaStatisticalMoment::skewness(&[1.0, 2.0]).is_err());
        assert!(ArithmaStatisticalMoment::kurtosis(&[1.0, 2.0, 3.0]).is_err());

        // Exactly at the minimum, they succeed.
        assert!(ArithmaStatisticalMoment::variance(&[1.0, 2.0]).is_ok());
        assert!(ArithmaStatisticalMoment::skewness(&[1.0, 2.0, 4.0]).is_ok());
        assert!(ArithmaStatisticalMoment::kurtosis(&[1.0, 2.0, 3.0, 5.0]).is_ok());
    }

    #[test]
    fn zero_spread_and_non_finite_inputs_are_rejected() {
        let flat = [7.0, 7.0, 7.0, 7.0];
        assert!(ArithmaStatisticalMoment::skewness(&flat).is_err());
        assert!(ArithmaStatisticalMoment::kurtosis(&flat).is_err());

        let dirty = [1.0, 2.0, f64::NAN, 4.0];
        assert!(ArithmaStatisticalMoment::mean(&dirty).is_err());
        assert!(ArithmaStatisticalMoment::variance(&dirty).is_err());
        assert!(ArithmaStatisticalMoment::skewness(&dirty).is_err());
        assert!(ArithmaStatisticalMoment::kurtosis(&dirty).is_err());
        assert!(ArithmaStatisticalMoment::mean(&[1.0, f64::INFINITY]).is_err());
    }
}
