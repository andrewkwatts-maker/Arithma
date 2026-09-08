//====== Arithma/rust/arithma_core/src/pyfacade/stats.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! PyO3 facade for the probability domain: the three concrete distributions,
//! the error function pair, sample moments, quantiles, confidence intervals
//! and the deterministic sampler.
//!
//! ## Parameters are validated at construction, not at first use
//!
//! `ArithmaNormal`, `ArithmaBinomial` and `ArithmaBernoulli` deliberately defer
//! parameter validation to `pdf`/`cdf` so that a hot-reloaded config file
//! cannot panic the engine mid-frame (see `distribution_factory.rs`). That
//! trade-off does not carry over to an interactive Python session, where
//! `Normal(0, 0)` succeeding and then raising on every method call is a worse
//! diagnostic than failing on the constructor line. So every class here
//! validates in `__new__`, and [`distribution_from_json`] inherits that: a spec
//! with an impossible parameter raises at parse time rather than at evaluation.
//!
//! ## Sample conventions
//!
//! [`sample_variance`], [`sample_skewness`] and [`sample_kurtosis`] are the
//! *sample* (unbiased-estimator) forms, and kurtosis is *excess*. They match
//! Excel `VAR.S` / `SKEW` / `KURT` and `scipy.stats.*(bias=False)`, not the
//! biased population defaults of `numpy`. The names carry the `sample_` prefix
//! for exactly that reason -- see `statistical_moment.rs` for the formulae.
//! `Distribution.variance()` is a different quantity: the exact population
//! variance of the distribution, not an estimate from data.
//!
//! ## Sampling is reproducible, always
//!
//! `arithma_core` carries no `rand` dependency and never seeds from the clock
//! or the OS. `sample(n)` uses the fixed [`DEFAULT_SEED`], so it is a pure
//! function of its arguments; `sample(n, seed=...)` selects an independent
//! stream. There is no way to ask for a non-deterministic draw, which is the
//! intended behaviour, not a missing feature. The generator is xorshift64* and
//! is **not** cryptographically secure.
//!
//! ## What the crate does not have
//!
//! No Student *t*, Poisson, exponential, uniform or chi-squared distribution,
//! and `confidence_interval.rs` is a value type with no constructor from data.
//! [`normal_confidence_interval`] therefore composes the pieces that do exist
//! (sample mean, sample SD, standard-normal quantile) into a *z* interval; it
//! is not a *t* interval and is only trustworthy for large `n`.

// `#[pymethods]` / `#[pyfunction]` on pyo3 0.22 expand `-> PyResult<T>` into a
// `PyErr::from` round-trip that clippy reads as a no-op conversion in *our*
// signature span. Matches the allow already carried by `core.rs`, `units.rs`
// and the rest of this directory.
#![allow(clippy::useless_conversion)]

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PySequence, PyString};

use crate::probabilities::binomial::binomial_coefficient as core_binomial_coefficient;
use crate::probabilities::distribution_factory::ArithmaDistributionSpec;
use crate::probabilities::normal::{erf as core_erf, erfc as core_erfc};
use crate::probabilities::statistical_sampler::DEFAULT_SEED;
use crate::probabilities::{
    ArithmaBernoulli, ArithmaBinomial, ArithmaConfidenceInterval, ArithmaDistribution,
    ArithmaDistributionFactory, ArithmaNormal, ArithmaQuantileFunction, ArithmaStatisticalMoment,
    ArithmaStatisticalSampler,
};
use crate::pyfacade::MAX_SEQUENCE_LEN;

/// Largest draw count accepted by `sample`.
///
/// Two orders of magnitude below [`MAX_SEQUENCE_LEN`] because every draw costs
/// a full inverse-CDF bisection -- up to 200 CDF evaluations, and a binomial
/// CDF is itself a summation. A million draws from `Binomial(1_000_000, p)`
/// would hold the GIL for hours; refusing is the honest answer.
const MAX_SAMPLE_COUNT: usize = 100_000;

/// Largest number of multiplicative terms [`binomial_coefficient`] will run.
///
/// `binomial_coefficient` loops `min(k, n-k)` times over `u64` arguments, so
/// `C(u64::MAX, u64::MAX/2)` would spin for the age of the universe. The bound
/// costs nothing real: `C(2m, m) > 2^m`, so any input needing more than ~1030
/// terms already overflows `f64` to infinity and is rejected below anyway.
const MAX_BINOMIAL_TERMS: u64 = 4096;

/// Largest JSON spec accepted by [`distribution_from_json`]. A distribution
/// spec is three fields; anything larger is a caller error, not a big config.
const MAX_SPEC_JSON_LEN: usize = 64 * 1024;

// ============================================================================
// Helpers.
// ============================================================================

/// The distribution types report bad *caller* input -- a NaN `x`, an
/// out-of-range parameter -- through `Err(String)`. Those are `ValueError`.
fn to_value_error(message: String) -> PyErr {
    PyValueError::new_err(message)
}

/// Failures that survive our own validation are internal, not caller error.
///
/// Once a distribution's parameters are checked in `__new__` and `p` is checked
/// against `(0, 1)`, a failed bisection or a failed draw means the quantile
/// search itself did not converge. Reporting that as `ValueError` would tell
/// the caller to fix an input that is already correct.
fn to_runtime_error(message: String) -> PyErr {
    PyRuntimeError::new_err(message)
}

/// A probability parameter must be a real number in `[0, 1]`.
fn validate_probability(p: f64, what: &str) -> PyResult<()> {
    if !p.is_finite() {
        return Err(PyValueError::new_err(format!(
            "{what} must be finite, got {p}"
        )));
    }
    if !(0.0..=1.0).contains(&p) {
        return Err(PyValueError::new_err(format!(
            "{what} must lie in [0, 1], got {p}"
        )));
    }
    Ok(())
}

/// A quantile argument is stricter than a probability parameter: the extreme
/// quantiles are `±∞` or a support endpoint, which bisection cannot express,
/// so `ArithmaQuantileFunction` refuses `p ≤ 0` and `p ≥ 1` outright.
fn validate_quantile_argument(p: f64) -> PyResult<()> {
    if !p.is_finite() {
        return Err(PyValueError::new_err(format!(
            "quantile p must be finite, got {p}"
        )));
    }
    if p <= 0.0 || p >= 1.0 {
        return Err(PyValueError::new_err(format!(
            "quantile p must lie strictly inside (0, 1), got {p}; \
             the 0 and 1 quantiles are unbounded or at the support edge"
        )));
    }
    Ok(())
}

/// A standard deviation must be strictly positive: `σ = 0` is a point mass,
/// which has no density at all, and a negative `σ` is meaningless.
fn validate_std_dev(std_dev: f64) -> PyResult<()> {
    if !std_dev.is_finite() {
        return Err(PyValueError::new_err(format!(
            "std_dev must be finite, got {std_dev}"
        )));
    }
    if std_dev <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "std_dev must be > 0, got {std_dev}; a zero-width normal is a point \
             mass and has no density"
        )));
    }
    Ok(())
}

/// Reject a non-finite value that Python's `float` accepts but the maths does
/// not. Python has no non-NaN float type, so this cannot be a signature check.
fn validate_finite(value: f64, what: &str) -> PyResult<()> {
    if !value.is_finite() {
        return Err(PyValueError::new_err(format!(
            "{what} must be finite, got {value}"
        )));
    }
    Ok(())
}

/// A confidence level names the mass *inside* the interval, so `0` and `1` are
/// both degenerate and excluded.
fn validate_level(level: f64) -> PyResult<()> {
    if !level.is_finite() {
        return Err(PyValueError::new_err(format!(
            "confidence level must be finite, got {level}"
        )));
    }
    if level <= 0.0 || level >= 1.0 {
        return Err(PyValueError::new_err(format!(
            "confidence level must lie strictly inside (0, 1), got {level}"
        )));
    }
    Ok(())
}

/// Bound the draw count before any allocation or bisection happens.
fn validate_sample_count(n: usize) -> PyResult<()> {
    debug_assert!(
        MAX_SAMPLE_COUNT <= MAX_SEQUENCE_LEN,
        "the sampler cap must stay inside the facade-wide sequence cap"
    );
    if n > MAX_SAMPLE_COUNT {
        return Err(PyValueError::new_err(format!(
            "requested {n} samples, over the {MAX_SAMPLE_COUNT} limit; each draw \
             costs a full inverse-CDF bisection"
        )));
    }
    Ok(())
}

/// Iteration count [`core_binomial_coefficient`] will run for `C(n, k)`, or an
/// error when that exceeds [`MAX_BINOMIAL_TERMS`]. Returns `0` for `k > n`,
/// which short-circuits to `0.0` without looping.
fn binomial_term_count(n: u64, k: u64) -> PyResult<u64> {
    if k > n {
        return Ok(0);
    }
    let terms = k.min(n - k);
    if terms > MAX_BINOMIAL_TERMS {
        return Err(PyValueError::new_err(format!(
            "C({n}, {k}) needs {terms} multiplicative terms, over the \
             {MAX_BINOMIAL_TERMS} limit; the value overflows f64 long before this"
        )));
    }
    Ok(terms)
}

/// Extract a dataset from a Python sequence.
///
/// The length is read and checked against [`MAX_SEQUENCE_LEN`] *before* the
/// extraction loop, so the iteration bound is fixed up front (CLAUDE.md
/// standard 3). `str` and `bool` are rejected: both satisfy the numeric or
/// sequence protocol well enough to produce nonsense silently.
fn extract_samples(data: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if data.is_instance_of::<PyString>() {
        return Err(PyTypeError::new_err(
            "expected a list or tuple of numbers, not a str",
        ));
    }
    let seq = data
        .downcast::<PySequence>()
        .map_err(|_| PyTypeError::new_err("expected a list or tuple of numbers"))?;
    let count = seq.len()?;
    if count > MAX_SEQUENCE_LEN {
        return Err(PyValueError::new_err(format!(
            "dataset of {count} points exceeds MAX_SEQUENCE_LEN ({MAX_SEQUENCE_LEN})"
        )));
    }
    let mut out: Vec<f64> = Vec::with_capacity(count);
    for index in 0..count {
        let item = seq.get_item(index)?;
        if item.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err(format!(
                "data[{index}] is a bool; pass 0.0 or 1.0 explicitly"
            )));
        }
        out.push(item.extract::<f64>()?);
    }
    debug_assert_eq!(out.len(), count, "the bounded loop must fill every slot");
    Ok(out)
}

/// Inverse CDF shared by the three distribution classes.
///
/// `p` is validated by the caller, so a failure here is a non-converging
/// bisection -- an internal fault, not bad input.
fn quantile_of(dist: &dyn ArithmaDistribution, p: f64) -> PyResult<f64> {
    let x = ArithmaQuantileFunction::inverse_cdf(dist, p).map_err(to_runtime_error)?;
    debug_assert!(x.is_finite(), "a quantile of a validated p must be finite");
    debug_assert!(
        dist.cdf(x).map(|c| c >= p).unwrap_or(false),
        "inverse_cdf must return an x with F(x) >= p"
    );
    Ok(x)
}

/// Seeded inverse-transform sampling shared by the three distribution classes.
fn sample_of(dist: &dyn ArithmaDistribution, n: usize, seed: Option<u64>) -> PyResult<Vec<f64>> {
    validate_sample_count(n)?;
    let draws = ArithmaStatisticalSampler::sample_with_seed(dist, n, seed.unwrap_or(DEFAULT_SEED))
        .map_err(to_runtime_error)?;
    debug_assert_eq!(draws.len(), n, "the sampler must return exactly n draws");
    debug_assert!(
        draws.iter().all(|v| v.is_finite()),
        "every inverse-transform draw must be finite"
    );
    Ok(draws)
}

// ============================================================================
// `Normal` pyclass.
// ============================================================================

/// The normal (Gaussian) distribution `N(mean, std_dev²)`.
///
/// `std_dev` is the standard deviation `σ`, not the variance. Raises
/// `ValueError` for a non-finite mean or a `std_dev` that is not strictly
/// positive.
#[pyclass(name = "Normal", module = "arithma")]
#[derive(Clone, Copy)]
pub struct Normal {
    pub(crate) inner: ArithmaNormal,
}

#[pymethods]
impl Normal {
    #[new]
    fn new(mean: f64, std_dev: f64) -> PyResult<Self> {
        validate_finite(mean, "mean")?;
        validate_std_dev(std_dev)?;
        let inner = ArithmaNormal::new(mean, std_dev);
        debug_assert!(inner.std_dev > 0.0, "validate_std_dev rejects sigma <= 0");
        debug_assert!(
            inner.mean.is_finite(),
            "validate_finite rejects NaN and inf"
        );
        Ok(Self { inner })
    }

    /// The standard normal `N(0, 1)`.
    #[staticmethod]
    fn standard() -> Self {
        let inner = ArithmaNormal::standard();
        debug_assert_eq!(inner.mean, 0.0, "the standard normal is centred at 0");
        debug_assert_eq!(inner.std_dev, 1.0, "the standard normal has unit width");
        Self { inner }
    }

    /// Probability density at `x`. Raises `ValueError` for a NaN `x`.
    fn pdf(&self, x: f64) -> PyResult<f64> {
        let density = self.inner.pdf(x).map_err(to_value_error)?;
        debug_assert!(density >= 0.0, "a density is never negative, and not NaN");
        debug_assert!(x.is_finite() || density == 0.0, "no density at infinity");
        Ok(density)
    }

    /// `P(X <= x)`. Raises `ValueError` for a NaN `x`.
    fn cdf(&self, x: f64) -> PyResult<f64> {
        let mass = self.inner.cdf(x).map_err(to_value_error)?;
        debug_assert!((0.0..=1.0).contains(&mass), "a CDF value is a probability");
        debug_assert!(mass.is_finite(), "a CDF value is never NaN for a finite x");
        Ok(mass)
    }

    /// Inverse CDF: the smallest `x` with `F(x) >= p`, by bisection.
    fn quantile(&self, p: f64) -> PyResult<f64> {
        validate_quantile_argument(p)?;
        quantile_of(&self.inner, p)
    }

    /// Expected value -- the `mean` the class was constructed with.
    fn mean(&self) -> PyResult<f64> {
        let value = self.inner.mean().map_err(to_value_error)?;
        debug_assert!(value.is_finite(), "a validated mean is finite");
        debug_assert_eq!(value, self.inner.mean, "the mean is a stored parameter");
        Ok(value)
    }

    /// Population variance `σ²`.
    fn variance(&self) -> PyResult<f64> {
        let value = self.inner.variance().map_err(to_value_error)?;
        debug_assert!(value >= 0.0, "a variance is never negative, and not NaN");
        let sigma = self.inner.std_dev;
        debug_assert_eq!(value, sigma * sigma, "the variance is sigma squared");
        Ok(value)
    }

    /// Population standard deviation `σ`.
    fn std_dev(&self) -> f64 {
        debug_assert!(self.inner.std_dev > 0.0, "checked in __new__");
        debug_assert!(self.inner.std_dev.is_finite(), "checked in __new__");
        self.inner.std_dev
    }

    /// `n` deterministic draws by inverse-transform sampling.
    ///
    /// Same arguments, same values -- always. Pass `seed` for an independent
    /// but equally reproducible stream.
    #[pyo3(signature = (n, seed=None))]
    fn sample(&self, n: usize, seed: Option<u64>) -> PyResult<Vec<f64>> {
        sample_of(&self.inner, n, seed)
    }

    fn __repr__(&self) -> String {
        format!("Normal({}, {})", self.inner.mean, self.inner.std_dev)
    }
}

// ============================================================================
// `Binomial` pyclass.
// ============================================================================

/// `Binomial(n, p)` -- successes in `n` independent trials.
///
/// The CDF sums the PMF term by term and refuses `n > 1_000_000`; that limit
/// lives in `binomial.rs`, and hitting it raises `ValueError`.
#[pyclass(name = "Binomial", module = "arithma")]
#[derive(Clone, Copy)]
pub struct Binomial {
    pub(crate) inner: ArithmaBinomial,
}

#[pymethods]
impl Binomial {
    #[new]
    fn new(n: u64, p: f64) -> PyResult<Self> {
        validate_probability(p, "p")?;
        let inner = ArithmaBinomial::new(n, p);
        debug_assert!((0.0..=1.0).contains(&inner.p), "validate_probability ran");
        debug_assert_eq!(inner.n, n, "the trial count is stored verbatim");
        Ok(Self { inner })
    }

    /// Trial count.
    #[getter]
    fn n(&self) -> u64 {
        self.inner.n
    }

    /// Per-trial success probability.
    #[getter]
    fn p(&self) -> f64 {
        self.inner.p
    }

    /// Probability mass `P(X = k)`. Off-support and non-integral `x` carry
    /// mass `0`; a NaN `x` raises `ValueError`.
    fn pdf(&self, x: f64) -> PyResult<f64> {
        let mass = self.inner.pdf(x).map_err(to_value_error)?;
        debug_assert!((0.0..=1.0).contains(&mass), "a PMF value is a probability");
        debug_assert!(mass.is_finite(), "a PMF value is never NaN for a finite x");
        Ok(mass)
    }

    /// Alias for [`Binomial::pdf`]. The distribution is discrete, so the
    /// textbook name is *probability mass function*; `pdf` is kept because the
    /// underlying trait spells it that way for every distribution.
    fn pmf(&self, x: f64) -> PyResult<f64> {
        self.pdf(x)
    }

    /// `P(X <= x)`, summed term by term.
    fn cdf(&self, x: f64) -> PyResult<f64> {
        let mass = self.inner.cdf(x).map_err(to_value_error)?;
        debug_assert!((0.0..=1.0).contains(&mass), "a CDF value is a probability");
        debug_assert!(mass.is_finite(), "the summation is clamped to [0, 1]");
        Ok(mass)
    }

    /// Inverse CDF. The CDF is a step function, so the result converges to the
    /// jump point *from above*: the 0.5-quantile of `Binomial(5, 0.5)` comes
    /// back as `2` plus up to `1e-12`. Round it if an exact integer is wanted.
    fn quantile(&self, p: f64) -> PyResult<f64> {
        validate_quantile_argument(p)?;
        quantile_of(&self.inner, p)
    }

    /// Expected value `n·p`.
    fn mean(&self) -> PyResult<f64> {
        let value = self.inner.mean().map_err(to_value_error)?;
        debug_assert!(value >= 0.0, "n and p are both non-negative");
        debug_assert!(value <= self.inner.n as f64, "the mean lies in the support");
        Ok(value)
    }

    /// Variance `n·p·(1-p)`.
    fn variance(&self) -> PyResult<f64> {
        let value = self.inner.variance().map_err(to_value_error)?;
        debug_assert!(value >= 0.0, "a variance is never negative");
        debug_assert!(value.is_finite(), "n is finite and p lies in [0, 1]");
        Ok(value)
    }

    /// Standard deviation `sqrt(n·p·(1-p))`.
    fn std_dev(&self) -> PyResult<f64> {
        let variance = self.inner.variance().map_err(to_value_error)?;
        debug_assert!(variance >= 0.0, "the sqrt below needs a non-negative input");
        let value = variance.sqrt();
        debug_assert!(!value.is_nan(), "sqrt of a non-negative value is not NaN");
        Ok(value)
    }

    /// `n` deterministic draws. Each costs a bisection over the CDF, which is
    /// itself a summation, so this is markedly slower than the normal case.
    #[pyo3(signature = (n, seed=None))]
    fn sample(&self, n: usize, seed: Option<u64>) -> PyResult<Vec<f64>> {
        sample_of(&self.inner, n, seed)
    }

    fn __repr__(&self) -> String {
        format!("Binomial({}, {})", self.inner.n, self.inner.p)
    }
}

// ============================================================================
// `Bernoulli` pyclass.
// ============================================================================

/// `Bernoulli(p)` -- one trial, success or failure.
///
/// The support is exactly `{0, 1}`. `pdf` compares `x` against `0.0` and `1.0`
/// exactly, so a value that merely rounds to an integer carries mass `0`.
#[pyclass(name = "Bernoulli", module = "arithma")]
#[derive(Clone, Copy)]
pub struct Bernoulli {
    pub(crate) inner: ArithmaBernoulli,
}

#[pymethods]
impl Bernoulli {
    #[new]
    fn new(p: f64) -> PyResult<Self> {
        validate_probability(p, "p")?;
        let inner = ArithmaBernoulli::new(p);
        debug_assert!((0.0..=1.0).contains(&inner.p), "validate_probability ran");
        debug_assert!(inner.p.is_finite(), "validate_probability rejects NaN");
        Ok(Self { inner })
    }

    /// Success probability.
    #[getter]
    fn p(&self) -> f64 {
        self.inner.p
    }

    /// Probability mass: `p` at `1`, `1-p` at `0`, `0` everywhere else.
    fn pdf(&self, x: f64) -> PyResult<f64> {
        let mass = self.inner.pdf(x).map_err(to_value_error)?;
        debug_assert!((0.0..=1.0).contains(&mass), "a PMF value is a probability");
        debug_assert!(
            x == 0.0 || x == 1.0 || mass == 0.0,
            "the support is exactly {{0, 1}}"
        );
        Ok(mass)
    }

    /// Alias for [`Bernoulli::pdf`]; see the note on [`Binomial::pmf`].
    fn pmf(&self, x: f64) -> PyResult<f64> {
        self.pdf(x)
    }

    /// `P(X <= x)`: `0` below `0`, `1-p` on `[0, 1)`, `1` from `1` up.
    fn cdf(&self, x: f64) -> PyResult<f64> {
        let mass = self.inner.cdf(x).map_err(to_value_error)?;
        debug_assert!((0.0..=1.0).contains(&mass), "a CDF value is a probability");
        debug_assert!(x >= 0.0 || mass == 0.0, "no mass below the support");
        Ok(mass)
    }

    /// Inverse CDF. As with [`Binomial::quantile`] the answer approaches the
    /// jump point from above rather than landing exactly on `0` or `1`.
    fn quantile(&self, p: f64) -> PyResult<f64> {
        validate_quantile_argument(p)?;
        quantile_of(&self.inner, p)
    }

    /// Expected value `p`.
    fn mean(&self) -> PyResult<f64> {
        let value = self.inner.mean().map_err(to_value_error)?;
        debug_assert!((0.0..=1.0).contains(&value), "the mean of Bernoulli is p");
        debug_assert_eq!(value, self.inner.p, "the mean is the stored parameter");
        Ok(value)
    }

    /// Variance `p(1-p)`, maximal at `p = 0.5`.
    fn variance(&self) -> PyResult<f64> {
        let value = self.inner.variance().map_err(to_value_error)?;
        debug_assert!(value >= 0.0, "a variance is never negative");
        debug_assert!(value <= 0.25 + f64::EPSILON, "p(1-p) peaks at 1/4");
        Ok(value)
    }

    /// Standard deviation `sqrt(p(1-p))`.
    fn std_dev(&self) -> PyResult<f64> {
        let variance = self.inner.variance().map_err(to_value_error)?;
        debug_assert!(variance >= 0.0, "the sqrt below needs a non-negative input");
        let value = variance.sqrt();
        debug_assert!(!value.is_nan(), "sqrt of a non-negative value is not NaN");
        Ok(value)
    }

    /// `n` deterministic draws, each landing at `0` or `1` up to the bisection
    /// tolerance.
    #[pyo3(signature = (n, seed=None))]
    fn sample(&self, n: usize, seed: Option<u64>) -> PyResult<Vec<f64>> {
        sample_of(&self.inner, n, seed)
    }

    fn __repr__(&self) -> String {
        format!("Bernoulli({})", self.inner.p)
    }
}

// ============================================================================
// `ConfidenceInterval` pyclass.
// ============================================================================

/// An interval estimate `[lower, upper]` at a stated confidence level.
///
/// A plain value type -- constructing one asserts nothing about where the
/// bounds came from. [`normal_confidence_interval`] builds one from data.
#[pyclass(name = "ConfidenceInterval", module = "arithma")]
#[derive(Clone, Copy)]
pub struct ConfidenceInterval {
    pub(crate) inner: ArithmaConfidenceInterval,
}

#[pymethods]
impl ConfidenceInterval {
    #[new]
    fn new(lower: f64, upper: f64, level: f64) -> PyResult<Self> {
        validate_finite(lower, "lower")?;
        validate_finite(upper, "upper")?;
        validate_level(level)?;
        if lower > upper {
            return Err(PyValueError::new_err(format!(
                "lower bound {lower} exceeds upper bound {upper}"
            )));
        }
        let inner = ArithmaConfidenceInterval::new(lower, upper, level);
        debug_assert!(inner.width() >= 0.0, "an ordered interval has width >= 0");
        debug_assert!(inner.level > 0.0 && inner.level < 1.0, "validate_level ran");
        Ok(Self { inner })
    }

    #[getter]
    fn lower(&self) -> f64 {
        self.inner.lower
    }

    #[getter]
    fn upper(&self) -> f64 {
        self.inner.upper
    }

    /// Confidence level, e.g. `0.95`.
    #[getter]
    fn level(&self) -> f64 {
        self.inner.level
    }

    /// `upper - lower`.
    fn width(&self) -> f64 {
        debug_assert!(self.inner.lower <= self.inner.upper, "ordered in __new__");
        let width = self.inner.width();
        debug_assert!(width >= 0.0, "an ordered interval has a non-negative width");
        width
    }

    /// The interval midpoint. For [`normal_confidence_interval`] this is the
    /// sample mean the interval was built around.
    fn midpoint(&self) -> f64 {
        debug_assert!(self.inner.lower <= self.inner.upper, "ordered in __new__");
        let mid = self.inner.lower + 0.5 * self.inner.width();
        debug_assert!(mid >= self.inner.lower, "the midpoint lies in the interval");
        mid
    }

    /// Whether `value` falls inside the closed interval. Raises `ValueError`
    /// for a non-finite `value` rather than answering `False`, which a caller
    /// would read as "a real number, and outside".
    fn contains(&self, value: f64) -> PyResult<bool> {
        validate_finite(value, "value")?;
        debug_assert!(self.inner.lower <= self.inner.upper, "ordered in __new__");
        debug_assert!(value.is_finite(), "validate_finite ran immediately above");
        Ok(value >= self.inner.lower && value <= self.inner.upper)
    }

    fn __repr__(&self) -> String {
        format!(
            "ConfidenceInterval({}, {}, level={})",
            self.inner.lower, self.inner.upper, self.inner.level
        )
    }
}

// ============================================================================
// Special functions.
// ============================================================================

/// Gauss error function `erf(x) = (2/√π) ∫₀ˣ e^{-t²} dt`.
///
/// Better than `3e-16` relative error over `[-6, 6]`, and saturated correctly
/// beyond. A NaN `x` raises `ValueError` instead of returning NaN, so a
/// propagated NaN cannot masquerade as an answer.
#[pyfunction]
fn erf(x: f64) -> PyResult<f64> {
    if x.is_nan() {
        return Err(PyValueError::new_err("erf: x must not be NaN"));
    }
    let value = core_erf(x);
    debug_assert!(!value.is_nan(), "erf of a non-NaN argument is not NaN");
    debug_assert!((-1.0..=1.0).contains(&value), "erf maps onto [-1, 1]");
    Ok(value)
}

/// Complementary error function `erfc(x) = 1 - erf(x)`.
///
/// Computed directly in the right tail rather than by subtraction, so
/// `erfc(6)` keeps full relative precision instead of collapsing to zero.
#[pyfunction]
fn erfc(x: f64) -> PyResult<f64> {
    if x.is_nan() {
        return Err(PyValueError::new_err("erfc: x must not be NaN"));
    }
    let value = core_erfc(x);
    debug_assert!(!value.is_nan(), "erfc of a non-NaN argument is not NaN");
    debug_assert!((0.0..=2.0).contains(&value), "erfc maps onto [0, 2]");
    Ok(value)
}

/// Binomial coefficient `C(n, k)`, computed multiplicatively.
///
/// Bit-exact for `n <= 50` and accurate to a few ULP well beyond that. Returns
/// `0.0` when `k > n`. Raises `ValueError` rather than returning `inf` when the
/// value overflows `f64` (somewhere past `n ≈ 1030` with `k` near `n/2`), and
/// rather than looping for hours when `min(k, n-k)` is astronomically large.
#[pyfunction]
fn binomial_coefficient(n: u64, k: u64) -> PyResult<f64> {
    let terms = binomial_term_count(n, k)?;
    debug_assert!(terms <= MAX_BINOMIAL_TERMS, "the loop below is bounded");
    let value = core_binomial_coefficient(n, k);
    debug_assert!(value >= 0.0, "a count of subsets is never negative");
    if !value.is_finite() {
        return Err(PyValueError::new_err(format!(
            "C({n}, {k}) overflows f64; use a log-scale formulation instead"
        )));
    }
    Ok(value)
}

// ============================================================================
// Sample moments.
// ============================================================================

/// Arithmetic mean. Raises `ValueError` for an empty dataset or one containing
/// a non-finite value, never `NaN`.
#[pyfunction]
fn sample_mean(data: &Bound<'_, PyAny>) -> PyResult<f64> {
    let values = extract_samples(data)?;
    let mean = ArithmaStatisticalMoment::mean(&values).map_err(to_value_error)?;
    debug_assert!(
        mean.is_finite(),
        "finite inputs cannot average to NaN or inf"
    );
    debug_assert!(!values.is_empty(), "an empty dataset is rejected upstream");
    Ok(mean)
}

/// Bessel-corrected **sample** variance `Σ(xᵢ-x̄)²/(n-1)`.
///
/// Matches Excel `VAR.S`, `numpy.var(ddof=1)` and R `var` -- not
/// `numpy.var()`'s population default. Needs at least 2 points.
#[pyfunction]
fn sample_variance(data: &Bound<'_, PyAny>) -> PyResult<f64> {
    let values = extract_samples(data)?;
    let variance = ArithmaStatisticalMoment::variance(&values).map_err(to_value_error)?;
    debug_assert!(variance >= 0.0, "a sum of squares is never negative");
    debug_assert!(values.len() >= 2, "n < 2 is rejected upstream");
    Ok(variance)
}

/// Sample standard deviation: the square root of [`sample_variance`].
///
/// `statistical_moment.rs` has no `std_dev` of its own, so this is the square
/// root taken here rather than a separate estimator. It is therefore the
/// square root of an unbiased variance, which is itself slightly biased low --
/// the usual and expected behaviour, matching Excel `STDEV.S`.
#[pyfunction]
fn sample_std_dev(data: &Bound<'_, PyAny>) -> PyResult<f64> {
    let values = extract_samples(data)?;
    let variance = ArithmaStatisticalMoment::variance(&values).map_err(to_value_error)?;
    debug_assert!(variance >= 0.0, "the sqrt below needs a non-negative input");
    let std_dev = variance.sqrt();
    debug_assert!(!std_dev.is_nan(), "sqrt of a non-negative value is not NaN");
    Ok(std_dev)
}

/// Adjusted Fisher–Pearson **sample** skewness `G₁`.
///
/// Matches Excel `SKEW` and `scipy.stats.skew(bias=False)`. Needs at least 3
/// points and a non-zero spread; a constant dataset raises `ValueError`
/// because its skewness is undefined, not zero.
#[pyfunction]
fn sample_skewness(data: &Bound<'_, PyAny>) -> PyResult<f64> {
    let values = extract_samples(data)?;
    let skewness = ArithmaStatisticalMoment::skewness(&values).map_err(to_value_error)?;
    debug_assert!(skewness.is_finite(), "a non-zero spread gives a finite G1");
    debug_assert!(values.len() >= 3, "n < 3 is rejected upstream");
    Ok(skewness)
}

/// **Excess** sample kurtosis `G₂` -- the `-3` is already applied, so a normal
/// distribution scores `0`.
///
/// Matches Excel `KURT` and `scipy.stats.kurtosis(fisher=True, bias=False)`.
/// Needs at least 4 points and a non-zero spread.
#[pyfunction]
fn sample_kurtosis(data: &Bound<'_, PyAny>) -> PyResult<f64> {
    let values = extract_samples(data)?;
    let kurtosis = ArithmaStatisticalMoment::kurtosis(&values).map_err(to_value_error)?;
    debug_assert!(kurtosis.is_finite(), "a non-zero spread gives a finite G2");
    debug_assert!(values.len() >= 4, "n < 4 is rejected upstream");
    Ok(kurtosis)
}

// ============================================================================
// Confidence intervals.
// ============================================================================

/// Two-sided *z* confidence interval for the mean of `data`.
///
/// `x̄ ± z · s/√n`, with `s` the Bessel-corrected sample SD and `z` the
/// standard-normal quantile at `1 - (1-level)/2`.
///
/// This is composed here rather than taken from `confidence_interval.rs`,
/// which is a value type with no constructor from data. It is a **z** interval,
/// not a **t** interval: the crate has no Student *t* distribution, so the
/// coverage is optimistic for small `n`. Use it above `n ≈ 30`, or take the
/// pieces (`sample_mean`, `sample_std_dev`) and apply your own critical value.
#[pyfunction]
#[pyo3(signature = (data, level=0.95))]
fn normal_confidence_interval(data: &Bound<'_, PyAny>, level: f64) -> PyResult<ConfidenceInterval> {
    validate_level(level)?;
    let values = extract_samples(data)?;
    let mean = ArithmaStatisticalMoment::mean(&values).map_err(to_value_error)?;
    let variance = ArithmaStatisticalMoment::variance(&values).map_err(to_value_error)?;
    debug_assert!(variance >= 0.0, "a sum of squares is never negative");
    debug_assert!(values.len() >= 2, "variance rejects n < 2 upstream");

    let tail = 0.5 * (1.0 - level);
    let z = quantile_of(&ArithmaNormal::standard(), 1.0 - tail)?;
    let margin = z * variance.sqrt() / (values.len() as f64).sqrt();
    if !margin.is_finite() {
        return Err(PyRuntimeError::new_err(format!(
            "confidence half-width is not finite (z = {z}, n = {})",
            values.len()
        )));
    }
    ConfidenceInterval::new(mean - margin, mean + margin, level)
}

/// The standard-normal critical value `z` for a two-sided interval at `level`,
/// i.e. `Φ⁻¹(1 - (1-level)/2)`. `z(0.95) ≈ 1.959963984540054`.
///
/// Exposed on its own because the *t*-interval a caller may actually want is
/// this same shape with a different critical value.
#[pyfunction]
fn normal_critical_value(level: f64) -> PyResult<f64> {
    validate_level(level)?;
    let z = quantile_of(&ArithmaNormal::standard(), 1.0 - 0.5 * (1.0 - level))?;
    debug_assert!(
        z > 0.0,
        "a two-sided critical value is positive above level 0"
    );
    debug_assert!(z.is_finite(), "quantile_of already refuses a non-finite z");
    Ok(z)
}

// ============================================================================
// Factory.
// ============================================================================

/// Build a distribution from a JSON spec, e.g. `{"kind":"normal","mean":0.0,
/// "std_dev":1.0}`. Returns a [`Normal`], [`Binomial`] or [`Bernoulli`].
///
/// Unlike `ArithmaDistributionFactory::create`, which defers parameter checks
/// so hot-reload cannot fail mid-frame, this validates immediately -- see the
/// module docstring. A malformed document or an unknown `kind` raises
/// `ValueError`.
#[pyfunction]
fn distribution_from_json(py: Python<'_>, json: &str) -> PyResult<PyObject> {
    if json.is_empty() {
        return Err(PyValueError::new_err("distribution spec must not be empty"));
    }
    if json.len() > MAX_SPEC_JSON_LEN {
        return Err(PyValueError::new_err(format!(
            "spec of {} bytes exceeds the {MAX_SPEC_JSON_LEN}-byte limit",
            json.len()
        )));
    }
    let spec = ArithmaDistributionFactory::from_json(json).map_err(to_value_error)?;
    let object = match spec {
        ArithmaDistributionSpec::Normal { mean, std_dev } => {
            Py::new(py, Normal::new(mean, std_dev)?)?.into_any()
        }
        ArithmaDistributionSpec::Binomial { n, p } => Py::new(py, Binomial::new(n, p)?)?.into_any(),
        ArithmaDistributionSpec::Bernoulli { p } => Py::new(py, Bernoulli::new(p)?)?.into_any(),
    };
    debug_assert!(!json.is_empty(), "the empty case returned above");
    debug_assert!(!object.is_none(py), "every arm constructs a real object");
    Ok(object)
}

// ============================================================================
// Registration
// ============================================================================

/// Register this module's surface on the extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Normal>()?;
    m.add_class::<Binomial>()?;
    m.add_class::<Bernoulli>()?;
    m.add_class::<ConfidenceInterval>()?;
    m.add_function(wrap_pyfunction!(erf, m)?)?;
    m.add_function(wrap_pyfunction!(erfc, m)?)?;
    m.add_function(wrap_pyfunction!(binomial_coefficient, m)?)?;
    m.add_function(wrap_pyfunction!(sample_mean, m)?)?;
    m.add_function(wrap_pyfunction!(sample_variance, m)?)?;
    m.add_function(wrap_pyfunction!(sample_std_dev, m)?)?;
    m.add_function(wrap_pyfunction!(sample_skewness, m)?)?;
    m.add_function(wrap_pyfunction!(sample_kurtosis, m)?)?;
    m.add_function(wrap_pyfunction!(normal_confidence_interval, m)?)?;
    m.add_function(wrap_pyfunction!(normal_critical_value, m)?)?;
    m.add_function(wrap_pyfunction!(distribution_from_json, m)?)?;
    // The seed `sample()` uses when none is given. Exposed so a caller can
    // reproduce a default-seeded draw through the explicit-seed path.
    m.add("DEFAULT_SAMPLE_SEED", DEFAULT_SEED)?;
    m.add("MAX_SAMPLE_COUNT", MAX_SAMPLE_COUNT)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Textbook dataset: mean 5, sample variance 32/7.
    const CLASSIC: [f64; 8] = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];

    #[test]
    fn accepts_any_probability_between_zero_and_one_inclusive() {
        for p in [0.0, 0.25, 0.5, 1.0] {
            assert!(
                validate_probability(p, "p").is_ok(),
                "p = {p} must validate"
            );
        }
    }

    #[test]
    fn rejects_a_probability_outside_the_unit_interval() {
        assert!(validate_probability(-0.001, "p").is_err());
        assert!(validate_probability(1.001, "p").is_err());
        assert!(validate_probability(f64::NAN, "p").is_err());
        assert!(validate_probability(f64::INFINITY, "p").is_err());
    }

    #[test]
    fn rejects_a_negative_standard_deviation() {
        assert!(validate_std_dev(-1.0).is_err());
    }

    #[test]
    fn rejects_a_zero_standard_deviation_because_a_point_mass_has_no_density() {
        assert!(validate_std_dev(0.0).is_err());
        assert!(validate_std_dev(f64::NAN).is_err());
        assert!(validate_std_dev(f64::INFINITY).is_err());
        assert!(validate_std_dev(1e-300).is_ok());
    }

    #[test]
    fn rejects_the_zero_and_one_quantiles_but_accepts_everything_between() {
        assert!(validate_quantile_argument(0.0).is_err());
        assert!(validate_quantile_argument(1.0).is_err());
        assert!(validate_quantile_argument(f64::NAN).is_err());
        assert!(validate_quantile_argument(0.5).is_ok());
        assert!(validate_quantile_argument(0.999_999).is_ok());
    }

    #[test]
    fn rejects_a_confidence_level_that_is_not_strictly_inside_zero_and_one() {
        assert!(validate_level(0.0).is_err());
        assert!(validate_level(1.0).is_err());
        assert!(validate_level(95.0).is_err());
        assert!(validate_level(f64::NAN).is_err());
        assert!(validate_level(0.95).is_ok());
    }

    #[test]
    fn rejects_a_draw_count_above_the_sampler_cap() {
        assert!(validate_sample_count(0).is_ok());
        assert!(validate_sample_count(MAX_SAMPLE_COUNT).is_ok());
        assert!(validate_sample_count(MAX_SAMPLE_COUNT + 1).is_err());
    }

    #[test]
    fn counts_binomial_terms_using_the_shorter_side() {
        assert_eq!(binomial_term_count(50, 2).expect("small k"), 2);
        // Symmetry: C(n, k) == C(n, n-k), so the loop takes the cheaper route.
        assert_eq!(binomial_term_count(50, 48).expect("large k"), 2);
        assert_eq!(binomial_term_count(50, 25).expect("middle k"), 25);
    }

    #[test]
    fn reports_zero_terms_when_k_exceeds_n() {
        assert_eq!(binomial_term_count(5, 6).expect("k > n short-circuits"), 0);
        // The pathological case the bound exists for: k > n, so no loop runs.
        assert_eq!(binomial_term_count(0, u64::MAX).expect("k > n"), 0);
    }

    #[test]
    fn rejects_a_binomial_coefficient_whose_loop_would_never_finish() {
        assert!(binomial_term_count(u64::MAX, u64::MAX / 2).is_err());
        assert!(binomial_term_count(100_000, 5_000).is_err());
    }

    #[test]
    fn binomial_coefficient_matches_known_values() {
        assert_eq!(binomial_coefficient(5, 2).expect("C(5,2)"), 10.0);
        assert_eq!(binomial_coefficient(52, 5).expect("C(52,5)"), 2_598_960.0);
        assert_eq!(binomial_coefficient(5, 6).expect("k > n"), 0.0);
        assert_eq!(binomial_coefficient(0, 0).expect("C(0,0)"), 1.0);
    }

    #[test]
    fn binomial_coefficient_refuses_to_return_infinity() {
        // C(4000, 2000) overflows f64 while staying inside the term cap.
        assert!(binomial_coefficient(4000, 2000).is_err());
    }

    #[test]
    fn erf_rejects_nan_and_stays_inside_its_range() {
        assert!(erf(f64::NAN).is_err());
        assert!(erfc(f64::NAN).is_err());
        assert_eq!(erf(0.0).expect("erf(0)"), 0.0);
        assert_eq!(erf(f64::INFINITY).expect("erf(inf)"), 1.0);
        assert_eq!(erfc(f64::INFINITY).expect("erfc(inf)"), 0.0);
        assert_eq!(erfc(f64::NEG_INFINITY).expect("erfc(-inf)"), 2.0);
    }

    #[test]
    fn constructing_a_normal_rejects_a_degenerate_width() {
        assert!(Normal::new(0.0, 1.0).is_ok());
        assert!(Normal::new(0.0, 0.0).is_err());
        assert!(Normal::new(0.0, -1.0).is_err());
        assert!(Normal::new(f64::NAN, 1.0).is_err());
    }

    #[test]
    fn the_standard_normal_has_unit_variance_and_a_half_median() {
        let n = Normal::standard();
        assert_eq!(n.variance().expect("variance"), 1.0);
        assert_eq!(n.std_dev(), 1.0);
        assert_eq!(n.cdf(0.0).expect("cdf(0)"), 0.5);
        assert!((n.quantile(0.5).expect("median")).abs() < 1e-9);
    }

    #[test]
    fn the_standard_normal_quantile_matches_the_textbook_critical_value() {
        let z = Normal::standard().quantile(0.975).expect("z(0.975)");
        assert!((z - 1.959_963_984_540_054).abs() < 1e-9, "got {z}");
    }

    #[test]
    fn a_normal_quantile_rejects_the_endpoints_rather_than_returning_infinity() {
        let n = Normal::standard();
        assert!(n.quantile(0.0).is_err());
        assert!(n.quantile(1.0).is_err());
        assert!(n.pdf(f64::NAN).is_err());
        assert!(n.cdf(f64::NAN).is_err());
    }

    #[test]
    fn constructing_a_binomial_rejects_an_impossible_success_probability() {
        assert!(Binomial::new(10, 0.5).is_ok());
        assert!(Binomial::new(10, 1.5).is_err());
        assert!(Binomial::new(10, -0.5).is_err());
        assert!(Bernoulli::new(0.5).is_ok());
        assert!(Bernoulli::new(f64::NAN).is_err());
    }

    #[test]
    fn the_binomial_pmf_and_alias_agree() {
        let b = Binomial::new(5, 0.5).expect("valid binomial");
        assert_eq!(b.pdf(2.0).expect("pdf"), 0.312_5);
        assert_eq!(b.pmf(2.0).expect("pmf"), b.pdf(2.0).expect("pdf"));
        assert_eq!(b.cdf(2.0).expect("cdf"), 0.5);
        assert_eq!(b.mean().expect("mean"), 2.5);
        assert_eq!(b.variance().expect("variance"), 1.25);
        assert_eq!(b.n(), 5);
    }

    #[test]
    fn the_bernoulli_surface_reports_the_two_point_support() {
        let b = Bernoulli::new(0.25).expect("valid bernoulli");
        assert_eq!(b.pdf(1.0).expect("pdf(1)"), 0.25);
        assert_eq!(b.pdf(0.5).expect("off-support"), 0.0);
        assert_eq!(b.cdf(0.0).expect("cdf(0)"), 0.75);
        assert_eq!(b.mean().expect("mean"), 0.25);
        assert_eq!(b.variance().expect("variance"), 0.187_5);
        assert!((b.std_dev().expect("sd") - 0.187_5_f64.sqrt()).abs() < 1e-15);
        assert_eq!(b.p(), 0.25);
    }

    #[test]
    fn sampling_the_same_distribution_twice_gives_the_same_draws() {
        let n = Normal::standard();
        let first = n.sample(16, None).expect("default seed");
        let second = n.sample(16, None).expect("default seed");
        assert_eq!(first, second, "the default seed must be fixed");
        let seeded = n.sample(16, Some(DEFAULT_SEED)).expect("explicit seed");
        assert_eq!(
            first, seeded,
            "the exposed default seed must be the real one"
        );
        let other = n.sample(16, Some(7)).expect("other seed");
        assert_ne!(
            first, other,
            "a different seed must give a different stream"
        );
    }

    #[test]
    fn sampling_refuses_a_draw_count_above_the_cap() {
        let n = Normal::standard();
        assert!(n.sample(MAX_SAMPLE_COUNT + 1, None).is_err());
        assert!(n.sample(0, None).expect("zero draws is legal").is_empty());
    }

    #[test]
    fn a_confidence_interval_rejects_inverted_bounds() {
        assert!(ConfidenceInterval::new(1.0, 0.0, 0.95).is_err());
        assert!(ConfidenceInterval::new(0.0, 1.0, 1.0).is_err());
        assert!(ConfidenceInterval::new(f64::NAN, 1.0, 0.95).is_err());
    }

    #[test]
    fn a_confidence_interval_reports_its_width_midpoint_and_membership() {
        let ci = ConfidenceInterval::new(-1.0, 3.0, 0.95).expect("valid interval");
        assert_eq!(ci.width(), 4.0);
        assert_eq!(ci.midpoint(), 1.0);
        assert!(ci.contains(3.0).expect("closed at the upper bound"));
        assert!(!ci.contains(3.001).expect("outside"));
        assert!(ci.contains(f64::NAN).is_err());
        assert_eq!(ci.level(), 0.95);
    }

    #[test]
    fn the_two_sided_critical_value_matches_the_textbook_z() {
        let z = normal_critical_value(0.95).expect("z(0.95)");
        assert!((z - 1.959_963_984_540_054).abs() < 1e-9, "got {z}");
        let z99 = normal_critical_value(0.99).expect("z(0.99)");
        assert!((z99 - 2.575_829_303_548_901).abs() < 1e-8, "got {z99}");
        assert!(normal_critical_value(1.0).is_err());
    }

    #[test]
    fn a_subnormal_width_saturates_instead_of_tripping_an_assertion() {
        // A sigma this small is finite and positive, so it passes validation,
        // but the peak density overflows to +inf and the variance underflows to
        // 0. Neither is an invariant violation, and neither may panic across
        // the FFI boundary.
        let n = Normal::new(0.0, 1e-320).expect("a subnormal sigma is still > 0");
        assert!(n.pdf(0.0).expect("pdf") > 0.0);
        assert_eq!(n.variance().expect("variance"), 0.0);
        assert_eq!(n.pdf(f64::INFINITY).expect("pdf at infinity"), 0.0);
    }

    #[test]
    fn bias_corrected_excess_kurtosis_can_fall_well_below_minus_two() {
        // The -2 floor belongs to the *population* form. G2 rescales it, and a
        // four-point two-valued dataset lands at exactly -6. Documented here
        // because a lower-bound assertion on `sample_kurtosis` would fire on
        // this perfectly ordinary input.
        let flat = [0.0, 0.0, 1.0, 1.0];
        let g2 = ArithmaStatisticalMoment::kurtosis(&flat).expect("n = 4 is enough");
        assert!((g2 + 6.0).abs() < 1e-12, "got {g2}");
    }

    #[test]
    fn the_sample_moment_wrappers_agree_with_the_core_helpers() {
        // The pyfunctions take a Python sequence, so exercise the same core
        // path they delegate to -- no interpreter is available under `cargo
        // test` with the `extension-module` feature.
        let mean = ArithmaStatisticalMoment::mean(&CLASSIC).expect("mean");
        assert_eq!(mean, 5.0);
        let variance = ArithmaStatisticalMoment::variance(&CLASSIC).expect("variance");
        assert!((variance - 32.0 / 7.0).abs() < 1e-14);
        assert!((variance.sqrt() - (32.0_f64 / 7.0).sqrt()).abs() < 1e-15);
    }
}
