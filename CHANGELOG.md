# Changelog

All notable changes to **Arithma** are documented here.

---

## [2.0.1] — 2026-05-17

Ecosystem alignment release. Brings Arithma into lock-step with the
v2.0.1 tag line shared by periodica, metaphysica, eml-math, and
eml-spectral.

### Changed

- **Package renamed `arithmos` → `arithma`** — restores the canonical
  name used throughout the design docs and the PlayTow master plan.
  - PyPI package: `arithma`
  - Rust crate: `arithma_core`
  - Python extension module: `arithma._arithma_core`
  - Cross-library interop methods: `to_arithma()` / `from_arithma()`
  - Feature flag on downstream libs: `with-arithma`
- **Version bumped to `2.0.1`** across `pyproject.toml` and
  `rust/arithma_core/Cargo.toml` to match ecosystem tag.
- `hello()` returns `"arithma"`.

### Notes

The `AR*` type-prefix convention (`ARExpression`, `ARFunction`,
`ARInteger`, `ARVariable`, ...) is **unchanged** — it was always the
intended short form and is not affected by the package rename.

---

## [1.4.1] — 2026-05-13

*(Intermediate rename to `arithmos` — superseded by v2.0.1 revert.)*

---

## [1.4.0] — 2026-05-10

Initial public release. Aligns the version line with the rest of the
EML / metaphysica / periodica family for the v1.4.0 ecosystem cut.

### Added

- **`arithma_core` Rust crate** (`cdylib + lib`) — the bottom of the
  symbolic-math dependency chain for the EML / metaphysica / periodica
  ecosystem. Modules:
  - `expression/` — `ARExpression` AST + iterative simplifier passes.
  - `function.rs` — operator catalogue (Add, Sub, Mul, Div, Pow, Sin,
    Cos, Tan, Limit, Sum, Product, Integral, FindRoots, Optimize, Mean,
    Variance, Geometry, ...).
  - `integer.rs` — `ARInteger` + `ARInternalInteger` with bit-flag
    specials (Negative / Rational / Infinity / NaN).
  - `variable.rs` / `constants.rs` — symbolic variables, constants
    registry, embedded `default_constants.json` (JSONC banner stripper
    included so PlayTow-style copyright headers parse cleanly).
  - `calculus/` — symbolic + iterative differentiation, integration.
  - `fourier.rs` / `equation_solver.rs` — placeholder modules.
  - `geometry/` — vector, line, plane, sphere, intersection.
  - `probabilities/` — normal, binomial, bernoulli, distribution
    factory, quantile function, confidence interval, statistical
    moments, statistical sampler.
  - `numerical/` — methods, critical points, interval analysis, root
    finding.
  - `matrix.rs` / `tensor.rs` — symbolic linear algebra.
  - `unit.rs` / `si_units.rs` — SI units registry (embedded JSON).
  - `lookup/` — `trig_hash` (canonical-angle hash slots 1000-1010,
    Pythagorean-identity-tested) and `math_hash` (~50 stable hash
    slots) plus `MathIdKind` classifier.
  - `fallback.rs` — fallback dispatch system.
  - `external/` — `ARExternalFunctionRegistry` for pluggable backends
    (PT*-typed engine glue, EML-Math, future C++/Python executors).
  - `arithmetic.rs` — internal lossless arithmetic helpers.
  - `pyfacade.rs` — PyO3 facade gated by the `python` feature.
- **`ARInterop` cross-library trait** — downstream libraries (eml-math,
  eml-spectral, metaphysica, periodica) implement this trait behind
  their own `with-arithma` feature flag to opt into Arithma as a
  foundational expression substrate, strictly via git submodule.
- **`arithma` Python package** — facade with `_HAS_RUST` guard around
  the maturin-built `_arithma_core` extension.
- **84 unit tests** covering expression construction, integer flags,
  constants JSON round-trip, lookup-table classifier behaviour, the
  Pythagorean identity over canonical angles, and external-function
  registry plumbing.

### Cargo features

| Feature | Effect |
|---|---|
| `default` | Pure Rust — no Python, no Arithma-bridged downstream. |
| `python` | Pulls PyO3 0.22 and exposes the `_arithma_core` extension. |
| `cpp-support` / `rust-support` | Reserved for SDK dynamic-loading executors. |
