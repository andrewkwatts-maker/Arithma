# Changelog

All notable changes to **Arithma** are documented here.

---

## [2.0.4] — 2026-09-01

### Fixed

- **The advertised `cpp-support` and `rust-support` features did not compile.**
  `external/mod.rs` declared `cpp_executor` and `rust_executor`, but neither
  file existed, so `cargo build --features cpp-support` failed with `E0583`.
  Both are now real `ArithmaBackend` implementations: `rust_executor` wraps an
  in-process closure, `cpp_executor` is a C-ABI seam exchanging expressions as
  JSON. An unbound handler reports `BackendUnavailable` so the router falls
  through, rather than panicking.

- **The constants registry never registered anything.**
  `load_constants_from_json` parsed the document, discarded the result and
  returned `Ok(())`, so `SYMBOL_REGISTRY` was permanently empty and every
  lookup returned `None`:

      lookup_symbol("π")   ->  None      (correct: Constant, 3.141592653589793)

  It now builds a `Constant` node per entry, skips `enabled: false`, rejects an
  entry with neither `cached_value` nor `expression`, and returns the count.
  Registration goes through `reregister_symbol`, so `initialize_defaults()` is
  genuinely idempotent as its docs claim. 30 constants now register.

- **The SI-unit registry never parsed its catalogue.** `si_units.rs` embedded
  `si_units.json` with `include_str!` but `REGISTRY` was a hardcoded empty
  `HashMap`, so `lookup()` always returned `None` and `len()` was always 0.
  Base and derived units now parse.

- Dead branch in `lookup::math_hash::lookup_exp`: the `is_nan()` arm and the
  fallback both returned `None`.

- Mojibake (UTF-8 read as CP-1252) across 50 files, some doubly encoded, plus
  UTF-8 BOMs including one on `rust/arithma_core/Cargo.toml`.

### Added

- `cargo test`, `cargo clippy -- -D warnings` and `cargo fmt --check` now run in
  CI across the default, `rust-support` and `cpp-support` feature sets. **None
  of the Rust tests had ever run in CI** — the workflow ran pytest only.
- `build.bat` / `build.sh` running the same gate locally, then the wheel.
- `docs/PHASE0.md` — the foundation-repair working brief.

### Changed

- **Finished the `Arithmos` → `Arithma` rename.** ~1,055 identifiers across 48
  files — every public struct, enum, trait, type alias and function that still
  carried the pre-rename `Arithmos*` prefix is now `Arithma*`. The old names
  are kept as `#[deprecated]` `pub use ... as ...` aliases (module-level and,
  for the crate-root re-exports, at the crate root too) for one release so
  downstream (eml-math, eml-spectral, metaphysica, periodica) has a migration
  window rather than a hard break.
- `tests/test_core.py::test_version` no longer hardcodes the version string; it
  compares `arithma.__version__` against the installed package's own
  `importlib.metadata` entry, so it can't silently drift on the next bump.
- `Cargo.lock` is now committed; this crate publishes a binary artifact and
  needs reproducible builds.
- Test count 124 → 133 (139 with `cpp-support`). The new tests assert real
  values — π, e, φ and the m/kg/s base units resolve — rather than merely
  asserting that a call did not error, which is what let the two dead
  registries above ship green.

### Known — not yet changed

- 41 `unimplemented!()` bodies remain (integration, equation solving, root
  finding, critical points, geometry, probability, Fourier, `Emit`). They panic
  on call; `grep -rn 'unimplemented!' rust/arithma_core/src` is the work list.
- `Emit::emit` is still a stub. A complete LaTeX emitter lives in `pyfacade.rs`
  and is what `to_latex()` actually uses.
- `ArithmaInternalInteger::to_f64` truncates above 2^256, and `from_f64` uses a
  fixed-scale rational rather than an exact dyadic conversion.

---

## [2.0.2] — 2026-05-17

### Changed

- Version bump for ecosystem tag alignment. No functional change.

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
