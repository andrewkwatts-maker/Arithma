# Changelog


## [Unreleased]

### Added -- like-term collection

`x + x -> 2*x`, `2*x + 3*x -> 5*x`, `3*x + (-3)*x -> 0`, `x * x -> x^2`,
`x^2 * x -> x^3`. Collection is not limited to variables: `sin(x) + sin(x)`
collects too.

- Needed structural equality, which `ArithmaExpression` does not derive.
  `structural_eq` is iterative and bounded like the rest of the module.
- `ArithmaFunction` deliberately does not derive `PartialEq` -- several
  variants carry payloads, including expressions. So `same_function` is an
  **explicit allow-list** of the payload-free arithmetic and transcendental
  variants plus the three with cheap comparable payloads. Anything unlisted
  compares unequal, which loses a simplification but never produces a wrong
  one. Matching on the discriminant and defaulting to "equal" would have been
  the unsound choice: a new payload variant would silently start comparing
  equal to itself regardless of its contents.
- **Comparison is syntactic.** `2*x` and `x*2` are not collected, because
  there is no canonical operand ordering. Stated in the docs and pinned by a
  test rather than left to be discovered.

### Fixed

- **`finish_commutative` had no case for a fully cancelled fold.** Collection
  can consume every operand -- `3x + (-3)x` leaves nothing, with no numeric
  accumulator either -- and the write-back asserted the operand list was
  non-empty. It now yields the identity element: zero for a sum, one for a
  product. Found by the cancellation test, not by inspection.

### Performance

- Collection is gated on a **borrow-only pre-check** rather than run
  speculatively. This is load-bearing, not an optimisation: `partition` pulls
  numeric operands to the front, so taking ownership of a sum that turns out
  to need no work would silently reorder `x + 2` into `2 + x`. Reporting "no
  change" after reordering is a lie the fixpoint loop cannot catch, and
  reporting "changed" every pass would stop it converging at all.
- A no-op pass over 1,000 symbolic nodes went **150 us -> 322 us**: roughly
  2x, for the per-node pre-check, and still linear. Recorded rather than
  glossed over -- the earlier figure is now wrong and the module doc says so.


### Added -- SI prefixes and unit conversion

`ArithmaUnit` carried dimensions but no magnitude, so there was no km->m.

- The 24 SI prefixes (quetta down to quecto) as a table carrying symbol, name,
  integer power of ten, and a link to the existing
  `expression::ArithmaSIPrefix` variant where one exists. It is **data, not a
  rival type**: the existing enum only spans yotta..yocto, so the four 2022
  CGPM prefixes have no variant, and a test asserts that only those four rows
  may be unlinked while the other twenty agree with the enum's multiplier.
- `split_unit_symbol`, `ArithmaUnit::scale_to_base`, `ArithmaUnit::convert`,
  and `ArithmaDimension::of_prefixed_unit`, so `km` reports length.
- **`kg` does not decompose.** Two documented rules settle every case: a whole
  symbol that is itself a known unit wins outright (which also protects `cd`,
  `mol`, `Pa`, `Gy`, `kat`, and `T` as tesla rather than tera), and otherwise
  the longest prefix whose remainder is a known unit wins (which decides
  `dam` as deca-metre). Since `g` is not a base unit, `mg` returns an unknown-
  unit error rather than a guess; adding gram is a separate change.
- Conversion applies a **single net power of ten**, multiplying for positive
  and dividing for negative, rather than `value * from_scale / to_scale`. So
  `convert(1.0, km, m)` is exactly `1000.0` and converting back is exactly
  `1.0` -- asserted with equality, not a tolerance. A finite input that would
  scale to infinity returns `Overflow` rather than an infinity that reads as a
  definite answer.
- Python: `convert`, `prefix_power`, `si_prefixes`, `split_unit_symbol`, and
  `Unit.scale_to_base` / `Unit.split_symbol`.

### Fixed -- two modules disagreed on how to spell micro

`expression::ArithmaSIPrefix::Micro.symbol()` returned the micro sign while
`unit`'s prefix table uses `"u"`. Same crate, same concept, two spellings --
so `si_prefix_power(Micro.symbol())` returned `None`, and a caller round-
tripping a unit through the enum silently lost the prefix. It also broke the
project's Latin-letters-only rule, which is what made it visible.

`symbol()` now returns `"u"`, and a test asserts every prefix symbol the enum
produces is ASCII **and** resolves in the `unit` table, so the two cannot drift
apart again.


### Added -- matrix inverse, eigenvalues and tensor contraction

Three gaps the previous release documented rather than closed.

- **`ArithmaMatrix::inverse`**, with `minor`, `cofactor` and `adjugate`.
  Computed as `adj(A) / det(A)`, not by Gauss-Jordan: elimination divides by
  pivots, and dividing symbolic entries introduces quotients that may never
  simplify away, so the result would be correct but unreadable. The adjugate
  route divides once per entry by a single shared determinant.
  - A determinant that is merely *symbolic* is **not** treated as singular. It
    may well be non-zero for the values the caller has in mind, so the
    quotient is returned unevaluated. Only an exactly-zero determinant raises
    `MatrixError::Singular`.
  - Inexact entries stay symbolic quotients rather than collapsing to rounded
    literals; with `det = 1` they fold to exact literals. Both halves of that
    rule are pinned by tests.
- **`ArithmaMatrix::characteristic_polynomial`** -- `det(A - x I)`, exact and
  symbolic, in a caller-named variable.
- **`ArithmaMatrix::eigenvalues_real`**, composed from the characteristic
  polynomial and the existing root finders rather than adding a separate
  eigen-decomposition. The scan window is the Gershgorin bound, which provably
  contains the whole spectrum, so "no root was missed" is a statement rather
  than a hope.
  - Two limits, documented rather than hidden: **real roots only** (a rotation
    matrix correctly returns none -- its spectrum is +/- i, and that is the
    right answer, not a convergence failure), and **repeated eigenvalues
    collapse**, because a double root does not change the polynomial's sign
    and so cannot be bracketed. Eigenvectors are not computed.
  - `trace = sum of eigenvalues` and `p(lambda) = 0` are both tested,
    independently of how the roots were found.
- **`ArithmaTensor::tensordot`, `trace` and `outer`.** `tensordot` with
  `axes_a=[1], axes_b=[0]` on two rank-2 tensors is matrix multiplication, and
  a test pins it against a hand-computed product. Both the result shape and
  the contracted shape go through the checked cell-count helper, so an
  over-cap contraction is refused before anything is allocated. Two new
  `TensorError` variants report malformed axis lists and mismatched extents.
- All of the above are on the Python surface: `Matrix.inverse/adjugate/minor/
  cofactor/characteristic_polynomial/eigenvalues_real` and
  `Tensor.tensordot/trace/outer`.


### Fixed -- defects surfaced while writing the bindings

All five were found by reading the modules closely enough to bind them. Each
carries a regression test.

- **`binomial_coefficient` had an unbounded loop.** It iterated
  `min(k, n-k)` times over `u64` arguments, so `C(u64::MAX, u64::MAX/2)` ran
  about 9e18 iterations and never returned -- reachable from Python.
  Now capped at `MAX_EXACT_BINOMIAL_TERMS = 1024`, which is a mathematical
  boundary rather than a compromise: with `m = min(k, n-k)` we have `n >= 2m`
  and `C(n, m) >= (n/m)^m >= 2^m`, and `f64::MAX` is just under `2^1024`. So
  every refused case genuinely is infinite and infinity is the right answer.
- **`ln_binomial_coefficient` had the same unbounded loop**, and capping it
  was not an option because that path exists precisely for large `n`. Added
  `probabilities::ln_gamma` (Lanczos, g = 7, n = 9, relative error under
  1e-13) and switched to the closed form
  `ln C(n, k) = lnGamma(n+1) - lnGamma(k+1) - lnGamma(n-k+1)` past 4096 terms.
  The worst case is now O(1), and a test pins the two paths agreeing to 1e-12
  where they overlap so the PMF cannot jump as `n` crosses the cap.
- **`mean` and `variance` never validated.** All three distributions returned
  `Ok` on an instance whose `pdf` returned `Err`: `ArithmaNormal::new(NAN, 0.0)
  .mean()` gave `Ok(NaN)`. The trait returns `Result` precisely so these can
  fail; one accessor reporting a broken instance as healthy is worse than
  either answer alone. Fixed in normal, binomial and bernoulli.
- **`critical_points::scan_for_roots` kept a stale `prev_y` across a domain
  gap.** `prev_x` and `prev_y` are meant to be a matched `(x, f(x))` pair, and
  the pole path updated only `prev_x` -- so the next successful sample was
  compared against a value from the *other side* of the discontinuity, which
  manufactures a sign change and brackets a pole. They are now a single
  `Option<(f64, f64)>` that is dropped rather than half-updated.
- **`methods.rs` returned a value where its docs promise an error.** The
  `bisect` and `brent` branches returned the current midpoint after
  `ARITHMA_SOLVE_MAX_ITERATIONS`, so an unconverged answer was
  indistinguishable from a converged one, contradicting `solve_with_method`'s
  documented contract. Both now check the residual through a shared
  `converged_or_err`, so the two bracketing methods cannot drift apart on what
  convergence means.


### Added -- solver, Fourier, numerical and statistics bindings

Four implemented Rust modules had no Python surface at all: `equation_solver`,
`fourier`, `numerical/` (4 files) and `probabilities/` (9 files). The work was
done and unreachable.

- `src/pyfacade/analysis.rs` -- `solve`, `solve_equation`, `solve_system`,
  `fourier_transform`, `fourier_window_weight`, `Solution`, `FourierConfig`,
  `FourierTransform`, plus `SOLVER_STRATEGIES` / `FOURIER_WINDOWS` so callers
  can enumerate the accepted spellings instead of learning them from an
  exception.
- `src/pyfacade/numerical.rs` -- `find_root_bisection`,
  `find_root_newton_raphson`, `find_root_secant`, `solve_with_method`,
  `find_stationary_points`, `find_inflection_points`, `find_extrema`,
  `analyze_point`, `classify_point`, `analyze_intervals`, `evaluate_interval`,
  `RootResult`, `CriticalPoint`, `FunctionAnalysis`, `Interval`.
  Non-convergence raises `RuntimeError` rather than returning a best guess, so
  a returned `RootResult.converged` is always true.
- `src/pyfacade/stats.rs` -- `Normal`, `Binomial`, `Bernoulli`,
  `ConfidenceInterval`, `erf`, `erfc`, `binomial_coefficient`, the `sample_*`
  moments, `normal_confidence_interval`, `normal_critical_value`,
  `distribution_from_json`.

The extension now exports **87 symbols, up from 42**.

### Fixed -- the pyfacade's own tests ran nowhere

- **`python` implied pyo3's `extension-module`.** That feature deliberately
  does not link libpython, so a test binary built with it cannot load -- which
  is why `cargo test --features python` appears nowhere in CI and why every
  `#[cfg(test)]` test inside `pyfacade/` was written and then never executed.
  `extension-module` is now a separate feature that only the wheel build turns
  on (`pyproject.toml` requests it). `cargo test --features python` runs
  **556 tests** against 416 on default features -- 140 were previously dead.
  A CI step now runs them.
- **The Python wrapper's re-export lists lagged the crate.** The extension
  exported 87 symbols while `__init__.py` listed 42, so the new bindings were
  compiled into the wheel and unreachable as `arithma.<name>`. Nothing failed;
  they were simply absent. Regenerated, and `test_core.py` now asserts the
  lists and the extension's surface agree in both directions, so this cannot
  drift again silently.
- Added a `_CONSTANTS` list. Plain values need separate handling: the
  graceful-degradation path substitutes a raising *callable* for a missing
  symbol, which would be misleading for `MAX_SAMPLE_COUNT`.
- Corrected `equation_solver.rs`'s module doc, which still claimed "Wave 2
  ships type signatures only". The solver passes are real -- closed-form
  polynomial fit with bisection fallback, and Gaussian elimination with
  partial pivoting for systems.

### Known limits recorded rather than hidden

Found while writing the bindings, documented in the Python docstring and left
in the crate:

- `binomial_coefficient` loops `min(k, n-k)` times over `u64` arguments, so a
  large `n` from Python would spin. The facade caps the term count and rejects
  beyond it, and converts the `inf` overflow return into `ValueError`.
- `confidence_interval.rs` is a 3-field struct with no way to build an
  interval from data, so `normal_confidence_interval` is composed in the
  facade and is a **z** interval, not a **t** -- unreliable for small samples.
- `ArithmaNormal::mean`/`variance` never call `validate`, so a NaN-mean
  instance returns `Ok(NaN)` while `pdf` on the same object errors. The facade
  closes this by validating in `__new__`.
- `critical_points.rs::scan_for_roots` keeps a stale `prev_y` across a failed
  sample, so a pole can manufacture a spurious bracket. Bounded: the
  subsequent bisection fails and the result is discarded.
- `methods.rs::solve_with_method`'s bisect and brent branches return a value
  after the iteration cap where the doc-comment promises an error.


### Added -- matrix, tensor and unit algebra

- **`ArithmaMatrix` algebra.** It was a shape and a flat cell list with no
  operation that combined two matrices. Added `zeros`, `identity`, `from_rows`,
  bounds-checked `get`/`set`, `transpose`, `add`, `sub`, `scalar_mul`,
  `matmul`, `trace`, `determinant` and `simplified`.
  - The determinant is exact and symbolic, by summation over permutations with
    an iterative next-permutation. Capped at order `MAX_DETERMINANT_ORDER`
    (6): the method is O(n!), and refusing beats running for an unbounded
    time. Gaussian elimination scales better but needs division, which
    introduces quotients that may not simplify away.
  - Shape errors are a `MatrixError` enum, not panics. A dimension mismatch in
    caller data is a condition to handle, and a panic crossing the PyO3
    boundary is undefined behaviour.
- **`ArithmaTensor` algebra.** Added `zeros`, `rank`, `strides`, `offset`,
  `get`/`set` by multi-index, `reshape`, `permute_axes`, `add`, `sub`,
  `hadamard`, `scalar_mul` and `simplified`.
  - Indexing is checked **per axis**, not against the flat length. For shape
    `[2, 3]` the index `(0, 3)` has a valid flat offset -- it is the cell
    `(1, 0)` -- so a flat-only check silently returns the wrong cell.
  - Shape products use checked multiplication. An unchecked product of
    caller-supplied dimensions can wrap to a small number and pass a naive
    length check, so `zeros([usize::MAX, 2, 2])` would allocate almost nothing
    and then index out of it.
  - There is no contraction and no einsum.
- **Dimensional analysis: `ArithmaDimension`.** `ArithmaUnit` carried a symbol
  and a name and nothing else, so nothing could answer "may these two
  quantities be added?". Added the exponent vector over the seven SI base
  dimensions, with `multiply`, `divide`, `powi`, `nth_root`,
  `is_compatible_with`, `derived_symbol`, `quantity` and an ASCII `Display`
  (`m*kg*s^-2`), plus a table of 19 standard derived dimensions.
  - This resolves `N`, `J`, `W`, `ohm` and the rest, which the SI catalogue
    deliberately omits -- so the base-units-only scope no longer leaves derived
    quantities unreachable.
  - Exponent arithmetic is checked, not wrapped, and refuses past
    `MAX_EXPONENT`. A root that would need a fractional exponent is refused
    rather than truncated: `sqrt(m)` is not a dimension this can express.
  - `ArithmaUnit::is_compatible_with` returns `Option<bool>`, because "I cannot
    tell" and "not compatible" are different answers and a caller would treat
    the latter as a definite error.
- Python surface: `Matrix.transpose/add/sub/scalar_mul/matmul/trace/
  determinant/simplified/is_square` with `@`, `+` and `-` operators;
  `Tensor.strides/reshape/permute_axes/add/sub/hadamard/scalar_mul/set`;
  `Unit.dimension/dimension_exponents/quantity/is_dimensionless/
  is_compatible_with`; and module-level `dimension_of`, `dimension_symbol` and
  `base_dimensions`. Shape errors raise `ValueError`, index errors `IndexError`.
- `Matrix.shape` is a property, matching the existing `Tensor.shape`. The two
  types should not differ on whether shape is called or read.


### Added -- expression simplification

- **`simplify()` is implemented.** It was a no-op that cleared its stack and
  returned `false` without touching the expression, and the public `Simplify`
  trait was a second stub returning a clone. Both now route to one iterative
  engine. Rules: constant folding, additive and multiplicative identities,
  absorbing zero, exact division, and the power rules. The three specification
  tests that shipped `#[ignore]`d now run as ordinary regression tests.
- **Exact integer arithmetic on `ArithmaInteger`.** The type advertised
  unlimited precision but had no arithmetic at all -- no add, no multiply,
  nothing that could combine two values -- so constant folding had nothing to
  fold with. Added `checked_add`, `checked_sub`, `checked_mul`,
  `checked_div_exact`, `checked_pow`, `checked_neg` and `to_u32` on the
  base-256 representation.
  - Every operation **refuses rather than approximates**. `7 / 2` returns
    `None` because the quotient is not exact; special values (NaN, infinity,
    rational, imaginary) are refused rather than folded; results past
    `MAX_DIGITS` (512 bytes) and exponents past `MAX_POW_EXPONENT` (1024) are
    refused rather than attempted. A refusal leaves the expression symbolic,
    which is always a safe answer.
  - Additive operations require matching units: metres plus seconds does not
    fold. Multiplication and division are unitless-only, because `m * m` is
    m^2 and this type cannot express that.
- `Expression.simplify()` and `Expression.is_simplifiable()` on the Python
  facade, with `max_iterations` and `allow_numeric_collapse` keyword arguments.
  An over-large `max_iterations` raises `ValueError` rather than being silently
  clamped.
- Benchmarks: `expression/simplify` (folding and no-op passes at 8/64/200/1000
  nodes) and `integer/arithmetic` (add, multiply, exact divide and pow at
  small and 256-bit widths).

### Fixed

- **`SimplificationConfig::default()` gave `max_iterations: 0`** -- a derived
  `Default` on a field where zero means "do nothing". Any caller using the
  default got a simplifier that was a no-op by construction, and it made the
  stub implementation indistinguishable from a working one. Now 32.
- **`si_units.rs` read a `derived_units` array that the catalogue has never
  contained.** `#[serde(default)]` made the absence silent, so a deliberate
  design decision looked like missing data. Derived units are meant to be
  expressions, not table entries -- `si_units.json` says so in its own
  `derived_units_info.note`. Removed the vestigial field, documented the scope
  on `lookup`, added `ArithmaSIUnits::scope_note()` so an empty result can be
  explained, and pinned the seven base units with tests.
- Corrected the "known gaps" section of the Python package docstring, which
  described the SI catalogue's base-units-only scope as a shipping defect.

### Performance

- **Simplification is O(n) per pass**, verified by benchmark rather than
  asserted. Two earlier designs of this same code were quadratic and both
  passed every test:
  1. Addressing nodes by a path from the root re-walked the spine per node.
  2. Cloning operands before deciding whether to rewrite copied whole subtrees
     per node.

  A no-op pass over 1,000 symbolic nodes went **36.5 ms -> 150 us (243x)**;
  folding 1,000 numeric nodes is 268 us. Both are now linear in node count.
  The `no_op_symbolic` benchmarks exist to catch a regression here, which the
  test suite cannot see.


### Performance
- Added `[profile.release]` (`lto = true`, `codegen-units = 1`, `opt-level = 3`).
  There were no profile tables at all, so `--release` built with no cross-crate
  inlining and 16 codegen units. Measured with the new benchmarks, every hot
  path improved: expression build 7-12% faster, tree traversal (`to_f64`)
  **~33% faster**, `Integer::from_i64` 19%, `Integer::to_f64` 25%,
  `Expression::var` 30%. All p = 0.00.
- Added `[profile.bench]` (thin LTO) and `[profile.test]` so benchmarks are
  representative without full LTO link time, and the recursive-simplification
  tests do not run unoptimised.

### Added
- `benches/hot_paths.rs` -- criterion benchmarks for expression build,
  tree traversal, integer conversion and atom construction, so "accelerated"
  is a measurement rather than a claim.
- `arithma.assert_rust_backend()` -- raises `RustBackendUnavailable` unless the
  extension is loaded *and* its version matches the Python package. `_HAS_RUST`
  alone only says the extension imported; a stale `_arithma_core` left in the
  source tree is easy to miss.
- `arithma.backend_report()` -- backend diagnostics for bug reports.
- Specification tests for `simplify()`, written while it was still a no-op.
  Nothing had caught that: the only existing test simplified an *atom*, where
  returning `false` is the correct answer. They are no longer `#[ignore]`d --
  see the simplification section above.

### Fixed
- Removed a stale "Wave-2 stub" marker from `ArithmaFunction::arity`, which is
  a real arity table and has been for some time.

### Added -- Python acceleration surface
- Split `pyfacade.rs` (1,393 lines) into `pyfacade/` with one module per
  domain: `core`, `calculus`, `constants`, `geometry`, `linalg`, `units`.
  The single file fought the "restrict functions to a single printed page"
  standard and made parallel work impossible.
- **Exported surface grew from 5 symbols to 42** -- 11 classes and 31
  functions. Calculus (`differentiate`, `differentiate_n`, `gradient`,
  `integrate`, `integrate_definite`, `integrate_numeric`), the constants
  registry, geometry (`Vector`, `Line`, `Plane`, `Sphere`, intersection
  routines), `Matrix`, `Tensor` and `Unit` are all reachable from Python now.
  Every one is a binding; no mathematics is implemented in Python.
- All new wrappers follow the safety-critical standards: two runtime
  assertions minimum per function, every caller-supplied sequence bounded by
  `MAX_SEQUENCE_LEN` before iteration, no recursion, functions under one page,
  every `Result` checked, and errors raised as Python exceptions rather than
  returned as defaults.
- Expression trees arriving from Python are walked with an explicit stack and
  a node budget. `Evaluable::evaluate` recurses, so a deep tree built in a
  Python loop would otherwise arrive as an uncatchable stack overflow.

### Fixed
- **21 of the 30 default constants were unreachable without typing a glyph.**
  They are keyed by their mathematical symbol, so `lookup("pi")` returned
  `None` and only `lookup("<pi glyph>")` worked -- awkward from a keyboard,
  fragile across source encodings, and impossible in an ASCII identifier
  position. Added a 29-entry ASCII alias table (`pi`, `tau`, `phi`, `gamma`,
  `sqrt2`, `zeta3`, `golden_ratio`, `feigenbaum_delta`, ...). The glyphs still
  work unchanged, and an exact match always beats an alias so a caller can
  still shadow one. Six tests, including one that asserts every alias target
  is a real registered constant.
- Cleared all 30 clippy warnings under `--features python` (format-arg
  inlining, `map_clone`), and documented the two that are unfixable at an FFI
  boundary: pyo3 0.22's `#[pymethods]` trampoline emits a `PyErr -> PyErr`
  conversion, and `Integer.to_string` must stay an inherent method because
  that is the Python API.

### CI
- Added a `clippy --features python` gate. Those 30 warnings had accumulated
  precisely because no job ever turned the feature on.
- Added a feature-matrix `cargo check` over `python`, `rust-support`,
  `cpp-support` and `--all-features`; `cargo test` builds default features
  only, so a feature-gated module could break with every test still passing.

### Known gaps (unchanged, restated for accuracy)
- `matrix.rs` and `tensor.rs` are containers -- the crate has no matrix
  algebra at all (no multiply, transpose, determinant or inverse), so the
  bindings expose construction and inspection only.
- `unit.rs` holds a symbol and a name; there is no dimensional analysis or
  unit conversion anywhere in the crate.
- `si_units.json` ships a `derived_units_info` prose note but no
  `derived_units` table, and the field is `#[serde(default)]`, so the SI
  registry silently resolves the 7 base units only -- `si_lookup("N")` returns
  `None`. Pinned by a test that should be updated when the data is fixed.
- `ArithmaTensor::cell_count` multiplies the shape unchecked and wraps
  silently in release; the facade re-validates with `checked_mul` first.

### Known gaps (from the earlier pass)
- `expression::iterative::simplify` is unimplemented (returns the input
  unchanged). This is the Wave-3 port from pt-arithmos' `pt_expression.rs`.
- `fallback::try_fallback` returns its argument unchanged.
- Python exposes 3 classes (`Expression`, `Integer`, `Variable`) out of roughly
  250 public items in the core; calculus, Fourier, geometry, matrices, units
  and probability are not reachable from Python yet.
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
