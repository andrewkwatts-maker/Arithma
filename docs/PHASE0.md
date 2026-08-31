# Phase 0 — Foundation repair

Working brief for finishing the Arithma foundation. Phase 0 fixes the base only:
**do not implement maths algorithms (Phase 4) and do not build the app (Phase 6).**

## Before starting

A previous session completed part of this. Check state first:

```bash
cargo test                                   # baseline: 124 passing
cargo check --features cpp-support           # must compile
cargo check --features rust-support          # must compile
grep -rn "Arithmos" rust/arithma_core/src | wc -l   # 1055 before the rename
grep -rn "unimplemented!" rust/arithma_core/src | wc -l  # 41 (Phase 4 work, leave alone)
```

Skip anything already done. Every task below needs a test proving it.

## 1. Unbreak the build

`external/mod.rs` declares `cpp_executor` (feature `cpp-support`) and
`rust_executor` (feature `rust-support`). If either `.rs` is missing, the
advertised feature fails to compile. Implement both as real `ArithmosBackend`
impls — the trait is in `external/registry.rs`:

- **rust_executor**: closure-based in-process backend.
- **cpp_executor**: C-ABI FFI seam. Serialise the expression to JSON (serde
  derives already exist on the AST), call a caller-supplied `extern "C"`
  function, parse the reply.

An unbound handler must return `BackendUnavailable` (the router's signal to fall
through), never panic. Unit-test both.

## 2. Revive the dead registries — silently wrong today

- `constants.rs::load_constants_from_json` parses the JSON then **registers
  nothing**, so `SYMBOL_REGISTRY` is permanently empty and π/e/φ never resolve.
  Make it populate the registry.
- `si_units.rs` embeds `si_units.json` via `include_str!` but **never parses
  it** — `REGISTRY` is a hardcoded empty `HashMap`. Parse it.

Test: `lookup_symbol("π")` ≈ 3.14159…, `ArithmosSIUnits::len() > 0`.

## 3. Finish the Arithmos → Arithma rename

~1,055 identifiers across 48 files, including every public type name. Scripted,
one commit. Keep `pub use` aliases for one release so downstream
(eml-math, eml-spectral, metaphysica, periodica) does not break. Also fix the
mojibake (`â€"`, `Façade`) in `constants.rs`/`arithmetic.rs` and strip the UTF-8
BOM from `rust/arithma_core/Cargo.toml`.

Do this **before** writing new code, or the new code inherits the old prefix.

## 4. CI that actually tests

`.github/workflows/ci.yml` runs pytest only — **all 124 Rust tests never run in
CI**. Add `cargo test`, `cargo clippy -- -D warnings`, `cargo fmt --check`, and
a coverage floor. Commit `Cargo.lock` (currently gitignored but a published
binary artifact needs it) and un-ignore `CLAUDE.md`.

## 5. Version hygiene

Code says 2.0.4 in four places (plus a fifth, `tests/test_core.py`, which
hardcodes it) but tags stop at **v2.0.2** — the entire PyO3 facade and the
`from_f64` precision fix are unreleased. Derive `__version__` from package
metadata so the test stops hardcoding it, then tag and release. Repoint
`origin/HEAD` at `main` (it currently points at a stale `master`).

## 6. build.bat + shell equivalent

CLAUDE.md §2 mandates both; neither exists.

## 7. Docs

`README.md` still says **"Arithmos"** and names `rust/arithmos_core/` and
`python/arithmos/` — both paths wrong. Rewrite it. `CHANGELOG.md` stops at
2.0.1: add entries for 2.0.2/2.0.3/2.0.4 covering the PyO3 facade
(Expression/Integer/Variable, operator dispatch, LaTeX, compact form) and the
precision fix.

## Definition of done

`cargo test` and `cargo clippy -D warnings` green; both features compile;
constants and SI registries resolve; zero `Arithmos` identifiers outside the
compatibility aliases; CI runs the Rust tests; README/CHANGELOG accurate.
Commit and push to `main` with no AI attribution in the message (CLAUDE.md rule).
