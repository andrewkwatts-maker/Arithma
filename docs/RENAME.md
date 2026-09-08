# Removing the `Arithmos*` aliases

## The short version

They can be removed from this repository alone. Nothing outside it uses them.

That contradicts what was planned, so the evidence is set out below rather than
asserted. The plan of record said the aliases were *"the migration mechanism for
four vendored dependants"* and had to drop *"in the same change that lands the
rename upstream across five repos"* — a coordinated edit that had been attempted
and reverted once. On checking, no such coordination is needed, and believing
otherwise is what made this look like a large job for months.

## What the aliases are

`arithma_core` was `arithmos_core`, and every public type was `Arithmos*` before
it became `Arithma*`. `src/lib.rs` re-exports the old spellings at the crate
root, and each defining module carries its own `#[deprecated(since = "2.0.4")]`
alias:

```
ArithmosConstants   ArithmosExpression       ArithmosFourierConfig
ArithmosFunction    ArithmosInteger          ArithmosSIUnits
ArithmosUnit        ArithmosVariable         ArithmosInterop
ArithmosExternalFunctionError                ArithmosExternalFunctionRegistry
```

plus roughly forty more at module scope (`ArithmosMatrix`, `ArithmosTensor`,
`ArithmosNormal`, `ArithmosCriticalPoint`, and so on).

## Why there is no downstream to coordinate with

The four consumers are eml-math, eml-spectral, metaphysica and periodica. Each
was checked, and none of them uses an `Arithmos*` type.

What each *does* carry is an `arithmos_bridge.rs` that imports the old **crate**
name. That is a different thing, and in every case the module is dead:

| Repo | State of its bridge |
|---|---|
| **EML-Math** | `arithmos_bridge.rs` exists; `lib.rs` says outright it "is NOT wired up". No `mod` declaration, `arithmos_core` is not a dependency. |
| **metaphysica** | `lib.rs`: *"present but is **not a module of this crate**"*. Same missing feature, same missing dependency. |
| **EML-Spectral** | `pub mod arithmos_bridge;` **is** declared, but the file opens `#![cfg(feature = "with-arithmos")]` and `Cargo.toml` states the feature "cannot be declared as a Cargo feature here". It compiles to nothing. |
| **periodica** | Already removed its `with-arithmos` / `with-physica` declarations — they gated modules that did not exist. |

So all four import a crate that is not among their dependencies, behind a
feature none of them declares. **None has ever compiled.** Removing the
`Arithmos*` aliases cannot break them, because nothing that would notice is
built.

## What to do

**In this repository, at the next minor or major version:**

1. Delete the alias block in `src/lib.rs` (the two sections headed
   *"Backward-compatibility aliases for the pre-rename `Arithmos*` names"*).
2. Delete the per-module `#[deprecated]` aliases — around fifty of them, all
   matching `pub use ... as Arithmos*` or `pub type Arithmos* =`.
3. Delete the `#[allow(deprecated)]` attributes that exist only to silence the
   re-exports.
4. `cargo test --no-default-features && cargo test --features python` and
   `grep -rn Arithmos rust/` — the only surviving hits should be in `docs/`.

They are already marked `#[deprecated(since = "2.0.4")]`, so 2.0.4 is a complete
deprecation cycle on its own. Ship it, then remove them in the release after.

**Separately, and not a blocker for the above** — the three dead bridges are
their own question. Each is a file that cannot compile, imports a crate that no
longer exists under that name, and is described in its own header as not wired
up. The options are to delete them, or to port them to `arithma_core` and wire
the feature and the dependency properly. Either is fine; leaving a module that
has never once been built is what should not continue, because it reads as an
integration point that exists.

## The lesson worth keeping

The blocker was a belief about the dependency graph, not the graph. Nothing
verified it, and it survived long enough to be recorded in a plan and used to
defer the work. A cross-repo claim of this kind is cheap to check — one grep for
the type names, one for the crate name, one look at whether the feature is
declared — and expensive to carry.
