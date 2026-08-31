# Arithma

Symbolic mathematics engine — a Rust core with Python bindings.

Arithma sits at the bottom of the dependency chain for the wider ecosystem
(eml-math, eml-spectral, metaphysica, periodica), which consume it as a git
submodule behind a `with-arithma` feature flag rather than as a PyPI dependency.

```bash
pip install arithma
```

```python
import arithma

x = arithma.Expression.variable("x")
expr = arithma.Expression.sin(x) + 2 * x        # operator dispatch builds an AST
expr.evaluate({"x": 0.5})                       # -> 1.479425538604203
expr.to_latex()                                 # -> '\\sin x + 2 x'

blob = expr.to_compact()                        # JSON-friendly tagged form
arithma.Expression.from_compact(blob)           # round-trips losslessly
```

`arithma.is_rust_backend()` reports whether the compiled extension loaded. The
package imports without it so pure-Python consumers can degrade gracefully, but
the three core classes raise an informative `ImportError` on use — there is no
pure-Python fallback maths.

## Layout

| Path | Contents |
|---|---|
| `rust/arithma_core/` | Rust core crate (`cdylib` + `lib`; PyO3 behind the `python` feature) |
| `python/arithma/` | Python package wrapping the extension via maturin |
| `tests/` | pytest suite for the Python facade |
| `docs/` | Working briefs (see `docs/PHASE0.md`) |

## Status

The **type surface is complete; many algorithm bodies are not.** Differentiation,
expression evaluation, the constants and SI-unit registries, exact-integer
arithmetic, the lookup tables and the PyO3 facade are implemented and tested.
Integration, equation solving, root finding, critical points, geometry,
probability distributions, Fourier transforms and `Emit` codegen are still
`unimplemented!()` and will panic if called — they are being filled in
progressively. `grep -rn 'unimplemented!' rust/arithma_core/src` is the current
work list.

Exposed to Python today: `Expression`, `Integer`, `Variable`. The Rust core
carries considerably more (notably a working iterative differentiator) that the
facade does not yet surface.

## Building

```bash
./build.sh          # Linux/macOS   — fmt, clippy, cargo test, pytest, wheel
build.bat           # Windows       — same gate
```

Or piecemeal:

```bash
cargo test                              # Rust core (all three feature sets in CI)
maturin develop --features python       # build the extension in place
pytest tests/ -v
```

## Features

| Feature | Effect |
|---|---|
| `python` | Builds the PyO3 extension module (`arithma._arithma_core`) |
| `rust-support` | In-process Rust backend for the external-function registry |
| `cpp-support` | C-ABI backend; exchanges expressions as JSON across the FFI seam |

## Licence

MIT — see [LICENSE](LICENSE).
