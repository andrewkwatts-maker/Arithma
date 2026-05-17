# Arithmos

Symbolic mathematics engine. Wave 1 scaffold; populated in Wave 2.

Upstream: https://github.com/andrewkwatts-maker/Arithmos.git

## Layout

- `rust/arithmos_core/` - Rust core crate (cdylib + rlib, optional `python` feature for PyO3)
- `python/arithmos/` - Python package wrapping the Rust extension via maturin
