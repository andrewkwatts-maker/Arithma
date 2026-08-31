#!/usr/bin/env bash
# ====== Arithma build script (Linux / macOS) ======
# Mirrors build.bat. Runs the full gate the CI workflow runs, then builds the
# wheel. Override the interpreter with:  PY=python3.12 ./build.sh
set -euo pipefail

PY="${PY:-python3}"

echo "=== Rust: format check ==="
if ! cargo fmt --all --check; then
    echo 'FAILED: run "cargo fmt --all" to fix formatting.' >&2
    exit 1
fi

echo "=== Rust: clippy ==="
cargo clippy --all-targets -- -D warnings

echo "=== Rust: tests (default) ==="
cargo test

echo "=== Rust: tests (rust-support) ==="
cargo test --features rust-support

echo "=== Rust: tests (cpp-support) ==="
cargo test --features cpp-support

echo "=== Python: install with dev extras ==="
"$PY" -m pip install --upgrade pip maturin
"$PY" -m pip install '.[dev]'

echo "=== Python: tests ==="
"$PY" -m pytest tests/ -v --tb=short

echo "=== Build wheel ==="
"$PY" -m maturin build --release

echo
echo "Build OK. Wheel is in target/wheels/."
