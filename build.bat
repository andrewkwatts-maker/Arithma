@echo off
REM ====== Arithma build script (Windows) ======
REM Mirrors build.sh. Runs the full gate the CI workflow runs, then builds
REM the wheel. Override the interpreter with:  set PY=path\to\python.exe

setlocal
if "%PY%"=="" set "PY=python"

echo === Rust: format check ===
cargo fmt --all --check
if errorlevel 1 (
    echo FAILED: run "cargo fmt --all" to fix formatting.
    exit /b 1
)

echo === Rust: clippy ===
cargo clippy --all-targets -- -D warnings
if errorlevel 1 exit /b 1

echo === Rust: tests (default) ===
cargo test
if errorlevel 1 exit /b 1

echo === Rust: tests (rust-support) ===
cargo test --features rust-support
if errorlevel 1 exit /b 1

echo === Rust: tests (cpp-support) ===
cargo test --features cpp-support
if errorlevel 1 exit /b 1

echo === Python: install with dev extras ===
"%PY%" -m pip install --upgrade pip maturin
if errorlevel 1 exit /b 1
"%PY%" -m pip install .[dev]
if errorlevel 1 exit /b 1

echo === Python: tests ===
"%PY%" -m pytest tests/ -v --tb=short
if errorlevel 1 exit /b 1

echo === Build wheel ===
"%PY%" -m maturin build --release
if errorlevel 1 exit /b 1

echo.
echo Build OK. Wheel is in target\wheels\.
endlocal
