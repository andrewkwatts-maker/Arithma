"""arithma - Symbolic mathematics engine (Python bindings).

Wave 1 scaffold; populated in Wave 2.
"""

__version__ = "2.0.1"

# Guard the native-extension import so the pure-Python package remains
# usable (with reduced functionality) on environments where the Rust
# extension hasn't been built yet.
try:
    from . import _arithmos_core  # type: ignore[attr-defined]  # noqa: F401
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

__all__ = ["__version__", "_HAS_RUST"]

