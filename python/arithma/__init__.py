"""arithma - Symbolic mathematics engine with Rust backend.

Core types (Wave 3 — not yet exposed via PyO3)
-----------------------------------------------
Expression, Integer, Variable will be available once the Wave-3 pyfacade
wrappers land. Until then they are ``None`` and ``_HAS_RUST`` guards all
Rust-dependent paths.

Availability
------------
When the Rust extension (``_arithma_core``) is present, ``is_rust_backend()``
returns ``True`` and ``_HAS_RUST`` is ``True``.  If the extension is absent
the module still imports successfully.
"""

__version__ = "2.0.2"

# ---------------------------------------------------------------------------
# Optional Rust extension
# ---------------------------------------------------------------------------
_HAS_RUST = False

try:
    from ._arithma_core import (  # type: ignore[attr-defined]
        is_rust_backend,
        version_rust,
    )
    _HAS_RUST = True
except ImportError:
    def is_rust_backend() -> bool:  # type: ignore[misc]
        return False

    def version_rust() -> str:  # type: ignore[misc]
        return __version__

# Wave-3: PyO3 wrapper types — not yet implemented in pyfacade.rs
Expression = None  # type: ignore[assignment]
Integer    = None  # type: ignore[assignment]
Variable   = None  # type: ignore[assignment]

__all__ = [
    "__version__",
    "_HAS_RUST",
    "Expression",
    "Integer",
    "Variable",
    "is_rust_backend",
    "version_rust",
]
