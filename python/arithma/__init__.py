"""arithma - Symbolic mathematics engine with Rust backend.

Core types
----------
Expression, Integer, Variable are PyO3-backed wrappers around the Rust
``arithma_core`` crate. They become available once the Rust extension
(``_arithma_core``) is loaded.

Availability
------------
When the Rust extension is present, ``is_rust_backend()`` returns ``True``
and ``_HAS_RUST`` is ``True``. If the extension is absent the module still
imports successfully and the three core classes raise an informative
``ImportError`` on use.
"""

__version__ = "2.0.4"

# ---------------------------------------------------------------------------
# Optional Rust extension
# ---------------------------------------------------------------------------
_HAS_RUST = False

try:
    from ._arithma_core import (  # type: ignore[attr-defined]
        Expression,
        Integer,
        Variable,
        is_rust_backend,
        version_rust,
    )
    _HAS_RUST = True
except ImportError:
    def is_rust_backend() -> bool:  # type: ignore[misc]
        return False

    def version_rust() -> str:  # type: ignore[misc]
        return __version__

    class _MissingRust:
        """Sentinel raised when the Rust extension didn't build.

        Importing :mod:`arithma` still works (so pure-Python downstreams can
        gracefully degrade) but instantiating ``Expression`` / ``Integer`` /
        ``Variable`` immediately raises so callers get a clear error.
        """

        _name: str

        def __init_subclass__(cls, **kwargs):  # pragma: no cover - defensive
            raise ImportError(
                "arithma._arithma_core not built — install via maturin / pip"
            )

        def __init__(self, *_args, **_kwargs):  # pragma: no cover - defensive
            raise ImportError(
                f"arithma.{self._name} unavailable: Rust extension "
                "(_arithma_core) failed to load. Install the prebuilt wheel "
                "or run `maturin develop --features python`."
            )

    class Expression(_MissingRust):  # type: ignore[no-redef]
        _name = "Expression"

    class Integer(_MissingRust):  # type: ignore[no-redef]
        _name = "Integer"

    class Variable(_MissingRust):  # type: ignore[no-redef]
        _name = "Variable"


__all__ = [
    "__version__",
    "_HAS_RUST",
    "Expression",
    "Integer",
    "Variable",
    "is_rust_backend",
    "version_rust",
]
