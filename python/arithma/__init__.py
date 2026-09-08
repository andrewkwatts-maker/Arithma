"""arithma - Symbolic mathematics engine with a Rust backend.

The Rust crate ``arithma_core`` is the implementation. This package is a thin
wrapper over it: every symbol below is a PyO3 binding, and there is no Python
implementation of the mathematics. The crate stands alone -- it does not
depend on Python, and the ``python`` cargo feature is purely additive.

Surface
-------
Core types
    :class:`Expression`, :class:`Integer`, :class:`Variable`
Calculus
    :func:`differentiate`, :func:`differentiate_n`, :func:`gradient`,
    :func:`integrate`, :func:`integrate_definite`, :func:`integrate_numeric`
Constants registry
    :func:`lookup`, :func:`register`, :func:`initialize_defaults`, ...
Geometry
    :class:`Vector`, :class:`Line`, :class:`Plane`, :class:`Sphere`,
    :func:`intersect_line_plane`, :func:`intersect_line_sphere`, ...
Linear algebra
    :class:`Matrix`, :class:`Tensor` -- transpose, matmul, determinant,
    reshape, axis permutation
Units and dimensions
    :class:`Unit`, :func:`si_lookup`, :func:`dimension_of`,
    :func:`dimension_symbol`, :func:`base_dimensions`
Equation solving and Fourier
    :func:`solve`, :func:`solve_equation`, :func:`solve_system`,
    :func:`fourier_transform`, :class:`FourierConfig`,
    :class:`FourierTransform`, :class:`Solution`
Numerical analysis
    :func:`find_root_bisection`, :func:`find_root_newton_raphson`,
    :func:`find_root_secant`, :func:`solve_with_method`,
    :func:`find_stationary_points`, :func:`find_inflection_points`,
    :func:`find_extrema`, :func:`analyze_intervals`, :class:`Interval`
Statistics
    :class:`Normal`, :class:`Binomial`, :class:`Bernoulli`,
    :class:`ConfidenceInterval`, :func:`erf`, :func:`sample_mean`,
    :func:`sample_variance`, :func:`normal_confidence_interval`

Availability
------------
When the Rust extension is present :func:`is_rust_backend` returns ``True``
and ``_HAS_RUST`` is ``True``. If it is absent the module still imports, but
every symbol raises an informative :class:`ImportError` on use -- it never
silently falls back to a slow or wrong Python implementation.

Use :func:`assert_rust_backend` to fail fast at startup when the accelerated
path is required.

Known gaps in the underlying crate
----------------------------------
These are limits of ``arithma_core``, not of the bindings. They are stated
here so callers are not surprised:

- :class:`Matrix` covers add, subtract, scalar multiply, transpose, matmul
  (``@``), trace, determinant, inverse, adjugate and eigenvalues. The
  determinant is exact and symbolic, computed by summation over permutations,
  so it is **capped at order 6**. :meth:`Matrix.eigenvalues_real` finds
  **real roots only** and collapses repeated ones -- a rotation matrix
  correctly returns ``[]``. **Eigenvectors are not computed.**
- :class:`Tensor` has indexing, strides, reshape, axis permutation, the
  elementwise operations, and contraction (:meth:`tensordot`,
  :meth:`trace`, :meth:`outer`). There is no einsum-style string API.
- Statistics covers normal, binomial and Bernoulli only. There is no Poisson,
  exponential, uniform, chi-squared or Student *t*, so
  :func:`normal_confidence_interval` is a **z** interval and is unreliable for
  small samples.
- :func:`solve_system` handles linear systems only, and ignores its
  ``strategy`` argument (the value is still validated, so a typo is an error
  rather than a silent no-op).
- :class:`Unit` handles dimensions and SI prefixes: metres and seconds
  refuse to combine, ``N`` is ``m*kg*s^-2``, and :func:`convert` moves
  between prefixed units exactly. **Only the SI base units and their
  prefixes** are known -- ``g``, ``min``, ``inch`` and other non-coherent
  units are not, so ``mg`` is an unknown unit rather than a guess.
- :func:`si_lookup` resolves the **7 SI base units only**. Derived units
  (``N``, ``J``, ``Hz``, ...) return ``None`` by design -- the catalogue
  states that they are meant to be represented as expressions, not table
  entries. This is a design decision, not missing data.

Simplification
--------------
:meth:`Expression.simplify` folds constants and applies the algebraic
identities. It is **exact**: numeric folding runs on unlimited-precision
integers, so ``7 / 2`` stays a quotient rather than rounding, and
``2 ** 64`` is computed exactly rather than through a float.

    >>> from arithma import Expression as E
    >>> E.number(1).add(E.number(1)).simplify().evaluate({})
    2.0
    >>> E.variable("x").add(E.number(0)).simplify().to_latex()
    'x'

Named constants are *not* collapsed to literals unless you pass
``allow_numeric_collapse=True``, because a constant's cached float is an
approximation and folding it would silently discard exactness.

Rewrites cover constant folding, additive and multiplicative identities,
absorbing zero, exact division, the power rules, and like-term collection
(``x + x -> 2*x``, ``2*x + 3*x -> 5*x``, ``x * x -> x**2``).

Collection is **syntactic**: ``2*x + x*2`` is not collected, because the two
operands are written in different orders and there is no canonical ordering.
Trigonometric identities are not implemented.
"""

__version__ = "2.0.4"

# ---------------------------------------------------------------------------
# Optional Rust extension
# ---------------------------------------------------------------------------
_HAS_RUST = False

#: Every symbol the Rust extension provides, grouped by domain. Used both for
#: the real import and to synthesise the informative failures below, so the two
#: paths cannot drift apart.
_CLASSES = (
    "Bernoulli",
    "Binomial",
    "ConfidenceInterval",
    "CriticalPoint",
    "Expression",
    "FourierConfig",
    "FourierTransform",
    "FunctionAnalysis",
    "Integer",
    "IntersectionResult",
    "Interval",
    "Line",
    "Matrix",
    "Normal",
    "Plane",
    "RootResult",
    "Solution",
    "Sphere",
    "Tensor",
    "Unit",
    "Variable",
    "Vector",
)

_FUNCTIONS = (
    "analyze_intervals",
    "analyze_point",
    "base_dimensions",
    "binomial_coefficient",
    "classify_point",
    "convert",
    "default_constants_json",
    "diff_chain",
    "diff_product",
    "diff_sum",
    "differentiate",
    "differentiate_iterative",
    "differentiate_n",
    "dimension_of",
    "dimension_symbol",
    "distribution_from_json",
    "erf",
    "erfc",
    "evaluate_interval",
    "find_extrema",
    "find_inflection_points",
    "find_root_bisection",
    "find_root_newton_raphson",
    "find_root_secant",
    "find_stationary_points",
    "fourier_transform",
    "fourier_window_weight",
    "gradient",
    "initialize_defaults",
    "integrate",
    "integrate_definite",
    "integrate_numeric",
    "intersect_line_plane",
    "intersect_line_sphere",
    "intersect_plane_plane",
    "is_registered",
    "is_rust_backend",
    "line_closest_point",
    "line_closest_point_param",
    "load_constants_from_json",
    "lookup",
    "lookup_value",
    "normal_confidence_interval",
    "normal_critical_value",
    "prefix_power",
    "register",
    "register_many",
    "registered_count",
    "reregister",
    "sample_kurtosis",
    "sample_mean",
    "sample_skewness",
    "sample_std_dev",
    "sample_variance",
    "si_base_units",
    "si_lookup",
    "si_prefixes",
    "si_unit_count",
    "si_units_json",
    "solve",
    "solve_equation",
    "solve_system",
    "solve_with_method",
    "split_unit_symbol",
    "version_rust",
)

#: Plain values (not classes or functions) the extension exports. They need a
#: separate list because the graceful-degradation path below substitutes a
#: raising callable for everything else, and a constant is not called.
_CONSTANTS = (
    "DEFAULT_SAMPLE_SEED",
    "FOURIER_WINDOWS",
    "MAX_SAMPLE_COUNT",
    "SOLVER_STRATEGIES",
)

try:
    from . import _arithma_core as _core  # type: ignore[attr-defined]

    for _name in _CLASSES + _FUNCTIONS + _CONSTANTS:
        globals()[_name] = getattr(_core, _name)
    del _name
    _HAS_RUST = True
except ImportError as _import_error:  # pragma: no cover - depends on build
    _REASON = (
        "arithma._arithma_core failed to load ({}). Install the prebuilt wheel "
        "or run `maturin develop --features python`."
    ).format(_import_error)

    def _missing(symbol: str):
        """Build a stand-in that raises rather than degrading silently."""

        def _raise(*_args, **_kwargs):
            raise ImportError(f"arithma.{symbol} unavailable: {_REASON}")

        return _raise

    def is_rust_backend() -> bool:  # type: ignore[misc]
        return False

    def version_rust() -> str:  # type: ignore[misc]
        return __version__

    for _name in _FUNCTIONS:
        if _name not in ("is_rust_backend", "version_rust"):
            globals()[_name] = _missing(_name)

    for _name in _CONSTANTS:
        globals()[_name] = None

    for _name in _CLASSES:
        globals()[_name] = type(
            _name,
            (),
            {
                "__init__": _missing(_name),
                "__doc__": f"Unavailable: {_REASON}",
            },
        )
    del _name


class RustBackendUnavailable(RuntimeError):
    """Raised when the Rust backend is required but is not usable."""


def assert_rust_backend() -> None:
    """Raise :class:`RustBackendUnavailable` unless the Rust backend is live.

    Call this at the top of any program that depends on the accelerated path,
    so a missing or stale extension fails at startup rather than silently
    costing orders of magnitude at runtime.

    ``_HAS_RUST`` alone only says the extension imported. This additionally
    checks the version handshake: a stale ``_arithma_core`` left in the source
    tree from an earlier build is easy to miss and produces baffling
    behaviour.
    """
    if not _HAS_RUST:
        raise RustBackendUnavailable(
            "arithma._arithma_core is not available. Build it with "
            "`maturin develop --features python`, or install the prebuilt wheel."
        )
    rust_version = version_rust()
    if str(rust_version) != str(__version__):
        raise RustBackendUnavailable(
            f"extension version {rust_version!r} does not match the Python "
            f"package version {__version__!r}; rebuild with "
            "`maturin develop --features python`."
        )


def backend_report() -> dict:
    """Describe the backend, for diagnostics and bug reports."""
    return {
        "has_rust": _HAS_RUST,
        "rust_version": version_rust() if _HAS_RUST else None,
        "python_version": __version__,
        "version_mismatch": (
            None
            if not _HAS_RUST or str(version_rust()) == str(__version__)
            else f"{version_rust()!r} != {__version__!r}"
        ),
        "classes": len(_CLASSES),
        "functions": len(_FUNCTIONS),
        "constants": len(_CONSTANTS),
    }


__all__ = [
    "__version__",
    "_HAS_RUST",
    "RustBackendUnavailable",
    "assert_rust_backend",
    "backend_report",
    *_CLASSES,
    *_FUNCTIONS,
    *_CONSTANTS,
]
