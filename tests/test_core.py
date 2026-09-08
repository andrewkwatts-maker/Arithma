"""Core smoke tests for the arithma package."""
from importlib.metadata import version as _installed_version

import pytest

import arithma


def test_version():
    # Compares against the installed package's own metadata (populated from
    # pyproject.toml at build time) rather than a second hardcoded literal,
    # so this test can't drift out of sync with a version bump on its own.
    assert arithma.__version__ == _installed_version("arithma")


def test_has_rust_flag_is_bool():
    assert isinstance(arithma._HAS_RUST, bool)


def test_rust_backend_available():
    """Rust extension must be present when installed via maturin."""
    assert arithma._HAS_RUST, (
        "arithma._arithma_core not found — maturin build may have failed"
    )


def test_is_rust_backend():
    assert arithma.is_rust_backend() is True


def test_version_rust_non_empty():
    assert arithma.version_rust() != ""


def test_version_rust_matches_package():
    assert arithma.version_rust() == arithma.__version__


# Wave-3 surface — Expression, Integer, Variable are now real PyO3 wrappers.

def test_expression_is_wave3_class():
    assert arithma.Expression is not None
    # Must be a usable class (instantiable via its factory).
    x = arithma.Expression.variable("x")
    assert x is not None


def test_integer_is_wave3_class():
    assert arithma.Integer is not None
    n = arithma.Integer.from_str("1")
    assert n.value() == 1


def test_variable_is_wave3_class():
    assert arithma.Variable is not None
    v = arithma.Variable("x")
    assert v.name == "x"


# ---------------------------------------------------------------------------
# Surface completeness
# ---------------------------------------------------------------------------

def test_wrapper_reexports_everything_the_extension_provides():
    """The wrapper's symbol lists must not lag behind the crate.

    They did: the extension exported 87 symbols while ``__init__`` re-exported
    42, so the solver, Fourier, numerical and statistics bindings were built
    into the wheel and unreachable as ``arithma.<name>``. Nothing failed --
    they were simply absent, which is the hardest kind of gap to notice.
    """
    if not arithma._HAS_RUST:  # pragma: no cover - depends on build
        pytest.skip("requires the built extension")
    exported = {n for n in dir(arithma._arithma_core) if not n.startswith("_")}
    listed = set(arithma._CLASSES) | set(arithma._FUNCTIONS) | set(arithma._CONSTANTS)
    missing = exported - listed
    assert not missing, f"extension exports these but the wrapper does not list them: {sorted(missing)}"
    stale = listed - exported
    assert not stale, f"wrapper lists these but the extension does not export them: {sorted(stale)}"


def test_every_listed_symbol_is_actually_importable():
    if not arithma._HAS_RUST:  # pragma: no cover - depends on build
        pytest.skip("requires the built extension")
    for name in arithma.__all__:
        assert hasattr(arithma, name), f"arithma.{name} is in __all__ but not defined"
