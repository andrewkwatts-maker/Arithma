"""Core smoke tests for the arithma package."""
import arithma


def test_version():
    assert arithma.__version__ == "2.0.3"


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
