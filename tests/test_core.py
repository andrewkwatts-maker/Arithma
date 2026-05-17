"""Core smoke tests for the arithma package — Wave-2 scaffold."""
import arithma


def test_version():
    assert arithma.__version__ == "2.0.1"


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


# Wave-3 stubs — Expression, Integer, Variable are not yet implemented.
# These tests document the expected None state until pyfacade Wave-3 lands.

def test_expression_is_none_until_wave3():
    assert arithma.Expression is None


def test_integer_is_none_until_wave3():
    assert arithma.Integer is None


def test_variable_is_none_until_wave3():
    assert arithma.Variable is None
