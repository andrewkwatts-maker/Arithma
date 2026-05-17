"""Core smoke tests for the arithma package."""
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


def test_expression_class_available():
    assert arithma.Expression is not None


def test_integer_class_available():
    assert arithma.Integer is not None


def test_variable_class_available():
    assert arithma.Variable is not None


def test_is_rust_backend():
    assert arithma.is_rust_backend() is True


def test_version_rust_non_empty():
    assert arithma.version_rust() != ""


def test_expression_number():
    expr = arithma.Expression.number(42)
    assert expr is not None
    assert expr.to_float() == 42.0


def test_expression_variable():
    x = arithma.Expression.variable("x")
    assert x is not None
    assert x.to_float() is None  # unbound variable has no float value


def test_expression_arithmetic():
    a = arithma.Expression.number(3)
    b = arithma.Expression.number(4)
    result = a + b
    assert result.to_float() == 7.0


def test_expression_multiply():
    a = arithma.Expression.number(6)
    b = arithma.Expression.number(7)
    assert (a * b).to_float() == 42.0


def test_integer_basic():
    n = arithma.Integer(10)
    assert n is not None
