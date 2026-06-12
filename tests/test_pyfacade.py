"""Wave-3 PyO3 facade tests — Expression / Integer / Variable wrappers."""
import math

import pytest

import arithma
from arithma import Expression, Integer, Variable


# ---------------------------------------------------------------------------
# Sanity / availability
# ---------------------------------------------------------------------------

def test_rust_backend_loaded():
    assert arithma._HAS_RUST, "Rust extension must be present for pyfacade tests"


def test_classes_are_not_none():
    assert Expression is not None
    assert Integer is not None
    assert Variable is not None


# ---------------------------------------------------------------------------
# Expression construction
# ---------------------------------------------------------------------------

def test_variable_construction():
    x = Expression.variable("x")
    assert x.kind() == "variable"
    assert not x.is_constant()


def test_number_construction_int():
    n = Expression.number(3)
    assert n.kind() == "number"
    assert n.is_constant()


def test_number_construction_float():
    n = Expression.number(3.14)
    assert n.is_constant()


def test_number_rejects_bool():
    with pytest.raises(TypeError):
        Expression.number(True)


def test_constant_construction():
    pi = Expression.constant("pi", math.pi)
    assert pi.kind() == "constant"
    assert pi.is_constant()


# ---------------------------------------------------------------------------
# Operator dispatch
# ---------------------------------------------------------------------------

def test_add_operator_produces_function_node():
    x = Expression.variable("x")
    y = Expression.variable("y")
    z = x + y
    assert z.kind().startswith("function:")
    kids = z.children()
    assert len(kids) == 2
    assert kids[0].kind() == "variable"
    assert kids[1].kind() == "variable"


def test_radd_with_python_int():
    x = Expression.variable("x")
    z = 2 + x  # noqa: invokes __radd__
    assert z.kind().startswith("function:")


def test_mul_with_python_int():
    x = Expression.variable("x")
    z = x * 2
    assert z.kind().startswith("function:")


def test_truediv_operator():
    x = Expression.variable("x")
    y = Expression.variable("y")
    z = x / y
    assert "Divide" in z.kind() or z.kind().startswith("function:")


def test_pow_operator():
    x = Expression.variable("x")
    z = x ** 2
    assert z.kind().startswith("function:")
    assert len(z.children()) == 2


def test_neg_operator():
    x = Expression.variable("x")
    z = -x
    assert z.kind().startswith("function:")
    assert len(z.children()) == 1


def test_sub_operator():
    x = Expression.variable("x")
    y = Expression.variable("y")
    z = x - y
    assert z.kind().startswith("function:")
    assert len(z.children()) == 2


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def test_evaluate_sin_zero():
    x = Expression.variable("x")
    expr = Expression.sin(x)
    val = expr.evaluate({"x": 0.0})
    assert val == pytest.approx(0.0, abs=1e-12)


def test_evaluate_exp_zero():
    y = Expression.variable("y")
    expr = Expression.exp(y)
    val = expr.evaluate({"y": 0.0})
    assert val == pytest.approx(1.0, abs=1e-12)


def test_evaluate_compound_expression():
    """sin(x) + 1 * exp(y) at x=0, y=1 → 0 + e."""
    x = Expression.variable("x")
    y = Expression.variable("y")
    e = Expression.sin(x) + Expression.number(1) * Expression.exp(y)
    val = e.evaluate({"x": 0.0, "y": 1.0})
    assert val == pytest.approx(math.e, rel=1e-9)


def test_evaluate_multiplication_by_zero():
    x = Expression.variable("x")
    expr = x * Expression.number(0)
    val = expr.evaluate({"x": 17.0})
    assert val == pytest.approx(0.0, abs=1e-12)


def test_evaluate_unbound_raises():
    x = Expression.variable("x")
    expr = x + Expression.number(1)
    with pytest.raises(KeyError):
        expr.evaluate({})


def test_evaluate_python_int_binding_accepted():
    x = Expression.variable("x")
    expr = x * Expression.number(2)
    val = expr.evaluate({"x": 3})
    assert val == pytest.approx(6.0, abs=1e-12)


# ---------------------------------------------------------------------------
# LaTeX rendering
# ---------------------------------------------------------------------------

def test_to_latex_variable():
    x = Expression.variable("x")
    assert x.to_latex() == "x"


def test_to_latex_number():
    n = Expression.number(7)
    assert n.to_latex() == "7"


def test_to_latex_add():
    x = Expression.variable("x")
    y = Expression.variable("y")
    s = (x + y).to_latex()
    assert "x" in s and "y" in s and "+" in s


def test_to_latex_pow():
    x = Expression.variable("x")
    s = (x ** 2).to_latex()
    assert "^" in s
    assert "x" in s


def test_to_latex_sin():
    x = Expression.variable("x")
    s = Expression.sin(x).to_latex()
    assert "\\sin" in s


def test_to_latex_div_is_frac():
    x = Expression.variable("x")
    y = Expression.variable("y")
    s = (x / y).to_latex()
    assert "\\frac" in s


def test_to_latex_nonempty():
    x = Expression.variable("x")
    y = Expression.variable("y")
    e = Expression.sin(x) + Expression.number(1) * Expression.exp(y)
    s = e.to_latex()
    assert s != ""
    assert "\\sin" in s
    assert "e^" in s


# ---------------------------------------------------------------------------
# Tree walking
# ---------------------------------------------------------------------------

def test_children_of_leaf_is_empty():
    x = Expression.variable("x")
    assert x.children() == []


def test_children_walks_one_layer():
    x = Expression.variable("x")
    y = Expression.variable("y")
    expr = x + y * Expression.number(3)
    kids = expr.children()
    assert len(kids) == 2
    # The right child should itself have children.
    rhs_kids = kids[1].children()
    assert len(rhs_kids) == 2


# ---------------------------------------------------------------------------
# Integer
# ---------------------------------------------------------------------------

def test_integer_from_str_small():
    n = Integer.from_str("42")
    assert n.value() == 42


def test_integer_from_str_negative():
    n = Integer.from_str("-123")
    assert n.value() == -123


def test_integer_from_str_zero():
    n = Integer.from_str("0")
    assert n.value() == 0


def test_integer_from_str_large():
    big = "9" * 100
    n = Integer.from_str(big)
    assert n.value() == int(big)


def test_integer_from_str_arbitrary_precision():
    s = "123456789012345678901234567890"
    n = Integer.from_str(s)
    assert n.value() == int(s)


def test_integer_from_str_invalid_raises():
    with pytest.raises(ValueError):
        Integer.from_str("not-a-number")


def test_integer_str_round_trip():
    n = Integer.from_str("987654321")
    assert str(n) == "987654321"


def test_integer_constructor_from_python_int():
    n = Integer(2 ** 70)
    assert n.value() == 2 ** 70


# ---------------------------------------------------------------------------
# Variable
# ---------------------------------------------------------------------------

def test_variable_unbound():
    v = Variable("alpha")
    assert v.name == "alpha"
    assert v.is_unbound()
    assert v.binding() is None


def test_variable_none_binding_is_unbound():
    v = Variable("alpha", binding=None)
    assert v.is_unbound()


def test_variable_float_binding():
    v = Variable("alpha", binding=0.5)
    assert not v.is_unbound()
    assert v.binding() == pytest.approx(0.5)


def test_variable_int_binding():
    v = Variable("beta", binding=7)
    assert not v.is_unbound()
    assert v.binding() == pytest.approx(7.0)


def test_variable_expression_binding():
    x = Expression.variable("x")
    expr = x + Expression.number(1)
    v = Variable("gamma", binding=expr)
    assert not v.is_unbound()
    bound = v.binding()
    assert isinstance(bound, Expression)


def test_variable_to_expression():
    v = Variable("delta")
    e = v.to_expression()
    assert e.kind() == "variable"
    assert e.evaluate({"delta": 3.14}) == pytest.approx(3.14)


def test_variable_set_binding():
    v = Variable("eps")
    v.set_binding(2.5)
    assert v.binding() == pytest.approx(2.5)
    v.set_binding(None)
    assert v.is_unbound()


# ---------------------------------------------------------------------------
# Compact-form serialisation (to_compact / from_compact)
# ---------------------------------------------------------------------------
#
# The compact form is a tagged Python list/list-of-lists that JSON-serialises
# directly. It backs the ``arithma_compact`` field shipped to the website in
# ``formulas.json``. Round-trip equivalence is checked by re-serialising the
# inflated expression and asserting deep equality on the JSON-friendly form.

import json


def _round_trip(expr):
    """Return both compact forms; equality of compact form ⇒ equality of AST."""
    blob = expr.to_compact()
    inflated = Expression.from_compact(blob)
    return blob, inflated.to_compact()


def test_compact_number_int():
    blob, blob_back = _round_trip(Expression.number(7))
    assert blob == ["num", "7"]
    assert blob == blob_back


def test_compact_number_float():
    # Floats route through ``from_f64`` and may decompose into ``num / num``.
    blob, blob_back = _round_trip(Expression.number(3.14))
    assert blob == blob_back


def test_compact_variable():
    blob, blob_back = _round_trip(Expression.variable("x"))
    assert blob == ["var", "x"]
    assert blob == blob_back


def test_compact_constant():
    pi = Expression.constant("pi", math.pi)
    blob, blob_back = _round_trip(pi)
    assert blob[0] == "const"
    assert blob[1] == "pi"
    assert blob[2] == pytest.approx(math.pi)
    assert blob == blob_back


def test_compact_constant_no_value():
    pi = Expression.constant("pi")
    blob, blob_back = _round_trip(pi)
    assert blob == ["const", "pi", None]
    assert blob == blob_back


def test_compact_sum():
    x = Expression.variable("x")
    y = Expression.variable("y")
    expr = x + y
    blob, blob_back = _round_trip(expr)
    assert blob[0] == "fn"
    assert blob[1] == "add"
    assert blob[2] == ["var", "x"]
    assert blob[3] == ["var", "y"]
    assert blob == blob_back


def test_compact_product():
    x = Expression.variable("x")
    y = Expression.variable("y")
    expr = x * y
    blob, blob_back = _round_trip(expr)
    assert blob == ["fn", "mul", ["var", "x"], ["var", "y"]]
    assert blob == blob_back


def test_compact_sin_x():
    x = Expression.variable("x")
    expr = Expression.sin(x)
    blob, blob_back = _round_trip(expr)
    assert blob == ["fn", "sin", ["var", "x"]]
    assert blob == blob_back


def test_compact_exp_xy():
    x = Expression.variable("x")
    y = Expression.variable("y")
    expr = Expression.exp(x * y)
    blob, blob_back = _round_trip(expr)
    assert blob == [
        "fn",
        "exp",
        ["fn", "mul", ["var", "x"], ["var", "y"]],
    ]
    assert blob == blob_back


def test_compact_deep_compound():
    """sin(x) + 1 * exp(y) — mixes leaves, scalars, products and transcendentals."""
    x = Expression.variable("x")
    y = Expression.variable("y")
    expr = Expression.sin(x) + Expression.number(1) * Expression.exp(y)
    blob, blob_back = _round_trip(expr)
    assert blob == blob_back


def test_compact_json_round_trip():
    """The compact form must survive a JSON serialise → parse cycle unchanged."""
    x = Expression.variable("x")
    y = Expression.variable("y")
    expr = Expression.sin(x) + Expression.exp(x * y)
    blob = expr.to_compact()
    payload = json.dumps(blob)
    parsed = json.loads(payload)
    inflated = Expression.from_compact(parsed)
    assert inflated.to_compact() == blob


def test_compact_inflated_evaluates_the_same():
    """Inflated expression must numerically match the original."""
    x = Expression.variable("x")
    y = Expression.variable("y")
    expr = Expression.sin(x) + Expression.exp(x * y)
    inflated = Expression.from_compact(expr.to_compact())
    env = {"x": 0.5, "y": 1.25}
    assert inflated.evaluate(env) == pytest.approx(expr.evaluate(env), rel=1e-9)


def test_compact_negate_and_division():
    x = Expression.variable("x")
    y = Expression.variable("y")
    expr = -(x / y)
    blob, blob_back = _round_trip(expr)
    assert blob[0:2] == ["fn", "neg"]
    assert blob == blob_back


def test_compact_pow_operator():
    x = Expression.variable("x")
    expr = x ** Expression.number(3)
    blob, blob_back = _round_trip(expr)
    assert blob[0:2] == ["fn", "pow"]
    assert blob == blob_back


def test_compact_large_integer_literal():
    """Very large integers must survive without f64 precision loss."""
    big = 10 ** 60
    expr = Expression.number(0) + Expression.variable("z")  # placeholder
    # Build via Integer-flavoured path instead: number() takes int, route through
    # the f64 fallback for huge ints would lose precision, so the test focuses on
    # the decimal-string serialiser by going via the underlying expression.
    # Use a literal small int to round-trip; large-integer-literal *parsing* is
    # the from_compact side and is verified by feeding the raw blob directly.
    blob = ["num", str(big)]
    inflated = Expression.from_compact(blob)
    assert inflated.to_compact() == blob


def test_compact_nan_sentinel():
    blob = ["num", "NaN"]
    inflated = Expression.from_compact(blob)
    assert inflated.to_compact() == blob


def test_compact_infinity_sentinels():
    for sentinel in ("Inf", "-Inf"):
        blob = ["num", sentinel]
        inflated = Expression.from_compact(blob)
        assert inflated.to_compact() == blob


def test_compact_tuple_input_accepted():
    """``from_compact`` accepts tuples as well as lists for ergonomics."""
    inflated = Expression.from_compact(("var", "x"))
    assert inflated.to_compact() == ["var", "x"]


def test_compact_rejects_unknown_tag():
    with pytest.raises(ValueError):
        Expression.from_compact(["zzz", "x"])


def test_compact_rejects_empty_node():
    with pytest.raises(ValueError):
        Expression.from_compact([])


def test_compact_rejects_unsupported_operator_tag():
    with pytest.raises(ValueError):
        Expression.from_compact(["fn", "derivative", ["var", "x"]])
