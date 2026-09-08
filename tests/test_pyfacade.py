"""Wave-3 PyO3 facade tests — Expression / Integer / Variable wrappers."""
import math

import pytest

import arithma
from arithma import Expression, Integer, Matrix, Tensor, Unit, Variable


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


# ---------------------------------------------------------------------------
# Simplification
#
# `Expression.simplify` was absent from the facade while the Rust simplifier
# was a stub. These pin the behaviour end to end, through the PyO3 boundary,
# so a regression in either layer fails here rather than silently returning
# the input unchanged.
# ---------------------------------------------------------------------------

def test_constant_sum_folds():
    expr = Expression.number(1).add(Expression.number(1))
    assert expr.simplify().evaluate({}) == 2.0


def test_folding_collapses_the_tree_not_just_the_value():
    # `evaluate` would return 2.0 either way; the point is that the tree is
    # now a literal, so the fold actually happened.
    expr = Expression.number(1).add(Expression.number(1))
    assert expr.kind().startswith("function")
    assert expr.simplify().kind() == "number"


def test_additive_and_multiplicative_identities():
    x = Expression.variable("x")
    assert x.add(Expression.number(0)).simplify().to_latex() == "x"
    assert x.mul(Expression.number(1)).simplify().to_latex() == "x"


def test_absorbing_zero():
    x = Expression.variable("x")
    assert x.mul(Expression.number(0)).simplify().evaluate({}) == 0.0


def test_power_rules():
    x = Expression.variable("x")
    assert x.pow_(Expression.number(1)).simplify().to_latex() == "x"
    assert Expression.number(2).pow_(Expression.number(10)).simplify().evaluate({}) == 1024.0


def test_nested_folding_completes_in_one_call():
    expr = Expression.number(1).add(Expression.number(1)).mul(Expression.number(3))
    assert expr.simplify().evaluate({}) == 6.0


def test_division_is_exact_or_left_alone():
    assert Expression.number(12).div(Expression.number(4)).simplify().evaluate({}) == 3.0
    # 7/2 has no integer value; rounding it would be a silent precision loss.
    seven_halves = Expression.number(7).div(Expression.number(2)).simplify()
    assert seven_halves.kind().startswith("function")


def test_large_powers_stay_exact_through_the_boundary():
    # 2**64 is past what an f64 can represent without loss, so this proves the
    # fold runs on the unlimited-precision integer path rather than a float.
    big = Expression.number(2).pow_(Expression.number(64)).simplify()
    assert big.to_latex() == str(2 ** 64)


def test_symbolic_expressions_are_not_mangled():
    expr = Expression.variable("x").add(Expression.number(1))
    simplified = expr.simplify()
    assert simplified.kind().startswith("function")
    with pytest.raises(Exception):
        simplified.evaluate({})


def test_simplify_reaches_a_fixpoint():
    once = Expression.number(2).add(Expression.number(3)).simplify()
    assert not once.is_simplifiable(), "a simplified expression must be stable"


def test_is_simplifiable_reports_honestly():
    assert Expression.number(1).add(Expression.number(1)).is_simplifiable()
    assert not Expression.variable("x").is_simplifiable()


def test_named_constants_need_an_explicit_opt_in():
    # A constant's cached float is an approximation; collapsing it by default
    # would discard exactness without the caller asking.
    two = Expression.constant("two", 2.0)
    assert two.simplify().kind() == "constant"
    assert two.simplify(allow_numeric_collapse=True).kind() == "number"


def test_an_oversized_iteration_budget_is_rejected():
    # Clamping silently would let a caller believe a budget was honoured.
    with pytest.raises(ValueError):
        Expression.number(1).add(Expression.number(1)).simplify(max_iterations=10 ** 9)


def test_a_zero_budget_is_a_no_op():
    expr = Expression.number(1).add(Expression.number(1))
    assert expr.simplify(max_iterations=0).kind().startswith("function")


# ---------------------------------------------------------------------------
# Matrix algebra
#
# `Matrix` and `Tensor` were containers on the Python side: shape, indexing and
# repr, with nothing that combined two of them. These pin the algebra through
# the PyO3 boundary.
# ---------------------------------------------------------------------------

def _m(rows):
    """Build a Matrix from a nested list of ints."""
    return Matrix.from_rows([[Expression.number(v) for v in r] for r in rows])


def _values(m):
    return [c.evaluate({}) for c in m.simplified().cells()]


def test_matrix_transpose():
    m = _m([[1, 2, 3], [4, 5, 6]])
    t = m.transpose()
    assert t.shape == (3, 2)
    assert _values(t) == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]


def test_matrix_add_and_sub():
    a = _m([[1, 2], [3, 4]])
    b = _m([[10, 20], [30, 40]])
    assert _values(a.add(b)) == [11.0, 22.0, 33.0, 44.0]
    assert _values(b.sub(a)) == [9.0, 18.0, 27.0, 36.0]


def test_matrix_shape_mismatch_raises():
    a = _m([[1, 2], [3, 4]])
    wrong = _m([[1, 2, 3]])
    with pytest.raises(ValueError):
        a.add(wrong)


def test_matrix_product():
    a = _m([[1, 2], [3, 4]])
    b = _m([[5, 6], [7, 8]])
    assert _values(a.matmul(b)) == [19.0, 22.0, 43.0, 50.0]
    # The `@` operator routes to the same code.
    assert _values(a @ b) == _values(a.matmul(b))


def test_matrix_operators():
    a = _m([[1, 2], [3, 4]])
    b = _m([[10, 20], [30, 40]])
    assert _values(a + b) == _values(a.add(b))
    assert _values(b - a) == _values(b.sub(a))


def test_matrix_scalar_multiplication():
    a = _m([[1, 2], [3, 4]])
    assert _values(a.scalar_mul(Expression.number(3))) == [3.0, 6.0, 9.0, 12.0]


def test_matrix_trace_and_determinant():
    a = _m([[1, 2], [3, 4]])
    assert a.trace().evaluate({}) == 5.0
    assert a.determinant().evaluate({}) == -2.0
    assert _m([[6, 1, 1], [4, -2, 5], [2, 8, 7]]).determinant().evaluate({}) == -306.0


def test_determinant_is_multiplicative_through_the_boundary():
    a = _m([[2, 0, 1], [3, -1, 2], [1, 4, 0]])
    b = _m([[1, 2, 0], [0, 1, 3], [2, 1, 1]])
    assert (a @ b).determinant().evaluate({}) == pytest.approx(
        a.determinant().evaluate({}) * b.determinant().evaluate({})
    )


def test_non_square_operations_raise():
    a = _m([[1, 2, 3], [4, 5, 6]])
    assert not a.is_square()
    with pytest.raises(ValueError):
        a.trace()
    with pytest.raises(ValueError):
        a.determinant()


def test_determinant_order_cap_raises_rather_than_hanging():
    big = Matrix.identity(9)
    with pytest.raises(ValueError):
        big.determinant()


# ---------------------------------------------------------------------------
# Tensor algebra
# ---------------------------------------------------------------------------

def _ramp(shape):
    count = 1
    for d in shape:
        count *= d
    return Tensor(shape, [Expression.number(i) for i in range(count)])


def test_tensor_strides_are_row_major():
    assert _ramp([2, 3, 4]).strides() == [12, 4, 1]


def test_tensor_reshape_preserves_order():
    t = _ramp([2, 6])
    r = t.reshape([3, 4])
    assert r.shape == [3, 4]
    assert [c.evaluate({}) for c in r.cells()] == [c.evaluate({}) for c in t.cells()]


def test_tensor_reshape_rejects_a_different_count():
    with pytest.raises(ValueError):
        _ramp([2, 6]).reshape([5, 5])


def test_tensor_permute_is_the_transpose_at_rank_two():
    t = _ramp([2, 3])
    p = t.permute_axes([1, 0])
    assert p.shape == [3, 2]
    assert [c.evaluate({}) for c in p.cells()] == [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]


def test_tensor_permute_validates_its_argument():
    t = _ramp([2, 3])
    for bad in ([0, 0], [0, 2], [0]):
        with pytest.raises(ValueError):
            t.permute_axes(bad)


def test_tensor_elementwise_operations():
    a = _ramp([2, 2])
    b = _ramp([2, 2])
    assert [c.evaluate({}) for c in a.add(b).simplified().cells()] == [0.0, 2.0, 4.0, 6.0]
    assert [c.evaluate({}) for c in a.hadamard(b).simplified().cells()] == [0.0, 1.0, 4.0, 9.0]
    with pytest.raises(ValueError):
        a.add(_ramp([4]))


def test_tensor_set_writes_where_get_reads():
    t = Tensor.zeros([2, 2, 2])
    t.set([1, 0, 1], Expression.number(9))
    assert t.get([1, 0, 1]).evaluate({}) == 9.0


# ---------------------------------------------------------------------------
# Dimensional analysis
# ---------------------------------------------------------------------------

def test_base_units_have_dimensions():
    assert Unit("m", "meter").dimension() == "m"
    assert Unit("kg", "kilogram").quantity() == "mass"


def test_derived_units_have_dimensions_even_though_the_catalogue_omits_them():
    # si_lookup("N") is None by design; the dimension still resolves.
    assert arithma.si_lookup("N") is None
    newton = Unit("N", "newton")
    assert newton.dimension() == "m*kg*s^-2"
    assert newton.quantity() == "force"
    assert newton.dimension_exponents() == [1, 1, -2, 0, 0, 0, 0]


def test_compatibility_distinguishes_unknown_from_incompatible():
    metre = Unit("m", "meter")
    second = Unit("s", "second")
    assert metre.is_compatible_with(metre) is True
    assert metre.is_compatible_with(second) is False
    # "I cannot tell" must not be reported as "incompatible".
    assert metre.is_compatible_with(Unit("zz", "unknown")) is None


def test_module_level_dimension_helpers():
    assert arithma.dimension_of("J") == "m^2*kg*s^-2"
    assert arithma.dimension_of("zz") is None
    assert arithma.dimension_symbol([1, 1, -2, 0, 0, 0, 0]) == "N"
    assert arithma.base_dimensions() == ["m", "kg", "s", "A", "K", "mol", "cd"]
    with pytest.raises(ValueError):
        arithma.dimension_symbol([1, 2, 3])


# ---------------------------------------------------------------------------
# Matrix inverse and eigenvalues
# ---------------------------------------------------------------------------

def test_matrix_inverse_round_trips_to_the_identity():
    m = _m([[4, 7], [2, 6]])
    product = m @ m.inverse()
    # Entries are exact symbolic quotients, but evaluating them to float
    # rounds, so a product of rounded values needs a tolerance. Demanding
    # bit-equality here would assert something false about floating point.
    assert _values(product) == pytest.approx([1.0, 0.0, 0.0, 1.0], abs=1e-12)


def test_matrix_inverse_of_a_three_by_three():
    m = _m([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    product = m @ m.inverse()
    assert _values(product) == pytest.approx(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], abs=1e-12
    )


def test_a_singular_matrix_has_no_inverse():
    m = _m([[1, 2], [2, 4]])
    assert m.determinant().evaluate({}) == 0.0
    with pytest.raises(ValueError):
        m.inverse()


def test_matrix_times_adjugate_is_determinant_times_identity():
    m = _m([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    det = m.determinant().evaluate({})
    product = m @ m.adjugate()
    assert _values(product) == pytest.approx(
        [det, 0.0, 0.0, 0.0, det, 0.0, 0.0, 0.0, det], abs=1e-9
    )


def test_cofactors_and_minors():
    m = _m([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    assert m.minor(1, 1).shape == (2, 2)
    assert _values(m.minor(1, 1)) == [1.0, 3.0, 7.0, 10.0]
    assert m.cofactor(0, 0).evaluate({}) == 2.0
    with pytest.raises(ValueError):
        m.minor(5, 0)


def test_eigenvalues_of_a_diagonal_matrix():
    m = _m([[2, 0, 0], [0, 3, 0], [0, 0, 7]])
    assert m.eigenvalues_real() == pytest.approx([2.0, 3.0, 7.0], abs=1e-9)


def test_eigenvalues_sum_to_the_trace():
    m = _m([[4, 1, 0], [1, 3, 1], [0, 1, 2]])
    values = m.eigenvalues_real()
    assert len(values) == 3
    assert sum(values) == pytest.approx(m.trace().evaluate({}), abs=1e-6)


def test_a_rotation_reports_no_real_eigenvalues():
    # Spectrum is +/- i. Returning [] is the correct answer, not a failure.
    m = _m([[0, -1], [1, 0]])
    assert m.eigenvalues_real() == []


def test_characteristic_polynomial_vanishes_at_the_eigenvalues():
    m = _m([[2, 0], [0, 3]])
    p = m.characteristic_polynomial("L")
    assert p.evaluate({"L": 2.0}) == pytest.approx(0.0, abs=1e-12)
    assert p.evaluate({"L": 3.0}) == pytest.approx(0.0, abs=1e-12)
    assert abs(p.evaluate({"L": 5.0})) > 1e-9


# ---------------------------------------------------------------------------
# Tensor contraction
# ---------------------------------------------------------------------------

def test_tensordot_over_inner_axes_is_matrix_multiplication():
    a = Tensor([2, 2], [Expression.number(v) for v in (1, 2, 3, 4)])
    b = Tensor([2, 2], [Expression.number(v) for v in (5, 6, 7, 8)])
    got = a.tensordot(b, [1], [0]).simplified()
    assert got.shape == [2, 2]
    assert [c.evaluate({}) for c in got.cells()] == [19.0, 22.0, 43.0, 50.0]


def test_contraction_shapes_compose():
    a = Tensor.zeros([2, 3, 4])
    b = Tensor.zeros([4, 5, 6])
    assert a.tensordot(b, [2], [0]).shape == [2, 3, 5, 6]


def test_trace_drops_two_axes():
    t = _ramp([3, 3])
    traced = t.trace(0, 1).simplified()
    assert traced.shape == []
    # ramp([3,3]) is 0..8 row-major, so the diagonal is 0 + 4 + 8.
    assert [c.evaluate({}) for c in traced.cells()] == [12.0]


def test_outer_product_concatenates_shapes():
    a = Tensor.zeros([2, 3])
    b = Tensor.zeros([4])
    assert a.outer(b).shape == [2, 3, 4]


def test_contraction_validates_its_axes():
    a = Tensor.zeros([2, 3])
    b = Tensor.zeros([3, 2])
    # Mismatched extents.
    with pytest.raises(ValueError):
        a.tensordot(b, [0], [0])
    # Out of range.
    with pytest.raises(ValueError):
        a.tensordot(b, [5], [0])
    # Unequal list lengths.
    with pytest.raises(ValueError):
        a.tensordot(b, [0, 1], [0])


# ---------------------------------------------------------------------------
# SI prefixes and unit conversion
# ---------------------------------------------------------------------------

def test_conversion_is_exact_for_powers_of_ten():
    # A single net power of ten is applied, so these are exact, not 999.9999.
    assert arithma.convert(1.0, "km", "m") == 1000.0
    assert arithma.convert(1000.0, "m", "km") == 1.0
    assert arithma.convert(1.0, "m", "mm") == 1000.0
    assert arithma.convert(2.5, "km", "mm") == 2_500_000.0


def test_conversion_refuses_incompatible_dimensions():
    # Refusing this is the entire point of tracking dimensions.
    with pytest.raises(ValueError):
        arithma.convert(1.0, "m", "s")


def test_conversion_refuses_an_unknown_unit():
    with pytest.raises(ValueError):
        arithma.convert(1.0, "zz", "m")


def test_kilogram_is_a_base_unit_not_kilo_grams():
    # The classic trap: kg is the SI base unit for mass, and g is not in the
    # base table at all, so kg must not decompose.
    assert arithma.split_unit_symbol("kg") == (0, "kg")
    assert arithma.split_unit_symbol("km") == (3, "m")
    assert arithma.split_unit_symbol("ms") == (-3, "s")


def test_prefix_micro_is_ascii():
    # The project's symbols are Latin letters; micro is "u", not the micro sign.
    assert arithma.prefix_power("u") == -6
    assert arithma.prefix_power("k") == 3
    assert arithma.prefix_power("zz") is None
    for symbol, name, power in arithma.si_prefixes():
        assert symbol.isascii(), f"prefix {symbol!r} must be ASCII"
        assert name.isascii(), f"prefix name {name!r} must be ASCII"


def test_prefixed_units_carry_the_base_dimension():
    assert Unit("km", "kilometre").dimension() == "m"
    assert Unit("km", "kilometre").scale_to_base() == 1000.0
    assert Unit("m", "metre").scale_to_base() == 1.0


# ---------------------------------------------------------------------------
# Like-term collection
# ---------------------------------------------------------------------------

def test_like_terms_collect():
    x = Expression.variable("x")
    got = x.add(x).simplify()
    # 2*x is symbolic, so it must not evaluate to a number.
    with pytest.raises(Exception):
        got.evaluate({})
    assert got.evaluate({"x": 5.0}) == 10.0


def test_existing_coefficients_add():
    x = Expression.variable("x")
    two_x = Expression.number(2).mul(x)
    three_x = Expression.number(3).mul(x)
    assert two_x.add(three_x).simplify().evaluate({"x": 2.0}) == 10.0


def test_opposite_terms_cancel():
    x = Expression.variable("x")
    got = Expression.number(3).mul(x).add(Expression.number(-3).mul(x)).simplify()
    assert got.kind() == "number"
    assert got.evaluate({}) == 0.0


def test_equal_factors_become_a_power():
    x = Expression.variable("x")
    got = x.mul(x).simplify()
    assert got.evaluate({"x": 3.0}) == 9.0


def test_collection_is_syntactic_and_stable():
    # x + y has nothing to collect, so it must report itself already simple --
    # otherwise a caller looping until no-change would never terminate.
    x, y = Expression.variable("x"), Expression.variable("y")
    assert not x.add(y).is_simplifiable()
    # And a collected result is stable on a second pass.
    assert not x.add(x).simplify().is_simplifiable()
