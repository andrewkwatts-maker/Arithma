//====== Arithma/rust/arithma_core/src/pyfacade/geometry.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! PyO3 facade for [`crate::geometry`].
//!
//! Exposes the symbolic 3D primitives — [`Vector`], [`Line`], [`Plane`],
//! [`Sphere`] — plus the closed-form intersection routines. Coordinates stay
//! `ArithmaExpression` throughout, so a Python caller can build a ray/plane
//! query out of free variables and bind numbers only at the end.
//!
//! ## Implementation status of the wrapped Rust surface
//!
//! Everything wrapped here is backed by a real implementation in
//! `src/geometry/*.rs` with real numeric tests; nothing in this file is a
//! facade over an `unimplemented!()` or an identity stub. Two methods are
//! *derived in this file* rather than delegated, and say so in their
//! docstrings:
//!
//! - [`Vector::magnitude`] — `ArithmaVector` offers only `magnitude_squared`,
//!   so the square root is applied here.
//! - [`line_closest_point`] — `ArithmaIntersection` offers only
//!   `closest_point_param`, so the `line.at(t)` substitution is applied here.
//!
//! ## Conventions
//!
//! Per `pyfacade/mod.rs`: at least two runtime assertions per function, fixed
//! loop bounds checked against [`MAX_SEQUENCE_LEN`], no recursion, every
//! non-void return checked, and every failure raised as a Python exception.
//!
//! Expression trees arriving from Python are walked with an explicit stack and
//! a node budget before being accepted, because
//! [`crate::expression::Evaluable`] descends them recursively: a pathological
//! tree built in a Python loop would otherwise reach Rust as a stack overflow,
//! which is not a catchable Python exception.

// `#[pymethods]` on pyo3 0.22 expands `-> PyResult<T>` into a `PyErr::from`
// round-trip that clippy reads as a no-op conversion in *our* signature span.
// `pyfacade::core` already carries 19 of these warnings; scoping the allow to
// this file keeps the new surface from adding more without editing the
// crate-wide lint policy in `lib.rs`.
#![allow(clippy::useless_conversion)]

use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::expression::{ArithmaBindings, ArithmaExpression, Evaluable};
use crate::geometry::intersection::{ArithmaIntersection, ArithmaIntersectionResult};
use crate::geometry::line::ArithmaLine;
use crate::geometry::plane::ArithmaPlane;
use crate::geometry::sphere::ArithmaSphere;
use crate::geometry::vector::ArithmaVector;
use crate::pyfacade::core::Expression;
use crate::pyfacade::MAX_SEQUENCE_LEN;

/// Node budget for a single expression tree crossing the Python boundary.
///
/// Shares [`MAX_SEQUENCE_LEN`] deliberately: a caller-supplied tree is
/// caller-supplied data exactly like a caller-supplied sequence, and one
/// number is easier to reason about than two.
const MAX_EXPRESSION_NODES: usize = MAX_SEQUENCE_LEN;

/// A vector has exactly three components. Named so the checks read as intent.
const VECTOR_ARITY: usize = 3;

// ============================================================================
// Helpers.
// ============================================================================

/// Count the nodes in an expression tree, giving up once `budget` is exceeded.
///
/// Iterative with an explicit stack (standard 4: no recursion) over a fixed
/// range (standard 3: fixed bounds). Returns `None` when the tree is larger
/// than `budget`, which callers turn into a `ValueError`.
fn expression_node_count_within(expr: &ArithmaExpression, budget: usize) -> Option<usize> {
    debug_assert!(budget > 0, "a zero node budget can never accept any tree");
    let mut stack: Vec<&ArithmaExpression> = Vec::with_capacity(16);
    stack.push(expr);
    debug_assert_eq!(stack.len(), 1, "the walk starts at the root alone");
    // `step` doubles as the number of nodes already popped, so the count needs
    // no separate mutable counter.
    for step in 0..budget {
        let Some(node) = stack.pop() else {
            debug_assert!(step < budget, "the loop index stays inside the budget");
            return Some(step);
        };
        match node {
            ArithmaExpression::Number(_)
            | ArithmaExpression::Variable(_)
            | ArithmaExpression::Constant { .. } => {}
            ArithmaExpression::Function(_, args) => stack.extend(args.iter()),
            ArithmaExpression::Sum {
                start,
                end,
                expression,
                ..
            }
            | ArithmaExpression::Product {
                start,
                end,
                expression,
                ..
            } => {
                stack.push(start);
                stack.push(end);
                stack.push(expression);
            }
            ArithmaExpression::Limit {
                approaching,
                expression,
                ..
            } => {
                stack.push(approaching);
                stack.push(expression);
            }
            ArithmaExpression::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                stack.push(condition);
                stack.push(then_expr);
                stack.push(else_expr);
            }
            ArithmaExpression::CachedValue { expr, .. }
            | ArithmaExpression::FourierOptimized { expr, .. } => stack.push(expr),
        }
    }
    None
}

/// True when `expr` fits inside the standing [`MAX_EXPRESSION_NODES`] budget.
fn expression_is_bounded(expr: &ArithmaExpression) -> bool {
    debug_assert!(MAX_EXPRESSION_NODES > 0, "budget constant must be positive");
    let counted = expression_node_count_within(expr, MAX_EXPRESSION_NODES);
    debug_assert!(
        counted.is_none_or(|n| n <= MAX_EXPRESSION_NODES),
        "a counted tree must be within budget"
    );
    counted.is_some()
}

/// True when all three components of `v` fit inside the node budget.
///
/// This is the invariant every `Vector`-reading wrapper asserts: Python hands
/// us pyclass instances by reference and the type system cannot state that
/// their components are finite-sized trees.
fn vector_is_bounded(v: &ArithmaVector) -> bool {
    debug_assert!(MAX_EXPRESSION_NODES > 0, "budget constant must be positive");
    debug_assert!(VECTOR_ARITY == 3, "vector arity is fixed at three");
    expression_is_bounded(&v.x) && expression_is_bounded(&v.y) && expression_is_bounded(&v.z)
}

/// Reject an over-large expression tree at the boundary with a `ValueError`.
fn check_expression(expr: &ArithmaExpression, who: &str) -> PyResult<()> {
    debug_assert!(!who.is_empty(), "error messages need a caller label");
    let counted = expression_node_count_within(expr, MAX_EXPRESSION_NODES);
    debug_assert!(
        counted.is_some() == expression_is_bounded(expr),
        "the count and the predicate must agree"
    );
    match counted {
        Some(_) => Ok(()),
        None => Err(PyValueError::new_err(format!(
            "{who}: expression tree exceeds the {MAX_EXPRESSION_NODES}-node limit"
        ))),
    }
}

/// Reject a vector whose components are over-large. Checked component by
/// component so the error names the offending axis.
fn check_vector(v: &ArithmaVector, who: &str) -> PyResult<()> {
    debug_assert!(!who.is_empty(), "error messages need a caller label");
    check_expression(&v.x, &format!("{who} (x component)"))?;
    check_expression(&v.y, &format!("{who} (y component)"))?;
    check_expression(&v.z, &format!("{who} (z component)"))?;
    debug_assert!(
        vector_is_bounded(v),
        "all three components passed the check"
    );
    Ok(())
}

/// Convert a Python object into an `ArithmaExpression`.
///
/// A near-duplicate of `core::coerce_to_expression`, which is private to that
/// module and therefore unreachable from here. Two deliberate differences:
/// `bool` is rejected (it is an `int` subclass in Python and would otherwise
/// become `0`/`1` silently), and non-finite floats are rejected because a NaN
/// or infinite coordinate poisons every downstream branch decision.
fn coerce_to_expression(obj: &Bound<'_, PyAny>, who: &str) -> PyResult<ArithmaExpression> {
    debug_assert!(!who.is_empty(), "error messages need a caller label");
    if obj.is_instance_of::<pyo3::types::PyBool>() {
        return Err(PyTypeError::new_err(format!(
            "{who}: bool is not a coordinate; pass 0 or 1 explicitly"
        )));
    }
    if let Ok(py_expr) = obj.extract::<PyRef<'_, Expression>>() {
        let inner = py_expr.inner.clone();
        check_expression(&inner, who)?;
        return Ok(inner);
    }
    if let Ok(int_val) = obj.extract::<i64>() {
        return Ok(ArithmaExpression::from_i64(int_val));
    }
    if let Ok(float_val) = obj.extract::<f64>() {
        if !float_val.is_finite() {
            return Err(PyValueError::new_err(format!(
                "{who}: coordinate must be finite, got {float_val}"
            )));
        }
        debug_assert!(float_val.is_finite(), "non-finite rejected above");
        return Ok(ArithmaExpression::from_f64(float_val));
    }
    Err(PyTypeError::new_err(format!(
        "{who}: expected Expression, int, or float"
    )))
}

/// Wrap a bare `ArithmaExpression` in the `Expression` pyclass.
///
/// `core::Expression::from_inner` is private to `core`, but the `inner` field
/// is `pub(crate)`, so the struct literal is the supported in-crate path.
fn wrap_expression(inner: ArithmaExpression) -> Expression {
    debug_assert!(
        expression_is_bounded(&inner),
        "only bounded trees reach the wrapper"
    );
    let wrapped = Expression { inner };
    debug_assert!(
        expression_is_bounded(&wrapped.inner),
        "wrapping must not alter the tree"
    );
    wrapped
}

/// Build an `ArithmaBindings` map from a Python `{name: number}` dict.
///
/// The iteration carries a hard trip count taken from the dict's declared
/// length, so a dict that mutates underneath us raises rather than spinning.
fn bindings_from_dict(env: &Bound<'_, PyDict>, who: &str) -> PyResult<ArithmaBindings> {
    debug_assert!(!who.is_empty(), "error messages need a caller label");
    let declared = env.len();
    if declared > MAX_SEQUENCE_LEN {
        return Err(PyValueError::new_err(format!(
            "{who}: binding map of {declared} entries exceeds the {MAX_SEQUENCE_LEN}-entry limit"
        )));
    }
    let mut bindings: ArithmaBindings = ArithmaBindings::new();
    let mut visited: usize = 0;
    for (key, val) in env.iter() {
        visited += 1;
        if visited > declared {
            return Err(PyRuntimeError::new_err(format!(
                "{who}: binding map changed size during iteration"
            )));
        }
        let name: String = key
            .extract()
            .map_err(|_| PyTypeError::new_err(format!("{who}: binding keys must be str")))?;
        let value: f64 = val.extract().map_err(|_| {
            PyTypeError::new_err(format!("{who}: binding {name:?} must be a real number"))
        })?;
        let _previous = bindings.insert(name, value);
    }
    debug_assert!(
        bindings.len() <= declared,
        "cannot collect more bindings than the map declared"
    );
    Ok(bindings)
}

/// Evaluate all three components of a vector against `bindings`.
///
/// A component that reduces to NaN is reported as an error rather than handed
/// back as a coordinate: a NaN position is a silently wrong answer, and
/// standard 7 forbids a wrapper returning one as if it were healthy.
fn evaluate_vector(v: &ArithmaVector, bindings: &ArithmaBindings) -> Result<[f64; 3], String> {
    debug_assert!(vector_is_bounded(v), "components must be bounded trees");
    let x = v.x.evaluate(bindings)?;
    let y = v.y.evaluate(bindings)?;
    let z = v.z.evaluate(bindings)?;
    if x.is_nan() || y.is_nan() || z.is_nan() {
        return Err("component evaluated to NaN".to_string());
    }
    debug_assert!(
        !x.is_nan() && !y.is_nan() && !z.is_nan(),
        "NaN components rejected above"
    );
    Ok([x, y, z])
}

/// Render a single component for a `__repr__`.
///
/// `core`'s LaTeX walker (`expression_to_latex`) and `Expression::to_latex`
/// are both private to `core`, so this file renders its own short form: the
/// numeric value when the component reduces against an empty binding map, and
/// `<symbolic>` otherwise. A repr must never fail, so there is no error path.
fn render_component(expr: &ArithmaExpression) -> String {
    debug_assert!(
        expression_is_bounded(expr),
        "only bounded trees reach the renderer"
    );
    let text = match expr.evaluate(&ArithmaBindings::new()) {
        Ok(value) if value.is_finite() => format!("{value}"),
        _ => "<symbolic>".to_string(),
    };
    debug_assert!(!text.is_empty(), "every component renders to something");
    text
}

/// Render a vector for a `__repr__`, delegating to [`render_component`].
fn render_vector(v: &ArithmaVector) -> String {
    debug_assert!(vector_is_bounded(v), "operand must be bounded");
    let text = format!(
        "Vector({}, {}, {})",
        render_component(&v.x),
        render_component(&v.y),
        render_component(&v.z)
    );
    debug_assert!(!text.is_empty(), "every vector renders to something");
    text
}

/// Discriminator string for an intersection outcome.
fn intersection_kind(result: &ArithmaIntersectionResult) -> &'static str {
    let kind = match result {
        ArithmaIntersectionResult::None => "none",
        ArithmaIntersectionResult::Point(_) => "point",
        ArithmaIntersectionResult::TwoPoints(_, _) => "two_points",
        ArithmaIntersectionResult::Continuous => "continuous",
    };
    debug_assert!(!kind.is_empty(), "every variant has a discriminator");
    debug_assert!(
        expected_point_count(kind) <= 2,
        "no variant carries more than two points"
    );
    kind
}

/// How many points the given [`intersection_kind`] is required to carry.
fn expected_point_count(kind: &str) -> usize {
    debug_assert!(!kind.is_empty(), "kind must be a discriminator string");
    let count = match kind {
        "point" => 1,
        "two_points" => 2,
        _ => 0,
    };
    debug_assert!(count <= 2, "no variant carries more than two points");
    count
}

/// Extract the hit points from an intersection outcome, in near-to-far order.
fn intersection_points(result: &ArithmaIntersectionResult) -> Vec<ArithmaVector> {
    let points = match result {
        ArithmaIntersectionResult::None | ArithmaIntersectionResult::Continuous => Vec::new(),
        ArithmaIntersectionResult::Point(p) => vec![p.clone()],
        ArithmaIntersectionResult::TwoPoints(near, far) => vec![near.clone(), far.clone()],
    };
    debug_assert!(points.len() <= 2, "at most two points per outcome");
    debug_assert_eq!(
        points.len(),
        expected_point_count(intersection_kind(result)),
        "point count must match the discriminator"
    );
    points
}

// ============================================================================
// `Vector` pyclass.
// ============================================================================

/// Symbolic 3-vector — Python wrapper around `ArithmaVector`.
///
/// Components are :class:`Expression` values, so a vector may be fully
/// numeric, fully symbolic, or a mix. Accepted component types are
/// :class:`Expression`, :class:`int`, and finite :class:`float`;
/// :class:`bool` and non-finite floats raise.
#[pyclass(name = "Vector", module = "arithma")]
#[derive(Clone)]
pub struct Vector {
    pub(crate) inner: ArithmaVector,
}

#[pymethods]
impl Vector {
    /// Construct from three components.
    #[new]
    fn new(x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>, z: &Bound<'_, PyAny>) -> PyResult<Self> {
        let cx = coerce_to_expression(x, "Vector(x)")?;
        let cy = coerce_to_expression(y, "Vector(y)")?;
        let cz = coerce_to_expression(z, "Vector(z)")?;
        let inner = ArithmaVector::new(cx, cy, cz);
        check_vector(&inner, "Vector()")?;
        debug_assert!(vector_is_bounded(&inner), "checked immediately above");
        debug_assert!(VECTOR_ARITY == 3, "vector arity is fixed at three");
        Ok(Self { inner })
    }

    /// The zero vector `(0, 0, 0)`.
    #[staticmethod]
    fn zero() -> Self {
        let inner = ArithmaVector::zero();
        debug_assert!(
            vector_is_bounded(&inner),
            "the zero vector is trivially bounded"
        );
        debug_assert!(
            matches!(inner.x, ArithmaExpression::Number(_)),
            "zero components must be numeric literals"
        );
        Self { inner }
    }

    /// Construct from any 3-element Python sequence.
    ///
    /// The sequence length is checked against ``MAX_SEQUENCE_LEN`` before the
    /// contents are touched, then against the required arity of three.
    #[staticmethod]
    fn from_sequence(components: &Bound<'_, PyAny>) -> PyResult<Self> {
        let declared = components.len().map_err(|_| {
            PyTypeError::new_err("Vector.from_sequence expects a sized sequence of 3 components")
        })?;
        if declared > MAX_SEQUENCE_LEN {
            return Err(PyValueError::new_err(format!(
                "Vector.from_sequence: {declared} elements exceeds the \
                 {MAX_SEQUENCE_LEN}-element limit"
            )));
        }
        if declared != VECTOR_ARITY {
            return Err(PyValueError::new_err(format!(
                "Vector.from_sequence expects exactly {VECTOR_ARITY} components, got {declared}"
            )));
        }
        debug_assert_eq!(declared, VECTOR_ARITY, "arity checked immediately above");
        let mut parts: Vec<ArithmaExpression> = Vec::with_capacity(VECTOR_ARITY);
        for index in 0..VECTOR_ARITY {
            let item = components.get_item(index)?;
            parts.push(coerce_to_expression(
                &item,
                &format!("Vector.from_sequence[{index}]"),
            )?);
        }
        debug_assert_eq!(parts.len(), VECTOR_ARITY, "loop filled every component");
        let inner = ArithmaVector::new(parts[0].clone(), parts[1].clone(), parts[2].clone());
        check_vector(&inner, "Vector.from_sequence")?;
        Ok(Self { inner })
    }

    /// The x component.
    #[getter]
    fn x(&self) -> Expression {
        debug_assert!(vector_is_bounded(&self.inner), "operand must be bounded");
        debug_assert!(expression_is_bounded(&self.inner.x), "x must be bounded");
        wrap_expression(self.inner.x.clone())
    }

    /// The y component.
    #[getter]
    fn y(&self) -> Expression {
        debug_assert!(vector_is_bounded(&self.inner), "operand must be bounded");
        debug_assert!(expression_is_bounded(&self.inner.y), "y must be bounded");
        wrap_expression(self.inner.y.clone())
    }

    /// The z component.
    #[getter]
    fn z(&self) -> Expression {
        debug_assert!(vector_is_bounded(&self.inner), "operand must be bounded");
        debug_assert!(expression_is_bounded(&self.inner.z), "z must be bounded");
        wrap_expression(self.inner.z.clone())
    }

    /// The three components as a list of :class:`Expression`.
    fn components(&self) -> Vec<Expression> {
        debug_assert!(vector_is_bounded(&self.inner), "operand must be bounded");
        let out = vec![
            wrap_expression(self.inner.x.clone()),
            wrap_expression(self.inner.y.clone()),
            wrap_expression(self.inner.z.clone()),
        ];
        debug_assert_eq!(out.len(), VECTOR_ARITY, "a vector has three components");
        out
    }

    /// Dot product ``self . other``, built symbolically.
    fn dot(&self, other: PyRef<'_, Vector>) -> Expression {
        debug_assert!(
            vector_is_bounded(&self.inner),
            "left operand must be bounded"
        );
        debug_assert!(
            vector_is_bounded(&other.inner),
            "right operand must be bounded"
        );
        wrap_expression(self.inner.dot(&other.inner))
    }

    /// Cross product ``self x other`` (right-handed: ``x_hat x y_hat = z_hat``).
    fn cross(&self, other: PyRef<'_, Vector>) -> Vector {
        debug_assert!(
            vector_is_bounded(&self.inner),
            "left operand must be bounded"
        );
        debug_assert!(
            vector_is_bounded(&other.inner),
            "right operand must be bounded"
        );
        Vector {
            inner: self.inner.cross(&other.inner),
        }
    }

    /// Component-wise sum ``self + other``.
    fn add(&self, other: PyRef<'_, Vector>) -> Vector {
        debug_assert!(
            vector_is_bounded(&self.inner),
            "left operand must be bounded"
        );
        debug_assert!(
            vector_is_bounded(&other.inner),
            "right operand must be bounded"
        );
        Vector {
            inner: self.inner.add_vec(&other.inner),
        }
    }

    /// Component-wise difference ``self - other``.
    fn sub(&self, other: PyRef<'_, Vector>) -> Vector {
        debug_assert!(
            vector_is_bounded(&self.inner),
            "left operand must be bounded"
        );
        debug_assert!(
            vector_is_bounded(&other.inner),
            "right operand must be bounded"
        );
        Vector {
            inner: self.inner.sub_vec(&other.inner),
        }
    }

    /// Scale every component by a scalar (``Expression``, ``int``, ``float``).
    fn scale(&self, scalar: &Bound<'_, PyAny>) -> PyResult<Vector> {
        debug_assert!(vector_is_bounded(&self.inner), "operand must be bounded");
        let factor = coerce_to_expression(scalar, "Vector.scale")?;
        debug_assert!(
            expression_is_bounded(&factor),
            "coercion enforces the budget"
        );
        Ok(Vector {
            inner: self.inner.scale(&factor),
        })
    }

    /// Squared magnitude ``self . self``. Exact — no square root is taken.
    fn magnitude_squared(&self) -> Expression {
        debug_assert!(vector_is_bounded(&self.inner), "operand must be bounded");
        let squared = self.inner.magnitude_squared();
        debug_assert!(expression_is_bounded(&squared), "result must stay bounded");
        wrap_expression(squared)
    }

    /// Magnitude ``sqrt(self . self)``.
    ///
    /// **Derived in the facade.** ``ArithmaVector`` exposes only
    /// ``magnitude_squared``; the ``sqrt`` node is added here. The result is a
    /// symbolic ``sqrt`` and is *not* simplified, so ``Vector(3, 4, 0)`` yields
    /// ``sqrt(3*3 + 4*4 + 0*0)``, which evaluates to ``5.0`` but does not print
    /// as ``5``.
    fn magnitude(&self) -> Expression {
        debug_assert!(vector_is_bounded(&self.inner), "operand must be bounded");
        let root = ArithmaExpression::sqrt(self.inner.magnitude_squared());
        debug_assert!(expression_is_bounded(&root), "result must stay bounded");
        wrap_expression(root)
    }

    /// Evaluate all three components against a ``{name: number}`` dict.
    ///
    /// Returns an ``(x, y, z)`` tuple of floats. Raises :class:`ValueError`
    /// for an unbound variable, a division by zero, or a component that
    /// reduces to NaN — never a placeholder coordinate.
    fn evaluate(&self, env: &Bound<'_, PyDict>) -> PyResult<(f64, f64, f64)> {
        debug_assert!(vector_is_bounded(&self.inner), "operand must be bounded");
        let bindings = bindings_from_dict(env, "Vector.evaluate")?;
        debug_assert!(
            bindings.len() <= MAX_SEQUENCE_LEN,
            "binding count is capped"
        );
        match evaluate_vector(&self.inner, &bindings) {
            Ok(parts) => Ok((parts[0], parts[1], parts[2])),
            Err(msg) => Err(PyValueError::new_err(format!("Vector.evaluate: {msg}"))),
        }
    }

    fn __add__(&self, other: PyRef<'_, Vector>) -> Vector {
        debug_assert!(
            vector_is_bounded(&self.inner),
            "left operand must be bounded"
        );
        debug_assert!(
            vector_is_bounded(&other.inner),
            "right operand must be bounded"
        );
        self.add(other)
    }

    fn __sub__(&self, other: PyRef<'_, Vector>) -> Vector {
        debug_assert!(
            vector_is_bounded(&self.inner),
            "left operand must be bounded"
        );
        debug_assert!(
            vector_is_bounded(&other.inner),
            "right operand must be bounded"
        );
        self.sub(other)
    }

    fn __mul__(&self, scalar: &Bound<'_, PyAny>) -> PyResult<Vector> {
        debug_assert!(vector_is_bounded(&self.inner), "operand must be bounded");
        debug_assert!(!scalar.is_none(), "None is not a scale factor");
        self.scale(scalar)
    }

    fn __rmul__(&self, scalar: &Bound<'_, PyAny>) -> PyResult<Vector> {
        debug_assert!(vector_is_bounded(&self.inner), "operand must be bounded");
        debug_assert!(!scalar.is_none(), "None is not a scale factor");
        self.scale(scalar)
    }

    fn __neg__(&self) -> Vector {
        debug_assert!(vector_is_bounded(&self.inner), "operand must be bounded");
        let minus_one = ArithmaExpression::from_i64(-1);
        debug_assert!(expression_is_bounded(&minus_one), "literal is bounded");
        Vector {
            inner: self.inner.scale(&minus_one),
        }
    }

    fn __len__(&self) -> usize {
        debug_assert!(vector_is_bounded(&self.inner), "operand must be bounded");
        debug_assert!(VECTOR_ARITY == 3, "vector arity is fixed at three");
        VECTOR_ARITY
    }

    fn __getitem__(&self, index: isize) -> PyResult<Expression> {
        debug_assert!(vector_is_bounded(&self.inner), "operand must be bounded");
        let arity = VECTOR_ARITY as isize;
        let resolved = if index < 0 { index + arity } else { index };
        if !(0..arity).contains(&resolved) {
            return Err(PyIndexError::new_err(format!(
                "Vector index {index} out of range (expected -{arity}..{arity})"
            )));
        }
        debug_assert!((0..arity).contains(&resolved), "index normalised above");
        match resolved {
            0 => Ok(self.x()),
            1 => Ok(self.y()),
            _ => Ok(self.z()),
        }
    }

    fn __repr__(&self) -> String {
        debug_assert!(vector_is_bounded(&self.inner), "operand must be bounded");
        debug_assert!(VECTOR_ARITY == 3, "vector arity is fixed at three");
        render_vector(&self.inner)
    }
}

// ============================================================================
// `Line` pyclass.
// ============================================================================

/// Parametric line ``P(t) = origin + t * direction`` — wraps `ArithmaLine`.
///
/// The direction need not be unit length. A zero direction is accepted at
/// construction (it may be symbolic and degenerate only for some bindings) but
/// leaves a division by zero in anything that normalises by
/// ``|direction| ** 2``.
#[pyclass(name = "Line", module = "arithma")]
#[derive(Clone)]
pub struct Line {
    pub(crate) inner: ArithmaLine,
}

#[pymethods]
impl Line {
    /// Construct from an origin point and a direction vector.
    #[new]
    fn new(origin: PyRef<'_, Vector>, direction: PyRef<'_, Vector>) -> PyResult<Self> {
        check_vector(&origin.inner, "Line(origin)")?;
        check_vector(&direction.inner, "Line(direction)")?;
        debug_assert!(vector_is_bounded(&origin.inner), "checked above");
        debug_assert!(vector_is_bounded(&direction.inner), "checked above");
        Ok(Self {
            inner: ArithmaLine::new(origin.inner.clone(), direction.inner.clone()),
        })
    }

    /// The origin point.
    #[getter]
    fn origin(&self) -> Vector {
        debug_assert!(
            vector_is_bounded(&self.inner.origin),
            "origin must be bounded"
        );
        debug_assert!(
            vector_is_bounded(&self.inner.direction),
            "direction must be bounded"
        );
        Vector {
            inner: self.inner.origin.clone(),
        }
    }

    /// The direction vector.
    #[getter]
    fn direction(&self) -> Vector {
        debug_assert!(
            vector_is_bounded(&self.inner.origin),
            "origin must be bounded"
        );
        debug_assert!(
            vector_is_bounded(&self.inner.direction),
            "direction must be bounded"
        );
        Vector {
            inner: self.inner.direction.clone(),
        }
    }

    /// Point at parameter ``t``: ``origin + t * direction``.
    ///
    /// ``t`` may be an :class:`Expression`, in which case the parameter stays
    /// symbolic in every coordinate of the result.
    fn at(&self, t: &Bound<'_, PyAny>) -> PyResult<Vector> {
        debug_assert!(
            vector_is_bounded(&self.inner.origin),
            "origin must be bounded"
        );
        debug_assert!(
            vector_is_bounded(&self.inner.direction),
            "direction must be bounded"
        );
        let param = coerce_to_expression(t, "Line.at(t)")?;
        Ok(Vector {
            inner: self.inner.at(param),
        })
    }

    fn __repr__(&self) -> String {
        debug_assert!(
            vector_is_bounded(&self.inner.origin),
            "origin must be bounded"
        );
        debug_assert!(
            vector_is_bounded(&self.inner.direction),
            "direction must be bounded"
        );
        format!(
            "Line(origin={}, direction={})",
            render_vector(&self.inner.origin),
            render_vector(&self.inner.direction)
        )
    }
}

// ============================================================================
// `Plane` pyclass.
// ============================================================================

/// Infinite plane in normal-and-offset form ``normal . p = offset``.
#[pyclass(name = "Plane", module = "arithma")]
#[derive(Clone)]
pub struct Plane {
    pub(crate) inner: ArithmaPlane,
}

#[pymethods]
impl Plane {
    /// Construct from a normal (need not be unit length) and a scalar offset.
    #[new]
    fn new(normal: PyRef<'_, Vector>, offset: &Bound<'_, PyAny>) -> PyResult<Self> {
        check_vector(&normal.inner, "Plane(normal)")?;
        let scalar = coerce_to_expression(offset, "Plane(offset)")?;
        debug_assert!(vector_is_bounded(&normal.inner), "checked above");
        debug_assert!(
            expression_is_bounded(&scalar),
            "coercion enforces the budget"
        );
        Ok(Self {
            inner: ArithmaPlane::new(normal.inner.clone(), scalar),
        })
    }

    /// The plane normal.
    #[getter]
    fn normal(&self) -> Vector {
        debug_assert!(
            vector_is_bounded(&self.inner.normal),
            "normal must be bounded"
        );
        debug_assert!(
            expression_is_bounded(&self.inner.offset),
            "offset must be bounded"
        );
        Vector {
            inner: self.inner.normal.clone(),
        }
    }

    /// The scalar offset along the normal.
    #[getter]
    fn offset(&self) -> Expression {
        debug_assert!(
            vector_is_bounded(&self.inner.normal),
            "normal must be bounded"
        );
        debug_assert!(
            expression_is_bounded(&self.inner.offset),
            "offset must be bounded"
        );
        wrap_expression(self.inner.offset.clone())
    }

    /// Signed distance from ``point`` to this plane.
    ///
    /// ``(normal . point - offset) / sqrt(normal . normal)``. Positive on the
    /// side the normal points toward. The division by ``|normal|`` is always
    /// emitted because the normal is not required to be unit length; with a
    /// zero normal the expression is still built, and the division by zero
    /// surfaces when the caller evaluates it.
    fn signed_distance(&self, point: PyRef<'_, Vector>) -> PyResult<Expression> {
        debug_assert!(
            vector_is_bounded(&self.inner.normal),
            "normal must be bounded"
        );
        check_vector(&point.inner, "Plane.signed_distance(point)")?;
        debug_assert!(vector_is_bounded(&point.inner), "checked above");
        Ok(wrap_expression(self.inner.signed_distance(&point.inner)))
    }

    fn __repr__(&self) -> String {
        debug_assert!(
            vector_is_bounded(&self.inner.normal),
            "normal must be bounded"
        );
        debug_assert!(
            expression_is_bounded(&self.inner.offset),
            "offset must be bounded"
        );
        format!(
            "Plane(normal={}, offset={})",
            render_vector(&self.inner.normal),
            render_component(&self.inner.offset)
        )
    }
}

// ============================================================================
// `Sphere` pyclass.
// ============================================================================

/// Sphere ``|p - centre| = radius``.
#[pyclass(name = "Sphere", module = "arithma")]
#[derive(Clone)]
pub struct Sphere {
    pub(crate) inner: ArithmaSphere,
}

#[pymethods]
impl Sphere {
    /// Construct from a centre point and a radius.
    ///
    /// A radius that reduces to a negative number is rejected: the underlying
    /// routines only ever use ``radius * radius``, so a negative radius would
    /// be silently indistinguishable from its absolute value. A radius that
    /// stays symbolic is accepted, because its sign is not yet decidable.
    #[new]
    fn new(centre: PyRef<'_, Vector>, radius: &Bound<'_, PyAny>) -> PyResult<Self> {
        check_vector(&centre.inner, "Sphere(centre)")?;
        let r = coerce_to_expression(radius, "Sphere(radius)")?;
        if let Ok(value) = r.evaluate(&ArithmaBindings::new()) {
            if value < 0.0 {
                return Err(PyValueError::new_err(format!(
                    "Sphere(radius): radius must be non-negative, got {value}"
                )));
            }
        }
        debug_assert!(vector_is_bounded(&centre.inner), "checked above");
        debug_assert!(expression_is_bounded(&r), "coercion enforces the budget");
        Ok(Self {
            inner: ArithmaSphere::new(centre.inner.clone(), r),
        })
    }

    /// The centre point.
    #[getter]
    fn centre(&self) -> Vector {
        debug_assert!(
            vector_is_bounded(&self.inner.centre),
            "centre must be bounded"
        );
        debug_assert!(
            expression_is_bounded(&self.inner.radius),
            "radius must be bounded"
        );
        Vector {
            inner: self.inner.centre.clone(),
        }
    }

    /// The centre point (US-spelling alias for :attr:`centre`).
    #[getter]
    fn center(&self) -> Vector {
        debug_assert!(
            vector_is_bounded(&self.inner.centre),
            "centre must be bounded"
        );
        debug_assert!(
            expression_is_bounded(&self.inner.radius),
            "radius must be bounded"
        );
        self.centre()
    }

    /// The radius.
    #[getter]
    fn radius(&self) -> Expression {
        debug_assert!(
            vector_is_bounded(&self.inner.centre),
            "centre must be bounded"
        );
        debug_assert!(
            expression_is_bounded(&self.inner.radius),
            "radius must be bounded"
        );
        wrap_expression(self.inner.radius.clone())
    }

    /// ``radius * radius``, built symbolically.
    fn radius_squared(&self) -> Expression {
        debug_assert!(
            vector_is_bounded(&self.inner.centre),
            "centre must be bounded"
        );
        let squared = self.inner.radius_squared();
        debug_assert!(expression_is_bounded(&squared), "result must stay bounded");
        wrap_expression(squared)
    }

    fn __repr__(&self) -> String {
        debug_assert!(
            vector_is_bounded(&self.inner.centre),
            "centre must be bounded"
        );
        debug_assert!(
            expression_is_bounded(&self.inner.radius),
            "radius must be bounded"
        );
        format!(
            "Sphere(centre={}, radius={})",
            render_vector(&self.inner.centre),
            render_component(&self.inner.radius)
        )
    }
}

// ============================================================================
// `IntersectionResult` pyclass.
// ============================================================================

/// Outcome of an intersection query.
///
/// ``kind`` is one of ``"none"``, ``"point"``, ``"two_points"`` or
/// ``"continuous"``. ``points`` holds 0, 1 or 2 :class:`Vector` hits, near to
/// far. ``"continuous"`` means the primitives overlap along a whole line and
/// carries no points. The object is falsey exactly when ``kind == "none"``.
#[pyclass(name = "IntersectionResult", module = "arithma")]
#[derive(Clone)]
pub struct IntersectionResult {
    kind: String,
    points: Vec<Vector>,
}

impl IntersectionResult {
    /// Convert a Rust intersection outcome into the Python-facing form.
    fn from_result(result: &ArithmaIntersectionResult) -> Self {
        let kind = intersection_kind(result);
        let raw = intersection_points(result);
        debug_assert_eq!(
            raw.len(),
            expected_point_count(kind),
            "point count must match the discriminator"
        );
        let points: Vec<Vector> = raw.into_iter().map(|inner| Vector { inner }).collect();
        debug_assert!(points.len() <= 2, "at most two points per outcome");
        Self {
            kind: kind.to_string(),
            points,
        }
    }
}

#[pymethods]
impl IntersectionResult {
    /// Discriminator: ``"none"``, ``"point"``, ``"two_points"``, ``"continuous"``.
    #[getter]
    fn kind(&self) -> String {
        debug_assert!(!self.kind.is_empty(), "kind is always populated");
        debug_assert_eq!(
            self.points.len(),
            expected_point_count(&self.kind),
            "point count must match the discriminator"
        );
        self.kind.clone()
    }

    /// The hit points, near to far.
    #[getter]
    fn points(&self) -> Vec<Vector> {
        debug_assert!(!self.kind.is_empty(), "kind is always populated");
        debug_assert!(self.points.len() <= 2, "at most two points per outcome");
        self.points.clone()
    }

    /// The first hit point, or ``None`` when there is none.
    fn point(&self) -> Option<Vector> {
        debug_assert!(!self.kind.is_empty(), "kind is always populated");
        debug_assert!(self.points.len() <= 2, "at most two points per outcome");
        self.points.first().cloned()
    }

    fn __bool__(&self) -> bool {
        debug_assert!(!self.kind.is_empty(), "kind is always populated");
        debug_assert!(self.points.len() <= 2, "at most two points per outcome");
        self.kind != "none"
    }

    fn __len__(&self) -> usize {
        debug_assert!(!self.kind.is_empty(), "kind is always populated");
        debug_assert_eq!(
            self.points.len(),
            expected_point_count(&self.kind),
            "point count must match the discriminator"
        );
        self.points.len()
    }

    fn __repr__(&self) -> String {
        debug_assert!(!self.kind.is_empty(), "kind is always populated");
        debug_assert!(self.points.len() <= 2, "at most two points per outcome");
        let rendered: Vec<String> = self.points.iter().map(|p| p.__repr__()).collect();
        format!(
            "IntersectionResult(kind={:?}, points=[{}])",
            self.kind,
            rendered.join(", ")
        )
    }
}

// ============================================================================
// Intersection module functions.
// ============================================================================

/// Line vs plane intersection.
///
/// Returns ``kind="point"`` for the general case, ``"continuous"`` when the
/// line lies in the plane, and ``"none"`` when it is parallel to but off the
/// plane. A query whose deciding scalar stays symbolic takes the general
/// branch and yields a symbolic point rather than being reported as a miss.
#[pyfunction]
fn intersect_line_plane(line: PyRef<'_, Line>, plane: PyRef<'_, Plane>) -> IntersectionResult {
    debug_assert!(
        vector_is_bounded(&line.inner.direction),
        "line direction must be bounded"
    );
    debug_assert!(
        vector_is_bounded(&plane.inner.normal),
        "plane normal must be bounded"
    );
    let result = ArithmaIntersection::line_plane(&line.inner, &plane.inner);
    IntersectionResult::from_result(&result)
}

/// Line vs sphere intersection.
///
/// Returns ``kind="two_points"`` (near, far) for a secant hit, ``"point"`` for
/// a tangent hit, and ``"none"`` for a miss or a zero-length direction. A
/// query whose discriminant stays symbolic is reported as ``"two_points"`` so
/// the symbolic answer survives; inspect the coordinates to settle the real
/// case.
#[pyfunction]
fn intersect_line_sphere(line: PyRef<'_, Line>, sphere: PyRef<'_, Sphere>) -> IntersectionResult {
    debug_assert!(
        vector_is_bounded(&line.inner.direction),
        "line direction must be bounded"
    );
    debug_assert!(
        vector_is_bounded(&sphere.inner.centre),
        "sphere centre must be bounded"
    );
    let result = ArithmaIntersection::line_sphere(&line.inner, &sphere.inner);
    IntersectionResult::from_result(&result)
}

/// Plane vs plane intersection, as a :class:`Line`.
///
/// Returns ``None`` when the normals are parallel — the planes are then either
/// coincident or disjoint and no unique line exists. A determinant that stays
/// symbolic is treated as non-degenerate and a symbolic line is returned.
#[pyfunction]
fn intersect_plane_plane(a: PyRef<'_, Plane>, b: PyRef<'_, Plane>) -> Option<Line> {
    debug_assert!(
        vector_is_bounded(&a.inner.normal),
        "first normal must be bounded"
    );
    debug_assert!(
        vector_is_bounded(&b.inner.normal),
        "second normal must be bounded"
    );
    ArithmaIntersection::plane_plane(&a.inner, &b.inner).map(|inner| Line { inner })
}

/// Parameter ``t`` of the point on ``line`` closest to ``point``.
///
/// ``((point - origin) . direction) / (direction . direction)``. A zero-length
/// direction leaves a division by zero in the returned expression, which fails
/// when the caller evaluates it rather than being rejected here — the
/// direction may be symbolic and degenerate only for some bindings.
#[pyfunction]
fn line_closest_point_param(line: PyRef<'_, Line>, point: PyRef<'_, Vector>) -> Expression {
    debug_assert!(
        vector_is_bounded(&line.inner.direction),
        "line direction must be bounded"
    );
    debug_assert!(
        vector_is_bounded(&point.inner),
        "query point must be bounded"
    );
    wrap_expression(ArithmaIntersection::closest_point_param(
        &line.inner,
        &point.inner,
    ))
}

/// Foot of the perpendicular from ``point`` onto ``line``.
///
/// **Derived in the facade** as ``line.at(closest_point_param(line, point))``;
/// `ArithmaIntersection` exposes only the parameter. Carries the same
/// zero-direction caveat as :func:`line_closest_point_param`.
#[pyfunction]
fn line_closest_point(line: PyRef<'_, Line>, point: PyRef<'_, Vector>) -> Vector {
    debug_assert!(
        vector_is_bounded(&line.inner.direction),
        "line direction must be bounded"
    );
    debug_assert!(
        vector_is_bounded(&point.inner),
        "query point must be bounded"
    );
    let t = ArithmaIntersection::closest_point_param(&line.inner, &point.inner);
    Vector {
        inner: line.inner.at(t),
    }
}

// ============================================================================
// Registration
// ============================================================================

/// Register this module's surface on the extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Vector>()?;
    m.add_class::<Line>()?;
    m.add_class::<Plane>()?;
    m.add_class::<Sphere>()?;
    m.add_class::<IntersectionResult>()?;
    m.add_function(wrap_pyfunction!(intersect_line_plane, m)?)?;
    m.add_function(wrap_pyfunction!(intersect_line_sphere, m)?)?;
    m.add_function(wrap_pyfunction!(intersect_plane_plane, m)?)?;
    m.add_function(wrap_pyfunction!(line_closest_point_param, m)?)?;
    m.add_function(wrap_pyfunction!(line_closest_point, m)?)?;
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================
//
// Everything below exercises the plain-Rust helpers, which are the whole of
// the logic this file adds on top of `crate::geometry`. The `#[pymethods]`
// bodies need a live interpreter and are covered by the Python-side suite.

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: i64, y: i64, z: i64) -> ArithmaVector {
        ArithmaVector::new(
            ArithmaExpression::from_i64(x),
            ArithmaExpression::from_i64(y),
            ArithmaExpression::from_i64(z),
        )
    }

    /// Build a left-leaning `((0+1)+1)+1...` chain of the requested depth
    /// without recursing.
    fn chain(depth: usize) -> ArithmaExpression {
        let mut expr = ArithmaExpression::zero();
        for _ in 0..depth {
            expr = ArithmaExpression::add(expr, ArithmaExpression::from_i64(1));
        }
        expr
    }

    #[test]
    fn node_count_of_a_leaf_is_one() {
        assert_eq!(
            expression_node_count_within(&ArithmaExpression::zero(), 16),
            Some(1)
        );
        assert_eq!(
            expression_node_count_within(&ArithmaExpression::var("t"), 16),
            Some(1)
        );
    }

    #[test]
    fn node_count_counts_every_node_in_a_chain() {
        // Depth d adds one Add node and one literal per level, atop one zero.
        for depth in [1_usize, 2, 5] {
            let expected = 2 * depth + 1;
            assert_eq!(
                expression_node_count_within(&chain(depth), 1024),
                Some(expected),
                "depth {depth}"
            );
        }
    }

    #[test]
    fn node_count_gives_up_past_the_budget() {
        let deep = chain(50);
        assert!(expression_node_count_within(&deep, 8).is_none());
        assert!(expression_node_count_within(&deep, 1024).is_some());
    }

    #[test]
    fn ordinary_expressions_and_vectors_are_bounded() {
        assert!(expression_is_bounded(&chain(64)));
        assert!(vector_is_bounded(&v(1, 2, 3)));
        assert!(vector_is_bounded(&ArithmaVector::zero()));
    }

    #[test]
    fn check_helpers_accept_well_formed_input() {
        assert!(check_expression(&chain(10), "test").is_ok());
        assert!(check_vector(&v(4, 5, 6), "test").is_ok());
    }

    #[test]
    fn evaluate_vector_returns_the_components() {
        let got = evaluate_vector(&v(3, -4, 7), &ArithmaBindings::new()).expect("numeric");
        assert!((got[0] - 3.0).abs() < 1e-12);
        assert!((got[1] + 4.0).abs() < 1e-12);
        assert!((got[2] - 7.0).abs() < 1e-12);
    }

    #[test]
    fn evaluate_vector_binds_free_variables() {
        let symbolic = ArithmaVector::new(
            ArithmaExpression::var("a"),
            ArithmaExpression::zero(),
            ArithmaExpression::zero(),
        );
        assert!(evaluate_vector(&symbolic, &ArithmaBindings::new()).is_err());
        let mut bindings = ArithmaBindings::new();
        bindings.insert("a".to_string(), 2.5);
        let got = evaluate_vector(&symbolic, &bindings).expect("bound");
        assert!((got[0] - 2.5).abs() < 1e-12);
    }

    #[test]
    fn evaluate_vector_never_returns_a_nan_coordinate() {
        // sqrt(-1) is NaN; it must surface as an error, not as a coordinate.
        let bad = ArithmaVector::new(
            ArithmaExpression::sqrt(ArithmaExpression::from_i64(-1)),
            ArithmaExpression::zero(),
            ArithmaExpression::zero(),
        );
        match evaluate_vector(&bad, &ArithmaBindings::new()) {
            Ok(parts) => panic!("NaN component was returned as {parts:?}"),
            Err(msg) => assert!(!msg.is_empty(), "errors must carry a message"),
        }
    }

    #[test]
    fn intersection_kinds_and_point_counts_agree() {
        let cases = [
            (ArithmaIntersectionResult::None, "none", 0_usize),
            (ArithmaIntersectionResult::Point(v(1, 2, 3)), "point", 1),
            (
                ArithmaIntersectionResult::TwoPoints(v(1, 0, 0), v(2, 0, 0)),
                "two_points",
                2,
            ),
            (ArithmaIntersectionResult::Continuous, "continuous", 0),
        ];
        for (result, kind, count) in cases.iter() {
            assert_eq!(intersection_kind(result), *kind);
            assert_eq!(expected_point_count(kind), *count);
            assert_eq!(intersection_points(result).len(), *count);
        }
    }

    #[test]
    fn intersection_points_preserve_near_to_far_order() {
        let outcome = ArithmaIntersectionResult::TwoPoints(v(7, 0, 0), v(13, 0, 0));
        let points = intersection_points(&outcome);
        let near = evaluate_vector(&points[0], &ArithmaBindings::new()).expect("numeric");
        let far = evaluate_vector(&points[1], &ArithmaBindings::new()).expect("numeric");
        assert!(near[0] < far[0], "near hit must come first");
    }

    #[test]
    fn kinds_without_points_report_zero() {
        assert_eq!(expected_point_count("continuous"), 0);
        assert_eq!(expected_point_count("none"), 0);
    }

    #[test]
    fn wrapped_primitives_behave_as_documented() {
        // Line down the z axis meeting the z = 0 plane at the origin.
        let line = ArithmaLine::new(v(0, 0, 10), v(0, 0, -1));
        let plane = ArithmaPlane::new(v(0, 0, 1), ArithmaExpression::zero());
        let hit = ArithmaIntersection::line_plane(&line, &plane);
        assert_eq!(intersection_kind(&hit), "point");
        let points = intersection_points(&hit);
        let coords = evaluate_vector(&points[0], &ArithmaBindings::new()).expect("numeric");
        assert!(coords.iter().all(|c| c.abs() < 1e-9), "got {coords:?}");
    }

    #[test]
    fn derived_magnitude_matches_the_squared_magnitude() {
        let root = ArithmaExpression::sqrt(v(3, 4, 0).magnitude_squared());
        let got = root.evaluate(&ArithmaBindings::new()).expect("numeric");
        assert!((got - 5.0).abs() < 1e-9, "got {got}");
    }

    #[test]
    fn derived_closest_point_lands_on_the_line() {
        let line = ArithmaLine::new(v(1, 2, 3), v(1, 2, 2));
        let point = v(4, 0, -1);
        let t = ArithmaIntersection::closest_point_param(&line, &point);
        let foot = line.at(t);
        let residual = point.sub_vec(&foot).dot(&line.direction);
        let got = residual.evaluate(&ArithmaBindings::new()).expect("numeric");
        assert!(got.abs() < 1e-6, "residual {got}");
    }
}
