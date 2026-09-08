//====== Arithma/rust/arithma_core/src/pyfacade/linalg.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! PyO3 facade for the linear-algebra domain: `ArithmaMatrix` and
//! `ArithmaTensor`.
//!
//! ## Scope -- read this before adding to the file
//!
//! Both underlying Rust types are Wave-2 **containers**, not algebra engines.
//! `crate::matrix::ArithmaMatrix` provides exactly `new` / `len` / `is_empty`
//! over a flat row-major `Vec<ArithmaExpression>`; `crate::tensor::ArithmaTensor`
//! provides exactly `new` / `cell_count`. There is no addition, multiplication,
//! transpose, determinant, inverse, trace, rank, decomposition, contraction, or
//! element accessor anywhere in the crate -- nothing outside `lib.rs`'s module
//! declaration even mentions these types.
//!
//! This facade therefore exposes **construction and inspection only**. It
//! deliberately does not offer `matmul`, `det`, `inv`, or friends: implementing
//! them here would put the mathematics in the Python facade, inverting the rule
//! that "Python is the thin wrapper, never the implementation" (see
//! `pyfacade/mod.rs`). When real algebra lands in `matrix.rs`, the wrappers
//! belong here and the docstrings below must be updated in the same commit.
//!
//! ## Hazards this wrapper absorbs
//!
//! - `ArithmaMatrix::new` and `ArithmaTensor::new` **panic** (`assert_eq!`) when
//!   the cell count disagrees with the declared shape. Every entry point here
//!   validates the count first and raises `ValueError`, so a caller mistake can
//!   never surface as a `pyo3_runtime.PanicException`.
//! - `ArithmaTensor::cell_count` is `shape.iter().product()`, which wraps
//!   silently in a release build. [`validate_shape`] performs the same product
//!   with `checked_mul` and rejects overflow before the core is ever called.
//! - `core::coerce_to_expression` is private to `core.rs`, so the
//!   int / float / `Expression` coercion rule is restated in [`coerce_cell`].
//!   It additionally rejects `bool`, matching the stricter behaviour of
//!   `Expression.number`, because a silent `True -> 1` inside a matrix is a
//!   data-corruption footgun rather than a convenience.

// `#[pymethods]` / `#[pyfunction]` on pyo3 0.22 expand `-> PyResult<T>` into a
// `PyErr::from` round-trip that clippy reads as a no-op conversion in *our*
// signature span. `pyfacade::core` already carries these warnings; scoping the
// allow to this file keeps the new surface from adding more without editing the
// crate-wide lint policy in `lib.rs`. Matches `calculus.rs` / `geometry.rs`.
#![allow(clippy::useless_conversion)]

use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PySequence, PyString};

use crate::expression::ArithmaExpression;
use crate::matrix::ArithmaMatrix;
use crate::pyfacade::core::Expression;
use crate::pyfacade::MAX_SEQUENCE_LEN;
use crate::tensor::ArithmaTensor;

/// Largest tensor rank accepted from Python. Bounds the shape loop (CLAUDE.md
/// standard 3) well above any physically meaningful tensor while keeping a
/// hostile `[1] * 10_000_000` shape from ever being materialised.
const MAX_TENSOR_RANK: usize = 32;

// ============================================================================
// Helpers.
// ============================================================================

/// Borrow a Python object as a sequence (list / tuple). `str` is rejected up
/// front: it satisfies `PySequence_Check` and would otherwise decompose into
/// single-character cells and fail with a confusing per-character error.
fn as_sequence<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PySequence>> {
    if obj.is_instance_of::<PyString>() {
        return Err(PyTypeError::new_err(
            "expected a list or tuple of cells, not a str",
        ));
    }
    let seq = obj
        .downcast::<PySequence>()
        .map_err(|_| PyTypeError::new_err("expected a list or tuple"))?;
    let len = seq.len()?;
    assert!(
        len <= isize::MAX as usize,
        "PySequence_Size returned garbage"
    );
    Ok(seq.clone())
}

/// Convert one Python object into an `ArithmaExpression` cell. Accepts an
/// existing `Expression`, an `int`, or a `float`; everything else (including
/// `bool`) is a `TypeError`.
fn coerce_cell(obj: &Bound<'_, PyAny>) -> PyResult<ArithmaExpression> {
    if obj.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(
            "matrix/tensor cells reject bool; pass 0 or 1 explicitly",
        ));
    }
    if let Ok(py_expr) = obj.extract::<PyRef<Expression>>() {
        return Ok(py_expr.inner.clone());
    }
    if let Ok(int_val) = obj.extract::<i64>() {
        return Ok(ArithmaExpression::from_i64(int_val));
    }
    if let Ok(float_val) = obj.extract::<f64>() {
        return Ok(ArithmaExpression::from_f64(float_val));
    }
    Err(PyTypeError::new_err(
        "cell must be an Expression, int, or float",
    ))
}

/// Extract a flat cell list from a Python sequence. The length is checked
/// against [`MAX_SEQUENCE_LEN`] *before* the loop runs, so the iteration bound
/// is fixed and known up front (CLAUDE.md standard 3).
fn extract_cells(obj: &Bound<'_, PyAny>) -> PyResult<Vec<ArithmaExpression>> {
    let seq = as_sequence(obj)?;
    let count = seq.len()?;
    if count > MAX_SEQUENCE_LEN {
        return Err(PyValueError::new_err(format!(
            "cell sequence of {count} exceeds MAX_SEQUENCE_LEN ({MAX_SEQUENCE_LEN})"
        )));
    }
    let mut cells: Vec<ArithmaExpression> = Vec::with_capacity(count);
    for index in 0..count {
        cells.push(coerce_cell(&seq.get_item(index)?)?);
    }
    assert_eq!(cells.len(), count, "bounded loop must fill every slot");
    Ok(cells)
}

/// `rows * cols` with overflow and size checks. Returns the validated cell
/// count so callers never recompute -- and never disagree with -- the product.
fn checked_area(rows: usize, cols: usize) -> PyResult<usize> {
    let area = rows.checked_mul(cols).ok_or_else(|| {
        PyValueError::new_err(format!("matrix shape {rows}x{cols} overflows usize"))
    })?;
    if area > MAX_SEQUENCE_LEN {
        return Err(PyValueError::new_err(format!(
            "matrix shape {rows}x{cols} needs {area} cells, over MAX_SEQUENCE_LEN ({MAX_SEQUENCE_LEN})"
        )));
    }
    assert!(
        area <= MAX_SEQUENCE_LEN,
        "area bound checked immediately above"
    );
    Ok(area)
}

/// Validate a tensor shape and return the product of its dimensions. Mirrors
/// `ArithmaTensor::cell_count` but with `checked_mul`, because the core version
/// wraps silently on overflow in a release build.
fn validate_shape(shape: &[usize]) -> PyResult<usize> {
    if shape.len() > MAX_TENSOR_RANK {
        return Err(PyValueError::new_err(format!(
            "tensor rank {} exceeds MAX_TENSOR_RANK ({})",
            shape.len(),
            MAX_TENSOR_RANK
        )));
    }
    let mut count: usize = 1;
    for (axis, &dim) in shape.iter().enumerate() {
        count = count.checked_mul(dim).ok_or_else(|| {
            PyValueError::new_err(format!("tensor shape overflows usize at axis {axis}"))
        })?;
        if count > MAX_SEQUENCE_LEN {
            return Err(PyValueError::new_err(format!(
                "tensor shape needs over {MAX_SEQUENCE_LEN} cells, above MAX_SEQUENCE_LEN"
            )));
        }
    }
    assert!(
        count <= MAX_SEQUENCE_LEN,
        "count bound enforced in the loop"
    );
    Ok(count)
}

/// Extract a shape / multi-index tuple from Python with the rank bound applied
/// before any allocation, so a hostile shape cannot force an unbounded `Vec`.
fn extract_shape(obj: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    let seq = as_sequence(obj)?;
    let rank = seq.len()?;
    if rank > MAX_TENSOR_RANK {
        return Err(PyValueError::new_err(format!(
            "tensor rank {rank} exceeds MAX_TENSOR_RANK ({MAX_TENSOR_RANK})"
        )));
    }
    let mut shape: Vec<usize> = Vec::with_capacity(rank);
    for axis in 0..rank {
        let dim: usize = seq.get_item(axis)?.extract().map_err(|_| {
            PyValueError::new_err(format!(
                "tensor shape axis {axis} must be a non-negative int"
            ))
        })?;
        shape.push(dim);
    }
    assert_eq!(shape.len(), rank, "bounded loop must fill every axis");
    Ok(shape)
}

/// Row-major offset of `indices` inside `shape`. Kept outside the pyclass so it
/// is unit-testable without a Python interpreter.
fn flat_index(shape: &[usize], indices: &[usize]) -> PyResult<usize> {
    if indices.len() != shape.len() {
        return Err(PyValueError::new_err(format!(
            "expected {} indices for a rank-{} tensor, got {}",
            shape.len(),
            shape.len(),
            indices.len()
        )));
    }
    assert!(
        shape.len() <= MAX_TENSOR_RANK,
        "rank bounded at construction"
    );
    let mut offset: usize = 0;
    for (axis, (&dim, &index)) in shape.iter().zip(indices.iter()).enumerate() {
        if index >= dim {
            return Err(PyIndexError::new_err(format!(
                "index {index} out of range for axis {axis} of length {dim}"
            )));
        }
        offset = offset
            .checked_mul(dim)
            .and_then(|scaled| scaled.checked_add(index))
            .ok_or_else(|| PyValueError::new_err("tensor index arithmetic overflowed"))?;
    }
    Ok(offset)
}

// ============================================================================
// `Matrix` pyclass.
// ============================================================================

/// Symbolic matrix -- Python wrapper around `ArithmaMatrix`.
///
/// A row-major container of `Expression` cells, with the algebra over them:
/// `add`, `sub`, `scalar_mul`, `matmul` (and `+`, `-`, `@`), `transpose`,
/// `trace`, `determinant`, `minor`, `cofactor`, `adjugate`, `inverse`,
/// `characteristic_polynomial` and `eigenvalues_real`.
///
/// Entries stay **symbolic**. `inverse()` is `adj(A)/det(A)`, so unless the
/// determinant divides each cofactor exactly the entries come back as exact
/// quotients rather than decimals -- `Matrix([[4,7],[2,6]]).inverse()` holds
/// `6/10`, not `0.6`. Evaluating those to `float` rounds; the quotient does
/// not. `determinant()` is exact cofactor expansion and is therefore capped at
/// `MAX_DETERMINANT_ORDER`, because the cost is factorial.
///
/// This docstring previously said *"No matrix arithmetic is available"* and
/// told the reader not to expect `det()` / `inv()` / `__matmul__`. That was
/// true of an earlier core and stopped being true when the algebra landed;
/// nothing failed, because a stale docstring is only read by people, and it is
/// read at exactly the moment they are deciding whether the method they want
/// exists.

#[pyclass(name = "Matrix", module = "arithma")]
#[derive(Clone)]
pub struct Matrix {
    inner: ArithmaMatrix,
}

#[pymethods]
impl Matrix {
    /// Build from an explicit shape and a flat row-major cell sequence.
    ///
    /// Raises `ValueError` if `len(cells) != rows * cols`, if the shape
    /// overflows, or if it would need more than `MAX_SEQUENCE_LEN` cells.
    #[new]
    fn new(rows: usize, cols: usize, cells: &Bound<'_, PyAny>) -> PyResult<Self> {
        let expected = checked_area(rows, cols)?;
        let values = extract_cells(cells)?;
        if values.len() != expected {
            return Err(PyValueError::new_err(format!(
                "Matrix({}, {}) expects {} cells, got {}",
                rows,
                cols,
                expected,
                values.len()
            )));
        }
        assert_eq!(values.len(), expected, "count equality checked just above");
        let inner = ArithmaMatrix::new(rows, cols, values);
        assert_eq!(inner.len(), expected, "ArithmaMatrix::new must keep cells");
        Ok(Self { inner })
    }

    /// Build from a nested sequence of rows, e.g. `[[1, 2], [3, x]]`.
    ///
    /// Raises `ValueError` for a ragged nested sequence -- the core has no
    /// concept of a ragged matrix and would panic on its shape assert.
    #[staticmethod]
    fn from_rows(rows: &Bound<'_, PyAny>) -> PyResult<Self> {
        let outer = as_sequence(rows)?;
        let row_count = outer.len()?;
        if row_count > MAX_SEQUENCE_LEN {
            return Err(PyValueError::new_err(format!(
                "row count {row_count} exceeds MAX_SEQUENCE_LEN ({MAX_SEQUENCE_LEN})"
            )));
        }
        let mut col_count: usize = 0;
        let mut cells: Vec<ArithmaExpression> = Vec::new();
        for index in 0..row_count {
            let row_cells = extract_cells(&outer.get_item(index)?)?;
            if index == 0 {
                col_count = row_cells.len();
            } else if row_cells.len() != col_count {
                return Err(PyValueError::new_err(format!(
                    "ragged rows: row 0 has {} cells, row {} has {}",
                    col_count,
                    index,
                    row_cells.len()
                )));
            }
            let _ = checked_area(row_count, col_count)?;
            cells.extend(row_cells);
        }
        let expected = checked_area(row_count, col_count)?;
        assert_eq!(cells.len(), expected, "row-major fill must match the shape");
        Ok(Self {
            inner: ArithmaMatrix::new(row_count, col_count, cells),
        })
    }

    /// Matrix of the given shape with every cell the literal `0`.
    #[staticmethod]
    fn zeros(rows: usize, cols: usize) -> PyResult<Self> {
        let count = checked_area(rows, cols)?;
        let cells = vec![ArithmaExpression::from_i64(0); count];
        assert_eq!(cells.len(), count, "vec! must honour the requested length");
        let inner = ArithmaMatrix::new(rows, cols, cells);
        assert_eq!(inner.len(), count, "ArithmaMatrix::new must keep cells");
        Ok(Self { inner })
    }

    /// `n x n` identity built from the literals `0` and `1`. Pure container
    /// construction -- it does not imply the core can multiply by it.
    #[staticmethod]
    fn identity(n: usize) -> PyResult<Self> {
        let count = checked_area(n, n)?;
        if n == 0 {
            return Ok(Self {
                inner: ArithmaMatrix::new(0, 0, Vec::new()),
            });
        }
        let mut cells: Vec<ArithmaExpression> = Vec::with_capacity(count);
        for index in 0..count {
            let on_diagonal = index / n == index % n;
            cells.push(ArithmaExpression::from_i64(i64::from(on_diagonal)));
        }
        assert_eq!(cells.len(), count, "bounded loop must fill every cell");
        Ok(Self {
            inner: ArithmaMatrix::new(n, n, cells),
        })
    }

    /// Row count.
    #[getter]
    fn rows(&self) -> usize {
        self.inner.rows
    }

    /// Column count.
    #[getter]
    fn cols(&self) -> usize {
        self.inner.cols
    }

    /// Total cell count. Wraps `ArithmaMatrix::len`.
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// True when the matrix holds no cells. Wraps `ArithmaMatrix::is_empty`.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Cell at `(row, col)`. Raises `IndexError` when out of range.
    fn get(&self, row: usize, col: usize) -> PyResult<Expression> {
        if row >= self.inner.rows || col >= self.inner.cols {
            return Err(PyIndexError::new_err(format!(
                "({}, {}) out of range for a {}x{} matrix",
                row, col, self.inner.rows, self.inner.cols
            )));
        }
        let offset = row
            .checked_mul(self.inner.cols)
            .and_then(|scaled| scaled.checked_add(col))
            .ok_or_else(|| PyValueError::new_err("matrix index arithmetic overflowed"))?;
        let cell = self
            .inner
            .cells
            .get(offset)
            .ok_or_else(|| PyIndexError::new_err("matrix cells shorter than its shape"))?;
        Ok(Expression {
            inner: cell.clone(),
        })
    }

    /// One row as a list of `Expression`. Raises `IndexError` when out of range.
    fn row(&self, row: usize) -> PyResult<Vec<Expression>> {
        if row >= self.inner.rows {
            return Err(PyIndexError::new_err(format!(
                "row {} out of range for a {}-row matrix",
                row, self.inner.rows
            )));
        }
        let start = row
            .checked_mul(self.inner.cols)
            .ok_or_else(|| PyValueError::new_err("matrix index arithmetic overflowed"))?;
        let end = start
            .checked_add(self.inner.cols)
            .filter(|stop| *stop <= self.inner.cells.len())
            .ok_or_else(|| PyIndexError::new_err("matrix cells shorter than its shape"))?;
        Ok(self.inner.cells[start..end]
            .iter()
            .map(|cell| Expression {
                inner: cell.clone(),
            })
            .collect())
    }

    /// Every cell in row-major order.
    fn cells(&self) -> Vec<Expression> {
        self.inner
            .cells
            .iter()
            .map(|cell| Expression {
                inner: cell.clone(),
            })
            .collect()
    }

    /// The matrix as a nested list of `Expression`, one inner list per row.
    fn to_rows(&self) -> PyResult<Vec<Vec<Expression>>> {
        let mut out: Vec<Vec<Expression>> = Vec::with_capacity(self.inner.rows);
        for index in 0..self.inner.rows {
            out.push(self.row(index)?);
        }
        assert_eq!(out.len(), self.inner.rows, "bounded loop fills every row");
        Ok(out)
    }

    // -------- algebra --------
    //
    // `Matrix` was a container on the Python side too: shape, indexing and
    // repr, with no operation that combined two matrices. Shape errors surface
    // as `ValueError` rather than a panic crossing the FFI boundary.

    /// True when the matrix has as many rows as columns.
    fn is_square(&self) -> bool {
        self.inner.is_square()
    }

    /// The shape as a `(rows, cols)` tuple.
    ///
    /// A property, matching `Tensor.shape` -- the two types should not differ
    /// on whether shape is called or read.
    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    /// The transpose: `result[i][j] == self[j][i]`.
    fn transpose(&self) -> Self {
        Self {
            inner: self.inner.transpose(),
        }
    }

    /// Elementwise sum. Shapes must match exactly.
    fn add(&self, other: &Self) -> PyResult<Self> {
        matrix_result(self.inner.add(&other.inner))
    }

    /// Elementwise difference. Shapes must match exactly.
    fn sub(&self, other: &Self) -> PyResult<Self> {
        matrix_result(self.inner.sub(&other.inner))
    }

    /// Multiply every cell by a scalar expression.
    fn scalar_mul(&self, scalar: &Expression) -> Self {
        Self {
            inner: self.inner.scalar_mul(&scalar.inner),
        }
    }

    /// Matrix product. `self.cols()` must equal `other.rows()`.
    fn matmul(&self, other: &Self) -> PyResult<Self> {
        matrix_result(self.inner.matmul(&other.inner))
    }

    /// Sum of the leading diagonal. Requires a square matrix.
    fn trace(&self) -> PyResult<Expression> {
        let value = self
            .inner
            .trace()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Expression { inner: value })
    }

    /// The determinant, exact and symbolic.
    ///
    /// Computed by summation over permutations, so it is restricted by order;
    /// a larger matrix raises `ValueError` rather than running unbounded.
    fn determinant(&self) -> PyResult<Expression> {
        let value = self
            .inner
            .determinant()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Expression { inner: value })
    }

    /// The `(row, col)` minor: this matrix with that row and column deleted.
    fn minor(&self, row: usize, col: usize) -> PyResult<Self> {
        matrix_result(self.inner.minor(row, col))
    }

    /// The `(row, col)` cofactor: the signed determinant of the minor.
    fn cofactor(&self, row: usize, col: usize) -> PyResult<Expression> {
        let value = self
            .inner
            .cofactor(row, col)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Expression { inner: value })
    }

    /// The adjugate: the transpose of the cofactor matrix.
    fn adjugate(&self) -> PyResult<Self> {
        matrix_result(self.inner.adjugate())
    }

    /// The inverse, as `adj(A) / det(A)`.
    ///
    /// Exact and symbolic. Raises `ValueError` when the matrix is singular or
    /// not square. A determinant that is merely *symbolic* is not singular --
    /// it may be non-zero for the values you have in mind -- so the quotient
    /// comes back unevaluated rather than refused.
    fn inverse(&self) -> PyResult<Self> {
        matrix_result(self.inner.inverse())
    }

    /// The characteristic polynomial `det(A - x I)` in the named variable.
    ///
    /// Its roots are the eigenvalues.
    #[pyo3(signature = (var = "lambda"))]
    fn characteristic_polynomial(&self, var: &str) -> PyResult<Expression> {
        if var.is_empty() {
            return Err(PyValueError::new_err("var must not be empty"));
        }
        debug_assert!(!var.is_empty(), "empty variable slipped past the guard");
        let value = self
            .inner
            .characteristic_polynomial(var)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Expression { inner: value })
    }

    /// Real eigenvalues, ascending.
    ///
    /// **Real roots only**, and repeated eigenvalues appear once. A rotation
    /// matrix returns `[]` because its spectrum is complex -- that is the
    /// correct answer, not a failure. Eigenvectors are not computed. Every
    /// cell must evaluate to a finite number.
    fn eigenvalues_real(&self) -> PyResult<Vec<f64>> {
        let values = self
            .inner
            .eigenvalues_real()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        debug_assert!(
            values.len() <= self.inner.rows,
            "more roots than the polynomial's degree"
        );
        debug_assert!(
            values.iter().all(|v| v.is_finite()),
            "non-finite eigenvalue"
        );
        Ok(values)
    }

    /// A copy with every cell simplified.
    fn simplified(&self) -> Self {
        Self {
            inner: self.inner.simplified(),
        }
    }

    fn __matmul__(&self, other: &Self) -> PyResult<Self> {
        self.matmul(other)
    }

    fn __add__(&self, other: &Self) -> PyResult<Self> {
        self.add(other)
    }

    fn __sub__(&self, other: &Self) -> PyResult<Self> {
        self.sub(other)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "Matrix(rows={}, cols={}, cells={})",
            self.inner.rows,
            self.inner.cols,
            self.inner.len()
        )
    }
}

// ============================================================================
// `Tensor` pyclass.
// ============================================================================

/// N-dimensional symbolic tensor -- Python wrapper around `ArithmaTensor`.
///
/// As with [`Matrix`], the Rust core is a Wave-2 container: it provides only
/// `new` and `cell_count`, with no contraction, product, reshape, or element
/// accessor. Row-major (C-order) layout is this facade's documented reading of
/// the flat `cells` vector; the core itself states no ordering, so treat
/// `Tensor.get` as the definition rather than an inherited guarantee.
#[pyclass(name = "Tensor", module = "arithma")]
#[derive(Clone)]
pub struct Tensor {
    inner: ArithmaTensor,
}

#[pymethods]
impl Tensor {
    /// Build from a shape sequence and a flat row-major cell sequence.
    ///
    /// Raises `ValueError` when `len(cells)` differs from the product of the
    /// shape, when the shape overflows `usize`, or when rank or cell count
    /// exceed their bounds.
    #[new]
    fn new(shape: &Bound<'_, PyAny>, cells: &Bound<'_, PyAny>) -> PyResult<Self> {
        let dims = extract_shape(shape)?;
        let expected = validate_shape(&dims)?;
        let values = extract_cells(cells)?;
        if values.len() != expected {
            return Err(PyValueError::new_err(format!(
                "Tensor with shape {:?} expects {} cells, got {}",
                dims,
                expected,
                values.len()
            )));
        }
        assert_eq!(values.len(), expected, "count equality checked just above");
        Ok(Self {
            inner: ArithmaTensor::new(dims, values),
        })
    }

    /// Tensor of the given shape with every cell the literal `0`.
    #[staticmethod]
    fn zeros(shape: &Bound<'_, PyAny>) -> PyResult<Self> {
        let dims = extract_shape(shape)?;
        let count = validate_shape(&dims)?;
        let cells = vec![ArithmaExpression::from_i64(0); count];
        assert_eq!(cells.len(), count, "vec! must honour the requested length");
        Ok(Self {
            inner: ArithmaTensor::new(dims, cells),
        })
    }

    /// Product of the shape dimensions, without building a tensor.
    ///
    /// Wraps `ArithmaTensor::cell_count`, but rejects shapes that would
    /// overflow `usize` (the core computes the product unchecked, which wraps
    /// silently in a release build) or exceed `MAX_SEQUENCE_LEN`.
    #[staticmethod]
    fn cell_count(shape: &Bound<'_, PyAny>) -> PyResult<usize> {
        let dims = extract_shape(shape)?;
        let checked = validate_shape(&dims)?;
        let native = ArithmaTensor::cell_count(&dims);
        assert_eq!(checked, native, "checked product must match core product");
        Ok(checked)
    }

    /// The tensor shape.
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape.clone()
    }

    /// Number of axes.
    #[getter]
    fn rank(&self) -> usize {
        self.inner.shape.len()
    }

    /// Total cell count.
    #[getter]
    fn size(&self) -> usize {
        self.inner.cells.len()
    }

    /// Cell at a multi-index, row-major. Raises `IndexError` when a component
    /// is out of range and `ValueError` when the index rank is wrong.
    fn get(&self, indices: &Bound<'_, PyAny>) -> PyResult<Expression> {
        let idx = extract_shape(indices)?;
        let offset = flat_index(&self.inner.shape, &idx)?;
        let cell = self
            .inner
            .cells
            .get(offset)
            .ok_or_else(|| PyIndexError::new_err("tensor cells shorter than its shape"))?;
        Ok(Expression {
            inner: cell.clone(),
        })
    }

    /// Cell at a flat row-major offset.
    fn cell(&self, index: usize) -> PyResult<Expression> {
        let cell = self.inner.cells.get(index).ok_or_else(|| {
            PyIndexError::new_err(format!(
                "flat index {} out of range for {} cells",
                index,
                self.inner.cells.len()
            ))
        })?;
        assert!(
            index < self.inner.cells.len(),
            "get() proved index in range"
        );
        Ok(Expression {
            inner: cell.clone(),
        })
    }

    /// Every cell in row-major order.
    fn cells(&self) -> Vec<Expression> {
        self.inner
            .cells
            .iter()
            .map(|cell| Expression {
                inner: cell.clone(),
            })
            .collect()
    }

    // -------- algebra --------

    /// Row-major strides: the flat step per unit step along each axis.
    fn strides(&self) -> Vec<usize> {
        self.inner.strides()
    }

    /// Reinterpret the cells under a new shape, preserving row-major order.
    fn reshape(&self, shape: &Bound<'_, PyAny>) -> PyResult<Self> {
        let dims = extract_shape(shape)?;
        tensor_result(self.inner.reshape(dims))
    }

    /// Reorder the axes. `axes` must be a permutation of `range(rank)`.
    ///
    /// For a rank-2 tensor this is the matrix transpose.
    fn permute_axes(&self, axes: &Bound<'_, PyAny>) -> PyResult<Self> {
        let order = extract_shape(axes)?;
        tensor_result(self.inner.permute_axes(&order))
    }

    /// Elementwise sum. Shapes must match exactly.
    fn add(&self, other: &Self) -> PyResult<Self> {
        tensor_result(self.inner.add(&other.inner))
    }

    /// Elementwise difference. Shapes must match exactly.
    fn sub(&self, other: &Self) -> PyResult<Self> {
        tensor_result(self.inner.sub(&other.inner))
    }

    /// Elementwise product. Not a tensor contraction.
    fn hadamard(&self, other: &Self) -> PyResult<Self> {
        tensor_result(self.inner.hadamard(&other.inner))
    }

    /// Multiply every cell by a scalar expression.
    fn scalar_mul(&self, scalar: &Expression) -> Self {
        Self {
            inner: self.inner.scalar_mul(&scalar.inner),
        }
    }

    /// A copy with every cell simplified.
    fn simplified(&self) -> Self {
        Self {
            inner: self.inner.simplified(),
        }
    }

    /// Self-contraction over two axes of equal extent. Rank drops by two.
    ///
    /// For a rank-2 tensor with axes `(0, 1)` this is the matrix trace, and
    /// the result is a rank-0 tensor holding one cell.
    fn trace(&self, axis_a: usize, axis_b: usize) -> PyResult<Self> {
        tensor_result(self.inner.trace(axis_a, axis_b))
    }

    /// Generalised contraction, as numpy's `tensordot`.
    ///
    /// With `axes_a=[1], axes_b=[0]` on two rank-2 tensors this is matrix
    /// multiplication. The paired axes must match in extent.
    fn tensordot(
        &self,
        other: &Self,
        axes_a: &Bound<'_, PyAny>,
        axes_b: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let a = extract_shape(axes_a)?;
        let b = extract_shape(axes_b)?;
        if a.len() != b.len() {
            return Err(PyValueError::new_err(format!(
                "axes_a and axes_b must pair up: got {} and {} entries",
                a.len(),
                b.len()
            )));
        }
        debug_assert_eq!(a.len(), b.len(), "arity check failed");
        tensor_result(self.inner.tensordot(&other.inner, &a, &b))
    }

    /// The outer product: the zero-axis case of `tensordot`. Shapes concatenate.
    fn outer(&self, other: &Self) -> PyResult<Self> {
        tensor_result(self.inner.outer(&other.inner))
    }

    /// Replace the cell at a multi-dimensional index.
    fn set(&mut self, indices: &Bound<'_, PyAny>, value: &Expression) -> PyResult<()> {
        let index = extract_shape(indices)?;
        self.inner
            .set(&index, value.inner.clone())
            .map_err(|e| PyIndexError::new_err(e.to_string()))
    }

    fn __add__(&self, other: &Self) -> PyResult<Self> {
        self.add(other)
    }

    fn __sub__(&self, other: &Self) -> PyResult<Self> {
        self.sub(other)
    }

    fn __len__(&self) -> usize {
        self.inner.cells.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "Tensor(shape={:?}, cells={})",
            self.inner.shape,
            self.inner.cells.len()
        )
    }
}

// ============================================================================
// Registration
// ============================================================================

/// Register this module's surface on the extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Matrix>()?;
    m.add_class::<Tensor>()?;
    Ok(())
}

/// Map a matrix result into a Python exception on failure.
///
/// Shape mismatches are caller errors, so they become `ValueError` with the
/// core error's own message rather than a panic unwinding through the FFI
/// boundary, which is undefined behaviour.
fn matrix_result(result: Result<ArithmaMatrix, crate::matrix::MatrixError>) -> PyResult<Matrix> {
    match result {
        Ok(inner) => Ok(Matrix { inner }),
        Err(e) => Err(PyValueError::new_err(e.to_string())),
    }
}

/// Map a tensor result into a Python exception on failure.
fn tensor_result(result: Result<ArithmaTensor, crate::tensor::TensorError>) -> PyResult<Tensor> {
    match result {
        Ok(inner) => Ok(Tensor { inner }),
        Err(e) => Err(PyValueError::new_err(e.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_area_rejects_overflow() {
        assert!(checked_area(usize::MAX, 2).is_err());
    }

    #[test]
    fn checked_area_rejects_oversize() {
        assert!(checked_area(MAX_SEQUENCE_LEN, 2).is_err());
    }

    #[test]
    fn checked_area_accepts_reasonable_shape() {
        assert_eq!(checked_area(3, 4).expect("3x4 is valid"), 12);
        assert_eq!(checked_area(0, 0).expect("0x0 is valid"), 0);
    }

    #[test]
    fn validate_shape_matches_core_cell_count() {
        let shape = [2usize, 3, 4];
        assert_eq!(validate_shape(&shape).expect("valid shape"), 24);
        assert_eq!(
            validate_shape(&shape).expect("valid shape"),
            ArithmaTensor::cell_count(&shape)
        );
    }

    #[test]
    fn validate_shape_rank_zero_is_one_cell() {
        assert_eq!(validate_shape(&[]).expect("rank-0 is valid"), 1);
    }

    #[test]
    fn validate_shape_rejects_overflow() {
        assert!(validate_shape(&[usize::MAX, 2]).is_err());
    }

    #[test]
    fn validate_shape_rejects_excess_rank() {
        let shape = vec![1usize; MAX_TENSOR_RANK + 1];
        assert!(validate_shape(&shape).is_err());
    }

    #[test]
    fn flat_index_is_row_major() {
        let shape = [2usize, 3, 4];
        assert_eq!(flat_index(&shape, &[0, 0, 0]).expect("in range"), 0);
        assert_eq!(flat_index(&shape, &[0, 0, 1]).expect("in range"), 1);
        assert_eq!(flat_index(&shape, &[0, 1, 0]).expect("in range"), 4);
        assert_eq!(flat_index(&shape, &[1, 0, 0]).expect("in range"), 12);
        assert_eq!(flat_index(&shape, &[1, 2, 3]).expect("in range"), 23);
    }

    #[test]
    fn flat_index_rejects_out_of_range_and_wrong_rank() {
        let shape = [2usize, 3];
        assert!(flat_index(&shape, &[2, 0]).is_err());
        assert!(flat_index(&shape, &[0, 3]).is_err());
        assert!(flat_index(&shape, &[0]).is_err());
        assert!(flat_index(&shape, &[0, 0, 0]).is_err());
    }

    #[test]
    fn zeros_has_the_requested_shape() {
        let m = Matrix::zeros(2, 3).expect("2x3 is valid");
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
        assert_eq!(m.len(), 6);
        assert!(!m.is_empty());
    }

    #[test]
    fn identity_puts_ones_on_the_diagonal() {
        let m = Matrix::identity(3).expect("3x3 is valid");
        assert_eq!(m.len(), 9);
        for row in 0..3usize {
            for col in 0..3usize {
                let expected = if row == col { 1.0 } else { 0.0 };
                match &m.inner.cells[row * 3 + col] {
                    ArithmaExpression::Number(n) => assert_eq!(n.to_f64(), expected),
                    other => panic!("identity cell must be a literal, got {other:?}"),
                }
            }
        }
    }

    #[test]
    fn identity_zero_is_empty() {
        let m = Matrix::identity(0).expect("0x0 is valid");
        assert!(m.is_empty());
        assert_eq!(m.rows(), 0);
    }

    #[test]
    fn matrix_get_rejects_out_of_range() {
        let m = Matrix::zeros(2, 2).expect("2x2 is valid");
        assert!(m.get(2, 0).is_err());
        assert!(m.get(0, 2).is_err());
    }

    #[test]
    fn matrix_row_length_matches_cols() {
        let m = Matrix::zeros(2, 3).expect("2x3 is valid");
        assert_eq!(m.row(1).expect("row 1 exists").len(), 3);
        assert!(m.row(2).is_err());
    }
}
