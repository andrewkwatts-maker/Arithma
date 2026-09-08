//====== Arithma/rust/arithma_core/src/matrix.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! Symbolic matrix algebra.
//!
//! Cells are `ArithmaExpression`, so every operation is symbolic and exact.
//! Products and determinants build unsimplified sums; [`ArithmaMatrix::simplified`]
//! folds them through the same simplifier the rest of the crate uses.
//!
//! Shape errors are [`MatrixError`] values rather than panics: a dimension
//! mismatch in caller data is a condition to handle, not a crash.

use crate::expression::iterative::simplify_iterative;
use crate::expression::{ArithmaExpression, SimplificationConfig};

/// Symbolic matrix of `ArithmaExpression` cells.
#[derive(Debug, Clone)]
pub struct ArithmaMatrix {
    pub rows: usize,
    pub cols: usize,
    pub cells: Vec<ArithmaExpression>,
}

impl ArithmaMatrix {
    /// Construct a matrix from a flat row-major cell list. Panics if the
    /// length does not match `rows * cols`.
    pub fn new(rows: usize, cols: usize, cells: Vec<ArithmaExpression>) -> Self {
        assert_eq!(
            cells.len(),
            rows * cols,
            "cell count must equal rows * cols"
        );
        Self { rows, cols, cells }
    }

    /// Total cell count.
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    /// Whether the matrix is empty.
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }
}

/// Largest number of rows or columns accepted.
///
/// Safety-critical standard 2: every loop below is bounded by the matrix
/// dimensions, so those dimensions must themselves be bounded. 4096 is far
/// beyond symbolic use while keeping `rows * cols` clear of overflow.
pub const MAX_DIMENSION: usize = 4096;

/// Largest matrix a determinant will be attempted on.
///
/// The determinant is computed by summing over permutations, which is O(n!).
/// At n = 6 that is 720 terms; beyond it the expression would be larger than
/// anything useful. Refusing is better than returning after an unbounded wait.
pub const MAX_DETERMINANT_ORDER: usize = 6;

/// Why a matrix operation could not be performed.
///
/// Operations return this rather than panicking, so a dimension mismatch in
/// caller data is a handleable error and not a crash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatrixError {
    /// Operand shapes are incompatible; carries both shapes.
    DimensionMismatch {
        left: (usize, usize),
        right: (usize, usize),
    },
    /// An index was outside the matrix.
    OutOfBounds { row: usize, col: usize },
    /// The operation requires a square matrix.
    NotSquare { rows: usize, cols: usize },
    /// The matrix is larger than the operation's cap.
    TooLarge { order: usize, limit: usize },
    /// The determinant is exactly zero, so there is no inverse.
    Singular,
    /// A cell does not evaluate to a finite number, and the operation needs
    /// one. Carries the first offending position.
    NotNumeric { row: usize, col: usize },
}

impl core::fmt::Display for MatrixError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DimensionMismatch { left, right } => write!(
                f,
                "dimension mismatch: {}x{} against {}x{}",
                left.0, left.1, right.0, right.1
            ),
            Self::OutOfBounds { row, col } => {
                write!(f, "index ({row}, {col}) is outside the matrix")
            }
            Self::NotSquare { rows, cols } => {
                write!(f, "operation requires a square matrix, got {rows}x{cols}")
            }
            Self::TooLarge { order, limit } => {
                write!(f, "order {order} exceeds the limit of {limit}")
            }
            Self::Singular => write!(f, "matrix is singular; determinant is zero"),
            Self::NotNumeric { row, col } => write!(
                f,
                "cell ({row}, {col}) is not a finite number, and this operation needs one"
            ),
        }
    }
}

impl std::error::Error for MatrixError {}

impl ArithmaMatrix {
    /// A `rows` x `cols` matrix of zeros.
    pub fn zeros(rows: usize, cols: usize) -> Result<Self, MatrixError> {
        if rows > MAX_DIMENSION || cols > MAX_DIMENSION {
            return Err(MatrixError::TooLarge {
                order: rows.max(cols),
                limit: MAX_DIMENSION,
            });
        }
        debug_assert!(rows <= MAX_DIMENSION, "row cap not enforced");
        debug_assert!(cols <= MAX_DIMENSION, "column cap not enforced");
        let cells = vec![ArithmaExpression::zero(); rows * cols];
        Ok(Self { rows, cols, cells })
    }

    /// The `n` x `n` identity matrix.
    pub fn identity(n: usize) -> Result<Self, MatrixError> {
        let mut out = Self::zeros(n, n)?;
        for i in 0..n {
            out.cells[i * n + i] = ArithmaExpression::from_i64(1);
        }
        debug_assert_eq!(out.rows, out.cols, "identity must be square");
        debug_assert_eq!(out.cells.len(), n * n, "identity has the wrong cell count");
        Ok(out)
    }

    /// Build from a list of rows, checking that they are all the same length.
    pub fn from_rows(rows: Vec<Vec<ArithmaExpression>>) -> Result<Self, MatrixError> {
        let row_count = rows.len();
        let col_count = rows.first().map_or(0, Vec::len);
        for (i, row) in rows.iter().enumerate() {
            if row.len() != col_count {
                return Err(MatrixError::DimensionMismatch {
                    left: (1, col_count),
                    right: (1, rows[i].len()),
                });
            }
        }
        if row_count > MAX_DIMENSION || col_count > MAX_DIMENSION {
            return Err(MatrixError::TooLarge {
                order: row_count.max(col_count),
                limit: MAX_DIMENSION,
            });
        }
        debug_assert!(row_count <= MAX_DIMENSION, "row cap not enforced");
        let cells: Vec<ArithmaExpression> = rows.into_iter().flatten().collect();
        debug_assert_eq!(cells.len(), row_count * col_count, "flatten lost cells");
        Ok(Self {
            rows: row_count,
            cols: col_count,
            cells,
        })
    }

    /// Flat index of `(row, col)` in row-major order.
    fn offset(&self, row: usize, col: usize) -> Option<usize> {
        if row >= self.rows || col >= self.cols {
            return None;
        }
        debug_assert!(row < self.rows && col < self.cols, "bounds check failed");
        let index = row * self.cols + col;
        debug_assert!(index < self.cells.len(), "offset escaped the cell buffer");
        Some(index)
    }

    /// Borrow the cell at `(row, col)`.
    pub fn get(&self, row: usize, col: usize) -> Result<&ArithmaExpression, MatrixError> {
        let index = self
            .offset(row, col)
            .ok_or(MatrixError::OutOfBounds { row, col })?;
        self.cells
            .get(index)
            .ok_or(MatrixError::OutOfBounds { row, col })
    }

    /// Replace the cell at `(row, col)`.
    pub fn set(
        &mut self,
        row: usize,
        col: usize,
        value: ArithmaExpression,
    ) -> Result<(), MatrixError> {
        let index = self
            .offset(row, col)
            .ok_or(MatrixError::OutOfBounds { row, col })?;
        debug_assert!(index < self.cells.len(), "offset escaped the cell buffer");
        self.cells[index] = value;
        Ok(())
    }

    /// True when the matrix has as many rows as columns.
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// The shape as `(rows, cols)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// The transpose: `result[i][j] == self[j][i]`.
    pub fn transpose(&self) -> Self {
        debug_assert_eq!(self.cells.len(), self.rows * self.cols, "shape mismatch");
        let mut cells = Vec::with_capacity(self.cells.len());
        for col in 0..self.cols {
            for row in 0..self.rows {
                cells.push(self.cells[row * self.cols + col].clone());
            }
        }
        debug_assert_eq!(cells.len(), self.cells.len(), "transpose lost cells");
        Self {
            rows: self.cols,
            cols: self.rows,
            cells,
        }
    }

    /// Elementwise combination of two identically-shaped matrices.
    fn zip_with(
        &self,
        other: &Self,
        combine: fn(ArithmaExpression, ArithmaExpression) -> ArithmaExpression,
    ) -> Result<Self, MatrixError> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(MatrixError::DimensionMismatch {
                left: self.shape(),
                right: other.shape(),
            });
        }
        debug_assert_eq!(self.cells.len(), other.cells.len(), "shape check failed");
        let mut cells = Vec::with_capacity(self.cells.len());
        for i in 0..self.cells.len() {
            cells.push(combine(self.cells[i].clone(), other.cells[i].clone()));
        }
        debug_assert_eq!(cells.len(), self.cells.len(), "zip lost cells");
        Ok(Self {
            rows: self.rows,
            cols: self.cols,
            cells,
        })
    }

    /// Matrix addition. Shapes must match exactly.
    pub fn add(&self, other: &Self) -> Result<Self, MatrixError> {
        self.zip_with(other, ArithmaExpression::add)
    }

    /// Matrix subtraction. Shapes must match exactly.
    pub fn sub(&self, other: &Self) -> Result<Self, MatrixError> {
        self.zip_with(other, ArithmaExpression::sub)
    }

    /// Multiply every cell by a scalar expression.
    pub fn scalar_mul(&self, scalar: &ArithmaExpression) -> Self {
        debug_assert_eq!(self.cells.len(), self.rows * self.cols, "shape mismatch");
        let cells = self
            .cells
            .iter()
            .map(|c| ArithmaExpression::mul(scalar.clone(), c.clone()))
            .collect::<Vec<_>>();
        debug_assert_eq!(cells.len(), self.cells.len(), "scalar_mul lost cells");
        Self {
            rows: self.rows,
            cols: self.cols,
            cells,
        }
    }

    /// Matrix product. `self.cols` must equal `other.rows`.
    ///
    /// Entries are symbolic, so the result is an unsimplified sum of products;
    /// run [`Self::simplified`] to fold it.
    pub fn matmul(&self, other: &Self) -> Result<Self, MatrixError> {
        if self.cols != other.rows {
            return Err(MatrixError::DimensionMismatch {
                left: self.shape(),
                right: other.shape(),
            });
        }
        debug_assert_eq!(self.cols, other.rows, "inner dimension check failed");
        let mut cells = Vec::with_capacity(self.rows * other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut acc: Option<ArithmaExpression> = None;
                for k in 0..self.cols {
                    let term = ArithmaExpression::mul(
                        self.cells[i * self.cols + k].clone(),
                        other.cells[k * other.cols + j].clone(),
                    );
                    acc = Some(match acc {
                        None => term,
                        Some(sum) => ArithmaExpression::add(sum, term),
                    });
                }
                // An inner dimension of zero yields an empty sum, which is 0.
                cells.push(acc.unwrap_or_else(ArithmaExpression::zero));
            }
        }
        debug_assert_eq!(cells.len(), self.rows * other.cols, "matmul lost cells");
        Ok(Self {
            rows: self.rows,
            cols: other.cols,
            cells,
        })
    }

    /// Sum of the leading diagonal. Requires a square matrix.
    pub fn trace(&self) -> Result<ArithmaExpression, MatrixError> {
        if !self.is_square() {
            return Err(MatrixError::NotSquare {
                rows: self.rows,
                cols: self.cols,
            });
        }
        debug_assert_eq!(self.rows, self.cols, "square check failed");
        let mut acc: Option<ArithmaExpression> = None;
        for i in 0..self.rows {
            let cell = self.cells[i * self.cols + i].clone();
            acc = Some(match acc {
                None => cell,
                Some(sum) => ArithmaExpression::add(sum, cell),
            });
        }
        Ok(simplified_expression(
            acc.unwrap_or_else(ArithmaExpression::zero),
        ))
    }

    /// The determinant, by summation over permutations.
    ///
    /// Exact and symbolic. Restricted to order [`MAX_DETERMINANT_ORDER`]
    /// because the method is O(n!); a larger matrix returns
    /// [`MatrixError::TooLarge`] rather than running for an unbounded time.
    ///
    /// Gaussian elimination would scale better but needs division, and
    /// dividing symbolic entries introduces quotients that may not simplify
    /// away -- exactness is worth more here than asymptotics at these sizes.
    pub fn determinant(&self) -> Result<ArithmaExpression, MatrixError> {
        if !self.is_square() {
            return Err(MatrixError::NotSquare {
                rows: self.rows,
                cols: self.cols,
            });
        }
        let n = self.rows;
        if n > MAX_DETERMINANT_ORDER {
            return Err(MatrixError::TooLarge {
                order: n,
                limit: MAX_DETERMINANT_ORDER,
            });
        }
        debug_assert!(n <= MAX_DETERMINANT_ORDER, "order cap not enforced");
        if n == 0 {
            // The determinant of the empty matrix is 1 by convention, which
            // keeps `det(A ⊕ B) = det(A) det(B)` true.
            return Ok(ArithmaExpression::from_i64(1));
        }

        let mut total: Option<ArithmaExpression> = None;
        let mut perm: Vec<usize> = (0..n).collect();
        let mut guard: usize = 0;
        let limit = factorial_bounded(n);
        loop {
            guard += 1;
            if guard > limit {
                break;
            }
            let mut product: Option<ArithmaExpression> = None;
            for (row, &col) in perm.iter().enumerate() {
                let cell = self.cells[row * self.cols + col].clone();
                product = Some(match product {
                    None => cell,
                    Some(p) => ArithmaExpression::mul(p, cell),
                });
            }
            let mut term = product.unwrap_or_else(|| ArithmaExpression::from_i64(1));
            if permutation_sign_is_negative(&perm) {
                term = ArithmaExpression::neg(term);
            }
            total = Some(match total {
                None => term,
                Some(sum) => ArithmaExpression::add(sum, term),
            });
            if !next_permutation(&mut perm) {
                break;
            }
        }
        debug_assert!(guard <= limit, "permutation loop exceeded n!");
        Ok(simplified_expression(
            total.unwrap_or_else(ArithmaExpression::zero),
        ))
    }

    /// The `(row, col)` minor: this matrix with that row and column deleted.
    ///
    /// Requires a square matrix of order at least 1.
    pub fn minor(&self, row: usize, col: usize) -> Result<Self, MatrixError> {
        if !self.is_square() {
            return Err(MatrixError::NotSquare {
                rows: self.rows,
                cols: self.cols,
            });
        }
        if row >= self.rows || col >= self.cols {
            return Err(MatrixError::OutOfBounds { row, col });
        }
        debug_assert!(self.rows >= 1, "a minor needs at least one row to delete");
        debug_assert_eq!(self.rows, self.cols, "square check failed");
        let n = self.rows;
        let mut cells = Vec::with_capacity((n - 1) * (n - 1));
        for i in 0..n {
            if i == row {
                continue;
            }
            for j in 0..n {
                if j == col {
                    continue;
                }
                cells.push(self.cells[i * n + j].clone());
            }
        }
        debug_assert_eq!(cells.len(), (n - 1) * (n - 1), "minor lost cells");
        Ok(Self {
            rows: n - 1,
            cols: n - 1,
            cells,
        })
    }

    /// The `(row, col)` cofactor: `(-1)^(row+col)` times the minor's
    /// determinant.
    pub fn cofactor(&self, row: usize, col: usize) -> Result<ArithmaExpression, MatrixError> {
        let minor = self.minor(row, col)?;
        debug_assert_eq!(minor.rows + 1, self.rows, "minor has the wrong order");
        let det = minor.determinant()?;
        let signed = if (row + col) % 2 == 0 {
            det
        } else {
            ArithmaExpression::neg(det)
        };
        Ok(simplified_expression(signed))
    }

    /// The adjugate: the transpose of the cofactor matrix.
    ///
    /// Bounded by [`MAX_DETERMINANT_ORDER`] + 1, because every entry is the
    /// determinant of an order-`n-1` minor.
    pub fn adjugate(&self) -> Result<Self, MatrixError> {
        if !self.is_square() {
            return Err(MatrixError::NotSquare {
                rows: self.rows,
                cols: self.cols,
            });
        }
        let n = self.rows;
        if n > MAX_DETERMINANT_ORDER + 1 {
            return Err(MatrixError::TooLarge {
                order: n,
                limit: MAX_DETERMINANT_ORDER + 1,
            });
        }
        debug_assert!(n <= MAX_DETERMINANT_ORDER + 1, "order cap not enforced");
        if n == 0 {
            return Self::zeros(0, 0);
        }
        if n == 1 {
            // The adjugate of a 1x1 is [1] by convention: it is the empty
            // product, and it keeps `A * adj(A) = det(A) I` true.
            return Self::from_rows(vec![vec![ArithmaExpression::from_i64(1)]]);
        }
        let mut cells = Vec::with_capacity(n * n);
        // Transposed on write: `adj[i][j]` is the cofactor at `(j, i)`.
        for i in 0..n {
            for j in 0..n {
                cells.push(self.cofactor(j, i)?);
            }
        }
        debug_assert_eq!(cells.len(), n * n, "adjugate lost cells");
        Ok(Self {
            rows: n,
            cols: n,
            cells,
        })
    }

    /// The inverse, as `adj(A) / det(A)`.
    ///
    /// Exact and symbolic. The adjugate route is chosen over Gauss-Jordan
    /// because elimination divides by pivots, and dividing symbolic entries
    /// introduces quotients that may not simplify away -- the result would be
    /// correct but unreadable. Here division happens once per entry, by a
    /// single shared determinant.
    ///
    /// Returns [`MatrixError::Singular`] when the determinant is exactly zero.
    /// A determinant that is merely *symbolic* is not treated as singular: it
    /// may be non-zero for the values the caller has in mind, so the quotient
    /// is returned unevaluated.
    pub fn inverse(&self) -> Result<Self, MatrixError> {
        let det = self.determinant()?;
        debug_assert!(self.is_square(), "determinant accepted a non-square matrix");
        if det.to_f64() == Some(0.0) {
            return Err(MatrixError::Singular);
        }
        // A 1x1 inverts to 1/a directly; the adjugate convention above makes
        // the general path agree, but this is clearer and avoids the detour.
        if self.rows == 1 {
            let entry = ArithmaExpression::div(ArithmaExpression::from_i64(1), det);
            return Self::from_rows(vec![vec![simplified_expression(entry)]]);
        }
        let adjugate = self.adjugate()?;
        debug_assert_eq!(adjugate.shape(), self.shape(), "adjugate changed the shape");
        let cells = adjugate
            .cells
            .into_iter()
            .map(|c| simplified_expression(ArithmaExpression::div(c, det.clone())))
            .collect::<Vec<_>>();
        debug_assert_eq!(cells.len(), self.rows * self.cols, "inverse lost cells");
        Ok(Self {
            rows: self.rows,
            cols: self.cols,
            cells,
        })
    }

    /// The characteristic polynomial `det(A - x I)`, as an expression in `var`.
    ///
    /// The roots of this are the eigenvalues. Building it symbolically and
    /// handing it to the numerical root finders is how eigenvalues are
    /// obtained here -- there is no separate eigen-decomposition, and this
    /// composes with machinery the crate already has.
    pub fn characteristic_polynomial(&self, var: &str) -> Result<ArithmaExpression, MatrixError> {
        if !self.is_square() {
            return Err(MatrixError::NotSquare {
                rows: self.rows,
                cols: self.cols,
            });
        }
        debug_assert!(!var.is_empty(), "the polynomial needs a variable name");
        let n = self.rows;
        debug_assert_eq!(self.cells.len(), n * n, "shape mismatch");
        let mut shifted = self.clone();
        for i in 0..n {
            let entry = &self.cells[i * n + i];
            shifted.cells[i * n + i] =
                ArithmaExpression::sub(entry.clone(), ArithmaExpression::var(var));
        }
        shifted.determinant()
    }

    /// Real eigenvalues, found numerically from the characteristic polynomial.
    ///
    /// The matrix must be numeric: every entry has to evaluate, because the
    /// roots are located by scanning for sign changes. A symbolic entry makes
    /// the polynomial unbounded in an unknown, and there is nothing to scan.
    ///
    /// # What this does and does not give you
    ///
    /// **Real roots only.** A rotation matrix has a complex spectrum and this
    /// reports none of it; that is a genuine limit, not a failure to converge.
    /// **Multiplicities are collapsed** -- a repeated eigenvalue appears once,
    /// because a double root does not change the sign of the polynomial and so
    /// cannot be bracketed. Both are consequences of locating roots by sign
    /// change, and both are stated rather than papered over.
    ///
    /// Eigenvectors are not computed.
    ///
    /// The search covers the Gershgorin disc bound, which is guaranteed to
    /// contain every eigenvalue, so no root is missed for being outside an
    /// arbitrary window.
    pub fn eigenvalues_real(&self) -> Result<Vec<f64>, MatrixError> {
        let poly = self.characteristic_polynomial(EIGEN_VAR)?;
        let n = self.rows;
        if n == 0 {
            return Ok(Vec::new());
        }
        debug_assert!(n <= MAX_DETERMINANT_ORDER, "order cap not enforced");
        let bound = self.gershgorin_bound()?;
        debug_assert!(bound.is_finite(), "Gershgorin bound must be finite");

        // Sample density scales with the bound so the scan resolves closely
        // spaced eigenvalues, but is capped so this stays a bounded loop.
        let steps = EIGEN_SCAN_STEPS;
        let lo = -bound - 1.0;
        let hi = bound + 1.0;
        let dx = (hi - lo) / steps as f64;
        let config = crate::numerical::root_finding::ArithmaRootFindingConfig::default();

        let mut roots: Vec<f64> = Vec::new();
        let mut prev: Option<(f64, f64)> = None;
        for i in 0..=steps {
            let x = lo + dx * i as f64;
            let y = match eval_poly(&poly, x) {
                Some(v) => v,
                None => {
                    prev = None;
                    continue;
                }
            };
            if y == 0.0 {
                push_root(&mut roots, x, dx);
            } else if let Some((px, py)) = prev {
                if py * y < 0.0 {
                    if let Ok(found) = crate::numerical::root_finding::find_root_bisection(
                        &poly, EIGEN_VAR, px, x, &config,
                    ) {
                        push_root(&mut roots, found.root, dx);
                    }
                }
            }
            prev = Some((x, y));
        }
        roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        debug_assert!(
            roots.len() <= n,
            "a degree-{n} polynomial has at most {n} roots"
        );
        Ok(roots)
    }

    /// Gershgorin bound: every eigenvalue lies within this distance of zero.
    ///
    /// `max_i (|a_ii| + sum_{j != i} |a_ij|)`. Using it as the scan window is
    /// what makes "no root was missed" a statement rather than a hope.
    fn gershgorin_bound(&self) -> Result<f64, MatrixError> {
        if !self.is_square() {
            return Err(MatrixError::NotSquare {
                rows: self.rows,
                cols: self.cols,
            });
        }
        let n = self.rows;
        let mut bound: f64 = 0.0;
        for i in 0..n {
            let mut row_sum = 0.0_f64;
            for j in 0..n {
                let cell = &self.cells[i * n + j];
                let value = cell
                    .to_f64()
                    .ok_or(MatrixError::NotNumeric { row: i, col: j })?;
                if !value.is_finite() {
                    return Err(MatrixError::NotNumeric { row: i, col: j });
                }
                row_sum += value.abs();
            }
            bound = bound.max(row_sum);
        }
        debug_assert!(bound >= 0.0, "an absolute row sum cannot be negative");
        debug_assert!(
            bound.is_finite(),
            "bound must be finite for a numeric matrix"
        );
        Ok(bound)
    }

    /// A copy with every cell simplified.
    ///
    /// [`Self::matmul`] and friends build unsimplified sums of products, which
    /// is correct but verbose; this folds them.
    pub fn simplified(&self) -> Self {
        debug_assert_eq!(self.cells.len(), self.rows * self.cols, "shape mismatch");
        let cells = self
            .cells
            .iter()
            .map(|c| simplified_expression(c.clone()))
            .collect::<Vec<_>>();
        debug_assert_eq!(cells.len(), self.cells.len(), "simplify lost cells");
        Self {
            rows: self.rows,
            cols: self.cols,
            cells,
        }
    }
}

/// Variable name used for the characteristic polynomial in eigenvalue search.
///
/// Deliberately unlikely to collide with a caller's own symbol appearing in a
/// matrix entry.
const EIGEN_VAR: &str = "__arithma_lambda";

/// Samples taken across the Gershgorin window when scanning for sign changes.
///
/// Safety-critical standard 2: a fixed bound. Dense enough to separate close
/// eigenvalues at the orders this crate supports.
const EIGEN_SCAN_STEPS: usize = 4096;

/// Evaluate the characteristic polynomial at `x`, or `None` if it does not
/// produce a finite value there.
fn eval_poly(poly: &ArithmaExpression, x: f64) -> Option<f64> {
    use crate::expression::Evaluable;
    debug_assert!(x.is_finite(), "scan produced a non-finite sample point");
    let mut bindings = std::collections::HashMap::with_capacity(1);
    bindings.insert(EIGEN_VAR.to_string(), x);
    let value = poly.evaluate(&bindings).ok()?;
    debug_assert!(!EIGEN_VAR.is_empty(), "binding key must not be empty");
    if value.is_finite() {
        Some(value)
    } else {
        None
    }
}

/// Record a root unless an equal one is already present.
///
/// A double root is not merely deduplicated here -- it is never found in the
/// first place, because it does not change the polynomial's sign. This guards
/// against the same simple root being reported twice by adjacent brackets.
fn push_root(roots: &mut Vec<f64>, x: f64, spacing: f64) {
    debug_assert!(spacing > 0.0, "scan spacing must be positive");
    let tol = (spacing.abs() * 0.5).max(1e-9);
    if !roots.iter().any(|r| (r - x).abs() <= tol) {
        roots.push(x);
    }
    debug_assert!(!roots.is_empty(), "push_root left the list empty");
}

/// `n!`, saturating. Used only as a loop bound, never as a result.
fn factorial_bounded(n: usize) -> usize {
    debug_assert!(n <= MAX_DETERMINANT_ORDER, "factorial called past the cap");
    let mut acc: usize = 1;
    for k in 1..=n.min(MAX_DETERMINANT_ORDER) {
        acc = acc.saturating_mul(k);
    }
    debug_assert!(acc >= 1, "factorial produced zero");
    acc
}

/// True when a permutation has odd parity, by counting inversions.
fn permutation_sign_is_negative(perm: &[usize]) -> bool {
    debug_assert!(perm.len() <= MAX_DETERMINANT_ORDER, "permutation too long");
    let mut inversions: usize = 0;
    for i in 0..perm.len() {
        for j in (i + 1)..perm.len() {
            if perm[i] > perm[j] {
                inversions += 1;
            }
        }
    }
    debug_assert!(
        inversions <= perm.len() * perm.len(),
        "inversion count wrong"
    );
    inversions % 2 == 1
}

/// Advance `perm` to the next permutation in lexicographic order.
///
/// Returns `false` when `perm` is the last one. Iterative by construction, so
/// it costs no stack depth regardless of length.
fn next_permutation(perm: &mut [usize]) -> bool {
    debug_assert!(perm.len() <= MAX_DETERMINANT_ORDER, "permutation too long");
    if perm.len() < 2 {
        return false;
    }
    // Find the rightmost ascent.
    let mut i = perm.len() - 1;
    while i > 0 && perm[i - 1] >= perm[i] {
        i -= 1;
    }
    if i == 0 {
        return false;
    }
    // Find the rightmost element greater than the pivot, and swap.
    let pivot = i - 1;
    let mut j = perm.len() - 1;
    while j > pivot && perm[j] <= perm[pivot] {
        j -= 1;
    }
    debug_assert!(j > pivot, "no successor found for the pivot");
    perm.swap(pivot, j);
    perm[i..].reverse();
    true
}

/// Simplify one expression with the default policy.
fn simplified_expression(mut expr: ArithmaExpression) -> ArithmaExpression {
    let config = SimplificationConfig::default();
    let _ = simplify_iterative(&mut expr, &config);
    expr
}

// ---------------------------------------------------------------------------
// Backward-compatibility aliases for the pre-rename `Arithmos*` names.
// Retained for one release; downstream (eml-math, eml-spectral, metaphysica,
// periodica) should migrate to the `Arithma*` names above.
// ---------------------------------------------------------------------------
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaMatrix`")]
#[allow(unused)]
pub use self::ArithmaMatrix as ArithmosMatrix;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_with_correct_count_works() {
        let m = ArithmaMatrix::new(0, 0, vec![]);
        assert!(m.is_empty());
    }

    #[test]
    #[should_panic]
    fn new_with_wrong_count_panics() {
        let _ = ArithmaMatrix::new(2, 2, vec![]);
    }
    // ─── algebra ───────────────────────────────────────────────────────────
    //
    // `ArithmaMatrix` was a container: a shape and a flat cell list, with no
    // operation that combined two of them. These pin the algebra.

    fn n(v: i64) -> ArithmaExpression {
        ArithmaExpression::from_i64(v)
    }

    /// Build from a nested integer literal list.
    fn mat(rows: &[&[i64]]) -> ArithmaMatrix {
        ArithmaMatrix::from_rows(
            rows.iter()
                .map(|r| r.iter().map(|v| n(*v)).collect::<Vec<_>>())
                .collect(),
        )
        .expect("literal matrix must be well-formed")
    }

    /// Read every cell as an f64, for comparison against an expected grid.
    fn values(m: &ArithmaMatrix) -> Vec<f64> {
        m.simplified()
            .cells
            .iter()
            .map(|c| c.to_f64().expect("numeric matrix cell must evaluate"))
            .collect()
    }

    #[test]
    fn identity_has_ones_on_the_diagonal() {
        let i3 = ArithmaMatrix::identity(3).expect("3x3 identity");
        assert_eq!(i3.shape(), (3, 3));
        assert!(i3.is_square());
        assert_eq!(
            values(&i3),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn get_and_set_are_bounds_checked() {
        let mut m = ArithmaMatrix::zeros(2, 3).expect("2x3");
        m.set(1, 2, n(7)).expect("in-bounds set");
        assert_eq!(m.get(1, 2).expect("in-bounds get").to_f64(), Some(7.0));
        // Out of bounds must be an error, not a panic and not a wrap onto the
        // next row -- (0, 3) and (1, 0) share a flat offset without the check.
        assert_eq!(
            m.get(0, 3).unwrap_err(),
            MatrixError::OutOfBounds { row: 0, col: 3 }
        );
        assert!(m.set(2, 0, n(1)).is_err());
    }

    #[test]
    fn transpose_swaps_the_axes() {
        let m = mat(&[&[1, 2, 3], &[4, 5, 6]]);
        let t = m.transpose();
        assert_eq!(t.shape(), (3, 2));
        assert_eq!(values(&t), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        // Transposing twice is the identity.
        assert_eq!(values(&t.transpose()), values(&m));
    }

    #[test]
    fn addition_requires_matching_shapes() {
        let a = mat(&[&[1, 2], &[3, 4]]);
        let b = mat(&[&[10, 20], &[30, 40]]);
        assert_eq!(
            values(&a.add(&b).expect("same shape")),
            vec![11.0, 22.0, 33.0, 44.0]
        );
        assert_eq!(
            values(&b.sub(&a).expect("same shape")),
            vec![9.0, 18.0, 27.0, 36.0]
        );
        let wrong = mat(&[&[1, 2, 3]]);
        assert_eq!(
            a.add(&wrong).unwrap_err(),
            MatrixError::DimensionMismatch {
                left: (2, 2),
                right: (1, 3)
            }
        );
    }

    #[test]
    fn scalar_multiplication_scales_every_cell() {
        let a = mat(&[&[1, 2], &[3, 4]]);
        assert_eq!(values(&a.scalar_mul(&n(3))), vec![3.0, 6.0, 9.0, 12.0]);
    }

    #[test]
    fn matrix_product_is_row_by_column() {
        let a = mat(&[&[1, 2], &[3, 4]]);
        let b = mat(&[&[5, 6], &[7, 8]]);
        // [1 2][5 6]   [19 22]
        // [3 4][7 8] = [43 50]
        assert_eq!(
            values(&a.matmul(&b).expect("2x2 by 2x2")),
            vec![19.0, 22.0, 43.0, 50.0]
        );
    }

    #[test]
    fn matrix_product_is_not_commutative() {
        let a = mat(&[&[1, 2], &[3, 4]]);
        let b = mat(&[&[5, 6], &[7, 8]]);
        assert_ne!(
            values(&a.matmul(&b).expect("ab")),
            values(&b.matmul(&a).expect("ba")),
            "a test that passed for a commutative product would not be testing much"
        );
    }

    #[test]
    fn product_with_the_identity_is_the_original() {
        let a = mat(&[&[1, 2, 3], &[4, 5, 6]]);
        let i3 = ArithmaMatrix::identity(3).expect("i3");
        let i2 = ArithmaMatrix::identity(2).expect("i2");
        assert_eq!(values(&a.matmul(&i3).expect("a*I")), values(&a));
        assert_eq!(values(&i2.matmul(&a).expect("I*a")), values(&a));
    }

    #[test]
    fn product_checks_the_inner_dimension() {
        let a = mat(&[&[1, 2, 3]]);
        let b = mat(&[&[1, 2, 3]]);
        assert_eq!(
            a.matmul(&b).unwrap_err(),
            MatrixError::DimensionMismatch {
                left: (1, 3),
                right: (1, 3)
            }
        );
    }

    #[test]
    fn non_square_operations_are_refused() {
        let a = mat(&[&[1, 2, 3], &[4, 5, 6]]);
        assert_eq!(
            a.trace().unwrap_err(),
            MatrixError::NotSquare { rows: 2, cols: 3 }
        );
        assert!(a.determinant().is_err());
    }

    #[test]
    fn trace_sums_the_diagonal() {
        let a = mat(&[&[1, 2], &[3, 4]]);
        assert_eq!(a.trace().expect("square").to_f64(), Some(5.0));
        assert_eq!(
            ArithmaMatrix::identity(4)
                .expect("i4")
                .trace()
                .expect("square")
                .to_f64(),
            Some(4.0)
        );
    }

    #[test]
    fn determinant_of_small_matrices() {
        // 1x1
        assert_eq!(mat(&[&[7]]).determinant().expect("1x1").to_f64(), Some(7.0));
        // 2x2: 1*4 - 2*3 = -2
        assert_eq!(
            mat(&[&[1, 2], &[3, 4]])
                .determinant()
                .expect("2x2")
                .to_f64(),
            Some(-2.0)
        );
        // 3x3, a standard worked example.
        let m = mat(&[&[6, 1, 1], &[4, -2, 5], &[2, 8, 7]]);
        assert_eq!(m.determinant().expect("3x3").to_f64(), Some(-306.0));
    }

    #[test]
    fn determinant_of_the_identity_is_one() {
        for n in 1..=MAX_DETERMINANT_ORDER {
            let i = ArithmaMatrix::identity(n).expect("identity");
            assert_eq!(
                i.determinant().expect("square").to_f64(),
                Some(1.0),
                "det(I_{n}) must be 1"
            );
        }
    }

    #[test]
    fn determinant_of_a_singular_matrix_is_zero() {
        // The second row is twice the first, so the rows are dependent.
        let m = mat(&[&[1, 2, 3], &[2, 4, 6], &[7, 8, 9]]);
        assert_eq!(m.determinant().expect("3x3").to_f64(), Some(0.0));
    }

    #[test]
    fn determinant_is_multiplicative() {
        // det(AB) = det(A)det(B) -- an identity that a sign error breaks.
        let a = mat(&[&[2, 0, 1], &[3, -1, 2], &[1, 4, 0]]);
        let b = mat(&[&[1, 2, 0], &[0, 1, 3], &[2, 1, 1]]);
        let da = a.determinant().expect("det a").to_f64().expect("numeric");
        let db = b.determinant().expect("det b").to_f64().expect("numeric");
        let dab = a
            .matmul(&b)
            .expect("ab")
            .determinant()
            .expect("det ab")
            .to_f64()
            .expect("numeric");
        assert_eq!(dab, da * db);
    }

    #[test]
    fn determinant_is_refused_past_its_order_cap() {
        let big = ArithmaMatrix::identity(MAX_DETERMINANT_ORDER + 1).expect("identity");
        assert_eq!(
            big.determinant().unwrap_err(),
            MatrixError::TooLarge {
                order: MAX_DETERMINANT_ORDER + 1,
                limit: MAX_DETERMINANT_ORDER
            }
        );
    }

    #[test]
    fn symbolic_entries_survive() {
        // det [[x, 0], [0, 1]] = x. The entry must stay symbolic, not become 0.
        let m = ArithmaMatrix::from_rows(vec![
            vec![ArithmaExpression::var("x"), n(0)],
            vec![n(0), n(1)],
        ])
        .expect("2x2");
        let det = m.determinant().expect("square");
        assert_eq!(det.to_f64(), None, "det must remain symbolic");
        assert!(
            matches!(&det, ArithmaExpression::Variable(v) if v == "x"),
            "det should reduce to x, got {det:?}"
        );
    }

    #[test]
    fn ragged_rows_are_rejected() {
        let ragged = ArithmaMatrix::from_rows(vec![vec![n(1), n(2)], vec![n(3)]]);
        assert!(ragged.is_err(), "rows of differing length must not build");
    }

    #[test]
    fn oversized_matrices_are_refused() {
        assert!(ArithmaMatrix::zeros(MAX_DIMENSION + 1, 1).is_err());
        assert!(ArithmaMatrix::zeros(1, MAX_DIMENSION + 1).is_err());
    }
    // ─── inverse and characteristic polynomial ─────────────────────────────

    use crate::expression::Evaluable;

    /// Compare evaluated cells with a tolerance.
    ///
    /// Inverse entries are exact *symbolically* -- `6/10` is kept as a
    /// quotient because `checked_div_exact` refuses an inexact integer
    /// division -- but evaluating that quotient to `f64` rounds, and a product
    /// of rounded values drifts. Demanding bit-equality here would be
    /// asserting something false about floating point, not about the algebra.
    fn assert_close(got: Vec<f64>, expected: Vec<f64>) {
        assert_eq!(got.len(), expected.len(), "shape mismatch: {got:?}");
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-12, "expected {expected:?}, got {got:?}");
        }
    }

    #[test]
    fn minors_delete_the_right_row_and_column() {
        let m = mat(&[&[1, 2, 3], &[4, 5, 6], &[7, 8, 9]]);
        let minor = m.minor(1, 1).expect("square");
        assert_eq!(minor.shape(), (2, 2));
        assert_eq!(values(&minor), vec![1.0, 3.0, 7.0, 9.0]);
        assert!(m.minor(3, 0).is_err(), "out-of-range row must be an error");
    }

    #[test]
    fn cofactors_carry_the_checkerboard_sign() {
        let m = mat(&[&[1, 2, 3], &[4, 5, 6], &[7, 8, 10]]);
        // C(0,0) = det[[5,6],[8,10]] = 50 - 48 = 2, sign +.
        assert_eq!(m.cofactor(0, 0).expect("square").to_f64(), Some(2.0));
        // C(0,1) = -det[[4,6],[7,10]] = -(40 - 42) = 2, sign -.
        assert_eq!(m.cofactor(0, 1).expect("square").to_f64(), Some(2.0));
        // C(1,0) = -det[[2,3],[8,10]] = -(20 - 24) = 4.
        assert_eq!(m.cofactor(1, 0).expect("square").to_f64(), Some(4.0));
    }

    #[test]
    fn a_matrix_times_its_adjugate_is_the_determinant_times_identity() {
        // The defining identity. A sign or transpose error breaks it.
        let m = mat(&[&[1, 2, 3], &[4, 5, 6], &[7, 8, 10]]);
        let det = m.determinant().expect("square").to_f64().expect("numeric");
        let product = m
            .matmul(&m.adjugate().expect("square"))
            .expect("same order");
        let expected = vec![det, 0.0, 0.0, 0.0, det, 0.0, 0.0, 0.0, det];
        assert_eq!(values(&product), expected);
    }

    #[test]
    fn a_matrix_times_its_inverse_is_the_identity() {
        let m = mat(&[&[4, 7], &[2, 6]]);
        let inv = m.inverse().expect("non-singular");
        let product = m.matmul(&inv).expect("same order");
        assert_close(values(&product), vec![1.0, 0.0, 0.0, 1.0]);
        // And on the other side, since matrix multiplication is not commutative.
        let other = inv.matmul(&m).expect("same order");
        assert_close(values(&other), vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn inverse_of_a_three_by_three_round_trips() {
        let m = mat(&[&[1, 2, 3], &[4, 5, 6], &[7, 8, 10]]);
        let product = m
            .matmul(&m.inverse().expect("non-singular"))
            .expect("same order");
        assert_close(
            values(&product),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        );
    }

    #[test]
    fn inexact_inverse_entries_stay_symbolic_quotients() {
        // det = 10, so the entries are tenths. `checked_div_exact` refuses an
        // inexact integer division, so each entry is kept as a quotient rather
        // than collapsed to a rounded literal -- the value is only rounded at
        // the point a caller asks for an f64.
        let m = mat(&[&[4, 7], &[2, 6]]);
        let inv = m.inverse().expect("non-singular");
        assert_close(values(&inv), vec![0.6, -0.7, -0.2, 0.4]);
        assert!(
            inv.cells
                .iter()
                .any(|c| matches!(c, ArithmaExpression::Function(..))),
            "an inexact entry must remain a quotient, not become a literal"
        );
    }

    #[test]
    fn exactly_divisible_inverse_entries_do_collapse_to_literals() {
        // det = 1 here, so every entry divides exactly and the quotients fold
        // away. This is the other half of the rule above.
        let m = mat(&[&[2, 1], &[1, 1]]);
        assert_eq!(m.determinant().expect("square").to_f64(), Some(1.0));
        let inv = m.inverse().expect("non-singular");
        assert_eq!(values(&inv), vec![1.0, -1.0, -1.0, 2.0]);
        assert!(
            inv.cells
                .iter()
                .all(|c| matches!(c, ArithmaExpression::Number(_))),
            "with det = 1 every entry should fold to a literal, got {:?}",
            inv.cells
        );
    }

    #[test]
    fn a_singular_matrix_has_no_inverse() {
        // Second row is twice the first.
        let m = mat(&[&[1, 2], &[2, 4]]);
        assert_eq!(m.determinant().expect("square").to_f64(), Some(0.0));
        assert_eq!(m.inverse().unwrap_err(), MatrixError::Singular);
    }

    #[test]
    fn the_identity_is_its_own_inverse() {
        for n in 1..=4 {
            let i = ArithmaMatrix::identity(n).expect("identity");
            assert_close(values(&i.inverse().expect("non-singular")), values(&i));
        }
    }

    #[test]
    fn non_square_matrices_have_no_inverse() {
        let m = mat(&[&[1, 2, 3], &[4, 5, 6]]);
        assert!(m.inverse().is_err());
        assert!(m.adjugate().is_err());
        assert!(m.minor(0, 0).is_err());
    }

    #[test]
    fn a_symbolic_determinant_is_not_treated_as_singular() {
        // det [[x, 0], [0, 1]] = x, which is unknown, not zero. Refusing to
        // invert would be wrong: the caller may well have a non-zero x.
        let m = ArithmaMatrix::from_rows(vec![
            vec![ArithmaExpression::var("x"), n(0)],
            vec![n(0), n(1)],
        ])
        .expect("2x2");
        let inv = m
            .inverse()
            .expect("an unknown determinant is not a zero one");
        assert_eq!(inv.shape(), (2, 2));
    }

    #[test]
    fn the_characteristic_polynomial_has_the_eigenvalues_as_roots() {
        // [[2, 0], [0, 3]] has eigenvalues 2 and 3, so p(2) = p(3) = 0.
        let m = mat(&[&[2, 0], &[0, 3]]);
        let p = m.characteristic_polynomial("L").expect("square");
        for (lambda, expected) in [(2.0, 0.0), (3.0, 0.0)] {
            let mut bindings = std::collections::HashMap::new();
            bindings.insert("L".to_string(), lambda);
            let value = p.evaluate(&bindings).expect("polynomial evaluates");
            assert!(
                (value - expected).abs() < 1e-12,
                "p({lambda}) should be 0, got {value}"
            );
        }
        // And a non-eigenvalue must not vanish.
        let mut bindings = std::collections::HashMap::new();
        bindings.insert("L".to_string(), 5.0);
        let value = p.evaluate(&bindings).expect("polynomial evaluates");
        assert!(
            value.abs() > 1e-9,
            "5 is not an eigenvalue, got p(5) = {value}"
        );
    }

    #[test]
    fn the_characteristic_polynomial_at_zero_is_the_determinant() {
        // p(0) = det(A - 0*I) = det(A). A cheap check that the shift is right.
        let m = mat(&[&[1, 2, 3], &[4, 5, 6], &[7, 8, 10]]);
        let p = m.characteristic_polynomial("L").expect("square");
        let mut bindings = std::collections::HashMap::new();
        bindings.insert("L".to_string(), 0.0);
        let at_zero = p.evaluate(&bindings).expect("evaluates");
        let det = m.determinant().expect("square").to_f64().expect("numeric");
        assert!(
            (at_zero - det).abs() < 1e-9,
            "p(0) = {at_zero}, det = {det}"
        );
    }
    // ─── eigenvalues ───────────────────────────────────────────────────────

    #[test]
    fn a_diagonal_matrix_has_its_diagonal_as_eigenvalues() {
        let m = mat(&[&[2, 0, 0], &[0, 3, 0], &[0, 0, 7]]);
        assert_close(m.eigenvalues_real().expect("numeric"), vec![2.0, 3.0, 7.0]);
    }

    #[test]
    fn a_triangular_matrix_has_its_diagonal_as_eigenvalues() {
        // The off-diagonal entries must not move the spectrum.
        let m = mat(&[&[1, 5, 9], &[0, 4, 6], &[0, 0, 8]]);
        assert_close(m.eigenvalues_real().expect("numeric"), vec![1.0, 4.0, 8.0]);
    }

    #[test]
    fn a_symmetric_matrix_gives_the_known_pair() {
        // [[2, 1], [1, 2]] has eigenvalues 1 and 3.
        let m = mat(&[&[2, 1], &[1, 2]]);
        assert_close(m.eigenvalues_real().expect("numeric"), vec![1.0, 3.0]);
    }

    #[test]
    fn eigenvalues_satisfy_the_characteristic_polynomial() {
        // The defining property, checked independently of how they were found.
        let m = mat(&[&[4, 1, 0], &[1, 3, 1], &[0, 1, 2]]);
        let poly = m.characteristic_polynomial("L").expect("square");
        for lambda in m.eigenvalues_real().expect("numeric") {
            let mut bindings = std::collections::HashMap::new();
            bindings.insert("L".to_string(), lambda);
            let value = poly.evaluate(&bindings).expect("evaluates");
            assert!(value.abs() < 1e-6, "p({lambda}) should vanish, got {value}");
        }
    }

    #[test]
    fn the_trace_is_the_sum_of_the_eigenvalues() {
        // True whenever the spectrum is entirely real and simple, which it is
        // for this symmetric matrix. A missed root would break it.
        let m = mat(&[&[4, 1, 0], &[1, 3, 1], &[0, 1, 2]]);
        let eigenvalues = m.eigenvalues_real().expect("numeric");
        assert_eq!(eigenvalues.len(), 3, "expected three real eigenvalues");
        let sum: f64 = eigenvalues.iter().sum();
        let trace = m.trace().expect("square").to_f64().expect("numeric");
        assert!((sum - trace).abs() < 1e-6, "sum {sum} vs trace {trace}");
    }

    #[test]
    fn a_rotation_has_no_real_eigenvalues() {
        // [[0, -1], [1, 0]] rotates by 90 degrees; its spectrum is +/- i.
        // Reporting none is correct, and is a limit worth pinning rather than
        // a convergence failure.
        let m = mat(&[&[0, -1], &[1, 0]]);
        assert!(
            m.eigenvalues_real().expect("numeric").is_empty(),
            "a rotation has no real eigenvalues"
        );
    }

    #[test]
    fn a_symbolic_matrix_cannot_have_numeric_eigenvalues() {
        let m = ArithmaMatrix::from_rows(vec![
            vec![ArithmaExpression::var("x"), n(0)],
            vec![n(0), n(1)],
        ])
        .expect("2x2");
        assert_eq!(
            m.eigenvalues_real().unwrap_err(),
            MatrixError::NotNumeric { row: 0, col: 0 },
            "a free variable leaves nothing to scan"
        );
    }
}
