//====== Arithma/rust/arithma_core/src/tensor.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! N-dimensional symbolic tensor.
//!
//! Cells are `ArithmaExpression` in row-major order. Indexing is checked per
//! axis, not just against the flat length: for shape `[2, 3]`, the index
//! `(0, 3)` has a valid flat offset (it is the cell `(1, 0)`), so a flat-only
//! check would silently return the wrong cell.
//!
//! Shape errors are [`TensorError`] values rather than panics.

use crate::expression::iterative::simplify_iterative;
use crate::expression::{ArithmaExpression, SimplificationConfig};

/// A general N-dimensional tensor of `ArithmaExpression` cells.
#[derive(Debug, Clone)]
pub struct ArithmaTensor {
    pub shape: Vec<usize>,
    pub cells: Vec<ArithmaExpression>,
}

impl ArithmaTensor {
    /// Total cell count = ∏(shape).
    pub fn cell_count(shape: &[usize]) -> usize {
        shape.iter().product()
    }

    /// Construct a tensor; panics if `cells.len()` ≠ ∏(shape).
    pub fn new(shape: Vec<usize>, cells: Vec<ArithmaExpression>) -> Self {
        assert_eq!(
            cells.len(),
            Self::cell_count(&shape),
            "cell count must equal ∏(shape)"
        );
        Self { shape, cells }
    }
}

/// Largest rank (number of axes) accepted.
pub const MAX_RANK: usize = 16;

/// Largest total cell count accepted.
///
/// Safety-critical standard 2 and 3: shape is caller-supplied, and
/// `[1000, 1000, 1000]` would otherwise ask for a billion cells. Refusing is
/// the honest answer.
pub const MAX_CELLS: usize = 1 << 22;

/// Why a tensor operation could not be performed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorError {
    /// Operand shapes are incompatible.
    ShapeMismatch { left: Vec<usize>, right: Vec<usize> },
    /// An index had the wrong number of components, or was out of range.
    BadIndex {
        index: Vec<usize>,
        shape: Vec<usize>,
    },
    /// A reshape did not preserve the cell count.
    ReshapeMismatch { from: usize, to: usize },
    /// An axis permutation was not a permutation of `0..rank`.
    BadPermutation { axes: Vec<usize>, rank: usize },
    /// A contraction axis list was malformed: an axis out of range for the
    /// tensor, an axis repeated, or the two lists of unequal length.
    BadAxes { axes: Vec<usize>, rank: usize },
    /// Two axes paired for contraction did not have the same extent.
    AxisMismatch {
        axis_a: usize,
        extent_a: usize,
        axis_b: usize,
        extent_b: usize,
    },
    /// The requested tensor exceeds [`MAX_RANK`] or [`MAX_CELLS`].
    TooLarge { cells: usize, rank: usize },
}

impl core::fmt::Display for TensorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ShapeMismatch { left, right } => {
                write!(f, "shape mismatch: {left:?} against {right:?}")
            }
            Self::BadIndex { index, shape } => {
                write!(f, "index {index:?} is not valid for shape {shape:?}")
            }
            Self::ReshapeMismatch { from, to } => {
                write!(f, "reshape would change the cell count from {from} to {to}")
            }
            Self::BadPermutation { axes, rank } => {
                write!(f, "{axes:?} is not a permutation of 0..{rank}")
            }
            Self::BadAxes { axes, rank } => {
                write!(
                    f,
                    "{axes:?} are not valid contraction axes for a rank-{rank} tensor"
                )
            }
            Self::AxisMismatch {
                axis_a,
                extent_a,
                axis_b,
                extent_b,
            } => {
                write!(
                    f,
                    "contracted axes {axis_a} (extent {extent_a}) and {axis_b} \
                     (extent {extent_b}) differ in extent"
                )
            }
            Self::TooLarge { cells, rank } => {
                write!(
                    f,
                    "tensor of rank {rank} with {cells} cells exceeds the limits \
                     (rank {MAX_RANK}, cells {MAX_CELLS})"
                )
            }
        }
    }
}

impl std::error::Error for TensorError {}

impl ArithmaTensor {
    /// A tensor of zeros with the given shape.
    pub fn zeros(shape: Vec<usize>) -> Result<Self, TensorError> {
        let cells = Self::checked_cell_count(&shape)?;
        debug_assert!(cells <= MAX_CELLS, "cell cap not enforced");
        debug_assert!(shape.len() <= MAX_RANK, "rank cap not enforced");
        Ok(Self {
            shape,
            cells: vec![ArithmaExpression::zero(); cells],
        })
    }

    /// Cell count for a shape, refusing anything past the caps.
    ///
    /// Uses checked multiplication: an unchecked product of caller-supplied
    /// dimensions can wrap to a small number and pass a naive length check.
    fn checked_cell_count(shape: &[usize]) -> Result<usize, TensorError> {
        if shape.len() > MAX_RANK {
            return Err(TensorError::TooLarge {
                cells: 0,
                rank: shape.len(),
            });
        }
        let mut acc: usize = 1;
        for dim in shape.iter() {
            acc = acc.checked_mul(*dim).ok_or(TensorError::TooLarge {
                cells: usize::MAX,
                rank: shape.len(),
            })?;
            if acc > MAX_CELLS {
                return Err(TensorError::TooLarge {
                    cells: acc,
                    rank: shape.len(),
                });
            }
        }
        debug_assert!(acc <= MAX_CELLS, "cell cap not enforced");
        Ok(acc)
    }

    /// Number of axes.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// The shape as a slice.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Row-major strides: the flat step per unit step along each axis.
    pub fn strides(&self) -> Vec<usize> {
        debug_assert!(self.shape.len() <= MAX_RANK, "rank past the cap");
        let mut strides = vec![1usize; self.shape.len()];
        let mut acc: usize = 1;
        let mut axis = self.shape.len();
        while axis > 0 {
            axis -= 1;
            strides[axis] = acc;
            acc = acc.saturating_mul(self.shape[axis]);
        }
        debug_assert_eq!(strides.len(), self.shape.len(), "stride arity wrong");
        strides
    }

    /// Flat offset of a multi-dimensional index, row-major.
    pub fn offset(&self, index: &[usize]) -> Result<usize, TensorError> {
        if index.len() != self.shape.len() {
            return Err(TensorError::BadIndex {
                index: index.to_vec(),
                shape: self.shape.clone(),
            });
        }
        let mut flat: usize = 0;
        for (axis, &i) in index.iter().enumerate() {
            if i >= self.shape[axis] {
                return Err(TensorError::BadIndex {
                    index: index.to_vec(),
                    shape: self.shape.clone(),
                });
            }
            flat = flat * self.shape[axis] + i;
        }
        debug_assert!(flat < self.cells.len().max(1), "offset escaped the buffer");
        Ok(flat)
    }

    /// Borrow the cell at a multi-dimensional index.
    pub fn get(&self, index: &[usize]) -> Result<&ArithmaExpression, TensorError> {
        let flat = self.offset(index)?;
        self.cells.get(flat).ok_or_else(|| TensorError::BadIndex {
            index: index.to_vec(),
            shape: self.shape.clone(),
        })
    }

    /// Replace the cell at a multi-dimensional index.
    pub fn set(&mut self, index: &[usize], value: ArithmaExpression) -> Result<(), TensorError> {
        let flat = self.offset(index)?;
        debug_assert!(flat < self.cells.len(), "offset escaped the buffer");
        self.cells[flat] = value;
        Ok(())
    }

    /// Reinterpret the cells under a new shape, preserving row-major order.
    pub fn reshape(&self, shape: Vec<usize>) -> Result<Self, TensorError> {
        let wanted = Self::checked_cell_count(&shape)?;
        if wanted != self.cells.len() {
            return Err(TensorError::ReshapeMismatch {
                from: self.cells.len(),
                to: wanted,
            });
        }
        debug_assert_eq!(wanted, self.cells.len(), "reshape count check failed");
        Ok(Self {
            shape,
            cells: self.cells.clone(),
        })
    }

    /// Reorder the axes. `axes` must be a permutation of `0..rank`.
    ///
    /// The rank-2 case is the matrix transpose.
    pub fn permute_axes(&self, axes: &[usize]) -> Result<Self, TensorError> {
        let rank = self.rank();
        if axes.len() != rank || !is_permutation(axes, rank) {
            return Err(TensorError::BadPermutation {
                axes: axes.to_vec(),
                rank,
            });
        }
        debug_assert_eq!(axes.len(), rank, "permutation arity check failed");
        let new_shape: Vec<usize> = axes.iter().map(|&a| self.shape[a]).collect();
        let old_strides = self.strides();
        let mut cells: Vec<ArithmaExpression> = Vec::with_capacity(self.cells.len());
        // Walk the output in row-major order, mapping each position back to
        // its source offset. An explicit odometer keeps this iterative.
        let mut counter = vec![0usize; rank];
        for _ in 0..self.cells.len() {
            let mut source: usize = 0;
            for (out_axis, &src_axis) in axes.iter().enumerate() {
                source += counter[out_axis] * old_strides[src_axis];
            }
            cells.push(self.cells[source].clone());
            advance_odometer(&mut counter, &new_shape);
        }
        debug_assert_eq!(cells.len(), self.cells.len(), "permute lost cells");
        Ok(Self {
            shape: new_shape,
            cells,
        })
    }

    /// Elementwise combination of two identically-shaped tensors.
    fn zip_with(
        &self,
        other: &Self,
        combine: fn(ArithmaExpression, ArithmaExpression) -> ArithmaExpression,
    ) -> Result<Self, TensorError> {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch {
                left: self.shape.clone(),
                right: other.shape.clone(),
            });
        }
        debug_assert_eq!(self.cells.len(), other.cells.len(), "shape check failed");
        let mut cells = Vec::with_capacity(self.cells.len());
        for i in 0..self.cells.len() {
            cells.push(combine(self.cells[i].clone(), other.cells[i].clone()));
        }
        debug_assert_eq!(cells.len(), self.cells.len(), "zip lost cells");
        Ok(Self {
            shape: self.shape.clone(),
            cells,
        })
    }

    /// Elementwise addition. Shapes must match exactly.
    pub fn add(&self, other: &Self) -> Result<Self, TensorError> {
        self.zip_with(other, ArithmaExpression::add)
    }

    /// Elementwise subtraction. Shapes must match exactly.
    pub fn sub(&self, other: &Self) -> Result<Self, TensorError> {
        self.zip_with(other, ArithmaExpression::sub)
    }

    /// Elementwise product (Hadamard). Shapes must match exactly.
    ///
    /// Not a tensor contraction -- see [`Self::tensordot`] for that.
    pub fn hadamard(&self, other: &Self) -> Result<Self, TensorError> {
        self.zip_with(other, ArithmaExpression::mul)
    }

    /// Contract two axes of this tensor against each other.
    ///
    /// The axes must be distinct and of equal extent. Both are consumed, so
    /// the rank falls by two and the surviving axes keep their relative order.
    /// For a rank-2 tensor with axes `(0, 1)` this is the matrix trace, and
    /// the result is a rank-0 tensor holding one cell.
    ///
    /// Cells come back as unsimplified sums, as in `ArithmaMatrix::matmul`;
    /// call [`Self::simplified`] to fold them.
    pub fn trace(&self, axis_a: usize, axis_b: usize) -> Result<Self, TensorError> {
        let rank = self.rank();
        let pair = [axis_a, axis_b];
        if !axes_are_distinct_and_in_range(&pair, rank) {
            return Err(TensorError::BadAxes {
                axes: pair.to_vec(),
                rank,
            });
        }
        if self.shape[axis_a] != self.shape[axis_b] {
            return Err(TensorError::AxisMismatch {
                axis_a,
                extent_a: self.shape[axis_a],
                axis_b,
                extent_b: self.shape[axis_b],
            });
        }
        debug_assert!(axis_a < rank && axis_b < rank, "axis range check failed");
        debug_assert_eq!(
            self.shape[axis_a], self.shape[axis_b],
            "extent check failed"
        );

        let kept = free_axes(rank, &pair);
        let out_shape: Vec<usize> = kept.iter().map(|&a| self.shape[a]).collect();
        // Through the checked helper like every other allocation here, even
        // though a trace can only shrink the tensor.
        let out_count = Self::checked_cell_count(&out_shape)?;
        debug_assert_eq!(out_shape.len() + 2, rank, "trace kept the wrong axes");

        let strides = self.strides();
        // One step along the diagonal moves along both axes at once, so their
        // strides add.
        let diagonal_step = strides[axis_a] + strides[axis_b];
        let extent = self.shape[axis_a];

        let mut cells = Vec::with_capacity(out_count);
        let mut counter = vec![0usize; kept.len()];
        for _ in 0..out_count {
            let mut base: usize = 0;
            for (slot, &axis) in kept.iter().enumerate() {
                base += counter[slot] * strides[axis];
            }
            let mut acc: Option<ArithmaExpression> = None;
            for step in 0..extent {
                let cell = self.cells[base + step * diagonal_step].clone();
                acc = Some(match acc {
                    None => cell,
                    Some(sum) => ArithmaExpression::add(sum, cell),
                });
            }
            // An extent of zero leaves an empty sum, which is 0.
            cells.push(acc.unwrap_or_else(ArithmaExpression::zero));
            advance_odometer(&mut counter, &out_shape);
        }
        debug_assert_eq!(cells.len(), out_count, "trace lost cells");
        Ok(Self {
            shape: out_shape,
            cells,
        })
    }

    /// Generalised contraction over paired axes -- the `tensordot` of numpy.
    ///
    /// Sums the product of the two tensors over each pair
    /// `axes_a[i]`/`axes_b[i]`, which must agree in extent. The result keeps
    /// the free axes of `self` followed by the free axes of `other`, each in
    /// ascending order. With `axes_a = [1]`, `axes_b = [0]` on two rank-2
    /// tensors this is exactly matrix multiplication; with both lists empty it
    /// is the outer product, which [`Self::outer`] names.
    ///
    /// Cells come back as unsimplified sums of products, as in
    /// `ArithmaMatrix::matmul`; call [`Self::simplified`] to fold them.
    pub fn tensordot(
        &self,
        other: &Self,
        axes_a: &[usize],
        axes_b: &[usize],
    ) -> Result<Self, TensorError> {
        let rank_a = self.rank();
        let rank_b = other.rank();
        if axes_a.len() != axes_b.len() {
            // Without a pairing there is nothing to check extents against.
            // Report the b list: it is the one out of step with a.
            return Err(TensorError::BadAxes {
                axes: axes_b.to_vec(),
                rank: rank_b,
            });
        }
        if !axes_are_distinct_and_in_range(axes_a, rank_a) {
            return Err(TensorError::BadAxes {
                axes: axes_a.to_vec(),
                rank: rank_a,
            });
        }
        if !axes_are_distinct_and_in_range(axes_b, rank_b) {
            return Err(TensorError::BadAxes {
                axes: axes_b.to_vec(),
                rank: rank_b,
            });
        }
        for (&a, &b) in axes_a.iter().zip(axes_b.iter()) {
            if self.shape[a] != other.shape[b] {
                return Err(TensorError::AxisMismatch {
                    axis_a: a,
                    extent_a: self.shape[a],
                    axis_b: b,
                    extent_b: other.shape[b],
                });
            }
        }
        debug_assert_eq!(axes_a.len(), axes_b.len(), "axis pairing check failed");
        debug_assert!(
            axes_a.len() <= rank_a && axes_b.len() <= rank_b,
            "axis range check failed"
        );

        let free_a = free_axes(rank_a, axes_a);
        let free_b = free_axes(rank_b, axes_b);
        let mut out_shape: Vec<usize> = Vec::with_capacity(free_a.len() + free_b.len());
        for &axis in free_a.iter() {
            out_shape.push(self.shape[axis]);
        }
        for &axis in free_b.iter() {
            out_shape.push(other.shape[axis]);
        }
        // The result can be far larger than either operand -- an outer product
        // multiplies the cell counts -- so it goes through the checked helper,
        // which also refuses a result past the rank cap.
        let out_count = Self::checked_cell_count(&out_shape)?;

        // The contracted extents are caller-chosen too, so the inner loop
        // bound is checked the same way rather than multiplied out blind.
        let con_shape: Vec<usize> = axes_a.iter().map(|&a| self.shape[a]).collect();
        let con_count = Self::checked_cell_count(&con_shape)?;

        let strides_a = self.strides();
        let strides_b = other.strides();
        let mut cells = Vec::with_capacity(out_count);
        let mut out_counter = vec![0usize; out_shape.len()];
        // Reused across output cells: a full odometer cycle ends back at all
        // zeros, so it needs no reset.
        let mut con_counter = vec![0usize; con_shape.len()];
        for _ in 0..out_count {
            // Split the output odometer: the leading slots index the free axes
            // of self, the trailing ones the free axes of other.
            let mut base_a: usize = 0;
            for (slot, &axis) in free_a.iter().enumerate() {
                base_a += out_counter[slot] * strides_a[axis];
            }
            let mut base_b: usize = 0;
            for (slot, &axis) in free_b.iter().enumerate() {
                base_b += out_counter[free_a.len() + slot] * strides_b[axis];
            }
            let mut acc: Option<ArithmaExpression> = None;
            for _ in 0..con_count {
                let mut offset_a = base_a;
                let mut offset_b = base_b;
                for (slot, &position) in con_counter.iter().enumerate() {
                    offset_a += position * strides_a[axes_a[slot]];
                    offset_b += position * strides_b[axes_b[slot]];
                }
                let term = ArithmaExpression::mul(
                    self.cells[offset_a].clone(),
                    other.cells[offset_b].clone(),
                );
                acc = Some(match acc {
                    None => term,
                    Some(sum) => ArithmaExpression::add(sum, term),
                });
                advance_odometer(&mut con_counter, &con_shape);
            }
            debug_assert!(
                con_counter.iter().all(|p| *p == 0),
                "inner odometer did not complete its cycle"
            );
            // A contracted extent of zero leaves an empty sum, which is 0.
            cells.push(acc.unwrap_or_else(ArithmaExpression::zero));
            advance_odometer(&mut out_counter, &out_shape);
        }
        debug_assert_eq!(cells.len(), out_count, "tensordot lost cells");
        Ok(Self {
            shape: out_shape,
            cells,
        })
    }

    /// Outer (tensor) product: the contraction over no axes at all.
    ///
    /// The shapes concatenate, so the ranks add and the cell counts multiply.
    pub fn outer(&self, other: &Self) -> Result<Self, TensorError> {
        let product = self.tensordot(other, &[], &[])?;
        debug_assert_eq!(
            product.rank(),
            self.rank() + other.rank(),
            "outer product changed the rank"
        );
        debug_assert_eq!(
            product.cells.len(),
            self.cells.len().saturating_mul(other.cells.len()),
            "outer product lost cells"
        );
        Ok(product)
    }

    /// Multiply every cell by a scalar expression.
    pub fn scalar_mul(&self, scalar: &ArithmaExpression) -> Self {
        debug_assert!(self.cells.len() <= MAX_CELLS, "cell cap exceeded");
        let cells = self
            .cells
            .iter()
            .map(|c| ArithmaExpression::mul(scalar.clone(), c.clone()))
            .collect::<Vec<_>>();
        debug_assert_eq!(cells.len(), self.cells.len(), "scalar_mul lost cells");
        Self {
            shape: self.shape.clone(),
            cells,
        }
    }

    /// A copy with every cell simplified.
    pub fn simplified(&self) -> Self {
        let config = SimplificationConfig::default();
        let cells = self
            .cells
            .iter()
            .map(|c| {
                let mut copy = c.clone();
                let _ = simplify_iterative(&mut copy, &config);
                copy
            })
            .collect::<Vec<_>>();
        debug_assert_eq!(cells.len(), self.cells.len(), "simplify lost cells");
        Self {
            shape: self.shape.clone(),
            cells,
        }
    }
}

/// True when `axes` contains each of `0..rank` exactly once.
fn is_permutation(axes: &[usize], rank: usize) -> bool {
    debug_assert!(rank <= MAX_RANK, "rank past the cap");
    if axes.len() != rank {
        return false;
    }
    let mut seen = vec![false; rank];
    for &a in axes.iter() {
        if a >= rank || seen[a] {
            return false;
        }
        seen[a] = true;
    }
    debug_assert!(
        seen.iter().all(|s| *s),
        "permutation check let a gap through"
    );
    true
}

/// True when every axis is inside `0..rank` and none of them repeats.
///
/// Unlike [`is_permutation`] the list may be shorter than the rank: a
/// contraction names only the axes it consumes.
fn axes_are_distinct_and_in_range(axes: &[usize], rank: usize) -> bool {
    debug_assert!(rank <= MAX_RANK, "rank past the cap");
    if axes.len() > rank {
        return false;
    }
    let mut seen = vec![false; rank];
    for &a in axes.iter() {
        if a >= rank || seen[a] {
            return false;
        }
        seen[a] = true;
    }
    debug_assert_eq!(
        seen.iter().filter(|s| **s).count(),
        axes.len(),
        "distinctness check let a repeat through"
    );
    true
}

/// The axes of `0..rank` absent from `used`, in ascending order.
fn free_axes(rank: usize, used: &[usize]) -> Vec<usize> {
    debug_assert!(rank <= MAX_RANK, "rank past the cap");
    debug_assert!(used.len() <= rank, "more contracted axes than axes");
    let mut free = Vec::with_capacity(rank.saturating_sub(used.len()));
    for axis in 0..rank {
        if !used.contains(&axis) {
            free.push(axis);
        }
    }
    free
}

/// Increment a row-major odometer in place, wrapping each axis at its extent.
fn advance_odometer(counter: &mut [usize], shape: &[usize]) {
    debug_assert_eq!(counter.len(), shape.len(), "odometer arity mismatch");
    let mut axis = counter.len();
    while axis > 0 {
        axis -= 1;
        counter[axis] += 1;
        if counter[axis] < shape[axis] {
            return;
        }
        counter[axis] = 0;
    }
}

// ---------------------------------------------------------------------------
// Backward-compatibility aliases for the pre-rename `Arithmos*` names.
// Retained for one release; downstream (eml-math, eml-spectral, metaphysica,
// periodica) should migrate to the `Arithma*` names above.
// ---------------------------------------------------------------------------
#[deprecated(since = "2.0.4", note = "renamed to `ArithmaTensor`")]
#[allow(unused)]
pub use self::ArithmaTensor as ArithmosTensor;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_count_zero_dimensional_is_one() {
        assert_eq!(ArithmaTensor::cell_count(&[]), 1);
    }

    #[test]
    fn cell_count_three_d() {
        assert_eq!(ArithmaTensor::cell_count(&[2, 3, 4]), 24);
    }
    // ─── indexing and algebra ──────────────────────────────────────────────

    fn n(v: i64) -> ArithmaExpression {
        ArithmaExpression::from_i64(v)
    }

    /// A tensor whose cells are 0, 1, 2, ... in row-major order.
    fn ramp(shape: Vec<usize>) -> ArithmaTensor {
        let count = ArithmaTensor::cell_count(&shape);
        ArithmaTensor::new(shape, (0..count as i64).map(n).collect())
    }

    fn values(t: &ArithmaTensor) -> Vec<f64> {
        t.simplified()
            .cells
            .iter()
            .map(|c| c.to_f64().expect("numeric cell must evaluate"))
            .collect()
    }

    #[test]
    fn strides_are_row_major() {
        let t = ramp(vec![2, 3, 4]);
        assert_eq!(t.rank(), 3);
        assert_eq!(t.strides(), vec![12, 4, 1]);
        assert_eq!(ramp(vec![5]).strides(), vec![1]);
    }

    #[test]
    fn indexing_matches_row_major_order() {
        let t = ramp(vec![2, 3]);
        // Cell (1, 2) is flat offset 1*3 + 2 = 5.
        assert_eq!(t.offset(&[1, 2]).expect("in range"), 5);
        assert_eq!(t.get(&[1, 2]).expect("in range").to_f64(), Some(5.0));
        assert_eq!(t.get(&[0, 0]).expect("in range").to_f64(), Some(0.0));
    }

    #[test]
    fn indexing_is_bounds_checked_per_axis() {
        let t = ramp(vec![2, 3]);
        // (0, 3) would be flat offset 3 without a per-axis check -- which is a
        // real cell, (1, 0). Silently returning it would be worse than an error.
        assert!(t.get(&[0, 3]).is_err());
        assert!(t.get(&[2, 0]).is_err());
        // Wrong number of index components.
        assert!(t.get(&[1]).is_err());
        assert!(t.get(&[1, 1, 1]).is_err());
    }

    #[test]
    fn set_writes_where_get_reads() {
        let mut t = ArithmaTensor::zeros(vec![2, 2, 2]).expect("2x2x2");
        t.set(&[1, 0, 1], n(9)).expect("in range");
        assert_eq!(t.get(&[1, 0, 1]).expect("in range").to_f64(), Some(9.0));
        assert_eq!(t.get(&[0, 0, 0]).expect("in range").to_f64(), Some(0.0));
    }

    #[test]
    fn reshape_preserves_cells_and_their_order() {
        let t = ramp(vec![2, 6]);
        let r = t.reshape(vec![3, 4]).expect("same cell count");
        assert_eq!(r.shape(), &[3, 4]);
        assert_eq!(values(&r), values(&t));
    }

    #[test]
    fn reshape_rejects_a_different_cell_count() {
        let t = ramp(vec![2, 6]);
        assert_eq!(
            t.reshape(vec![5, 5]).unwrap_err(),
            TensorError::ReshapeMismatch { from: 12, to: 25 }
        );
    }

    #[test]
    fn permuting_a_rank_two_tensor_is_the_matrix_transpose() {
        let t = ramp(vec![2, 3]); // [[0,1,2],[3,4,5]]
        let p = t.permute_axes(&[1, 0]).expect("valid permutation");
        assert_eq!(p.shape(), &[3, 2]);
        assert_eq!(values(&p), vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }

    #[test]
    fn permuting_twice_by_the_inverse_restores_the_original() {
        let t = ramp(vec![2, 3, 4]);
        let p = t.permute_axes(&[2, 0, 1]).expect("valid");
        assert_eq!(p.shape(), &[4, 2, 3]);
        // The inverse of (2,0,1) is (1,2,0).
        let back = p.permute_axes(&[1, 2, 0]).expect("valid");
        assert_eq!(back.shape(), t.shape());
        assert_eq!(values(&back), values(&t));
    }

    #[test]
    fn permutation_arguments_are_validated() {
        let t = ramp(vec![2, 3]);
        // Repeated axis, out-of-range axis, and wrong arity must all fail.
        assert!(t.permute_axes(&[0, 0]).is_err());
        assert!(t.permute_axes(&[0, 2]).is_err());
        assert!(t.permute_axes(&[0]).is_err());
    }

    #[test]
    fn elementwise_operations_require_matching_shapes() {
        let a = ramp(vec![2, 2]);
        let b = ramp(vec![2, 2]);
        assert_eq!(
            values(&a.add(&b).expect("same shape")),
            vec![0.0, 2.0, 4.0, 6.0]
        );
        assert_eq!(
            values(&a.sub(&b).expect("same shape")),
            vec![0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            values(&a.hadamard(&b).expect("same shape")),
            vec![0.0, 1.0, 4.0, 9.0]
        );
        let wrong = ramp(vec![4]);
        assert_eq!(
            a.add(&wrong).unwrap_err(),
            TensorError::ShapeMismatch {
                left: vec![2, 2],
                right: vec![4]
            }
        );
    }

    #[test]
    fn scalar_multiplication_scales_every_cell() {
        let t = ramp(vec![2, 2]);
        assert_eq!(values(&t.scalar_mul(&n(10))), vec![0.0, 10.0, 20.0, 30.0]);
    }

    #[test]
    fn oversized_shapes_are_refused_rather_than_allocated() {
        // Past the cell cap.
        assert!(ArithmaTensor::zeros(vec![MAX_CELLS + 1]).is_err());
        // Past the rank cap.
        assert!(ArithmaTensor::zeros(vec![1; MAX_RANK + 1]).is_err());
        // A product that would wrap a usize must be caught, not wrapped: an
        // unchecked multiply here yields a small number and a tiny allocation.
        assert!(ArithmaTensor::zeros(vec![usize::MAX, 2, 2]).is_err());
    }

    #[test]
    fn a_rank_zero_tensor_holds_one_cell() {
        let t = ArithmaTensor::zeros(vec![]).expect("scalar tensor");
        assert_eq!(t.rank(), 0);
        assert_eq!(t.cells.len(), 1);
        assert_eq!(t.offset(&[]).expect("scalar index"), 0);
    }
    // ─── contraction ───────────────────────────────────────────────────────

    /// The order-`n` identity, as a rank-2 tensor.
    fn identity(n_axis: usize) -> ArithmaTensor {
        let mut t = ArithmaTensor::zeros(vec![n_axis, n_axis]).expect("square");
        for i in 0..n_axis {
            t.set(&[i, i], n(1)).expect("diagonal is in range");
        }
        t
    }

    #[test]
    fn tensordot_over_the_inner_axes_is_matrix_multiplication() {
        // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]], by hand.
        let a = ArithmaTensor::new(vec![2, 2], vec![n(1), n(2), n(3), n(4)]);
        let b = ArithmaTensor::new(vec![2, 2], vec![n(5), n(6), n(7), n(8)]);
        let product = a.tensordot(&b, &[1], &[0]).expect("inner axes agree");
        assert_eq!(product.shape(), &[2, 2]);
        assert_eq!(values(&product), vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn multiplying_by_the_identity_leaves_a_tensor_unchanged() {
        let a = ramp(vec![2, 3]);
        let product = a
            .tensordot(&identity(3), &[1], &[0])
            .expect("inner axes agree");
        assert_eq!(product.shape(), a.shape());
        assert_eq!(values(&product), values(&a));
    }

    #[test]
    fn the_trace_of_a_rank_two_identity_tensor_counts_its_diagonal() {
        let scalar = identity(3).trace(0, 1).expect("square");
        // Contracting both axes leaves nothing, so the result is rank 0.
        assert_eq!(scalar.rank(), 0);
        assert_eq!(scalar.cells.len(), 1);
        assert_eq!(values(&scalar), vec![3.0]);
        // And the ordinary trace of [[0,1],[2,3]] is 0 + 3.
        assert_eq!(
            values(&ramp(vec![2, 2]).trace(0, 1).expect("square")),
            vec![3.0]
        );
    }

    #[test]
    fn tracing_a_rank_three_tensor_drops_the_two_contracted_axes() {
        // Shape [2,3,2]: axes 0 and 2 match in extent, axis 1 survives.
        let t = ramp(vec![2, 3, 2]);
        let traced = t.trace(0, 2).expect("axes 0 and 2 both have extent 2");
        assert_eq!(traced.shape(), &[3]);
        // Diagonal steps are 6 + 1 = 7 apart, so cell j is t[2j] + t[2j + 7].
        assert_eq!(values(&traced), vec![7.0, 11.0, 15.0]);
    }

    #[test]
    fn contracting_axes_of_differing_extents_is_an_error() {
        let a = ramp(vec![2, 3]);
        let b = ramp(vec![4, 5]);
        assert_eq!(
            a.tensordot(&b, &[1], &[0]).unwrap_err(),
            TensorError::AxisMismatch {
                axis_a: 1,
                extent_a: 3,
                axis_b: 0,
                extent_b: 4,
            }
        );
        // The same check guards a self-contraction: a non-square trace.
        assert_eq!(
            a.trace(0, 1).unwrap_err(),
            TensorError::AxisMismatch {
                axis_a: 0,
                extent_a: 2,
                axis_b: 1,
                extent_b: 3,
            }
        );
    }

    #[test]
    fn repeated_or_out_of_range_contraction_axes_are_errors() {
        let t = ramp(vec![2, 2]);
        // An axis cannot be contracted against itself.
        assert_eq!(
            t.trace(0, 0).unwrap_err(),
            TensorError::BadAxes {
                axes: vec![0, 0],
                rank: 2
            }
        );
        assert_eq!(
            t.trace(0, 5).unwrap_err(),
            TensorError::BadAxes {
                axes: vec![0, 5],
                rank: 2
            }
        );
        assert!(t.tensordot(&t, &[0, 0], &[0, 1]).is_err());
        assert!(t.tensordot(&t, &[0, 1], &[1, 1]).is_err());
        assert!(t.tensordot(&t, &[2], &[0]).is_err());
        assert!(t.tensordot(&t, &[0], &[2]).is_err());
        // Unequal list lengths leave the axes unpaired.
        assert!(t.tensordot(&t, &[0, 1], &[0]).is_err());
        assert!(t.tensordot(&t, &[], &[0]).is_err());
    }

    #[test]
    fn contracting_one_axis_of_two_rank_three_tensors_gives_the_expected_shape() {
        // Free axes of the left come first, then those of the right.
        let a = ramp(vec![2, 3, 4]);
        let b = ramp(vec![4, 5, 6]);
        let c = a.tensordot(&b, &[2], &[0]).expect("extent 4 on both");
        assert_eq!(c.shape(), &[2, 3, 5, 6]);
        assert_eq!(c.cells.len(), 180);

        // Small enough to check by hand: shapes [1,2,2] and [2,2,1].
        let small_a = ramp(vec![1, 2, 2]);
        let small_b = ramp(vec![2, 2, 1]);
        let small = small_a
            .tensordot(&small_b, &[2], &[0])
            .expect("extent 2 on both");
        assert_eq!(small.shape(), &[1, 2, 2, 1]);
        assert_eq!(values(&small), vec![2.0, 3.0, 6.0, 11.0]);
    }

    #[test]
    fn contracting_every_axis_of_both_operands_yields_a_scalar() {
        // The Frobenius inner product of [[0,1],[2,3]] with itself.
        let t = ramp(vec![2, 2]);
        let scalar = t.tensordot(&t, &[0, 1], &[0, 1]).expect("shapes agree");
        assert_eq!(scalar.rank(), 0);
        assert_eq!(values(&scalar), vec![14.0]);
    }

    #[test]
    fn the_outer_product_concatenates_the_shapes() {
        let a = ramp(vec![2]); // [0, 1]
        let b = ramp(vec![3]); // [0, 1, 2]
        let o = a.outer(&b).expect("no axes to contract");
        assert_eq!(o.shape(), &[2, 3]);
        assert_eq!(values(&o), vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0]);
        // outer is just tensordot with both axis lists empty.
        assert_eq!(
            values(&a.tensordot(&b, &[], &[]).expect("no axes")),
            values(&o)
        );
    }

    #[test]
    fn a_contraction_result_past_the_caps_is_refused_rather_than_allocated() {
        // Ranks add under an outer product, so two rank-9 tensors would give a
        // rank-18 result -- past MAX_RANK even though each holds one cell.
        let tall = ArithmaTensor::zeros(vec![1; 9]).expect("rank 9 is allowed");
        assert!(tall.outer(&tall).is_err());
        // Cell counts multiply, so two small legal operands can still ask for
        // more cells than the cap allows: 4096 * 2048 is twice MAX_CELLS.
        let wide = ArithmaTensor::zeros(vec![4096]).expect("under the cap");
        let narrow = ArithmaTensor::zeros(vec![2048]).expect("under the cap");
        assert!(wide.outer(&narrow).is_err());
    }
}
