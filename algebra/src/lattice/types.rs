//! Vectors and matrices over rings.

use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use core::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

use grid_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError, Valid};

use crate::arith::ring::Ring;
use crate::lattice::params::{LargeNormStats, LargeNormedRing, NormStats, NormedRing};

/// A vector of ring elements.
#[derive(Clone, PartialEq, Eq)]
pub struct RingVec<R: Ring> {
    entries: Vec<R>,
}

impl<R: Ring> RingVec<R> {
    /// Create a vector from entries.
    pub fn new(entries: Vec<R>) -> Self {
        Self { entries }
    }

    /// Return the vector length.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Borrow entry `i`.
    pub fn get(&self, i: usize) -> &R {
        &self.entries[i]
    }

    /// Return whether the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Borrow the entries.
    pub fn entries(&self) -> &[R] {
        &self.entries
    }

    /// Mutably borrow the entries.
    pub fn entries_mut(&mut self) -> &mut [R] {
        &mut self.entries
    }

    /// Return the all-zero vector of length `n`.
    pub fn zero(n: usize) -> Self {
        Self {
            entries: vec![R::zero(); n],
        }
    }

    /// Compute the inner product with another vector.
    pub fn dot(&self, other: &Self) -> R {
        assert_eq!(self.len(), other.len(), "vector lengths must match");
        R::dot_product(&self.entries, &other.entries)
    }

    /// Multiply all entries by a scalar.
    pub fn scale(&self, s: &R) -> Self {
        let mut entries = self.entries.clone();
        R::scalar_mul_slice(&mut entries, s);
        Self { entries }
    }

    /// In-place `self += s * other`.
    pub fn add_assign_scaled(&mut self, other: &Self, s: &R) {
        assert_eq!(self.len(), other.len(), "vector lengths must match");
        R::add_assign_scaled_slice(&mut self.entries, &other.entries, s);
    }

    /// Compute the exact squared `L2` norm.
    pub fn l2_norm_sq(&self) -> u128
    where
        R: NormedRing,
    {
        NormStats::compute(self).l2_sq
    }

    /// Compute the exact `L∞` norm.
    pub fn linf_norm(&self) -> u64
    where
        R: NormedRing,
    {
        NormStats::compute(self).linf
    }

    /// Compute the exact squared `L2` norm in the large companion representation.
    pub fn l2_norm_sq_large(&self) -> <R as LargeNormedRing>::Norm
    where
        R: LargeNormedRing,
    {
        LargeNormStats::compute(self).l2_sq
    }

    /// Compute the exact `L∞` norm in the large companion representation.
    pub fn linf_norm_large(&self) -> <R as LargeNormedRing>::Norm
    where
        R: LargeNormedRing,
    {
        LargeNormStats::compute(self).linf
    }
}

impl<R: Ring> fmt::Debug for RingVec<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("RingVec").field(&self.entries).finish()
    }
}

impl<R: Ring> Add for RingVec<R> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        assert_eq!(self.len(), rhs.len(), "vector lengths must match");
        let mut entries = self.entries;
        R::add_assign_slice(&mut entries, &rhs.entries);
        Self { entries }
    }
}

impl<R: Ring> Add<&Self> for RingVec<R> {
    type Output = Self;

    fn add(self, rhs: &Self) -> Self {
        assert_eq!(self.len(), rhs.len(), "vector lengths must match");
        let mut entries = self.entries;
        R::add_assign_slice(&mut entries, rhs.entries());
        Self { entries }
    }
}

impl<R: Ring> AddAssign for RingVec<R> {
    fn add_assign(&mut self, rhs: Self) {
        assert_eq!(self.len(), rhs.len(), "vector lengths must match");
        R::add_assign_slice(&mut self.entries, &rhs.entries);
    }
}

impl<R: Ring> Sub for RingVec<R> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        assert_eq!(self.len(), rhs.len(), "vector lengths must match");
        let mut entries = self.entries;
        R::sub_assign_slice(&mut entries, &rhs.entries);
        Self { entries }
    }
}

impl<R: Ring> Sub<&Self> for RingVec<R> {
    type Output = Self;

    fn sub(self, rhs: &Self) -> Self {
        assert_eq!(self.len(), rhs.len(), "vector lengths must match");
        let mut entries = self.entries;
        R::sub_assign_slice(&mut entries, rhs.entries());
        Self { entries }
    }
}

impl<R: Ring> SubAssign for RingVec<R> {
    fn sub_assign(&mut self, rhs: Self) {
        assert_eq!(self.len(), rhs.len(), "vector lengths must match");
        R::sub_assign_slice(&mut self.entries, &rhs.entries);
    }
}

impl<R: Ring> Neg for RingVec<R> {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            entries: self.entries.into_iter().map(Neg::neg).collect(),
        }
    }
}

impl<R: Ring + CanonicalSerialize> CanonicalSerialize for RingVec<R> {
    fn serialized_size(&self) -> usize {
        8 + self
            .entries
            .iter()
            .map(CanonicalSerialize::serialized_size)
            .sum::<usize>()
    }

    fn serialize_into(&self, buf: &mut Vec<u8>) -> Result<(), SerializationError> {
        buf.extend_from_slice(&(self.entries.len() as u64).to_le_bytes());
        for entry in &self.entries {
            entry.serialize_into(buf)?;
        }
        Ok(())
    }
}

impl<R: Ring + CanonicalDeserialize> CanonicalDeserialize for RingVec<R> {
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        if data.len() < 8 {
            return Err(SerializationError::UnexpectedEnd);
        }
        let len = usize::try_from(u64::from_le_bytes(data[..8].try_into().unwrap()))
            .map_err(|_| SerializationError::InvalidData("vector length too large".into()))?;
        let mut consumed = 8;
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(len)
            .map_err(|_| SerializationError::InvalidData("vector length too large".into()))?;
        for _ in 0..len {
            let (entry, used) = R::deserialize(&data[consumed..])?;
            entries.push(entry);
            consumed += used;
        }
        Ok((Self { entries }, consumed))
    }
}

impl<R: Ring + Valid> Valid for RingVec<R> {
    fn is_valid(&self) -> bool {
        self.entries.iter().all(Valid::is_valid)
    }
}

/// A row-major matrix of ring elements.
#[derive(Clone, PartialEq, Eq)]
pub struct RingMat<R: Ring> {
    rows: usize,
    cols: usize,
    entries: Vec<R>,
}

impl<R: Ring> RingMat<R> {
    /// Create a matrix from row-major entries.
    pub fn new(rows: usize, cols: usize, entries: Vec<R>) -> Self {
        let expected = rows
            .checked_mul(cols)
            .expect("matrix shape overflowed usize");
        assert_eq!(expected, entries.len(), "matrix shape does not match");
        Self {
            rows,
            cols,
            entries,
        }
    }

    /// Return the row count.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Return the column count.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Borrow the row-major entries.
    pub fn entries(&self) -> &[R] {
        &self.entries
    }

    /// Return the all-zero matrix.
    pub fn zero(rows: usize, cols: usize) -> Self {
        let len = rows
            .checked_mul(cols)
            .expect("matrix shape overflowed usize");
        Self {
            rows,
            cols,
            entries: vec![R::zero(); len],
        }
    }

    fn index(&self, i: usize, j: usize) -> usize {
        assert!(i < self.rows && j < self.cols, "matrix index out of bounds");
        i * self.cols + j
    }

    /// Borrow entry `(i, j)`.
    pub fn get(&self, i: usize, j: usize) -> &R {
        &self.entries[self.index(i, j)]
    }

    /// Mutably borrow entry `(i, j)`.
    pub fn get_mut(&mut self, i: usize, j: usize) -> &mut R {
        let idx = self.index(i, j);
        &mut self.entries[idx]
    }

    /// Return row `i`.
    pub fn row(&self, i: usize) -> RingVec<R> {
        assert!(i < self.rows, "row index out of bounds");
        let start = i * self.cols;
        RingVec::new(self.entries[start..start + self.cols].to_vec())
    }

    /// Return column `j`.
    pub fn col(&self, j: usize) -> RingVec<R> {
        assert!(j < self.cols, "column index out of bounds");
        RingVec::new(
            (0..self.rows)
                .map(|i| self.get(i, j).clone())
                .collect::<Vec<_>>(),
        )
    }

    /// Return the transpose.
    pub fn transpose(&self) -> Self {
        let mut out = Vec::with_capacity(self.entries.len());
        for j in 0..self.cols {
            for i in 0..self.rows {
                out.push(self.get(i, j).clone());
            }
        }
        Self::new(self.cols, self.rows, out)
    }

    /// Multiply the matrix by a RingVec.
    pub fn mul_vec(&self, v: &RingVec<R>) -> RingVec<R> {
        assert_eq!(self.cols, v.len(), "matrix/vector shape mismatch");
        let mut out = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            let start = i * self.cols;
            let end = start + self.cols;
            out.push(R::dot_product(&self.entries[start..end], v.entries()));
        }
        RingVec::new(out)
    }

    /// Multiply the matrix by a slice (avoids wrapping in RingVec + cloning).
    pub fn mul_slice(&self, v: &[R]) -> RingVec<R> {
        assert_eq!(self.cols, v.len(), "matrix/vector shape mismatch");
        let mut out = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            let start = i * self.cols;
            let end = start + self.cols;
            out.push(R::dot_product(&self.entries[start..end], v));
        }
        RingVec::new(out)
    }
}

impl<R: Ring> fmt::Debug for RingMat<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RingMat")
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field("entries", &self.entries)
            .finish()
    }
}

impl<R: Ring> Add for RingMat<R> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        assert_eq!(self.rows, rhs.rows, "matrix row counts must match");
        assert_eq!(self.cols, rhs.cols, "matrix col counts must match");
        let mut entries = self.entries;
        R::add_assign_slice(&mut entries, &rhs.entries);
        Self::new(self.rows, self.cols, entries)
    }
}

impl<R: Ring> Add<&Self> for RingMat<R> {
    type Output = Self;

    fn add(self, rhs: &Self) -> Self {
        assert_eq!(self.rows, rhs.rows, "matrix row counts must match");
        assert_eq!(self.cols, rhs.cols, "matrix col counts must match");
        let mut entries = self.entries;
        R::add_assign_slice(&mut entries, rhs.entries());
        Self::new(self.rows, self.cols, entries)
    }
}

impl<R: Ring> AddAssign for RingMat<R> {
    fn add_assign(&mut self, rhs: Self) {
        assert_eq!(self.rows, rhs.rows, "matrix row counts must match");
        assert_eq!(self.cols, rhs.cols, "matrix col counts must match");
        R::add_assign_slice(&mut self.entries, &rhs.entries);
    }
}

impl<R: Ring> Sub for RingMat<R> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        assert_eq!(self.rows, rhs.rows, "matrix row counts must match");
        assert_eq!(self.cols, rhs.cols, "matrix col counts must match");
        let mut entries = self.entries;
        R::sub_assign_slice(&mut entries, &rhs.entries);
        Self::new(self.rows, self.cols, entries)
    }
}

impl<R: Ring> Sub<&Self> for RingMat<R> {
    type Output = Self;

    fn sub(self, rhs: &Self) -> Self {
        assert_eq!(self.rows, rhs.rows, "matrix row counts must match");
        assert_eq!(self.cols, rhs.cols, "matrix col counts must match");
        let mut entries = self.entries;
        R::sub_assign_slice(&mut entries, rhs.entries());
        Self::new(self.rows, self.cols, entries)
    }
}

impl<R: Ring> SubAssign for RingMat<R> {
    fn sub_assign(&mut self, rhs: Self) {
        assert_eq!(self.rows, rhs.rows, "matrix row counts must match");
        assert_eq!(self.cols, rhs.cols, "matrix col counts must match");
        R::sub_assign_slice(&mut self.entries, &rhs.entries);
    }
}

impl<R: Ring> Neg for RingMat<R> {
    type Output = Self;

    fn neg(self) -> Self {
        Self::new(
            self.rows,
            self.cols,
            self.entries.into_iter().map(Neg::neg).collect(),
        )
    }
}

impl<R: Ring> Mul<&RingVec<R>> for RingMat<R> {
    type Output = RingVec<R>;

    fn mul(self, rhs: &RingVec<R>) -> RingVec<R> {
        self.mul_vec(rhs)
    }
}

impl<R: Ring + CanonicalSerialize> CanonicalSerialize for RingMat<R> {
    fn serialized_size(&self) -> usize {
        16 + self
            .entries
            .iter()
            .map(CanonicalSerialize::serialized_size)
            .sum::<usize>()
    }

    fn serialize_into(&self, buf: &mut Vec<u8>) -> Result<(), SerializationError> {
        buf.extend_from_slice(&(self.rows as u64).to_le_bytes());
        buf.extend_from_slice(&(self.cols as u64).to_le_bytes());
        for entry in &self.entries {
            entry.serialize_into(buf)?;
        }
        Ok(())
    }
}

impl<R: Ring + CanonicalDeserialize> CanonicalDeserialize for RingMat<R> {
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        if data.len() < 16 {
            return Err(SerializationError::UnexpectedEnd);
        }
        let rows = usize::try_from(u64::from_le_bytes(data[..8].try_into().unwrap()))
            .map_err(|_| SerializationError::InvalidData("matrix row count too large".into()))?;
        let cols = usize::try_from(u64::from_le_bytes(data[8..16].try_into().unwrap()))
            .map_err(|_| SerializationError::InvalidData("matrix column count too large".into()))?;
        let entry_count = rows
            .checked_mul(cols)
            .ok_or_else(|| SerializationError::InvalidData("matrix shape too large".into()))?;
        let mut consumed = 16;
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(entry_count)
            .map_err(|_| SerializationError::InvalidData("matrix shape too large".into()))?;
        for _ in 0..entry_count {
            let (entry, used) = R::deserialize(&data[consumed..])?;
            entries.push(entry);
            consumed += used;
        }
        Ok((
            Self {
                rows,
                cols,
                entries,
            },
            consumed,
        ))
    }
}

impl<R: Ring + Valid> Valid for RingMat<R> {
    fn is_valid(&self) -> bool {
        if self.rows.checked_mul(self.cols) != Some(self.entries.len()) {
            return false;
        }
        self.entries.iter().all(Valid::is_valid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arith::prime::PrimeField;
    use crate::arith::ring::IntegerRing;
    use grid_serialize::{CanonicalDeserialize, CanonicalSerialize};

    type F17 = PrimeField<17>;
    #[test]
    fn test_ring_vec_arithmetic() {
        let a = RingVec::new(vec![F17::from_u64(1), F17::from_u64(2), F17::from_u64(3)]);
        let b = RingVec::new(vec![F17::from_u64(4), F17::from_u64(5), F17::from_u64(6)]);
        assert_eq!(a.dot(&b).to_u64(), 15);
        let scaled = a.scale(&F17::from_u64(2));
        assert_eq!(scaled.entries()[1].to_u64(), 4);

        let mut add_assign = a.clone();
        add_assign += b.clone();
        assert_eq!(add_assign, a.clone() + b.clone());

        let mut sub_assign = a.clone();
        sub_assign -= b.clone();
        assert_eq!(sub_assign, a - b);
    }

    #[test]
    fn test_ring_vec_add_assign_scaled_and_norms() {
        let mut a = RingVec::new(vec![F17::from_u64(1), F17::from_u64(2), F17::from_u64(3)]);
        let b = RingVec::new(vec![F17::from_u64(4), F17::from_u64(5), F17::from_u64(6)]);
        a.add_assign_scaled(&b, &F17::from_u64(2));
        assert_eq!(a.entries()[0].to_u64(), 9);
        assert_eq!(a.entries()[1].to_u64(), 12);
        assert_eq!(a.l2_norm_sq(), 93);
        assert_eq!(a.linf_norm(), 8);
    }

    #[test]
    fn test_ring_mat_mul_vec_and_transpose() {
        let mat = RingMat::new(
            2,
            3,
            vec![
                F17::from_u64(1),
                F17::from_u64(2),
                F17::from_u64(3),
                F17::from_u64(4),
                F17::from_u64(5),
                F17::from_u64(6),
            ],
        );
        let vec = RingVec::new(vec![F17::from_u64(1), F17::from_u64(2), F17::from_u64(3)]);
        let prod = mat.mul_vec(&vec);
        assert_eq!(prod.entries()[0].to_u64(), 14);
        assert_eq!(prod.entries()[1].to_u64(), 15);
        assert_eq!(mat.transpose().transpose(), mat);

        let other = RingMat::new(
            2,
            3,
            vec![
                F17::from_u64(6),
                F17::from_u64(5),
                F17::from_u64(4),
                F17::from_u64(3),
                F17::from_u64(2),
                F17::from_u64(1),
            ],
        );
        let mut add_assign = mat.clone();
        add_assign += other.clone();
        assert_eq!(add_assign, mat.clone() + other.clone());

        let mut sub_assign = mat.clone();
        sub_assign -= other.clone();
        assert_eq!(sub_assign, mat - other);
    }

    #[test]
    fn test_serialization_round_trip() {
        let vec = RingVec::new(vec![F17::from_u64(1), F17::from_u64(2), F17::from_u64(3)]);
        let bytes = vec.serialize().unwrap();
        let decoded = RingVec::<F17>::deserialize_exact(&bytes).unwrap();
        assert_eq!(decoded, vec);

        let mat = RingMat::new(
            2,
            2,
            vec![
                F17::from_u64(1),
                F17::from_u64(2),
                F17::from_u64(3),
                F17::from_u64(4),
            ],
        );
        let bytes = mat.serialize().unwrap();
        let decoded = RingMat::<F17>::deserialize_exact(&bytes).unwrap();
        assert_eq!(decoded, mat);
    }

    #[test]
    fn test_matrix_deserialize_rejects_overflow_shape() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        let err = RingMat::<F17>::deserialize(&bytes).unwrap_err();
        assert!(matches!(err, SerializationError::InvalidData(_)));
    }
}
