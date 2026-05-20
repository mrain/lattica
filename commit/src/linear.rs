//! Shared linear commitment shapes and validation helpers.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::any::Any;

use grid_algebra::arith::large_modulus::{LargePrimeProfile, LargeRnsProfile};
use grid_algebra::arith::large_prime::LargePrimeField;
use grid_algebra::arith::large_rns::LargeRns;
use grid_algebra::arith::ntt::NTTRing;
use grid_algebra::arith::limb::UintLimb;
use grid_algebra::arith::prime::PrimeField;
use grid_algebra::arith::ring::{Field, Ring};
use grid_algebra::arith::z2k::Z2K;
use grid_algebra::lattice::params::NormBound;
use grid_algebra::lattice::types::{RingMat, RingVec};
use grid_algebra::poly::ring::{CyclotomicPolyRing, NegacyclicMulRing};
use grid_algebra::poly::{
    TwistedNttPoly, finish_twisted_ring_vec, prepare_twisted_ring_mat, prepare_twisted_ring_vec,
};
use grid_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError, Valid};

use crate::error::CommitmentError;

/// Type-erased runtime cache for prepared (NTT-domain) commitment key material.
///
/// Safety: each `PreparedLinearOps` impl builds and consumes its own caches with
/// matching types. The `CyclotomicPolyRing<C, N>` impl stores `PreparedLinearCommitmentKey<TwistedNttPoly<C, N>>`
/// and `RingMat<TwistedNttPoly<C, N>>`, and downcasts back to those exact types.
/// Cross-type contamination is impossible because the scheme is monomorphized per `(C, N)`.
pub(crate) type PreparedRuntimeCache = Box<dyn Any>;

/// Common dimensions used by linear commitments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommitmentDimensions {
    /// Message length.
    pub message_len: usize,
    /// Opening/randomness length.
    pub opening_len: usize,
    /// Commitment vector length.
    pub commitment_len: usize,
}

impl CommitmentDimensions {
    /// Validate basic dimension sanity and checked products.
    pub fn validate(&self) -> Result<(), CommitmentError> {
        if self.message_len == 0 || self.opening_len == 0 || self.commitment_len == 0 {
            return Err(CommitmentError::InvalidParameters);
        }
        // Check that matrix dimension products don't overflow usize.
        // Actual allocation size is the caller's responsibility.
        self.commitment_len
            .checked_mul(self.message_len)
            .ok_or(CommitmentError::InvalidParameters)?;
        self.commitment_len
            .checked_mul(self.opening_len)
            .ok_or(CommitmentError::InvalidParameters)?;
        Ok(())
    }
}

impl CanonicalSerialize for CommitmentDimensions {
    fn serialized_size(&self) -> usize {
        24
    }

    fn serialize_into(&self, buf: &mut Vec<u8>) -> Result<(), SerializationError> {
        buf.extend_from_slice(&(self.message_len as u64).to_le_bytes());
        buf.extend_from_slice(&(self.opening_len as u64).to_le_bytes());
        buf.extend_from_slice(&(self.commitment_len as u64).to_le_bytes());
        Ok(())
    }
}

impl CanonicalDeserialize for CommitmentDimensions {
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        if data.len() < 24 {
            return Err(SerializationError::UnexpectedEnd);
        }
        let message_len = usize::try_from(u64::from_le_bytes(data[..8].try_into().unwrap()))
            .map_err(|_| SerializationError::InvalidData("message length too large".into()))?;
        let opening_len = usize::try_from(u64::from_le_bytes(data[8..16].try_into().unwrap()))
            .map_err(|_| SerializationError::InvalidData("opening length too large".into()))?;
        let commitment_len = usize::try_from(u64::from_le_bytes(data[16..24].try_into().unwrap()))
            .map_err(|_| SerializationError::InvalidData("commitment length too large".into()))?;
        Ok((
            Self {
                message_len,
                opening_len,
                commitment_len,
            },
            24,
        ))
    }
}

impl Valid for CommitmentDimensions {
    fn is_valid(&self) -> bool {
        self.validate().is_ok()
    }
}

/// Shared parameter scaffold for linear commitments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearCommitmentParams<B = NormBound> {
    /// Message/opening/commitment dimensions.
    pub dims: CommitmentDimensions,
    /// Bound on acceptable openings.
    pub opening_bound: B,
}

impl<B> CanonicalSerialize for LinearCommitmentParams<B>
where
    B: CanonicalSerialize,
{
    fn serialized_size(&self) -> usize {
        self.dims.serialized_size() + self.opening_bound.serialized_size()
    }

    fn serialize_into(&self, buf: &mut Vec<u8>) -> Result<(), SerializationError> {
        self.dims.serialize_into(buf)?;
        self.opening_bound.serialize_into(buf)?;
        Ok(())
    }
}

impl<B> CanonicalDeserialize for LinearCommitmentParams<B>
where
    B: CanonicalDeserialize,
{
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        let (dims, used_dims) = CommitmentDimensions::deserialize(data)?;
        let (opening_bound, used_bound) = B::deserialize(&data[used_dims..])?;
        Ok((
            Self {
                dims,
                opening_bound,
            },
            used_dims + used_bound,
        ))
    }
}

impl<B> Valid for LinearCommitmentParams<B>
where
    B: Valid,
{
    fn is_valid(&self) -> bool {
        self.dims.is_valid() && self.opening_bound.is_valid()
    }
}

/// Shared linear commitment key scaffold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearCommitmentKey<R: Ring> {
    /// Message matrix.
    pub a_msg: RingMat<R>,
    /// Opening/randomness matrix.
    pub a_open: RingMat<R>,
}

/// Runtime-prepared linear commitment key material in the twisted domain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PreparedLinearCommitmentKey<P: Ring> {
    pub a_msg: RingMat<P>,
    pub a_open: RingMat<P>,
}

impl<R: Ring + CanonicalSerialize> CanonicalSerialize for LinearCommitmentKey<R> {
    fn serialized_size(&self) -> usize {
        self.a_msg.serialized_size() + self.a_open.serialized_size()
    }

    fn serialize_into(&self, buf: &mut Vec<u8>) -> Result<(), SerializationError> {
        self.a_msg.serialize_into(buf)?;
        self.a_open.serialize_into(buf)?;
        Ok(())
    }
}

impl<R: Ring + CanonicalDeserialize> CanonicalDeserialize for LinearCommitmentKey<R> {
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        let (a_msg, used_msg) = RingMat::<R>::deserialize(data)?;
        let (a_open, used_open) = RingMat::<R>::deserialize(&data[used_msg..])?;
        Ok((Self { a_msg, a_open }, used_msg + used_open))
    }
}

impl<R: Ring + Valid> Valid for LinearCommitmentKey<R> {
    fn is_valid(&self) -> bool {
        self.a_msg.is_valid() && self.a_open.is_valid() && self.a_msg.rows() == self.a_open.rows()
    }
}

/// Shared commitment wrapper.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearCommitment<R: Ring> {
    /// Commitment vector value.
    pub value: RingVec<R>,
}

impl<R: Ring + CanonicalSerialize> CanonicalSerialize for LinearCommitment<R> {
    fn serialized_size(&self) -> usize {
        self.value.serialized_size()
    }

    fn serialize_into(&self, buf: &mut Vec<u8>) -> Result<(), SerializationError> {
        self.value.serialize_into(buf)
    }
}

impl<R: Ring + CanonicalDeserialize> CanonicalDeserialize for LinearCommitment<R> {
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        let (value, used) = RingVec::<R>::deserialize(data)?;
        Ok((Self { value }, used))
    }
}

impl<R: Ring + Valid> Valid for LinearCommitment<R> {
    fn is_valid(&self) -> bool {
        self.value.is_valid()
    }
}

/// Shared opening wrapper.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearOpening<R: Ring> {
    /// Opening randomness.
    pub randomness: RingVec<R>,
}

impl<R: Ring + CanonicalSerialize> CanonicalSerialize for LinearOpening<R> {
    fn serialized_size(&self) -> usize {
        self.randomness.serialized_size()
    }

    fn serialize_into(&self, buf: &mut Vec<u8>) -> Result<(), SerializationError> {
        self.randomness.serialize_into(buf)
    }
}

impl<R: Ring + CanonicalDeserialize> CanonicalDeserialize for LinearOpening<R> {
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        let (randomness, used) = RingVec::<R>::deserialize(data)?;
        Ok((Self { randomness }, used))
    }
}

impl<R: Ring + Valid> Valid for LinearOpening<R> {
    fn is_valid(&self) -> bool {
        self.randomness.is_valid()
    }
}

/// Check that a linear commitment key matches the declared dimensions.
pub fn validate_linear_matrices<R: Ring>(
    a_msg: &RingMat<R>,
    a_open: &RingMat<R>,
    dims: CommitmentDimensions,
) -> Result<(), CommitmentError> {
    dims.validate()?;
    if a_msg.rows() != dims.commitment_len
        || a_msg.cols() != dims.message_len
        || a_open.rows() != dims.commitment_len
        || a_open.cols() != dims.opening_len
    {
        return Err(CommitmentError::DimensionMismatch);
    }
    Ok(())
}

pub fn validate_linear_key<R: Ring>(
    key: &LinearCommitmentKey<R>,
    dims: CommitmentDimensions,
) -> Result<(), CommitmentError> {
    validate_linear_matrices(&key.a_msg, &key.a_open, dims)
}

/// Check that a message length matches the declared dimensions.
pub fn validate_message<R: Ring>(
    message: &RingVec<R>,
    dims: CommitmentDimensions,
) -> Result<(), CommitmentError> {
    dims.validate()?;
    if message.len() != dims.message_len {
        return Err(CommitmentError::DimensionMismatch);
    }
    Ok(())
}

/// Check that an opening length matches the declared dimensions.
pub fn validate_opening_randomness<R: Ring>(
    randomness: &RingVec<R>,
    dims: CommitmentDimensions,
) -> Result<(), CommitmentError> {
    dims.validate()?;
    if randomness.len() != dims.opening_len {
        return Err(CommitmentError::DimensionMismatch);
    }
    Ok(())
}

pub fn validate_opening<R: Ring>(
    opening: &LinearOpening<R>,
    dims: CommitmentDimensions,
) -> Result<(), CommitmentError> {
    validate_opening_randomness(&opening.randomness, dims)
}

/// Check that a commitment length matches the declared dimensions.
pub fn validate_commitment_value<R: Ring>(
    value: &RingVec<R>,
    dims: CommitmentDimensions,
) -> Result<(), CommitmentError> {
    dims.validate()?;
    if value.len() != dims.commitment_len {
        return Err(CommitmentError::DimensionMismatch);
    }
    Ok(())
}

pub fn validate_commitment<R: Ring>(
    commitment: &LinearCommitment<R>,
    dims: CommitmentDimensions,
) -> Result<(), CommitmentError> {
    validate_commitment_value(&commitment.value, dims)
}

pub(crate) fn prepare_poly_vector<C, const N: usize>(
    vector: &RingVec<CyclotomicPolyRing<C, N>>,
) -> Result<RingVec<TwistedNttPoly<C, N>>, CommitmentError>
where
    C: Field
        + NTTRing
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize
        + Send
        + Sync
        + 'static,
{
    prepare_twisted_ring_vec(vector).map_err(|_| CommitmentError::InvalidParameters)
}

pub(crate) fn prepare_poly_matrix<C, const N: usize>(
    matrix: &RingMat<CyclotomicPolyRing<C, N>>,
) -> Result<RingMat<TwistedNttPoly<C, N>>, CommitmentError>
where
    C: Field
        + NTTRing
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize
        + Send
        + Sync
        + 'static,
{
    prepare_twisted_ring_mat(matrix).map_err(|_| CommitmentError::InvalidParameters)
}

pub(crate) fn finish_prepared_vector<C, const N: usize>(
    vector: &RingVec<TwistedNttPoly<C, N>>,
) -> Result<RingVec<CyclotomicPolyRing<C, N>>, CommitmentError>
where
    C: Field
        + NTTRing
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize
        + Send
        + Sync
        + 'static,
{
    finish_twisted_ring_vec(vector).map_err(|_| CommitmentError::InvalidParameters)
}

pub(crate) fn prepare_linear_key<C, const N: usize>(
    a_msg: &RingMat<CyclotomicPolyRing<C, N>>,
    a_open: &RingMat<CyclotomicPolyRing<C, N>>,
) -> Result<PreparedLinearCommitmentKey<TwistedNttPoly<C, N>>, CommitmentError>
where
    C: Field
        + NTTRing
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize
        + Send
        + Sync
        + 'static,
{
    Ok(PreparedLinearCommitmentKey {
        a_msg: prepare_poly_matrix(a_msg)?,
        a_open: prepare_poly_matrix(a_open)?,
    })
}

pub(crate) fn recompute_linear_commitment_prepared<C, const N: usize>(
    key: &PreparedLinearCommitmentKey<TwistedNttPoly<C, N>>,
    message: &RingVec<TwistedNttPoly<C, N>>,
    randomness: &RingVec<TwistedNttPoly<C, N>>,
    dims: CommitmentDimensions,
) -> Result<LinearCommitment<CyclotomicPolyRing<C, N>>, CommitmentError>
where
    C: Field
        + NTTRing
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize
        + Send
        + Sync
        + 'static,
{
    let prepared =
        recompute_linear_commitment_parts(&key.a_msg, &key.a_open, message, randomness, dims)?;
    Ok(LinearCommitment {
        value: finish_prepared_vector(&prepared.value)?,
    })
}

pub(crate) fn mul_poly_matrix_prepared<C, const N: usize>(
    matrix: &RingMat<CyclotomicPolyRing<C, N>>,
    vector: &RingVec<CyclotomicPolyRing<C, N>>,
) -> Result<RingVec<CyclotomicPolyRing<C, N>>, CommitmentError>
where
    C: Field
        + NTTRing
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize
        + Send
        + Sync
        + 'static,
{
    let prepared_matrix = prepare_poly_matrix(matrix)?;
    let prepared_vector = prepare_poly_vector(vector)?;
    let prepared_out = prepared_matrix.mul_vec(&prepared_vector);
    finish_prepared_vector(&prepared_out)
}

pub(crate) fn mul_prepared_matrix_vector<C, const N: usize>(
    matrix: &RingMat<TwistedNttPoly<C, N>>,
    vector: &RingVec<TwistedNttPoly<C, N>>,
) -> Result<RingVec<CyclotomicPolyRing<C, N>>, CommitmentError>
where
    C: Field
        + NTTRing
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize
        + Send
        + Sync
        + 'static,
{
    let prepared_out = matrix.mul_vec(vector);
    finish_prepared_vector(&prepared_out)
}

#[inline(always)]
fn mul_matrix_vector_trusted<R: Ring>(matrix: &RingMat<R>, vector: &RingVec<R>) -> RingVec<R> {
    let rows = matrix.rows();
    let cols = matrix.cols();
    let entries = matrix.entries();
    let vector_entries = vector.entries();
    let mut out = Vec::with_capacity(rows);
    for row in 0..rows {
        let start = row * cols;
        out.push(R::dot_product(
            &entries[start..start + cols],
            vector_entries,
        ));
    }
    RingVec::new(out)
}

#[inline(always)]
fn recompute_linear_commitment_parts_trusted<R: Ring>(
    a_msg: &RingMat<R>,
    a_open: &RingMat<R>,
    message: &RingVec<R>,
    randomness: &RingVec<R>,
) -> LinearCommitment<R> {
    let rows = a_msg.rows();
    let msg_cols = a_msg.cols();
    let open_cols = a_open.cols();
    let a_msg_entries = a_msg.entries();
    let a_open_entries = a_open.entries();
    let message_entries = message.entries();
    let randomness_entries = randomness.entries();
    let mut value = Vec::with_capacity(rows);
    for row in 0..rows {
        let msg_start = row * msg_cols;
        let open_start = row * open_cols;
        let msg_term = R::dot_product(
            &a_msg_entries[msg_start..msg_start + msg_cols],
            message_entries,
        );
        let open_term = R::dot_product(
            &a_open_entries[open_start..open_start + open_cols],
            randomness_entries,
        );
        value.push(msg_term + &open_term);
    }
    LinearCommitment {
        value: RingVec::new(value),
    }
}

#[inline(always)]
fn recompute_linear_commitment_prepared_trusted<C, const N: usize>(
    key: &PreparedLinearCommitmentKey<TwistedNttPoly<C, N>>,
    message: &RingVec<TwistedNttPoly<C, N>>,
    randomness: &RingVec<TwistedNttPoly<C, N>>,
) -> Result<LinearCommitment<CyclotomicPolyRing<C, N>>, CommitmentError>
where
    C: Field
        + NTTRing
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize
        + Send
        + Sync
        + 'static,
{
    let prepared =
        recompute_linear_commitment_parts_trusted(&key.a_msg, &key.a_open, message, randomness);
    Ok(LinearCommitment {
        value: finish_prepared_vector(&prepared.value)?,
    })
}

#[inline(always)]
fn mul_prepared_matrix_vector_trusted<C, const N: usize>(
    matrix: &RingMat<TwistedNttPoly<C, N>>,
    vector: &RingVec<TwistedNttPoly<C, N>>,
) -> Result<RingVec<CyclotomicPolyRing<C, N>>, CommitmentError>
where
    C: Field
        + NTTRing
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize
        + Send
        + Sync
        + 'static,
{
    let prepared_out = mul_matrix_vector_trusted(matrix, vector);
    finish_prepared_vector(&prepared_out)
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn recompute_bdlop_parts_runtime_fallback<R: PreparedLinearOps>(
    prepared_mask: Option<&dyn Any>,
    a_mask: &RingMat<R>,
    prepared_key: Option<&dyn Any>,
    a_msg: &RingMat<R>,
    a_open: &RingMat<R>,
    message: &RingVec<R>,
    randomness: &RingVec<R>,
    dims: CommitmentDimensions,
) -> Result<(RingVec<R>, RingVec<R>), CommitmentError> {
    let u = R::mul_matrix_vector_runtime(prepared_mask, a_mask, randomness)?;
    let v = R::recompute_linear_commitment_runtime(
        prepared_key,
        a_msg,
        a_open,
        message,
        randomness,
        dims,
    )?
    .value;
    Ok((u, v))
}

#[doc(hidden)]
pub trait PreparedLinearOps: Ring {
    fn build_linear_key_cache(
        _a_msg: &RingMat<Self>,
        _a_open: &RingMat<Self>,
    ) -> Option<PreparedRuntimeCache> {
        None
    }

    fn build_matrix_cache(_matrix: &RingMat<Self>) -> Option<PreparedRuntimeCache> {
        None
    }

    fn recompute_linear_commitment_runtime(
        _prepared_key: Option<&dyn Any>,
        a_msg: &RingMat<Self>,
        a_open: &RingMat<Self>,
        message: &RingVec<Self>,
        randomness: &RingVec<Self>,
        _dims: CommitmentDimensions,
    ) -> Result<LinearCommitment<Self>, CommitmentError> {
        Ok(recompute_linear_commitment_parts_trusted(
            a_msg, a_open, message, randomness,
        ))
    }

    fn mul_matrix_vector_runtime(
        _prepared_matrix: Option<&dyn Any>,
        matrix: &RingMat<Self>,
        vector: &RingVec<Self>,
    ) -> Result<RingVec<Self>, CommitmentError> {
        Ok(mul_matrix_vector_trusted(matrix, vector))
    }

    #[allow(clippy::too_many_arguments)]
    fn recompute_bdlop_parts_runtime(
        prepared_mask: Option<&dyn Any>,
        a_mask: &RingMat<Self>,
        prepared_key: Option<&dyn Any>,
        a_msg: &RingMat<Self>,
        a_open: &RingMat<Self>,
        message: &RingVec<Self>,
        randomness: &RingVec<Self>,
        dims: CommitmentDimensions,
    ) -> Result<(RingVec<Self>, RingVec<Self>), CommitmentError> {
        recompute_bdlop_parts_runtime_fallback::<Self>(
            prepared_mask,
            a_mask,
            prepared_key,
            a_msg,
            a_open,
            message,
            randomness,
            dims,
        )
    }
}

impl<const Q: u64, L: UintLimb> PreparedLinearOps for PrimeField<Q, L> {
    #[inline(always)]
    fn recompute_linear_commitment_runtime(
        _prepared_key: Option<&dyn Any>,
        a_msg: &RingMat<Self>,
        a_open: &RingMat<Self>,
        message: &RingVec<Self>,
        randomness: &RingVec<Self>,
        _dims: CommitmentDimensions,
    ) -> Result<LinearCommitment<Self>, CommitmentError> {
        Ok(recompute_linear_commitment_parts_trusted(
            a_msg, a_open, message, randomness,
        ))
    }

    #[inline(always)]
    fn mul_matrix_vector_runtime(
        _prepared_matrix: Option<&dyn Any>,
        matrix: &RingMat<Self>,
        vector: &RingVec<Self>,
    ) -> Result<RingVec<Self>, CommitmentError> {
        Ok(mul_matrix_vector_trusted(matrix, vector))
    }
}

impl<const K: u32> PreparedLinearOps for Z2K<K> {
    #[inline(always)]
    fn recompute_linear_commitment_runtime(
        _prepared_key: Option<&dyn Any>,
        a_msg: &RingMat<Self>,
        a_open: &RingMat<Self>,
        message: &RingVec<Self>,
        randomness: &RingVec<Self>,
        _dims: CommitmentDimensions,
    ) -> Result<LinearCommitment<Self>, CommitmentError> {
        Ok(recompute_linear_commitment_parts_trusted(
            a_msg, a_open, message, randomness,
        ))
    }

    #[inline(always)]
    fn mul_matrix_vector_runtime(
        _prepared_matrix: Option<&dyn Any>,
        matrix: &RingMat<Self>,
        vector: &RingVec<Self>,
    ) -> Result<RingVec<Self>, CommitmentError> {
        Ok(mul_matrix_vector_trusted(matrix, vector))
    }
}

impl<P, const LIMBS: usize> PreparedLinearOps for LargePrimeField<P, LIMBS> where
    P: LargePrimeProfile<LIMBS>
{
}

impl<P, const LIMBS: usize> PreparedLinearOps for LargeRns<P, LIMBS> where P: LargeRnsProfile<LIMBS> {}

impl<C, const N: usize> PreparedLinearOps for CyclotomicPolyRing<C, N>
where
    C: Field
        + NTTRing
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize
        + Send
        + Sync
        + 'static,
{
    fn build_linear_key_cache(
        a_msg: &RingMat<Self>,
        a_open: &RingMat<Self>,
    ) -> Option<PreparedRuntimeCache> {
        prepare_linear_key(a_msg, a_open)
            .ok()
            .map(|prepared| Box::new(prepared) as PreparedRuntimeCache)
    }

    fn build_matrix_cache(matrix: &RingMat<Self>) -> Option<PreparedRuntimeCache> {
        prepare_poly_matrix(matrix)
            .ok()
            .map(|prepared| Box::new(prepared) as PreparedRuntimeCache)
    }

    #[inline(always)]
    fn recompute_linear_commitment_runtime(
        prepared_key: Option<&dyn Any>,
        a_msg: &RingMat<Self>,
        a_open: &RingMat<Self>,
        message: &RingVec<Self>,
        randomness: &RingVec<Self>,
        _dims: CommitmentDimensions,
    ) -> Result<LinearCommitment<Self>, CommitmentError> {
        let prepared_key_ref = prepared_key.and_then(|cache| {
            let r = cache.downcast_ref::<PreparedLinearCommitmentKey<TwistedNttPoly<C, N>>>();
            debug_assert!(
                r.is_some(),
                "prepared cache type mismatch: expected PreparedLinearCommitmentKey<TwistedNttPoly<C, N>>"
            );
            r
        });
        if let Some(prepared_key) = prepared_key_ref
            && let Ok(prepared_message) = prepare_poly_vector(message)
            && let Ok(prepared_randomness) = prepare_poly_vector(randomness)
            && let Ok(commitment) = recompute_linear_commitment_prepared_trusted(
                prepared_key,
                &prepared_message,
                &prepared_randomness,
            )
        {
            return Ok(commitment);
        }

        let Ok(prepared_key) = prepare_linear_key(a_msg, a_open) else {
            return Ok(recompute_linear_commitment_parts_trusted(
                a_msg, a_open, message, randomness,
            ));
        };
        let Ok(prepared_message) = prepare_poly_vector(message) else {
            return Ok(recompute_linear_commitment_parts_trusted(
                a_msg, a_open, message, randomness,
            ));
        };
        let Ok(prepared_randomness) = prepare_poly_vector(randomness) else {
            return Ok(recompute_linear_commitment_parts_trusted(
                a_msg, a_open, message, randomness,
            ));
        };
        recompute_linear_commitment_prepared_trusted(
            &prepared_key,
            &prepared_message,
            &prepared_randomness,
        )
        .or_else(|_| {
            Ok(recompute_linear_commitment_parts_trusted(
                a_msg, a_open, message, randomness,
            ))
        })
    }

    #[inline(always)]
    fn mul_matrix_vector_runtime(
        prepared_matrix: Option<&dyn Any>,
        matrix: &RingMat<Self>,
        vector: &RingVec<Self>,
    ) -> Result<RingVec<Self>, CommitmentError> {
        let prepared_matrix_ref = prepared_matrix.and_then(|cache| {
            let r = cache.downcast_ref::<RingMat<TwistedNttPoly<C, N>>>();
            debug_assert!(
                r.is_some(),
                "prepared cache type mismatch: expected RingMat<TwistedNttPoly<C, N>>"
            );
            r
        });
        if let Some(prepared_matrix) = prepared_matrix_ref
            && let Ok(prepared_vector) = prepare_poly_vector(vector)
        {
            return mul_prepared_matrix_vector_trusted(prepared_matrix, &prepared_vector)
                .or_else(|_| Ok(mul_matrix_vector_trusted(matrix, vector)));
        }

        match mul_poly_matrix_prepared(matrix, vector) {
            Ok(result) => Ok(result),
            Err(_) => Ok(mul_matrix_vector_trusted(matrix, vector)),
        }
    }

    #[inline(always)]
    fn recompute_bdlop_parts_runtime(
        prepared_mask: Option<&dyn Any>,
        a_mask: &RingMat<Self>,
        prepared_key: Option<&dyn Any>,
        a_msg: &RingMat<Self>,
        a_open: &RingMat<Self>,
        message: &RingVec<Self>,
        randomness: &RingVec<Self>,
        dims: CommitmentDimensions,
    ) -> Result<(RingVec<Self>, RingVec<Self>), CommitmentError> {
        let prepared_mask_cache = prepared_mask;
        let prepared_key_cache = prepared_key;
        let prepared_message = match prepare_poly_vector(message) {
            Ok(prepared) => prepared,
            Err(_) => {
                return recompute_bdlop_parts_runtime_fallback::<Self>(
                    prepared_mask_cache,
                    a_mask,
                    prepared_key_cache,
                    a_msg,
                    a_open,
                    message,
                    randomness,
                    dims,
                );
            }
        };
        let prepared_randomness = match prepare_poly_vector(randomness) {
            Ok(prepared) => prepared,
            Err(_) => {
                return recompute_bdlop_parts_runtime_fallback::<Self>(
                    prepared_mask_cache,
                    a_mask,
                    prepared_key_cache,
                    a_msg,
                    a_open,
                    message,
                    randomness,
                    dims,
                );
            }
        };

        let owned_prepared_mask;
        let prepared_mask_ref = prepared_mask_cache.and_then(|cache| {
            let r = cache.downcast_ref::<RingMat<TwistedNttPoly<C, N>>>();
            debug_assert!(
                r.is_some(),
                "prepared mask cache type mismatch: expected RingMat<TwistedNttPoly<C, N>>"
            );
            r
        });
        let prepared_mask_ref = if let Some(prepared) = prepared_mask_ref {
            prepared
        } else {
            owned_prepared_mask = match prepare_poly_matrix(a_mask) {
                Ok(prepared) => prepared,
                Err(_) => {
                    return recompute_bdlop_parts_runtime_fallback::<Self>(
                        prepared_mask_cache,
                        a_mask,
                        prepared_key_cache,
                        a_msg,
                        a_open,
                        message,
                        randomness,
                        dims,
                    );
                }
            };
            &owned_prepared_mask
        };

        let owned_prepared_key;
        let prepared_key_ref = prepared_key_cache.and_then(|cache| {
            let r = cache.downcast_ref::<PreparedLinearCommitmentKey<TwistedNttPoly<C, N>>>();
            debug_assert!(
                r.is_some(),
                "prepared key cache type mismatch: expected PreparedLinearCommitmentKey<TwistedNttPoly<C, N>>"
            );
            r
        });
        let prepared_key_ref = if let Some(prepared) = prepared_key_ref {
            prepared
        } else {
            owned_prepared_key = match prepare_linear_key(a_msg, a_open) {
                Ok(prepared) => prepared,
                Err(_) => {
                    return recompute_bdlop_parts_runtime_fallback::<Self>(
                        prepared_mask_cache,
                        a_mask,
                        prepared_key_cache,
                        a_msg,
                        a_open,
                        message,
                        randomness,
                        dims,
                    );
                }
            };
            &owned_prepared_key
        };

        let u = match mul_prepared_matrix_vector_trusted(prepared_mask_ref, &prepared_randomness) {
            Ok(result) => result,
            Err(_) => {
                return recompute_bdlop_parts_runtime_fallback::<Self>(
                    prepared_mask_cache,
                    a_mask,
                    prepared_key_cache,
                    a_msg,
                    a_open,
                    message,
                    randomness,
                    dims,
                );
            }
        };
        let v = match recompute_linear_commitment_prepared_trusted(
            prepared_key_ref,
            &prepared_message,
            &prepared_randomness,
        ) {
            Ok(result) => result.value,
            Err(_) => {
                return recompute_bdlop_parts_runtime_fallback::<Self>(
                    prepared_mask_cache,
                    a_mask,
                    prepared_key_cache,
                    a_msg,
                    a_open,
                    message,
                    randomness,
                    dims,
                );
            }
        };
        Ok((u, v))
    }
}

/// Recompute a linear commitment from a key, message, and opening.
#[inline(always)]
pub fn recompute_linear_commitment_parts<R: Ring>(
    a_msg: &RingMat<R>,
    a_open: &RingMat<R>,
    message: &RingVec<R>,
    randomness: &RingVec<R>,
    dims: CommitmentDimensions,
) -> Result<LinearCommitment<R>, CommitmentError> {
    validate_linear_matrices(a_msg, a_open, dims)?;
    validate_message(message, dims)?;
    validate_opening_randomness(randomness, dims)?;
    Ok(recompute_linear_commitment_parts_trusted(
        a_msg, a_open, message, randomness,
    ))
}

pub fn recompute_linear_commitment<R: Ring>(
    key: &LinearCommitmentKey<R>,
    message: &RingVec<R>,
    opening: &LinearOpening<R>,
    dims: CommitmentDimensions,
) -> Result<LinearCommitment<R>, CommitmentError> {
    recompute_linear_commitment_parts(&key.a_msg, &key.a_open, message, &opening.randomness, dims)
}

#[cfg(test)]
mod tests {
    use super::*;

    use grid_algebra::arith::bigint::BigUint;
    use grid_algebra::arith::prime::PrimeField;
    use grid_algebra::arith::ring::IntegerRing;
    use grid_algebra::lattice::params::LargeNormBound;
    use grid_algebra::poly::ring::CyclotomicPolyRing;

    type F17 = PrimeField<17>;
    type Poly8 = CyclotomicPolyRing<F17, 8>;

    fn test_dims() -> CommitmentDimensions {
        CommitmentDimensions {
            message_len: 2,
            opening_len: 3,
            commitment_len: 2,
        }
    }

    #[test]
    fn test_commitment_dimensions_reject_zero_lengths() {
        let dims = CommitmentDimensions {
            message_len: 0,
            opening_len: 1,
            commitment_len: 1,
        };
        assert_eq!(dims.validate(), Err(CommitmentError::InvalidParameters));
        assert!(!dims.is_valid());
    }

    #[test]
    fn test_commitment_dimensions_reject_overflow() {
        let dims = CommitmentDimensions {
            message_len: usize::MAX,
            opening_len: 1,
            commitment_len: 2,
        };
        assert_eq!(dims.validate(), Err(CommitmentError::InvalidParameters));
        assert!(!dims.is_valid());
    }

    #[test]
    fn test_commitment_dimensions_accept_large_valid_dims() {
        // Large dimensions that fit in usize should be accepted
        let dims = CommitmentDimensions {
            message_len: 100_000,
            opening_len: 100_000,
            commitment_len: 10_000,
        };
        assert!(dims.validate().is_ok());
    }

    #[test]
    fn test_commitment_dimensions_serialize_round_trip() {
        let dims = test_dims();
        let bytes = dims.serialize().unwrap();
        let decoded = CommitmentDimensions::deserialize_and_validate_exact(&bytes).unwrap();
        assert_eq!(decoded, dims);
    }

    #[test]
    fn test_linear_commitment_params_large_bound_round_trip() {
        let params = LinearCommitmentParams::<LargeNormBound<BigUint<8>>> {
            dims: test_dims(),
            opening_bound: LargeNormBound {
                max_l2_sq: BigUint::<8>::from_u64(123),
                max_linf: BigUint::<8>::from_u64(7),
            },
        };
        let bytes = params.serialize().unwrap();
        let decoded =
            LinearCommitmentParams::<LargeNormBound<BigUint<8>>>::deserialize_and_validate_exact(
                &bytes,
            )
            .unwrap();
        assert_eq!(decoded, params);
    }

    #[test]
    fn test_recompute_linear_commitment() {
        let dims = test_dims();
        let key = LinearCommitmentKey {
            a_msg: RingMat::new(
                2,
                2,
                vec![
                    F17::from_u64(1),
                    F17::from_u64(2),
                    F17::from_u64(3),
                    F17::from_u64(4),
                ],
            ),
            a_open: RingMat::new(
                2,
                3,
                vec![
                    F17::from_u64(1),
                    F17::from_u64(0),
                    F17::from_u64(1),
                    F17::from_u64(0),
                    F17::from_u64(1),
                    F17::from_u64(1),
                ],
            ),
        };
        let message = RingVec::new(vec![F17::from_u64(5), F17::from_u64(6)]);
        let opening = LinearOpening {
            randomness: RingVec::new(vec![F17::from_u64(1), F17::from_u64(2), F17::from_u64(3)]),
        };
        let commitment: LinearCommitment<F17> =
            recompute_linear_commitment(&key, &message, &opening, dims).expect("valid shapes");
        assert_eq!(commitment.value.len(), 2);
        assert_eq!(commitment.value.get(0).to_u64(), 4);
        assert_eq!(commitment.value.get(1).to_u64(), 10);
    }

    #[test]
    fn test_recompute_rejects_dimension_mismatch() {
        let dims = test_dims();
        let key = LinearCommitmentKey {
            a_msg: RingMat::zero(2, 2),
            a_open: RingMat::zero(2, 3),
        };
        let message = RingVec::new(vec![F17::from_u64(1)]);
        let opening = LinearOpening {
            randomness: RingVec::zero(3),
        };
        let err = recompute_linear_commitment(&key, &message, &opening, dims).unwrap_err();
        assert_eq!(err, CommitmentError::DimensionMismatch);
    }

    #[test]
    fn test_linear_key_deserialize_and_validate_rejects_row_mismatch() {
        let malformed = LinearCommitmentKey {
            a_msg: RingMat::new(1, 2, vec![F17::from_u64(1), F17::from_u64(2)]),
            a_open: RingMat::new(
                2,
                3,
                vec![
                    F17::from_u64(3),
                    F17::from_u64(4),
                    F17::from_u64(5),
                    F17::from_u64(6),
                    F17::from_u64(7),
                    F17::from_u64(8),
                ],
            ),
        };
        assert!(!malformed.is_valid());

        let bytes = malformed.serialize().unwrap();
        let err = LinearCommitmentKey::<F17>::deserialize_and_validate(&bytes).unwrap_err();
        assert_eq!(
            err,
            SerializationError::InvalidData("deserialized value is invalid".into())
        );
    }

    #[test]
    fn test_prepared_linear_recompute_matches_coefficient_path() {
        let dims = CommitmentDimensions {
            message_len: 2,
            opening_len: 2,
            commitment_len: 2,
        };
        let a_msg = RingMat::new(
            2,
            2,
            vec![
                Poly8::one(),
                Poly8::zero(),
                Poly8::from_array(core::array::from_fn(|i| F17::from_u64((i as u64) + 1))),
                Poly8::from_array(core::array::from_fn(|i| F17::from_u64((2 * i as u64) + 1))),
            ],
        );
        let a_open = RingMat::new(
            2,
            2,
            vec![
                Poly8::zero(),
                Poly8::one(),
                Poly8::from_array(core::array::from_fn(|i| F17::from_u64((3 * i as u64) + 1))),
                Poly8::from_array(core::array::from_fn(|i| F17::from_u64((4 * i as u64) + 1))),
            ],
        );
        let message = RingVec::new(vec![
            Poly8::one(),
            Poly8::from_array(core::array::from_fn(|i| F17::from_u64((5 * i as u64) + 1))),
        ]);
        let randomness = RingVec::new(vec![
            Poly8::zero(),
            Poly8::from_array(core::array::from_fn(|i| F17::from_u64((6 * i as u64) + 1))),
        ]);

        let prepared_key = prepare_linear_key(&a_msg, &a_open).unwrap();
        let prepared_message = prepare_poly_vector(&message).unwrap();
        let prepared_randomness = prepare_poly_vector(&randomness).unwrap();

        let coeff = recompute_linear_commitment_parts(&a_msg, &a_open, &message, &randomness, dims)
            .unwrap();
        let prepared = recompute_linear_commitment_prepared(
            &prepared_key,
            &prepared_message,
            &prepared_randomness,
            dims,
        )
        .unwrap();

        assert_eq!(prepared, coeff);
    }
}
