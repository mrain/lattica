//! Gadget commitment parameter and data types.

use core::fmt;

use grid_algebra::arith::ring::Ring;
use grid_algebra::lattice::params::NormBound;
use grid_algebra::lattice::types::RingVec;
use grid_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError, Valid};

use crate::linear::CommitmentDimensions;
use crate::linear::PreparedRuntimeCache;

/// Setup parameters for gadget commitments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GadgetParams<B = NormBound> {
    /// Message/opening/commitment dimensions.
    pub dims: CommitmentDimensions,
    /// CBD opening sampler parameter.
    pub opening_eta: usize,
    /// Bound on accepted openings.
    pub opening_bound: B,
    /// Gadget base.
    pub base: u64,
    /// Number of gadget digits.
    pub digits: usize,
    /// Metadata used for docs and benchmark labels.
    pub security_bits: usize,
}

impl<B> CanonicalSerialize for GadgetParams<B>
where
    B: CanonicalSerialize,
{
    fn serialized_size(&self) -> usize {
        self.dims.serialized_size() + 8 + self.opening_bound.serialized_size() + 8 + 8 + 8
    }

    fn serialize_into(&self, buf: &mut alloc::vec::Vec<u8>) -> Result<(), SerializationError> {
        self.dims.serialize_into(buf)?;
        buf.extend_from_slice(&(self.opening_eta as u64).to_le_bytes());
        self.opening_bound.serialize_into(buf)?;
        buf.extend_from_slice(&self.base.to_le_bytes());
        buf.extend_from_slice(&(self.digits as u64).to_le_bytes());
        buf.extend_from_slice(&(self.security_bits as u64).to_le_bytes());
        Ok(())
    }
}

impl<B> CanonicalDeserialize for GadgetParams<B>
where
    B: CanonicalDeserialize,
{
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        let (dims, used_dims) = CommitmentDimensions::deserialize(data)?;
        if data.len() < used_dims + 8 {
            return Err(SerializationError::UnexpectedEnd);
        }
        let opening_eta = usize::try_from(u64::from_le_bytes(
            data[used_dims..used_dims + 8].try_into().unwrap(),
        ))
        .map_err(|_| SerializationError::InvalidData("opening eta too large".into()))?;
        let (opening_bound, used_bound) = B::deserialize(&data[used_dims + 8..])?;
        let base_offset = used_dims + 8 + used_bound;
        if data.len() < base_offset + 24 {
            return Err(SerializationError::UnexpectedEnd);
        }
        let base = u64::from_le_bytes(data[base_offset..base_offset + 8].try_into().unwrap());
        let digits = usize::try_from(u64::from_le_bytes(
            data[base_offset + 8..base_offset + 16].try_into().unwrap(),
        ))
        .map_err(|_| SerializationError::InvalidData("digit count too large".into()))?;
        let security_bits = usize::try_from(u64::from_le_bytes(
            data[base_offset + 16..base_offset + 24].try_into().unwrap(),
        ))
        .map_err(|_| SerializationError::InvalidData("security bits too large".into()))?;
        Ok((
            Self {
                dims,
                opening_eta,
                opening_bound,
                base,
                digits,
                security_bits,
            },
            base_offset + 24,
        ))
    }
}

impl<B> Valid for GadgetParams<B>
where
    B: Valid,
{
    fn is_valid(&self) -> bool {
        self.dims.is_valid()
            && self.opening_eta > 0
            && self.opening_bound.is_valid()
            && self.base >= 2
            && self.digits > 0
            && self.dims.message_len == self.dims.commitment_len
    }
}

/// Commitment output for gadget commitments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GadgetCommitment<R: Ring> {
    /// Commitment vector.
    pub value: RingVec<R>,
}

impl<R: Ring + CanonicalSerialize> CanonicalSerialize for GadgetCommitment<R> {
    fn serialized_size(&self) -> usize {
        self.value.serialized_size()
    }

    fn serialize_into(&self, buf: &mut alloc::vec::Vec<u8>) -> Result<(), SerializationError> {
        self.value.serialize_into(buf)
    }
}

impl<R: Ring + CanonicalDeserialize> CanonicalDeserialize for GadgetCommitment<R> {
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        let (value, used) = RingVec::<R>::deserialize(data)?;
        Ok((Self { value }, used))
    }
}

impl<R: Ring + Valid> Valid for GadgetCommitment<R> {
    fn is_valid(&self) -> bool {
        self.value.is_valid()
    }
}

/// Opening material for gadget commitments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GadgetOpening<R: Ring> {
    /// Opening randomness vector.
    pub randomness: RingVec<R>,
    /// Gadget digit vector.
    pub digits: RingVec<R>,
}

impl<R: Ring + CanonicalSerialize> CanonicalSerialize for GadgetOpening<R> {
    fn serialized_size(&self) -> usize {
        self.randomness.serialized_size() + self.digits.serialized_size()
    }

    fn serialize_into(&self, buf: &mut alloc::vec::Vec<u8>) -> Result<(), SerializationError> {
        self.randomness.serialize_into(buf)?;
        self.digits.serialize_into(buf)?;
        Ok(())
    }
}

impl<R: Ring + CanonicalDeserialize> CanonicalDeserialize for GadgetOpening<R> {
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        let (randomness, used_rand) = RingVec::<R>::deserialize(data)?;
        let (digits, used_digits) = RingVec::<R>::deserialize(&data[used_rand..])?;
        Ok((Self { randomness, digits }, used_rand + used_digits))
    }
}

impl<R: Ring + Valid> Valid for GadgetOpening<R> {
    fn is_valid(&self) -> bool {
        self.randomness.is_valid() && self.digits.is_valid()
    }
}

/// A concrete gadget commitment scheme instance.
pub struct GadgetCommitmentScheme<R: Ring, B = NormBound> {
    pub(crate) params: GadgetParams<B>,
    pub(crate) a_open: grid_algebra::lattice::types::RingMat<R>,
    pub(crate) g_matrix: grid_algebra::lattice::types::RingMat<R>,
    pub(crate) prepared_a_open: Option<PreparedRuntimeCache>,
}

impl<R: Ring, B: Clone> Clone for GadgetCommitmentScheme<R, B> {
    fn clone(&self) -> Self {
        Self {
            params: self.params.clone(),
            a_open: self.a_open.clone(),
            g_matrix: self.g_matrix.clone(),
            prepared_a_open: None,
        }
    }
}

impl<R: Ring, B: PartialEq> PartialEq for GadgetCommitmentScheme<R, B> {
    fn eq(&self, other: &Self) -> bool {
        self.params == other.params
            && self.a_open == other.a_open
            && self.g_matrix == other.g_matrix
    }
}

impl<R: Ring, B: Eq> Eq for GadgetCommitmentScheme<R, B> {}

impl<R: Ring, B: fmt::Debug> fmt::Debug for GadgetCommitmentScheme<R, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GadgetCommitmentScheme")
            .field("params", &self.params)
            .field("a_open", &self.a_open)
            .field("g_matrix", &self.g_matrix)
            .finish_non_exhaustive()
    }
}
