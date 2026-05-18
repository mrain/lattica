//! Ajtai parameter and data types.

use core::fmt;

use grid_algebra::arith::ring::Ring;
use grid_algebra::lattice::params::NormBound;
use grid_algebra::lattice::types::{RingMat, RingVec};
use grid_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError, Valid};

use crate::linear::CommitmentDimensions;
use crate::linear::PreparedRuntimeCache;

/// Setup parameters for Ajtai commitments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AjtaiParams<B = NormBound> {
    /// Message/opening/commitment dimensions.
    pub dims: CommitmentDimensions,
    /// CBD opening sampler parameter.
    pub opening_eta: usize,
    /// Bound on accepted openings.
    pub opening_bound: B,
    /// Metadata used for docs and benchmark labels.
    pub security_bits: usize,
}

impl<B> CanonicalSerialize for AjtaiParams<B>
where
    B: CanonicalSerialize,
{
    fn serialized_size(&self) -> usize {
        self.dims.serialized_size() + 8 + self.opening_bound.serialized_size() + 8
    }

    fn serialize_into(&self, buf: &mut alloc::vec::Vec<u8>) -> Result<(), SerializationError> {
        self.dims.serialize_into(buf)?;
        buf.extend_from_slice(&(self.opening_eta as u64).to_le_bytes());
        self.opening_bound.serialize_into(buf)?;
        buf.extend_from_slice(&(self.security_bits as u64).to_le_bytes());
        Ok(())
    }
}

impl<B> CanonicalDeserialize for AjtaiParams<B>
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
        let security_offset = used_dims + 8 + used_bound;
        if data.len() < security_offset + 8 {
            return Err(SerializationError::UnexpectedEnd);
        }
        let security_bits = usize::try_from(u64::from_le_bytes(
            data[security_offset..security_offset + 8]
                .try_into()
                .unwrap(),
        ))
        .map_err(|_| SerializationError::InvalidData("security bits too large".into()))?;
        Ok((
            Self {
                dims,
                opening_eta,
                opening_bound,
                security_bits,
            },
            security_offset + 8,
        ))
    }
}

impl<B> Valid for AjtaiParams<B>
where
    B: Valid,
{
    fn is_valid(&self) -> bool {
        self.dims.is_valid() && self.opening_eta > 0 && self.opening_bound.is_valid()
    }
}

/// Public commitment key for Ajtai commitments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AjtaiCommitmentKey<R: Ring> {
    /// Message matrix.
    pub a_msg: RingMat<R>,
    /// Opening/randomness matrix.
    pub a_open: RingMat<R>,
}

impl<R: Ring + CanonicalSerialize> CanonicalSerialize for AjtaiCommitmentKey<R> {
    fn serialized_size(&self) -> usize {
        self.a_msg.serialized_size() + self.a_open.serialized_size()
    }

    fn serialize_into(&self, buf: &mut alloc::vec::Vec<u8>) -> Result<(), SerializationError> {
        self.a_msg.serialize_into(buf)?;
        self.a_open.serialize_into(buf)?;
        Ok(())
    }
}

impl<R: Ring + CanonicalDeserialize> CanonicalDeserialize for AjtaiCommitmentKey<R> {
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        let (a_msg, used_msg) = RingMat::<R>::deserialize(data)?;
        let (a_open, used_open) = RingMat::<R>::deserialize(&data[used_msg..])?;
        Ok((Self { a_msg, a_open }, used_msg + used_open))
    }
}

impl<R: Ring + Valid> Valid for AjtaiCommitmentKey<R> {
    fn is_valid(&self) -> bool {
        self.a_msg.is_valid() && self.a_open.is_valid() && self.a_msg.rows() == self.a_open.rows()
    }
}

/// Commitment output for Ajtai commitments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AjtaiCommitment<R: Ring> {
    /// Commitment vector.
    pub value: RingVec<R>,
}

impl<R: Ring + CanonicalSerialize> CanonicalSerialize for AjtaiCommitment<R> {
    fn serialized_size(&self) -> usize {
        self.value.serialized_size()
    }

    fn serialize_into(&self, buf: &mut alloc::vec::Vec<u8>) -> Result<(), SerializationError> {
        self.value.serialize_into(buf)
    }
}

impl<R: Ring + CanonicalDeserialize> CanonicalDeserialize for AjtaiCommitment<R> {
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        let (value, used) = RingVec::<R>::deserialize(data)?;
        Ok((Self { value }, used))
    }
}

impl<R: Ring + Valid> Valid for AjtaiCommitment<R> {
    fn is_valid(&self) -> bool {
        self.value.is_valid()
    }
}

/// Opening material for Ajtai commitments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AjtaiOpening<R: Ring> {
    /// Opening randomness vector.
    pub randomness: RingVec<R>,
}

impl<R: Ring + CanonicalSerialize> CanonicalSerialize for AjtaiOpening<R> {
    fn serialized_size(&self) -> usize {
        self.randomness.serialized_size()
    }

    fn serialize_into(&self, buf: &mut alloc::vec::Vec<u8>) -> Result<(), SerializationError> {
        self.randomness.serialize_into(buf)
    }
}

impl<R: Ring + CanonicalDeserialize> CanonicalDeserialize for AjtaiOpening<R> {
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        let (randomness, used) = RingVec::<R>::deserialize(data)?;
        Ok((Self { randomness }, used))
    }
}

impl<R: Ring + Valid> Valid for AjtaiOpening<R> {
    fn is_valid(&self) -> bool {
        self.randomness.is_valid()
    }
}

/// A concrete Ajtai commitment scheme instance.
pub struct AjtaiCommitmentScheme<R: Ring, B = NormBound> {
    pub(crate) params: AjtaiParams<B>,
    pub(crate) key: AjtaiCommitmentKey<R>,
    pub(crate) prepared_key: Option<PreparedRuntimeCache>,
}

impl<R: Ring, B: Clone> Clone for AjtaiCommitmentScheme<R, B> {
    fn clone(&self) -> Self {
        Self {
            params: self.params.clone(),
            key: self.key.clone(),
            prepared_key: None,
        }
    }
}

impl<R: Ring, B: PartialEq> PartialEq for AjtaiCommitmentScheme<R, B> {
    fn eq(&self, other: &Self) -> bool {
        self.params == other.params && self.key == other.key
    }
}

impl<R: Ring, B: Eq> Eq for AjtaiCommitmentScheme<R, B> {}

impl<R: Ring, B: fmt::Debug> fmt::Debug for AjtaiCommitmentScheme<R, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AjtaiCommitmentScheme")
            .field("params", &self.params)
            .field("key", &self.key)
            .finish_non_exhaustive()
    }
}
