//! BDLOP parameter and data types.

use core::fmt;

use grid_algebra::arith::ring::Ring;
use grid_algebra::lattice::params::NormBound;
use grid_algebra::lattice::types::{RingMat, RingVec};
use grid_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError, Valid};

use crate::linear::CommitmentDimensions;
use crate::linear::PreparedRuntimeCache;

/// Setup parameters for BDLOP commitments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BdlopParams<B = NormBound> {
    /// Message/opening/commitment dimensions.
    pub dims: CommitmentDimensions,
    /// CBD opening sampler parameter.
    pub opening_eta: usize,
    /// Bound on accepted openings.
    pub opening_bound: B,
    /// Metadata used for docs and benchmark labels.
    pub security_bits: usize,
}

impl<B> CanonicalSerialize for BdlopParams<B>
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

impl<B> CanonicalDeserialize for BdlopParams<B>
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

impl<B> Valid for BdlopParams<B>
where
    B: Valid,
{
    fn is_valid(&self) -> bool {
        self.dims.is_valid() && self.opening_eta > 0 && self.opening_bound.is_valid()
    }
}

/// Public commitment key for BDLOP commitments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BdlopCommitmentKey<R: Ring> {
    /// Masking matrix for the `u` component.
    pub a_mask: RingMat<R>,
    /// Message matrix for the `v` component.
    pub a_msg: RingMat<R>,
    /// Opening/randomness matrix for the `v` component.
    pub a_open: RingMat<R>,
}

impl<R: Ring + CanonicalSerialize> CanonicalSerialize for BdlopCommitmentKey<R> {
    fn serialized_size(&self) -> usize {
        self.a_mask.serialized_size() + self.a_msg.serialized_size() + self.a_open.serialized_size()
    }

    fn serialize_into(&self, buf: &mut alloc::vec::Vec<u8>) -> Result<(), SerializationError> {
        self.a_mask.serialize_into(buf)?;
        self.a_msg.serialize_into(buf)?;
        self.a_open.serialize_into(buf)?;
        Ok(())
    }
}

impl<R: Ring + CanonicalDeserialize> CanonicalDeserialize for BdlopCommitmentKey<R> {
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        let (a_mask, used_mask) = RingMat::<R>::deserialize(data)?;
        let (a_msg, used_msg) = RingMat::<R>::deserialize(&data[used_mask..])?;
        let (a_open, used_open) = RingMat::<R>::deserialize(&data[used_mask + used_msg..])?;
        Ok((
            Self {
                a_mask,
                a_msg,
                a_open,
            },
            used_mask + used_msg + used_open,
        ))
    }
}

impl<R: Ring + Valid> Valid for BdlopCommitmentKey<R> {
    fn is_valid(&self) -> bool {
        self.a_mask.is_valid()
            && self.a_msg.is_valid()
            && self.a_open.is_valid()
            && self.a_mask.rows() == self.a_msg.rows()
            && self.a_msg.rows() == self.a_open.rows()
            && self.a_mask.cols() == self.a_open.cols()
    }
}

/// Commitment output for BDLOP commitments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BdlopCommitment<R: Ring> {
    /// First commitment component.
    pub u: RingVec<R>,
    /// Second commitment component.
    pub v: RingVec<R>,
}

impl<R: Ring + CanonicalSerialize> CanonicalSerialize for BdlopCommitment<R> {
    fn serialized_size(&self) -> usize {
        self.u.serialized_size() + self.v.serialized_size()
    }

    fn serialize_into(&self, buf: &mut alloc::vec::Vec<u8>) -> Result<(), SerializationError> {
        self.u.serialize_into(buf)?;
        self.v.serialize_into(buf)?;
        Ok(())
    }
}

impl<R: Ring + CanonicalDeserialize> CanonicalDeserialize for BdlopCommitment<R> {
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        let (u, used_u) = RingVec::<R>::deserialize(data)?;
        let (v, used_v) = RingVec::<R>::deserialize(&data[used_u..])?;
        Ok((Self { u, v }, used_u + used_v))
    }
}

impl<R: Ring + Valid> Valid for BdlopCommitment<R> {
    fn is_valid(&self) -> bool {
        self.u.is_valid() && self.v.is_valid() && self.u.len() == self.v.len()
    }
}

/// Opening material for BDLOP commitments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BdlopOpening<R: Ring> {
    /// Opening randomness vector.
    pub randomness: RingVec<R>,
}

impl<R: Ring + CanonicalSerialize> CanonicalSerialize for BdlopOpening<R> {
    fn serialized_size(&self) -> usize {
        self.randomness.serialized_size()
    }

    fn serialize_into(&self, buf: &mut alloc::vec::Vec<u8>) -> Result<(), SerializationError> {
        self.randomness.serialize_into(buf)
    }
}

impl<R: Ring + CanonicalDeserialize> CanonicalDeserialize for BdlopOpening<R> {
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        let (randomness, used) = RingVec::<R>::deserialize(data)?;
        Ok((Self { randomness }, used))
    }
}

impl<R: Ring + Valid> Valid for BdlopOpening<R> {
    fn is_valid(&self) -> bool {
        self.randomness.is_valid()
    }
}

/// A concrete BDLOP commitment scheme instance.
pub struct BdlopCommitmentScheme<R: Ring, B = NormBound> {
    pub(crate) params: BdlopParams<B>,
    pub(crate) key: BdlopCommitmentKey<R>,
    pub(crate) prepared_key: Option<PreparedRuntimeCache>,
    pub(crate) prepared_mask: Option<PreparedRuntimeCache>,
}

impl<R: Ring, B: Clone> Clone for BdlopCommitmentScheme<R, B> {
    fn clone(&self) -> Self {
        Self {
            params: self.params.clone(),
            key: self.key.clone(),
            prepared_key: None,
            prepared_mask: None,
        }
    }
}

impl<R: Ring, B: PartialEq> PartialEq for BdlopCommitmentScheme<R, B> {
    fn eq(&self, other: &Self) -> bool {
        self.params == other.params && self.key == other.key
    }
}

impl<R: Ring, B: Eq> Eq for BdlopCommitmentScheme<R, B> {}

impl<R: Ring, B: fmt::Debug> fmt::Debug for BdlopCommitmentScheme<R, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BdlopCommitmentScheme")
            .field("params", &self.params)
            .field("key", &self.key)
            .finish_non_exhaustive()
    }
}
