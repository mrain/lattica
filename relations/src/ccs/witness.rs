//! CCS witness containers.

use grid_algebra::arith::ring::Ring;
use grid_algebra::lattice::types::RingVec;
use grid_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError, Valid};

use crate::witness::{WitnessNormMetadata, WitnessNorms};

/// Private witness assignment for a CCS instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CcsWitness<R: Ring, N = WitnessNorms> {
    /// Private witness coordinates.
    pub private_witness: RingVec<R>,
    /// Exact norms of the private witness coordinates.
    pub norms: N,
}

impl<R: Ring, N: WitnessNormMetadata<R>> CcsWitness<R, N> {
    /// Build a witness with exact norm metadata computed from the private assignment.
    pub fn new(private_witness: RingVec<R>) -> Self {
        let norms = N::from_private_witness(&private_witness);
        Self {
            private_witness,
            norms,
        }
    }
}

impl<R: Ring + CanonicalSerialize, N: CanonicalSerialize> CanonicalSerialize for CcsWitness<R, N> {
    fn serialized_size(&self) -> usize {
        self.private_witness.serialized_size() + self.norms.serialized_size()
    }

    fn serialize_into(&self, buf: &mut alloc::vec::Vec<u8>) -> Result<(), SerializationError> {
        self.private_witness.serialize_into(buf)?;
        self.norms.serialize_into(buf)?;
        Ok(())
    }
}

impl<R: Ring + CanonicalDeserialize, N: CanonicalDeserialize> CanonicalDeserialize
    for CcsWitness<R, N>
{
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        let (private_witness, used_witness) = RingVec::<R>::deserialize(data)?;
        let (norms, used_norms) = N::deserialize(&data[used_witness..])?;
        Ok((
            Self {
                private_witness,
                norms,
            },
            used_witness + used_norms,
        ))
    }
}

impl<R: Ring + Valid, N: WitnessNormMetadata<R>> Valid for CcsWitness<R, N> {
    fn is_valid(&self) -> bool {
        self.private_witness.is_valid() && self.norms.matches_private_witness(&self.private_witness)
    }
}
