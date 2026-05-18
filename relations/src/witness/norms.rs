//! Explicit witness norm metadata.

use grid_algebra::arith::ring::Ring;
use grid_algebra::lattice::params::{
    LargeNormBound, LargeNormStats, LargeNormValue, LargeNormedRing, NormBound, NormStats,
    NormedRing,
};
use grid_algebra::lattice::types::RingVec;
use grid_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError, Valid};

/// Exact witness norms stored alongside a private witness assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WitnessNorms {
    /// Exact squared `L2` norm of the private witness assignment.
    pub private_l2_sq: u128,
    /// Exact `L∞` norm of the private witness assignment.
    pub private_linf: u64,
}

/// Norm metadata carried alongside a private witness assignment.
pub trait WitnessNormMetadata<R: Ring>:
    CanonicalSerialize + CanonicalDeserialize + Valid + Clone + PartialEq + Eq
{
    /// Compute exact witness norms from a private witness vector.
    fn from_private_witness(private_witness: &RingVec<R>) -> Self;

    /// Return whether these norms match the supplied private witness exactly.
    fn matches_private_witness(&self, private_witness: &RingVec<R>) -> bool;
}

impl WitnessNorms {
    /// Compute witness norms from a private witness vector.
    pub fn from_private_witness<R: NormedRing>(private_witness: &RingVec<R>) -> Self {
        let stats = NormStats::compute(private_witness);
        Self {
            private_l2_sq: stats.l2_sq,
            private_linf: stats.linf,
        }
    }

    /// Return whether these norms match the supplied private witness exactly.
    pub fn matches_private_witness<R: NormedRing>(&self, private_witness: &RingVec<R>) -> bool {
        *self == Self::from_private_witness(private_witness)
    }
}

impl<R: NormedRing> WitnessNormMetadata<R> for WitnessNorms {
    fn from_private_witness(private_witness: &RingVec<R>) -> Self {
        Self::from_private_witness(private_witness)
    }

    fn matches_private_witness(&self, private_witness: &RingVec<R>) -> bool {
        self.matches_private_witness(private_witness)
    }
}

impl CanonicalSerialize for WitnessNorms {
    fn serialized_size(&self) -> usize {
        24
    }

    fn serialize_into(&self, buf: &mut alloc::vec::Vec<u8>) -> Result<(), SerializationError> {
        buf.extend_from_slice(&self.private_l2_sq.to_le_bytes());
        buf.extend_from_slice(&self.private_linf.to_le_bytes());
        Ok(())
    }
}

impl CanonicalDeserialize for WitnessNorms {
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        if data.len() < 24 {
            return Err(SerializationError::UnexpectedEnd);
        }
        Ok((
            Self {
                private_l2_sq: u128::from_le_bytes(data[..16].try_into().unwrap()),
                private_linf: u64::from_le_bytes(data[16..24].try_into().unwrap()),
            },
            24,
        ))
    }
}

impl Valid for WitnessNorms {
    fn is_valid(&self) -> bool {
        true
    }
}

/// Exact witness norms stored alongside a private witness assignment for large-modulus backends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LargeWitnessNorms<T> {
    /// Exact squared `L2` norm of the private witness assignment.
    pub private_l2_sq: T,
    /// Exact `L∞` norm of the private witness assignment.
    pub private_linf: T,
}

impl<T: LargeNormValue> LargeWitnessNorms<T> {
    /// Compute witness norms from a private witness vector.
    pub fn from_private_witness<R>(private_witness: &RingVec<R>) -> Self
    where
        R: LargeNormedRing<Norm = T>,
    {
        let stats = LargeNormStats::compute(private_witness);
        Self {
            private_l2_sq: stats.l2_sq,
            private_linf: stats.linf,
        }
    }

    /// Return whether these norms match the supplied private witness exactly.
    pub fn matches_private_witness<R>(&self, private_witness: &RingVec<R>) -> bool
    where
        R: LargeNormedRing<Norm = T>,
    {
        self == &Self::from_private_witness(private_witness)
    }
}

impl<R, T> WitnessNormMetadata<R> for LargeWitnessNorms<T>
where
    R: LargeNormedRing<Norm = T>,
    T: LargeNormValue + CanonicalSerialize + CanonicalDeserialize + Valid,
{
    fn from_private_witness(private_witness: &RingVec<R>) -> Self {
        Self::from_private_witness(private_witness)
    }

    fn matches_private_witness(&self, private_witness: &RingVec<R>) -> bool {
        self.matches_private_witness(private_witness)
    }
}

impl<T> CanonicalSerialize for LargeWitnessNorms<T>
where
    T: CanonicalSerialize,
{
    fn serialized_size(&self) -> usize {
        self.private_l2_sq.serialized_size() + self.private_linf.serialized_size()
    }

    fn serialize_into(&self, buf: &mut alloc::vec::Vec<u8>) -> Result<(), SerializationError> {
        self.private_l2_sq.serialize_into(buf)?;
        self.private_linf.serialize_into(buf)?;
        Ok(())
    }
}

impl<T> CanonicalDeserialize for LargeWitnessNorms<T>
where
    T: CanonicalDeserialize,
{
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        let (private_l2_sq, used_l2) = T::deserialize(data)?;
        let (private_linf, used_linf) = T::deserialize(&data[used_l2..])?;
        Ok((
            Self {
                private_l2_sq,
                private_linf,
            },
            used_l2 + used_linf,
        ))
    }
}

impl<T> Valid for LargeWitnessNorms<T>
where
    T: Valid,
{
    fn is_valid(&self) -> bool {
        self.private_l2_sq.is_valid() && self.private_linf.is_valid()
    }
}

/// Explicit witness norm bounds carried by a relation instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WitnessNormBounds {
    /// Accepted bound for the private witness assignment.
    pub private_witness: NormBound,
}

/// Bound metadata carried by a relation instance.
pub trait WitnessBoundsMetadata<R: Ring>:
    CanonicalSerialize + CanonicalDeserialize + Valid + Clone + PartialEq + Eq
{
    /// Exact witness norm metadata type paired with this bound object.
    type Norms: WitnessNormMetadata<R>;

    /// Check whether a private witness satisfies the declared bound.
    fn check_private_witness(&self, private_witness: &RingVec<R>) -> bool;
}

impl WitnessNormBounds {
    /// Check whether a private witness satisfies the declared bound.
    pub fn check_private_witness<R: NormedRing>(&self, private_witness: &RingVec<R>) -> bool {
        self.private_witness.check(private_witness)
    }
}

impl<R: NormedRing> WitnessBoundsMetadata<R> for WitnessNormBounds {
    type Norms = WitnessNorms;

    fn check_private_witness(&self, private_witness: &RingVec<R>) -> bool {
        self.check_private_witness(private_witness)
    }
}

impl CanonicalSerialize for WitnessNormBounds {
    fn serialized_size(&self) -> usize {
        self.private_witness.serialized_size()
    }

    fn serialize_into(&self, buf: &mut alloc::vec::Vec<u8>) -> Result<(), SerializationError> {
        self.private_witness.serialize_into(buf)
    }
}

impl CanonicalDeserialize for WitnessNormBounds {
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        let (private_witness, used) = NormBound::deserialize(data)?;
        Ok((Self { private_witness }, used))
    }
}

impl Valid for WitnessNormBounds {
    fn is_valid(&self) -> bool {
        self.private_witness.is_valid()
    }
}

/// Explicit witness norm bounds carried by a relation instance for large-modulus backends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LargeWitnessNormBounds<T> {
    /// Accepted bound for the private witness assignment.
    pub private_witness: LargeNormBound<T>,
}

impl<T: LargeNormValue> LargeWitnessNormBounds<T> {
    /// Check whether a private witness satisfies the declared bound.
    pub fn check_private_witness<R>(&self, private_witness: &RingVec<R>) -> bool
    where
        R: LargeNormedRing<Norm = T>,
    {
        self.private_witness.check(private_witness)
    }
}

impl<R, T> WitnessBoundsMetadata<R> for LargeWitnessNormBounds<T>
where
    R: LargeNormedRing<Norm = T>,
    T: LargeNormValue + CanonicalSerialize + CanonicalDeserialize + Valid,
{
    type Norms = LargeWitnessNorms<T>;

    fn check_private_witness(&self, private_witness: &RingVec<R>) -> bool {
        self.check_private_witness(private_witness)
    }
}

impl<T> CanonicalSerialize for LargeWitnessNormBounds<T>
where
    T: CanonicalSerialize,
{
    fn serialized_size(&self) -> usize {
        self.private_witness.serialized_size()
    }

    fn serialize_into(&self, buf: &mut alloc::vec::Vec<u8>) -> Result<(), SerializationError> {
        self.private_witness.serialize_into(buf)
    }
}

impl<T> CanonicalDeserialize for LargeWitnessNormBounds<T>
where
    T: CanonicalDeserialize,
{
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        let (private_witness, used) = LargeNormBound::<T>::deserialize(data)?;
        Ok((Self { private_witness }, used))
    }
}

impl<T> Valid for LargeWitnessNormBounds<T>
where
    T: Valid,
{
    fn is_valid(&self) -> bool {
        self.private_witness.is_valid()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use grid_algebra::arith::LargeCanonicalRing;
    use grid_algebra::arith::bigint::BigUint;
    use grid_algebra::arith::large_prime::Bn254Fr;
    use grid_algebra::arith::large_rns::Rns3V0;
    use grid_algebra::arith::prime::PrimeField;
    use grid_algebra::arith::ring::IntegerRing;
    use grid_algebra::lattice::types::RingVec;

    type F17 = PrimeField<17>;

    #[test]
    fn test_witness_norms_match_private_witness() {
        let private = RingVec::new(vec![F17::from_u64(1), F17::from_u64(16)]);
        let norms = WitnessNorms::from_private_witness(&private);
        assert!(norms.matches_private_witness(&private));
    }

    #[test]
    fn test_witness_norm_bounds_check_private_witness() {
        let private = RingVec::new(vec![F17::from_u64(1), F17::from_u64(16)]);
        let bounds = WitnessNormBounds {
            private_witness: NormBound {
                max_l2_sq: 2,
                max_linf: 1,
            },
        };
        assert!(bounds.check_private_witness(&private));
    }

    #[test]
    fn test_large_witness_norms_match_private_witness_over_large_prime() {
        let modulus = Bn254Fr::modulus_canonical();
        let (modulus_minus_one, borrow) = modulus.sub_small(1);
        assert!(!borrow);
        let private = RingVec::new(vec![
            Bn254Fr::from_u64(5),
            Bn254Fr::from_canonical(&modulus_minus_one),
        ]);
        let norms = LargeWitnessNorms::<BigUint<8>>::from_private_witness(&private);
        assert_eq!(norms.private_l2_sq, BigUint::<8>::from_u64(26));
        assert_eq!(norms.private_linf, BigUint::<8>::from_u64(5));
        assert!(norms.matches_private_witness(&private));
    }

    #[test]
    fn test_large_witness_norm_bounds_check_private_witness_over_large_rns() {
        let modulus = Rns3V0::modulus_canonical();
        let (modulus_minus_two, borrow) = modulus.sub_small(2);
        assert!(!borrow);
        let private = RingVec::new(vec![
            Rns3V0::from_u64(7),
            Rns3V0::from_canonical(&modulus_minus_two),
        ]);
        let bounds = LargeWitnessNormBounds {
            private_witness: LargeNormBound {
                max_l2_sq: BigUint::<6>::from_u64(53),
                max_linf: BigUint::<6>::from_u64(7),
            },
        };
        assert!(bounds.check_private_witness(&private));
        assert!(bounds.is_valid());
    }

    #[test]
    fn test_large_witness_norm_metadata_round_trip() {
        let norms = LargeWitnessNorms {
            private_l2_sq: BigUint::<8>::from_u64(123),
            private_linf: BigUint::<8>::from_u64(7),
        };
        let bytes = norms.serialize().unwrap();
        let decoded = LargeWitnessNorms::<BigUint<8>>::deserialize_exact(&bytes).unwrap();
        assert_eq!(decoded, norms);

        let bounds = LargeWitnessNormBounds {
            private_witness: LargeNormBound {
                max_l2_sq: BigUint::<8>::from_u64(123),
                max_linf: BigUint::<8>::from_u64(7),
            },
        };
        let bytes = bounds.serialize().unwrap();
        let decoded = LargeWitnessNormBounds::<BigUint<8>>::deserialize_exact(&bytes).unwrap();
        assert_eq!(decoded, bounds);
    }
}
