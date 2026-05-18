//! R1CS instance containers and satisfiability checks.

use alloc::vec::Vec;

use grid_algebra::arith::ring::Ring;
use grid_algebra::lattice::types::{RingMat, RingVec};
use grid_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError, Valid};

use crate::error::RelationsError;
use crate::r1cs::R1csWitness;
use crate::traits::ConstraintSystem;
use crate::witness::{WitnessBoundsMetadata, WitnessNormBounds};

/// A toy R1CS instance over a generic ring backend.
///
/// This implementation uses the standard witness layout
/// `z = [1 || public_inputs || private_witness]` and checks `(A z) o (B z) = C z`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct R1csInstance<R: Ring, B = WitnessNormBounds> {
    /// Number of constraint rows.
    pub num_constraints: usize,
    /// Length of the full witness vector `z`.
    pub num_variables: usize,
    /// Public inputs included in `z` immediately after the leading constant `1`.
    pub public_inputs: RingVec<R>,
    /// Accepted witness norm bounds.
    pub witness_bounds: B,
    /// Left multiplicand matrix.
    pub a: RingMat<R>,
    /// Right multiplicand matrix.
    pub b: RingMat<R>,
    /// Output matrix.
    pub c: RingMat<R>,
}

impl<R: Ring, B> R1csInstance<R, B> {
    /// Create a validated R1CS instance from dense matrices.
    pub fn new(
        public_inputs: RingVec<R>,
        witness_bounds: B,
        a: RingMat<R>,
        b: RingMat<R>,
        c: RingMat<R>,
    ) -> Result<Self, RelationsError> {
        let instance = Self {
            num_constraints: a.rows(),
            num_variables: a.cols(),
            public_inputs,
            witness_bounds,
            a,
            b,
            c,
        };
        instance.validate()?;
        Ok(instance)
    }

    /// Validate the declared dimensions and layout invariants.
    pub fn validate(&self) -> Result<(), RelationsError> {
        if self.num_constraints == 0 || self.num_variables == 0 {
            return Err(RelationsError::InvalidParameters);
        }
        if self.a.rows() != self.num_constraints
            || self.b.rows() != self.num_constraints
            || self.c.rows() != self.num_constraints
            || self.a.cols() != self.num_variables
            || self.b.cols() != self.num_variables
            || self.c.cols() != self.num_variables
        {
            return Err(RelationsError::DimensionMismatch);
        }
        if self.public_inputs.len() + 1 >= self.num_variables {
            return Err(RelationsError::InvalidParameters);
        }
        Ok(())
    }

    fn witness_private_len(&self) -> Result<usize, RelationsError> {
        self.num_variables
            .checked_sub(self.public_inputs.len() + 1)
            .ok_or(RelationsError::InvalidParameters)
    }

    fn witness_vector<N>(&self, witness: &R1csWitness<R, N>) -> Result<RingVec<R>, RelationsError> {
        let expected_private_len = self.witness_private_len()?;
        if witness.private_witness.len() != expected_private_len {
            return Err(RelationsError::DimensionMismatch);
        }

        let mut entries = Vec::with_capacity(self.num_variables);
        entries.push(R::one());
        entries.extend_from_slice(self.public_inputs.entries());
        entries.extend_from_slice(witness.private_witness.entries());
        Ok(RingVec::new(entries))
    }
}

impl<R, B> ConstraintSystem<R> for R1csInstance<R, B>
where
    R: Ring + Valid,
    B: WitnessBoundsMetadata<R>,
{
    type Witness = R1csWitness<R, B::Norms>;

    fn is_satisfied(&self, witness: &Self::Witness) -> Result<bool, RelationsError> {
        self.validate()?;
        if !witness.is_valid() {
            return Err(RelationsError::InvalidWitness);
        }
        if !self
            .witness_bounds
            .check_private_witness(&witness.private_witness)
        {
            return Err(RelationsError::WitnessNormExceeded);
        }

        let z = self.witness_vector(witness)?;
        let az = self.a.mul_vec(&z);
        let bz = self.b.mul_vec(&z);
        let cz = self.c.mul_vec(&z);

        Ok(az
            .entries()
            .iter()
            .zip(bz.entries().iter())
            .zip(cz.entries().iter())
            .all(|((lhs, rhs), out)| lhs.clone() * rhs == *out))
    }
}

impl<R: Ring + CanonicalSerialize, B: CanonicalSerialize> CanonicalSerialize
    for R1csInstance<R, B>
{
    fn serialized_size(&self) -> usize {
        16 + self.public_inputs.serialized_size()
            + self.witness_bounds.serialized_size()
            + self.a.serialized_size()
            + self.b.serialized_size()
            + self.c.serialized_size()
    }

    fn serialize_into(&self, buf: &mut Vec<u8>) -> Result<(), SerializationError> {
        buf.extend_from_slice(&(self.num_constraints as u64).to_le_bytes());
        buf.extend_from_slice(&(self.num_variables as u64).to_le_bytes());
        self.public_inputs.serialize_into(buf)?;
        self.witness_bounds.serialize_into(buf)?;
        self.a.serialize_into(buf)?;
        self.b.serialize_into(buf)?;
        self.c.serialize_into(buf)?;
        Ok(())
    }
}

impl<R: Ring + CanonicalDeserialize, B: CanonicalDeserialize> CanonicalDeserialize
    for R1csInstance<R, B>
{
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        if data.len() < 16 {
            return Err(SerializationError::UnexpectedEnd);
        }
        let num_constraints = usize::try_from(u64::from_le_bytes(data[..8].try_into().unwrap()))
            .map_err(|_| SerializationError::InvalidData("constraint count too large".into()))?;
        let num_variables = usize::try_from(u64::from_le_bytes(data[8..16].try_into().unwrap()))
            .map_err(|_| SerializationError::InvalidData("variable count too large".into()))?;

        let (public_inputs, used_inputs) = RingVec::<R>::deserialize(&data[16..])?;
        let (witness_bounds, used_bounds) = B::deserialize(&data[16 + used_inputs..])?;
        let offset = 16 + used_inputs + used_bounds;
        let (a, used_a) = RingMat::<R>::deserialize(&data[offset..])?;
        let (b, used_b) = RingMat::<R>::deserialize(&data[offset + used_a..])?;
        let (c, used_c) = RingMat::<R>::deserialize(&data[offset + used_a + used_b..])?;

        Ok((
            Self {
                num_constraints,
                num_variables,
                public_inputs,
                witness_bounds,
                a,
                b,
                c,
            },
            offset + used_a + used_b + used_c,
        ))
    }
}

impl<R: Ring + Valid, B: Valid> Valid for R1csInstance<R, B> {
    fn is_valid(&self) -> bool {
        self.public_inputs.is_valid()
            && self.witness_bounds.is_valid()
            && self.a.is_valid()
            && self.b.is_valid()
            && self.c.is_valid()
            && self.validate().is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::witness::{LargeWitnessNormBounds, LargeWitnessNorms};
    use grid_algebra::arith::bigint::BigUint;
    use grid_algebra::arith::large_prime::Bn254Fr;
    use grid_algebra::arith::prime::PrimeField;
    use grid_algebra::arith::ring::{IntegerRing, Ring};
    use grid_algebra::poly::ring::CyclotomicPolyRing;
    use grid_serialize::CanonicalSerialize;

    type F17 = PrimeField<17>;
    type Rq23Np8 = CyclotomicPolyRing<PrimeField<8380417>, 256>;

    fn toy_instance() -> R1csInstance<F17> {
        // z = [1, y, x, w], with x * w = y
        let public_inputs = RingVec::new(vec![F17::from_u64(15)]);
        let witness_bounds = WitnessNormBounds {
            private_witness: grid_algebra::lattice::params::NormBound {
                max_l2_sq: 34,
                max_linf: 5,
            },
        };
        let a = RingMat::new(
            1,
            4,
            vec![F17::zero(), F17::zero(), F17::one(), F17::zero()],
        );
        let b = RingMat::new(
            1,
            4,
            vec![F17::zero(), F17::zero(), F17::zero(), F17::one()],
        );
        let c = RingMat::new(
            1,
            4,
            vec![F17::zero(), F17::one(), F17::zero(), F17::zero()],
        );
        R1csInstance::new(public_inputs, witness_bounds, a, b, c).unwrap()
    }

    #[test]
    fn test_r1cs_accepts_satisfying_witness() {
        let instance = toy_instance();
        let witness = R1csWitness::new(RingVec::new(vec![F17::from_u64(3), F17::from_u64(5)]));
        assert!(instance.is_satisfied(&witness).unwrap());
    }

    #[test]
    fn test_r1cs_rejects_invalid_witness() {
        let instance = toy_instance();
        let witness = R1csWitness::new(RingVec::new(vec![F17::from_u64(1), F17::from_u64(5)]));
        assert!(!instance.is_satisfied(&witness).unwrap());
    }

    #[test]
    fn test_r1cs_rejects_over_norm_witness() {
        let instance = toy_instance();
        let witness = R1csWitness::new(RingVec::new(vec![F17::from_u64(7), F17::from_u64(5)]));
        assert_eq!(
            instance.is_satisfied(&witness),
            Err(RelationsError::WitnessNormExceeded)
        );
    }

    #[test]
    fn test_r1cs_round_trip_rq23_np8() {
        let public_inputs = RingVec::new(vec![Rq23Np8::one()]);
        let witness_bounds = WitnessNormBounds {
            private_witness: grid_algebra::lattice::params::NormBound {
                max_l2_sq: u128::MAX,
                max_linf: u64::MAX,
            },
        };
        let zero = Rq23Np8::zero();
        let one = Rq23Np8::one();
        let a = RingMat::new(1, 3, vec![zero.clone(), zero.clone(), one.clone()]);
        let b = RingMat::new(1, 3, vec![zero.clone(), zero.clone(), one.clone()]);
        let c = RingMat::new(1, 3, vec![zero.clone(), one.clone(), zero.clone()]);
        let instance = R1csInstance::new(public_inputs, witness_bounds, a, b, c).unwrap();
        let bytes = instance.serialize().unwrap();
        let decoded = R1csInstance::<Rq23Np8>::deserialize_and_validate_exact(&bytes).unwrap();
        assert_eq!(decoded, instance);
    }

    #[test]
    fn test_r1cs_round_trip_large_prime_large_bounds() {
        let public_inputs = RingVec::new(vec![Bn254Fr::from_u64(15)]);
        let witness_bounds = LargeWitnessNormBounds {
            private_witness: grid_algebra::lattice::params::LargeNormBound {
                max_l2_sq: BigUint::<8>::from_u64(34),
                max_linf: BigUint::<8>::from_u64(5),
            },
        };
        let a = RingMat::new(
            1,
            4,
            vec![
                Bn254Fr::zero(),
                Bn254Fr::zero(),
                Bn254Fr::one(),
                Bn254Fr::zero(),
            ],
        );
        let b = RingMat::new(
            1,
            4,
            vec![
                Bn254Fr::zero(),
                Bn254Fr::zero(),
                Bn254Fr::zero(),
                Bn254Fr::one(),
            ],
        );
        let c = RingMat::new(
            1,
            4,
            vec![
                Bn254Fr::zero(),
                Bn254Fr::one(),
                Bn254Fr::zero(),
                Bn254Fr::zero(),
            ],
        );
        let instance: R1csInstance<Bn254Fr, LargeWitnessNormBounds<BigUint<8>>> =
            R1csInstance::new(public_inputs, witness_bounds, a, b, c).unwrap();
        let bytes = instance.serialize().unwrap();
        let decoded =
            R1csInstance::<Bn254Fr, LargeWitnessNormBounds<BigUint<8>>>::deserialize_and_validate_exact(&bytes)
                .unwrap();
        let witness: R1csWitness<Bn254Fr, LargeWitnessNorms<BigUint<8>>> =
            R1csWitness::new(RingVec::new(vec![
                Bn254Fr::from_u64(3),
                Bn254Fr::from_u64(5),
            ]));
        assert!(decoded.is_satisfied(&witness).unwrap());
        assert_eq!(witness.norms.private_l2_sq, BigUint::<8>::from_u64(34));
        assert_eq!(witness.norms.private_linf, BigUint::<8>::from_u64(5));
        assert_eq!(decoded.public_inputs, instance.public_inputs);
    }
}
