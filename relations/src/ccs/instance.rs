//! CCS instance containers and satisfiability checks.

use alloc::vec::Vec;

use grid_algebra::arith::ring::Ring;
use grid_algebra::lattice::types::{RingMat, RingVec};
use grid_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError, Valid};

use crate::ccs::CcsWitness;
use crate::error::RelationsError;
use crate::traits::ConstraintSystem;
use crate::witness::{WitnessBoundsMetadata, WitnessNormBounds};

/// A single CCS term `selector * Π_i (M_{idx_i} z)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CcsTerm<R> {
    /// Scalar selector for this term.
    pub selector: R,
    /// Matrices whose row evaluations are multiplied together.
    pub matrix_indices: Vec<usize>,
}

impl<R> CcsTerm<R> {
    fn validate(&self, num_matrices: usize) -> Result<(), RelationsError> {
        if self.matrix_indices.is_empty()
            || self.matrix_indices.iter().any(|&idx| idx >= num_matrices)
        {
            return Err(RelationsError::InvalidParameters);
        }
        Ok(())
    }
}

impl<R: CanonicalSerialize> CanonicalSerialize for CcsTerm<R> {
    fn serialized_size(&self) -> usize {
        self.selector.serialized_size() + 8 + self.matrix_indices.len() * 8
    }

    fn serialize_into(&self, buf: &mut Vec<u8>) -> Result<(), SerializationError> {
        self.selector.serialize_into(buf)?;
        buf.extend_from_slice(&(self.matrix_indices.len() as u64).to_le_bytes());
        for &idx in &self.matrix_indices {
            buf.extend_from_slice(&(idx as u64).to_le_bytes());
        }
        Ok(())
    }
}

impl<R: CanonicalDeserialize> CanonicalDeserialize for CcsTerm<R> {
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        let (selector, used_selector) = R::deserialize(data)?;
        if data.len() < used_selector + 8 {
            return Err(SerializationError::UnexpectedEnd);
        }
        let len = usize::try_from(u64::from_le_bytes(
            data[used_selector..used_selector + 8].try_into().unwrap(),
        ))
        .map_err(|_| SerializationError::InvalidData("term length too large".into()))?;
        let mut used = used_selector + 8;
        let remaining = data.len() - used;
        if len > remaining / 8 {
            return Err(SerializationError::UnexpectedEnd);
        }
        let mut matrix_indices = Vec::new();
        matrix_indices
            .try_reserve_exact(len)
            .map_err(|_| SerializationError::InvalidData("term length too large".into()))?;
        for _ in 0..len {
            if data.len() < used + 8 {
                return Err(SerializationError::UnexpectedEnd);
            }
            matrix_indices.push(
                usize::try_from(u64::from_le_bytes(data[used..used + 8].try_into().unwrap()))
                    .map_err(|_| {
                        SerializationError::InvalidData("matrix index too large".into())
                    })?,
            );
            used += 8;
        }
        Ok((
            Self {
                selector,
                matrix_indices,
            },
            used,
        ))
    }
}

impl<R: Valid> Valid for CcsTerm<R> {
    fn is_valid(&self) -> bool {
        self.selector.is_valid() && !self.matrix_indices.is_empty()
    }
}

/// A toy CCS instance over a generic ring backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CcsInstance<R: Ring, B = WitnessNormBounds> {
    /// Number of constraint rows.
    pub num_constraints: usize,
    /// Length of the full witness vector `z`.
    pub num_variables: usize,
    /// Public inputs included in `z` immediately after the leading constant `1`.
    pub public_inputs: RingVec<R>,
    /// Accepted witness norm bounds.
    pub witness_bounds: B,
    /// Dense matrices used by the CCS relation.
    pub matrices: Vec<RingMat<R>>,
    /// Terms whose row-wise sum must vanish.
    pub terms: Vec<CcsTerm<R>>,
}

impl<R: Ring, B> CcsInstance<R, B> {
    /// Create a validated CCS instance from dense matrices and terms.
    pub fn new(
        public_inputs: RingVec<R>,
        witness_bounds: B,
        matrices: Vec<RingMat<R>>,
        terms: Vec<CcsTerm<R>>,
    ) -> Result<Self, RelationsError> {
        let Some(first) = matrices.first() else {
            return Err(RelationsError::InvalidParameters);
        };
        let instance = Self {
            num_constraints: first.rows(),
            num_variables: first.cols(),
            public_inputs,
            witness_bounds,
            matrices,
            terms,
        };
        instance.validate()?;
        Ok(instance)
    }

    /// Validate the declared dimensions and term references.
    pub fn validate(&self) -> Result<(), RelationsError> {
        if self.num_constraints == 0 || self.num_variables == 0 {
            return Err(RelationsError::InvalidParameters);
        }
        if self.public_inputs.len() + 1 >= self.num_variables {
            return Err(RelationsError::InvalidParameters);
        }
        if self.matrices.is_empty() || self.terms.is_empty() {
            return Err(RelationsError::InvalidParameters);
        }
        for matrix in &self.matrices {
            if matrix.rows() != self.num_constraints || matrix.cols() != self.num_variables {
                return Err(RelationsError::DimensionMismatch);
            }
        }
        for term in &self.terms {
            term.validate(self.matrices.len())?;
        }
        Ok(())
    }

    fn witness_private_len(&self) -> Result<usize, RelationsError> {
        self.num_variables
            .checked_sub(self.public_inputs.len() + 1)
            .ok_or(RelationsError::InvalidParameters)
    }

    fn witness_vector<N>(&self, witness: &CcsWitness<R, N>) -> Result<RingVec<R>, RelationsError> {
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

impl<R, B> ConstraintSystem<R> for CcsInstance<R, B>
where
    R: Ring + Valid,
    B: WitnessBoundsMetadata<R>,
{
    type Witness = CcsWitness<R, B::Norms>;

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
        let evaluations: Vec<_> = self
            .matrices
            .iter()
            .map(|matrix| matrix.mul_vec(&z))
            .collect();
        for row in 0..self.num_constraints {
            let mut acc = R::zero();
            for term in &self.terms {
                let mut term_value = term.selector.clone();
                for &matrix_idx in &term.matrix_indices {
                    term_value *= evaluations[matrix_idx].get(row).clone();
                }
                acc += term_value;
            }
            if !acc.is_zero() {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

impl<R: Ring + CanonicalSerialize, B: CanonicalSerialize> CanonicalSerialize for CcsInstance<R, B> {
    fn serialized_size(&self) -> usize {
        16 + self.public_inputs.serialized_size()
            + self.witness_bounds.serialized_size()
            + 8
            + self
                .matrices
                .iter()
                .map(CanonicalSerialize::serialized_size)
                .sum::<usize>()
            + 8
            + self
                .terms
                .iter()
                .map(CanonicalSerialize::serialized_size)
                .sum::<usize>()
    }

    fn serialize_into(&self, buf: &mut Vec<u8>) -> Result<(), SerializationError> {
        buf.extend_from_slice(&(self.num_constraints as u64).to_le_bytes());
        buf.extend_from_slice(&(self.num_variables as u64).to_le_bytes());
        self.public_inputs.serialize_into(buf)?;
        self.witness_bounds.serialize_into(buf)?;
        buf.extend_from_slice(&(self.matrices.len() as u64).to_le_bytes());
        for matrix in &self.matrices {
            matrix.serialize_into(buf)?;
        }
        buf.extend_from_slice(&(self.terms.len() as u64).to_le_bytes());
        for term in &self.terms {
            term.serialize_into(buf)?;
        }
        Ok(())
    }
}

impl<R: Ring + CanonicalDeserialize, B: CanonicalDeserialize> CanonicalDeserialize
    for CcsInstance<R, B>
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
        let mut used = 16 + used_inputs + used_bounds;
        if data.len() < used + 8 {
            return Err(SerializationError::UnexpectedEnd);
        }
        let matrix_len =
            usize::try_from(u64::from_le_bytes(data[used..used + 8].try_into().unwrap()))
                .map_err(|_| SerializationError::InvalidData("matrix count too large".into()))?;
        used += 8;
        if matrix_len > data.len() - used {
            return Err(SerializationError::UnexpectedEnd);
        }
        let mut matrices = Vec::new();
        matrices
            .try_reserve_exact(matrix_len)
            .map_err(|_| SerializationError::InvalidData("matrix count too large".into()))?;
        for _ in 0..matrix_len {
            let (matrix, matrix_used) = RingMat::<R>::deserialize(&data[used..])?;
            matrices.push(matrix);
            used += matrix_used;
        }
        if data.len() < used + 8 {
            return Err(SerializationError::UnexpectedEnd);
        }
        let term_len =
            usize::try_from(u64::from_le_bytes(data[used..used + 8].try_into().unwrap()))
                .map_err(|_| SerializationError::InvalidData("term count too large".into()))?;
        used += 8;
        if term_len > data.len() - used {
            return Err(SerializationError::UnexpectedEnd);
        }
        let mut terms = Vec::new();
        terms
            .try_reserve_exact(term_len)
            .map_err(|_| SerializationError::InvalidData("term count too large".into()))?;
        for _ in 0..term_len {
            let (term, term_used) = CcsTerm::<R>::deserialize(&data[used..])?;
            terms.push(term);
            used += term_used;
        }

        Ok((
            Self {
                num_constraints,
                num_variables,
                public_inputs,
                witness_bounds,
                matrices,
                terms,
            },
            used,
        ))
    }
}

impl<R: Ring + Valid, B: Valid> Valid for CcsInstance<R, B> {
    fn is_valid(&self) -> bool {
        self.public_inputs.is_valid()
            && self.witness_bounds.is_valid()
            && self.matrices.iter().all(Valid::is_valid)
            && self.terms.iter().all(Valid::is_valid)
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
    use grid_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError};

    type F17 = PrimeField<17>;

    fn toy_instance() -> CcsInstance<F17> {
        // z = [1, y, x, w], with x * w - y = 0
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
        let terms = vec![
            CcsTerm {
                selector: F17::one(),
                matrix_indices: vec![0, 1],
            },
            CcsTerm {
                selector: -F17::one(),
                matrix_indices: vec![2],
            },
        ];
        CcsInstance::new(public_inputs, witness_bounds, vec![a, b, c], terms).unwrap()
    }

    #[test]
    fn test_ccs_accepts_satisfying_witness() {
        let instance = toy_instance();
        let witness = CcsWitness::new(RingVec::new(vec![F17::from_u64(3), F17::from_u64(5)]));
        assert!(instance.is_satisfied(&witness).unwrap());
    }

    #[test]
    fn test_ccs_rejects_invalid_witness() {
        let instance = toy_instance();
        let witness = CcsWitness::new(RingVec::new(vec![F17::from_u64(1), F17::from_u64(5)]));
        assert!(!instance.is_satisfied(&witness).unwrap());
    }

    #[test]
    fn test_ccs_rejects_over_norm_witness() {
        let instance = toy_instance();
        let witness = CcsWitness::new(RingVec::new(vec![F17::from_u64(7), F17::from_u64(5)]));
        assert_eq!(
            instance.is_satisfied(&witness),
            Err(RelationsError::WitnessNormExceeded)
        );
    }

    #[test]
    fn test_ccs_instance_serialize_round_trip() {
        let instance = toy_instance();
        let bytes = instance.serialize().unwrap();
        assert_eq!(bytes.len(), instance.serialized_size());
        let decoded = CcsInstance::<F17>::deserialize_exact(&bytes).unwrap();
        assert_eq!(decoded, instance);
    }

    #[test]
    fn test_ccs_term_deserialize_rejects_impossible_index_count() {
        let mut bytes = F17::one().serialize().unwrap();
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());
        let err = CcsTerm::<F17>::deserialize(&bytes).unwrap_err();
        assert_eq!(err, SerializationError::UnexpectedEnd);
    }

    #[test]
    fn test_ccs_instance_deserialize_rejects_impossible_matrix_count() {
        let instance = toy_instance();
        let mut bytes = instance.serialize().unwrap();
        let matrix_count_offset = 16
            + instance.public_inputs.serialized_size()
            + instance.witness_bounds.serialized_size();
        bytes.truncate(matrix_count_offset + 8);
        bytes[matrix_count_offset..matrix_count_offset + 8]
            .copy_from_slice(&u64::MAX.to_le_bytes());
        let err = CcsInstance::<F17>::deserialize(&bytes).unwrap_err();
        assert_eq!(err, SerializationError::UnexpectedEnd);
    }

    #[test]
    fn test_ccs_accepts_satisfying_witness_with_large_bounds() {
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
        let terms = vec![
            CcsTerm {
                selector: Bn254Fr::one(),
                matrix_indices: vec![0, 1],
            },
            CcsTerm {
                selector: -Bn254Fr::one(),
                matrix_indices: vec![2],
            },
        ];
        let instance: CcsInstance<Bn254Fr, LargeWitnessNormBounds<BigUint<8>>> =
            CcsInstance::new(public_inputs, witness_bounds, vec![a, b, c], terms).unwrap();
        let witness: CcsWitness<Bn254Fr, LargeWitnessNorms<BigUint<8>>> =
            CcsWitness::new(RingVec::new(vec![
                Bn254Fr::from_u64(3),
                Bn254Fr::from_u64(5),
            ]));
        assert!(instance.is_satisfied(&witness).unwrap());
    }
}
