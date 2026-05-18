//! Toy SIS/LWE/RLWE/MLWE problem definitions and verifier helpers.

use grid_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError, Valid};

use crate::arith::ring::{IntegerRing, Ring};
use crate::lattice::params::NormBound;
use crate::lattice::sampling::CoeffSampler;
use crate::lattice::sampling::toy::{sample_mat, sample_vec};
use crate::lattice::types::{RingMat, RingVec};
use crate::poly::ring::PolyRing;

/// An LWE instance `(A, b = A*s + e)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LweInstance<R: IntegerRing> {
    /// Matrix `A` of shape `m x n`.
    pub a: RingMat<R>,
    /// Vector `b` of length `m`.
    pub b: RingVec<R>,
}

/// An LWE witness `(s, e)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LweWitness<R: IntegerRing> {
    /// Secret vector of length `n`.
    pub secret: RingVec<R>,
    /// Error vector of length `m`.
    pub error: RingVec<R>,
}

/// A SIS instance consisting of a matrix and a norm bound.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SisInstance<R: IntegerRing> {
    /// Matrix `A` of shape `m x n`.
    pub a: RingMat<R>,
    /// Bound on acceptable witnesses.
    pub bound: NormBound,
}

/// A SIS witness `x`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SisWitness<R: IntegerRing> {
    /// Witness vector of length `n`.
    pub x: RingVec<R>,
}

/// An RLWE instance `(a, b = a*s + e)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RlweInstance<P: PolyRing> {
    /// Public ring element `a`.
    pub a: P,
    /// Public ring element `b`.
    pub b: P,
}

/// An RLWE witness `(s, e)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RlweWitness<P: PolyRing> {
    /// Secret ring element.
    pub secret: P,
    /// Error ring element.
    pub error: P,
}

/// An MLWE instance `(A, b = A*s + e)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlweInstance<P: PolyRing> {
    /// Matrix over the polynomial ring.
    pub a: RingMat<P>,
    /// Right-hand side vector.
    pub b: RingVec<P>,
}

/// An MLWE witness `(s, e)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlweWitness<P: PolyRing> {
    /// Secret vector.
    pub secret: RingVec<P>,
    /// Error vector.
    pub error: RingVec<P>,
}

impl<R> CanonicalSerialize for LweInstance<R>
where
    R: IntegerRing + CanonicalSerialize,
{
    fn serialized_size(&self) -> usize {
        self.a.serialized_size() + self.b.serialized_size()
    }

    fn serialize_into(&self, buf: &mut alloc::vec::Vec<u8>) -> Result<(), SerializationError> {
        self.a.serialize_into(buf)?;
        self.b.serialize_into(buf)?;
        Ok(())
    }
}

impl<R> CanonicalDeserialize for LweInstance<R>
where
    R: IntegerRing + CanonicalDeserialize,
{
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        let (a, used_a) = RingMat::<R>::deserialize(data)?;
        let (b, used_b) = RingVec::<R>::deserialize(&data[used_a..])?;
        Ok((Self { a, b }, used_a + used_b))
    }
}

impl<R> Valid for LweInstance<R>
where
    R: IntegerRing + Valid,
{
    fn is_valid(&self) -> bool {
        self.a.is_valid() && self.b.is_valid() && self.a.rows() == self.b.len()
    }
}

impl<R> CanonicalSerialize for LweWitness<R>
where
    R: IntegerRing + CanonicalSerialize,
{
    fn serialized_size(&self) -> usize {
        self.secret.serialized_size() + self.error.serialized_size()
    }

    fn serialize_into(&self, buf: &mut alloc::vec::Vec<u8>) -> Result<(), SerializationError> {
        self.secret.serialize_into(buf)?;
        self.error.serialize_into(buf)?;
        Ok(())
    }
}

impl<R> CanonicalDeserialize for LweWitness<R>
where
    R: IntegerRing + CanonicalDeserialize,
{
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        let (secret, used_s) = RingVec::<R>::deserialize(data)?;
        let (error, used_e) = RingVec::<R>::deserialize(&data[used_s..])?;
        Ok((Self { secret, error }, used_s + used_e))
    }
}

impl<R> Valid for LweWitness<R>
where
    R: IntegerRing + Valid,
{
    fn is_valid(&self) -> bool {
        self.secret.is_valid() && self.error.is_valid()
    }
}

impl<R> CanonicalSerialize for SisInstance<R>
where
    R: IntegerRing + CanonicalSerialize,
{
    fn serialized_size(&self) -> usize {
        self.a.serialized_size() + 24
    }

    fn serialize_into(&self, buf: &mut alloc::vec::Vec<u8>) -> Result<(), SerializationError> {
        self.a.serialize_into(buf)?;
        buf.extend_from_slice(&self.bound.max_l2_sq.to_le_bytes());
        buf.extend_from_slice(&self.bound.max_linf.to_le_bytes());
        Ok(())
    }
}

impl<R> CanonicalDeserialize for SisInstance<R>
where
    R: IntegerRing + CanonicalDeserialize,
{
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        let (a, used_a) = RingMat::<R>::deserialize(data)?;
        if data.len() < used_a + 24 {
            return Err(SerializationError::UnexpectedEnd);
        }
        let max_l2_sq = u128::from_le_bytes(data[used_a..used_a + 16].try_into().unwrap());
        let max_linf = u64::from_le_bytes(data[used_a + 16..used_a + 24].try_into().unwrap());
        Ok((
            Self {
                a,
                bound: NormBound {
                    max_l2_sq,
                    max_linf,
                },
            },
            used_a + 24,
        ))
    }
}

impl<R> Valid for SisInstance<R>
where
    R: IntegerRing + Valid,
{
    fn is_valid(&self) -> bool {
        self.a.is_valid()
    }
}

impl<R> CanonicalSerialize for SisWitness<R>
where
    R: IntegerRing + CanonicalSerialize,
{
    fn serialized_size(&self) -> usize {
        self.x.serialized_size()
    }

    fn serialize_into(&self, buf: &mut alloc::vec::Vec<u8>) -> Result<(), SerializationError> {
        self.x.serialize_into(buf)
    }
}

impl<R> CanonicalDeserialize for SisWitness<R>
where
    R: IntegerRing + CanonicalDeserialize,
{
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        let (x, used) = RingVec::<R>::deserialize(data)?;
        Ok((Self { x }, used))
    }
}

impl<R> Valid for SisWitness<R>
where
    R: IntegerRing + Valid,
{
    fn is_valid(&self) -> bool {
        self.x.is_valid()
    }
}

impl<P> CanonicalSerialize for RlweInstance<P>
where
    P: PolyRing,
{
    fn serialized_size(&self) -> usize {
        self.a.serialized_size() + self.b.serialized_size()
    }

    fn serialize_into(&self, buf: &mut alloc::vec::Vec<u8>) -> Result<(), SerializationError> {
        self.a.serialize_into(buf)?;
        self.b.serialize_into(buf)?;
        Ok(())
    }
}

impl<P> CanonicalDeserialize for RlweInstance<P>
where
    P: PolyRing,
{
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        let (a, used_a) = P::deserialize(data)?;
        let (b, used_b) = P::deserialize(&data[used_a..])?;
        Ok((Self { a, b }, used_a + used_b))
    }
}

impl<P> Valid for RlweInstance<P>
where
    P: PolyRing + Valid,
{
    fn is_valid(&self) -> bool {
        self.a.is_valid() && self.b.is_valid()
    }
}

impl<P> CanonicalSerialize for RlweWitness<P>
where
    P: PolyRing,
{
    fn serialized_size(&self) -> usize {
        self.secret.serialized_size() + self.error.serialized_size()
    }

    fn serialize_into(&self, buf: &mut alloc::vec::Vec<u8>) -> Result<(), SerializationError> {
        self.secret.serialize_into(buf)?;
        self.error.serialize_into(buf)?;
        Ok(())
    }
}

impl<P> CanonicalDeserialize for RlweWitness<P>
where
    P: PolyRing,
{
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        let (secret, used_s) = P::deserialize(data)?;
        let (error, used_e) = P::deserialize(&data[used_s..])?;
        Ok((Self { secret, error }, used_s + used_e))
    }
}

impl<P> Valid for RlweWitness<P>
where
    P: PolyRing + Valid,
{
    fn is_valid(&self) -> bool {
        self.secret.is_valid() && self.error.is_valid()
    }
}

impl<P> CanonicalSerialize for MlweInstance<P>
where
    P: PolyRing,
{
    fn serialized_size(&self) -> usize {
        self.a.serialized_size() + self.b.serialized_size()
    }

    fn serialize_into(&self, buf: &mut alloc::vec::Vec<u8>) -> Result<(), SerializationError> {
        self.a.serialize_into(buf)?;
        self.b.serialize_into(buf)?;
        Ok(())
    }
}

impl<P> CanonicalDeserialize for MlweInstance<P>
where
    P: PolyRing,
{
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        let (a, used_a) = RingMat::<P>::deserialize(data)?;
        let (b, used_b) = RingVec::<P>::deserialize(&data[used_a..])?;
        Ok((Self { a, b }, used_a + used_b))
    }
}

impl<P> Valid for MlweInstance<P>
where
    P: PolyRing + Valid,
{
    fn is_valid(&self) -> bool {
        self.a.is_valid() && self.b.is_valid() && self.a.rows() == self.b.len()
    }
}

impl<P> CanonicalSerialize for MlweWitness<P>
where
    P: PolyRing,
{
    fn serialized_size(&self) -> usize {
        self.secret.serialized_size() + self.error.serialized_size()
    }

    fn serialize_into(&self, buf: &mut alloc::vec::Vec<u8>) -> Result<(), SerializationError> {
        self.secret.serialize_into(buf)?;
        self.error.serialize_into(buf)?;
        Ok(())
    }
}

impl<P> CanonicalDeserialize for MlweWitness<P>
where
    P: PolyRing,
{
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        let (secret, used_s) = RingVec::<P>::deserialize(data)?;
        let (error, used_e) = RingVec::<P>::deserialize(&data[used_s..])?;
        Ok((Self { secret, error }, used_s + used_e))
    }
}

impl<P> Valid for MlweWitness<P>
where
    P: PolyRing + Valid,
{
    fn is_valid(&self) -> bool {
        self.secret.is_valid() && self.error.is_valid()
    }
}

/// Generate a toy LWE instance and witness.
pub fn lwe_generate<R, SA, SS, SE, T>(
    rng: &mut T,
    n: usize,
    m: usize,
    a_sampler: &SA,
    secret_sampler: &SS,
    error_sampler: &SE,
) -> (LweInstance<R>, LweWitness<R>)
where
    R: IntegerRing,
    SA: CoeffSampler<R>,
    SS: CoeffSampler<R>,
    SE: CoeffSampler<R>,
    T: grid_std::rand::Rng,
{
    let a = sample_mat(a_sampler, rng, m, n);
    let secret = sample_vec(secret_sampler, rng, n);
    let error = sample_vec(error_sampler, rng, m);
    let b = a.mul_vec(&secret) + &error;
    (LweInstance { a, b }, LweWitness { secret, error })
}

/// Verify a toy LWE witness.
pub fn lwe_verify<R>(instance: &LweInstance<R>, witness: &LweWitness<R>) -> bool
where
    R: IntegerRing,
{
    if instance.a.rows() != instance.b.len()
        || instance.a.cols() != witness.secret.len()
        || instance.a.rows() != witness.error.len()
    {
        return false;
    }
    instance.a.mul_vec(&witness.secret) + &witness.error == instance.b
}

/// Verify a SIS witness against a matrix and norm bound.
pub fn sis_verify<R>(a: &RingMat<R>, x: &RingVec<R>, bound: &NormBound) -> bool
where
    R: IntegerRing + crate::lattice::params::NormedRing,
{
    if a.cols() != x.len() {
        return false;
    }
    let syndrome = a.mul_vec(x);
    syndrome.entries().iter().all(Ring::is_zero) && bound.check(x)
}

/// Verify an RLWE witness.
pub fn rlwe_verify<P: PolyRing>(instance: &RlweInstance<P>, witness: &RlweWitness<P>) -> bool {
    P::mul_ref(&instance.a, &witness.secret) + &witness.error == instance.b
}

/// Verify an MLWE witness.
pub fn mlwe_verify<P: PolyRing>(instance: &MlweInstance<P>, witness: &MlweWitness<P>) -> bool {
    if instance.a.rows() != instance.b.len()
        || instance.a.cols() != witness.secret.len()
        || instance.a.rows() != witness.error.len()
    {
        return false;
    }
    instance.a.mul_vec(&witness.secret) + &witness.error == instance.b
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arith::large_prime::Bn254Fr;
    use crate::arith::prime::PrimeField;
    use crate::arith::ring::IntegerRing;
    use crate::lattice::params::NormBound;
    use crate::lattice::sampling::toy::{
        CBDSampler, LargeCBDSampler, LargeTernarySampler, LargeUniformSampler, TernarySampler,
        UniformSampler, sample_poly,
    };
    use grid_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError, Valid};

    type F17 = PrimeField<17>;
    #[test]
    fn test_lwe_generate_and_verify() {
        let mut rng = grid_std::test_rng();
        let a_sampler = UniformSampler::<F17>::new();
        let secret_sampler = TernarySampler::<F17>::new();
        let error_sampler = CBDSampler::<F17>::new(2);
        let (instance, witness) =
            lwe_generate(&mut rng, 4, 5, &a_sampler, &secret_sampler, &error_sampler);
        assert!(lwe_verify(&instance, &witness));

        let mut bad = witness.clone();
        bad.error.entries_mut()[0] += F17::one();
        assert!(!lwe_verify(&instance, &bad));

        let malformed = LweWitness {
            secret: RingVec::zero(3),
            error: RingVec::zero(5),
        };
        assert!(!lwe_verify(&instance, &malformed));
    }

    #[test]
    fn test_lwe_generate_and_verify_over_large_prime_backend() {
        let mut rng = grid_std::test_rng();
        let a_sampler = LargeUniformSampler::<Bn254Fr>::new();
        let secret_sampler = LargeTernarySampler::<Bn254Fr>::new();
        let error_sampler = LargeCBDSampler::<Bn254Fr>::new(2);
        let (instance, witness) =
            lwe_generate(&mut rng, 4, 5, &a_sampler, &secret_sampler, &error_sampler);
        assert!(lwe_verify(&instance, &witness));

        let mut bad = witness.clone();
        bad.error.entries_mut()[0] += Bn254Fr::one();
        assert!(!lwe_verify(&instance, &bad));
    }

    #[test]
    fn test_sis_verify() {
        let a = RingMat::new(
            2,
            2,
            vec![
                F17::from_u64(1),
                F17::from_u64(0),
                F17::from_u64(0),
                F17::from_u64(1),
            ],
        );
        let x = RingVec::zero(2);
        let bound = NormBound {
            max_l2_sq: 0,
            max_linf: 0,
        };
        assert!(sis_verify(&a, &x, &bound));

        let bad = RingVec::new(vec![F17::from_u64(1), F17::from_u64(0)]);
        assert!(!sis_verify(&a, &bad, &bound));

        let malformed = RingVec::zero(1);
        assert!(!sis_verify(&a, &malformed, &bound));
    }

    #[test]
    fn test_rlwe_and_mlwe_verify() {
        let mut rng = grid_std::test_rng();
        let sampler = UniformSampler::<F17>::new();
        let a = sample_poly::<F17, _, _, 8>(&sampler, &mut rng);
        let secret = sample_poly::<F17, _, _, 8>(&sampler, &mut rng);
        let error = sample_poly::<F17, _, _, 8>(&sampler, &mut rng);
        let b = a.clone() * &secret + &error;
        let instance = RlweInstance { a: a.clone(), b };
        let witness = RlweWitness {
            secret: secret.clone(),
            error: error.clone(),
        };
        assert!(rlwe_verify(&instance, &witness));

        let mat = RingMat::new(1, 1, vec![a]);
        let secret_vec = RingVec::new(vec![secret]);
        let error_vec = RingVec::new(vec![error]);
        let b_vec = mat.mul_vec(&secret_vec) + &error_vec;
        let instance = MlweInstance { a: mat, b: b_vec };
        let witness = MlweWitness {
            secret: secret_vec,
            error: error_vec,
        };
        assert!(mlwe_verify(&instance, &witness));

        let malformed = MlweWitness {
            secret: RingVec::zero(2),
            error: RingVec::zero(1),
        };
        assert!(!mlwe_verify(&instance, &malformed));
    }

    #[test]
    fn test_problem_serialization_round_trip() {
        let instance = LweInstance {
            a: RingMat::new(1, 2, vec![F17::from_u64(1), F17::from_u64(2)]),
            b: RingVec::new(vec![F17::from_u64(3)]),
        };
        let bytes = instance.serialize().unwrap();
        let decoded = LweInstance::<F17>::deserialize_exact(&bytes).unwrap();
        assert_eq!(decoded, instance);

        let sis = SisInstance {
            a: RingMat::new(1, 1, vec![F17::from_u64(1)]),
            bound: NormBound {
                max_l2_sq: 9,
                max_linf: 3,
            },
        };
        let bytes = sis.serialize().unwrap();
        assert_eq!(sis.serialized_size(), bytes.len());
    }

    #[test]
    fn test_problem_validity_checks_shapes() {
        let invalid_lwe = LweInstance {
            a: RingMat::new(2, 1, vec![F17::from_u64(1), F17::from_u64(2)]),
            b: RingVec::new(vec![F17::from_u64(3)]),
        };
        assert!(!invalid_lwe.is_valid());

        let sampler = UniformSampler::<F17>::new();
        let mut rng = grid_std::test_rng();
        let invalid_mlwe = MlweInstance {
            a: RingMat::new(
                2,
                1,
                vec![
                    sample_poly::<F17, _, _, 8>(&sampler, &mut rng),
                    sample_poly::<F17, _, _, 8>(&sampler, &mut rng),
                ],
            ),
            b: RingVec::new(vec![sample_poly::<F17, _, _, 8>(&sampler, &mut rng)]),
        };
        assert!(!invalid_mlwe.is_valid());
    }

    #[test]
    fn test_lwe_deserialize_and_validate_rejects_bad_shape() {
        let invalid = LweInstance {
            a: RingMat::new(2, 1, vec![F17::from_u64(1), F17::from_u64(2)]),
            b: RingVec::new(vec![F17::from_u64(3)]),
        };
        let bytes = invalid.serialize().unwrap();
        let err = LweInstance::<F17>::deserialize_and_validate(&bytes).unwrap_err();
        assert_eq!(
            err,
            SerializationError::InvalidData("deserialized value is invalid".into())
        );
    }

    #[test]
    fn test_sis_deserialize_rejects_truncated_bound() {
        let sis = SisInstance {
            a: RingMat::new(1, 1, vec![F17::from_u64(1)]),
            bound: NormBound {
                max_l2_sq: 9,
                max_linf: 3,
            },
        };
        let bytes = sis.serialize().unwrap();
        let err = SisInstance::<F17>::deserialize(&bytes[..bytes.len() - 1]).unwrap_err();
        assert_eq!(err, SerializationError::UnexpectedEnd);
    }
}
