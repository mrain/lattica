//! Polynomial ring trait and cyclotomic polynomial ring implementation.

use alloc::vec::Vec;
use core::array::from_fn;
use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use grid_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError, Valid};

use crate::arith::ntt::{NTTRing, NttError};
use crate::arith::ring::{IntegerRing, Ring};
use crate::arith::z2k::Z2K;
use crate::poly::ntt::{cached_twisted_plan, poly_mul_ntt};

/// Construction errors for polynomial ring elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolyError {
    /// The cyclotomic degree is not supported.
    InvalidDegree { degree: usize },
    /// The coefficient slice length does not match the degree.
    WrongCoeffCount { expected: usize, got: usize },
}

/// A polynomial ring `R_q = Z_q[X] / f(X)`.
pub trait PolyRing: Ring + CanonicalSerialize + CanonicalDeserialize {
    /// The coefficient ring.
    type Coeff: IntegerRing + CanonicalSerialize + CanonicalDeserialize;

    /// The degree `n` of the quotient polynomial.
    fn degree() -> usize;

    /// Get the coefficient at index `i`.
    fn coeff(&self, i: usize) -> Self::Coeff;

    /// Set the coefficient at index `i`.
    fn set_coeff(&mut self, i: usize, val: Self::Coeff);

    /// Create from a coefficient slice.
    fn try_from_coeffs(coeffs: &[Self::Coeff]) -> Result<Self, PolyError>;

    /// Borrow the coefficient slice.
    fn coeffs(&self) -> &[Self::Coeff];

    /// Borrow the coefficient slice mutably.
    fn coeffs_mut(&mut self) -> &mut [Self::Coeff];
}

/// Coefficient rings that can multiply negacyclic polynomials of degree `N`.
pub trait NegacyclicMulRing<const N: usize>:
    IntegerRing + CanonicalSerialize + CanonicalDeserialize
{
    /// Multiply two negacyclic coefficient arrays.
    fn negacyclic_mul_coeffs(a: &[Self; N], b: &[Self; N]) -> [Self; N];
}

const fn is_valid_degree(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

fn naive_negacyclic_mul_coeffs<R, const N: usize>(a: &[R; N], b: &[R; N]) -> [R; N]
where
    R: IntegerRing + CanonicalSerialize + CanonicalDeserialize,
{
    let mut out = from_fn(|_| R::zero());
    for (i, a_i) in a.iter().enumerate() {
        for (j, b_j) in b.iter().enumerate() {
            let prod = R::mul_ref(a_i, b_j);
            let idx = i + j;
            if idx < N {
                out[idx] += prod;
            } else {
                out[idx - N] -= prod;
            }
        }
    }
    out
}

impl<R, const N: usize> NegacyclicMulRing<N> for R
where
    R: IntegerRing + NTTRing + CanonicalSerialize + CanonicalDeserialize + Send + Sync + 'static,
{
    fn negacyclic_mul_coeffs(a: &[Self; N], b: &[Self; N]) -> [Self; N] {
        if let Some(twice_n) = N.checked_mul(2)
            && Self::supports_ntt(twice_n)
        {
            if let Ok(plan) = cached_twisted_plan::<Self>(N) {
                let mut lhs = a.clone();
                let mut rhs = b.clone();
                if plan.multiply_in_place(&mut lhs, &mut rhs).is_ok() {
                    return lhs;
                }
            }
            if let Ok(coeffs) = poly_mul_ntt(a, b)
                && let Ok(coeffs) = <Vec<Self> as TryInto<[Self; N]>>::try_into(coeffs)
            {
                return coeffs;
            }
        }
        naive_negacyclic_mul_coeffs(a, b)
    }
}

impl<const K: u32, const N: usize> NegacyclicMulRing<N> for Z2K<K> {
    fn negacyclic_mul_coeffs(a: &[Self; N], b: &[Self; N]) -> [Self; N] {
        naive_negacyclic_mul_coeffs(a, b)
    }
}

/// `Z_q[X] / (X^N + 1)` for power-of-two `N`.
#[derive(Clone, PartialEq, Eq)]
pub struct CyclotomicPolyRing<
    R: IntegerRing + CanonicalSerialize + CanonicalDeserialize,
    const N: usize,
> {
    coeffs: [R; N],
}

impl<R: IntegerRing + CanonicalSerialize + CanonicalDeserialize, const N: usize>
    CyclotomicPolyRing<R, N>
{
    /// The validated cyclotomic degree.
    pub const DEGREE: usize = {
        assert!(
            is_valid_degree(N),
            "cyclotomic degree must be a non-zero power of two"
        );
        N
    };

    #[inline(always)]
    fn check_degree() {
        let _ = Self::DEGREE;
    }

    /// Create from a trusted coefficient array.
    pub fn from_array(coeffs: [R; N]) -> Self {
        Self::check_degree();
        Self { coeffs }
    }

    /// Borrow the underlying coefficient array.
    pub(crate) fn coeff_array(&self) -> &[R; N] {
        &self.coeffs
    }

    /// Multiply every coefficient by a scalar.
    pub fn scalar_mul(&self, s: &R) -> Self {
        let mut coeffs = self.coeffs.clone();
        R::scalar_mul_slice(&mut coeffs, s);
        Self::from_array(coeffs)
    }

    /// Multiply every coefficient by a scalar.
    pub fn scalar_mul_assign(&mut self, s: &R) {
        R::scalar_mul_slice(&mut self.coeffs, s);
    }

    /// Evaluate the polynomial at `x` using Horner's rule.
    pub fn evaluate(&self, x: &R) -> R {
        let mut acc = R::zero();
        for coeff in self.coeffs.iter().rev() {
            acc *= x;
            acc += coeff;
        }
        acc
    }
}

impl<R, const N: usize> CyclotomicPolyRing<R, N>
where
    R: NegacyclicMulRing<N>,
{
    /// Naive negacyclic convolution in `Z_q[X] / (X^N + 1)`.
    pub fn neg_cyclic_mul(a: &Self, b: &Self) -> Self {
        Self::from_array(naive_negacyclic_mul_coeffs(&a.coeffs, &b.coeffs))
    }

    fn add_impl(&self, rhs: &Self) -> Self {
        let mut coeffs = self.coeffs.clone();
        R::add_assign_slice(&mut coeffs, &rhs.coeffs);
        Self::from_array(coeffs)
    }

    fn sub_impl(&self, rhs: &Self) -> Self {
        let mut coeffs = self.coeffs.clone();
        R::sub_assign_slice(&mut coeffs, &rhs.coeffs);
        Self::from_array(coeffs)
    }

    fn mul_impl(&self, rhs: &Self) -> Self {
        Self::from_array(R::negacyclic_mul_coeffs(&self.coeffs, &rhs.coeffs))
    }
}

impl<R: IntegerRing + CanonicalSerialize + CanonicalDeserialize, const N: usize> fmt::Debug
    for CyclotomicPolyRing<R, N>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("CyclotomicPolyRing")
            .field(&self.coeffs)
            .finish()
    }
}

impl<R, const N: usize> Ring for CyclotomicPolyRing<R, N>
where
    R: NegacyclicMulRing<N>,
{
    fn zero() -> Self {
        Self::from_array(from_fn(|_| R::zero()))
    }

    fn one() -> Self {
        Self::from_array(from_fn(|i| if i == 0 { R::one() } else { R::zero() }))
    }

    fn add_ref(a: &Self, b: &Self) -> Self {
        a + b
    }

    fn sub_ref(a: &Self, b: &Self) -> Self {
        a - b
    }

    fn mul_ref(a: &Self, b: &Self) -> Self {
        a * b
    }
}

impl<R, const N: usize> PolyRing for CyclotomicPolyRing<R, N>
where
    R: NegacyclicMulRing<N>,
{
    type Coeff = R;

    fn degree() -> usize {
        Self::DEGREE
    }

    fn coeff(&self, i: usize) -> Self::Coeff {
        self.coeffs[i].clone()
    }

    fn set_coeff(&mut self, i: usize, val: Self::Coeff) {
        self.coeffs[i] = val;
    }

    fn try_from_coeffs(coeffs: &[Self::Coeff]) -> Result<Self, PolyError> {
        if !is_valid_degree(N) {
            return Err(PolyError::InvalidDegree { degree: N });
        }
        if coeffs.len() != N {
            return Err(PolyError::WrongCoeffCount {
                expected: N,
                got: coeffs.len(),
            });
        }
        Ok(Self::from_array(from_fn(|i| coeffs[i].clone())))
    }

    fn coeffs(&self) -> &[Self::Coeff] {
        &self.coeffs
    }

    fn coeffs_mut(&mut self) -> &mut [Self::Coeff] {
        &mut self.coeffs
    }
}

impl<R, const N: usize> Add for CyclotomicPolyRing<R, N>
where
    R: NegacyclicMulRing<N>,
{
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self {
        self += rhs;
        self
    }
}

impl<R, const N: usize> Add<&Self> for CyclotomicPolyRing<R, N>
where
    R: NegacyclicMulRing<N>,
{
    type Output = Self;

    fn add(self, rhs: &Self) -> Self {
        self.add_impl(rhs)
    }
}

impl<R, const N: usize> Add<Self> for &CyclotomicPolyRing<R, N>
where
    R: NegacyclicMulRing<N>,
{
    type Output = CyclotomicPolyRing<R, N>;

    fn add(self, rhs: Self) -> Self::Output {
        self.add_impl(rhs)
    }
}

impl<R, const N: usize> Add<CyclotomicPolyRing<R, N>> for &CyclotomicPolyRing<R, N>
where
    R: NegacyclicMulRing<N>,
{
    type Output = CyclotomicPolyRing<R, N>;

    fn add(self, rhs: CyclotomicPolyRing<R, N>) -> Self::Output {
        self.add_impl(&rhs)
    }
}

impl<R, const N: usize> AddAssign for CyclotomicPolyRing<R, N>
where
    R: NegacyclicMulRing<N>,
{
    fn add_assign(&mut self, rhs: Self) {
        R::add_assign_slice(&mut self.coeffs, &rhs.coeffs);
    }
}

impl<R, const N: usize> AddAssign<&Self> for CyclotomicPolyRing<R, N>
where
    R: NegacyclicMulRing<N>,
{
    fn add_assign(&mut self, rhs: &Self) {
        R::add_assign_slice(&mut self.coeffs, &rhs.coeffs);
    }
}

impl<R, const N: usize> Sub for CyclotomicPolyRing<R, N>
where
    R: NegacyclicMulRing<N>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        self.sub_impl(&rhs)
    }
}

impl<R, const N: usize> Sub<&Self> for CyclotomicPolyRing<R, N>
where
    R: NegacyclicMulRing<N>,
{
    type Output = Self;

    fn sub(self, rhs: &Self) -> Self {
        self.sub_impl(rhs)
    }
}

impl<R, const N: usize> Sub<Self> for &CyclotomicPolyRing<R, N>
where
    R: NegacyclicMulRing<N>,
{
    type Output = CyclotomicPolyRing<R, N>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.sub_impl(rhs)
    }
}

impl<R, const N: usize> Sub<CyclotomicPolyRing<R, N>> for &CyclotomicPolyRing<R, N>
where
    R: NegacyclicMulRing<N>,
{
    type Output = CyclotomicPolyRing<R, N>;

    fn sub(self, rhs: CyclotomicPolyRing<R, N>) -> Self::Output {
        self.sub_impl(&rhs)
    }
}

impl<R, const N: usize> SubAssign for CyclotomicPolyRing<R, N>
where
    R: NegacyclicMulRing<N>,
{
    fn sub_assign(&mut self, rhs: Self) {
        R::sub_assign_slice(&mut self.coeffs, &rhs.coeffs);
    }
}

impl<R, const N: usize> SubAssign<&Self> for CyclotomicPolyRing<R, N>
where
    R: NegacyclicMulRing<N>,
{
    fn sub_assign(&mut self, rhs: &Self) {
        R::sub_assign_slice(&mut self.coeffs, &rhs.coeffs);
    }
}

impl<R, const N: usize> Mul for CyclotomicPolyRing<R, N>
where
    R: NegacyclicMulRing<N>,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        self.mul_impl(&rhs)
    }
}

impl<R, const N: usize> Mul<&Self> for CyclotomicPolyRing<R, N>
where
    R: NegacyclicMulRing<N>,
{
    type Output = Self;

    fn mul(self, rhs: &Self) -> Self {
        self.mul_impl(rhs)
    }
}

impl<R, const N: usize> Mul<Self> for &CyclotomicPolyRing<R, N>
where
    R: NegacyclicMulRing<N>,
{
    type Output = CyclotomicPolyRing<R, N>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.mul_impl(rhs)
    }
}

impl<R, const N: usize> Mul<CyclotomicPolyRing<R, N>> for &CyclotomicPolyRing<R, N>
where
    R: NegacyclicMulRing<N>,
{
    type Output = CyclotomicPolyRing<R, N>;

    fn mul(self, rhs: CyclotomicPolyRing<R, N>) -> Self::Output {
        self.mul_impl(&rhs)
    }
}

impl<R, const N: usize> MulAssign for CyclotomicPolyRing<R, N>
where
    R: NegacyclicMulRing<N>,
{
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.mul_impl(&rhs);
    }
}

impl<R, const N: usize> MulAssign<&Self> for CyclotomicPolyRing<R, N>
where
    R: NegacyclicMulRing<N>,
{
    fn mul_assign(&mut self, rhs: &Self) {
        *self = self.mul_impl(rhs);
    }
}

impl<R, const N: usize> Neg for CyclotomicPolyRing<R, N>
where
    R: NegacyclicMulRing<N>,
{
    type Output = Self;

    fn neg(self) -> Self {
        Self::from_array(from_fn(|i| -self.coeffs[i].clone()))
    }
}

impl<R: IntegerRing + CanonicalSerialize + CanonicalDeserialize, const N: usize> CanonicalSerialize
    for CyclotomicPolyRing<R, N>
{
    fn serialized_size(&self) -> usize {
        self.coeffs
            .iter()
            .map(CanonicalSerialize::serialized_size)
            .sum()
    }

    fn serialize_into(&self, buf: &mut Vec<u8>) -> Result<(), SerializationError> {
        Self::check_degree();
        for coeff in &self.coeffs {
            coeff.serialize_into(buf)?;
        }
        Ok(())
    }
}

impl<R: IntegerRing + CanonicalSerialize + CanonicalDeserialize, const N: usize>
    CanonicalDeserialize for CyclotomicPolyRing<R, N>
{
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        Self::check_degree();
        let mut coeffs = Vec::with_capacity(N);
        let mut consumed = 0;
        for _ in 0..N {
            let (coeff, used) = R::deserialize(&data[consumed..])?;
            coeffs.push(coeff);
            consumed += used;
        }
        let coeffs = <Vec<R> as TryInto<[R; N]>>::try_into(coeffs).map_err(|_| {
            SerializationError::InvalidData("failed to build cyclotomic coefficient array".into())
        })?;
        Ok((Self::from_array(coeffs), consumed))
    }
}

impl<R, const N: usize> Valid for CyclotomicPolyRing<R, N>
where
    R: IntegerRing + CanonicalSerialize + CanonicalDeserialize + Valid,
{
    fn is_valid(&self) -> bool {
        let _ = Self::DEGREE;
        self.coeffs.iter().all(Valid::is_valid)
    }
}

impl<R, const N: usize> grid_std::UniformRand for CyclotomicPolyRing<R, N>
where
    R: NegacyclicMulRing<N> + grid_std::UniformRand,
{
    fn rand<T: grid_std::rand::Rng + ?Sized>(rng: &mut T) -> Self {
        Self::from_array(from_fn(|_| R::rand(rng)))
    }
}

/// Multiply two cyclotomic polynomials using NTT when the coefficient ring supports it.
pub fn mul_with_ntt<R, const N: usize>(
    a: &CyclotomicPolyRing<R, N>,
    b: &CyclotomicPolyRing<R, N>,
) -> Result<CyclotomicPolyRing<R, N>, NttError>
where
    R: IntegerRing + NTTRing + CanonicalSerialize + CanonicalDeserialize + Send + Sync + 'static,
{
    let coeffs = poly_mul_ntt(&a.coeffs, &b.coeffs)?;
    let got = coeffs.len();
    let coeffs =
        <Vec<R> as TryInto<[R; N]>>::try_into(coeffs).map_err(|_| NttError::LengthMismatch {
            left: N,
            right: got,
        })?;
    Ok(CyclotomicPolyRing::from_array(coeffs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arith::prime::PrimeField;
    use crate::arith::ring::tests::test_ring_axioms;
    use grid_serialize::{CanonicalDeserialize, CanonicalSerialize};
    use grid_std::UniformRand;

    type F17 = PrimeField<17>;
    type Poly8 = CyclotomicPolyRing<F17, 8>;

    #[test]
    fn test_ring_axioms_poly() {
        let a = Poly8::from_array([
            F17::from_u64(1),
            F17::from_u64(2),
            F17::from_u64(3),
            F17::from_u64(4),
            F17::from_u64(0),
            F17::from_u64(1),
            F17::from_u64(0),
            F17::from_u64(2),
        ]);
        let b = Poly8::from_array([
            F17::from_u64(3),
            F17::from_u64(1),
            F17::from_u64(4),
            F17::from_u64(1),
            F17::from_u64(5),
            F17::from_u64(9),
            F17::from_u64(2),
            F17::from_u64(6),
        ]);
        let c = Poly8::from_array([
            F17::from_u64(5),
            F17::from_u64(3),
            F17::from_u64(5),
            F17::from_u64(8),
            F17::from_u64(9),
            F17::from_u64(7),
            F17::from_u64(9),
            F17::from_u64(3),
        ]);
        test_ring_axioms(a, b, c);
    }

    #[test]
    fn test_scalar_mul_and_evaluate() {
        let poly = Poly8::from_array([
            F17::from_u64(1),
            F17::from_u64(2),
            F17::from_u64(3),
            F17::from_u64(0),
            F17::from_u64(0),
            F17::from_u64(0),
            F17::from_u64(0),
            F17::from_u64(0),
        ]);
        let scaled = poly.scalar_mul(&F17::from_u64(2));
        assert_eq!(scaled.coeff(0).to_u64(), 2);
        assert_eq!(scaled.coeff(2).to_u64(), 6);
        assert_eq!(poly.evaluate(&F17::from_u64(2)).to_u64(), 0);
    }

    #[test]
    fn test_in_place_arithmetic_matches_out_of_place() {
        let a = Poly8::from_array([
            F17::from_u64(1),
            F17::from_u64(2),
            F17::from_u64(3),
            F17::from_u64(4),
            F17::from_u64(0),
            F17::from_u64(1),
            F17::from_u64(0),
            F17::from_u64(2),
        ]);
        let b = Poly8::from_array([
            F17::from_u64(3),
            F17::from_u64(1),
            F17::from_u64(4),
            F17::from_u64(1),
            F17::from_u64(5),
            F17::from_u64(9),
            F17::from_u64(2),
            F17::from_u64(6),
        ]);

        let mut add_assign = a.clone();
        add_assign += b.clone();
        assert_eq!(add_assign, a.clone() + b.clone());

        let mut sub_assign = a.clone();
        sub_assign -= b.clone();
        assert_eq!(sub_assign, a.clone() - b.clone());

        let mut mul_assign = a.clone();
        mul_assign *= b.clone();
        assert_eq!(mul_assign, a * b);
    }

    #[test]
    fn test_serialization_round_trip() {
        let poly = Poly8::from_array([
            F17::from_u64(0),
            F17::from_u64(1),
            F17::from_u64(2),
            F17::from_u64(3),
            F17::from_u64(4),
            F17::from_u64(5),
            F17::from_u64(6),
            F17::from_u64(7),
        ]);
        let bytes = poly.serialize().unwrap();
        let decoded = Poly8::deserialize_exact(&bytes).unwrap();
        assert_eq!(decoded, poly);
    }

    #[test]
    fn test_deserialize_rejects_truncated_coefficients() {
        let poly = Poly8::from_array([
            F17::from_u64(0),
            F17::from_u64(1),
            F17::from_u64(2),
            F17::from_u64(3),
            F17::from_u64(4),
            F17::from_u64(5),
            F17::from_u64(6),
            F17::from_u64(7),
        ]);
        let bytes = poly.serialize().unwrap();
        let err = Poly8::deserialize(&bytes[..bytes.len() - 1]).unwrap_err();
        assert_eq!(err, SerializationError::UnexpectedEnd);
    }

    #[test]
    fn test_try_from_coeffs_rejects_wrong_length() {
        let err = Poly8::try_from_coeffs(&[F17::from_u64(1), F17::from_u64(2), F17::from_u64(3)])
            .unwrap_err();
        assert_eq!(
            err,
            PolyError::WrongCoeffCount {
                expected: 8,
                got: 3,
            }
        );
    }

    #[test]
    fn test_uniform_rand() {
        let mut rng = grid_std::test_rng();
        let poly = Poly8::rand(&mut rng);
        assert_eq!(poly.coeffs().len(), 8);
    }
}
