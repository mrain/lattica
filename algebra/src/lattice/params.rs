//! Norm helpers for lattice objects.

use crate::arith::bigint::BigUint;
use crate::arith::large_modulus::LargeCanonicalRing;
use crate::arith::ring::{IntegerRing, Ring};
use crate::lattice::types::RingVec;
use crate::poly::ring::{CyclotomicPolyRing, NegacyclicMulRing, PolyRing};
use grid_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError, Valid};

/// A ring element with exact centered norms.
pub trait NormedRing: Ring {
    /// Exact squared `L2` norm.
    fn l2_norm_sq(&self) -> u128;

    /// Exact `L∞` norm.
    fn linf_norm(&self) -> u64;
}

/// A comparable norm value used by the large-modulus companion surface.
pub trait LargeNormValue: Clone + core::fmt::Debug + PartialEq + Eq + PartialOrd + Ord {
    /// The additive identity.
    fn zero() -> Self;

    /// Embed a small `u64`.
    fn from_u64(value: u64) -> Self;

    /// Add two norm values, saturating if the representation overflows.
    fn add(lhs: &Self, rhs: &Self) -> Self;

    /// Multiply by a small non-negative scalar, saturating on overflow.
    fn scale_u64(value: &Self, scalar: u64) -> Self;

    /// Square the value, saturating on overflow.
    fn square(value: &Self) -> Self;

    /// Return the smallest integer `x` such that `x^2 >= value`.
    fn sqrt_ceil(value: &Self) -> Self;
}

/// Maps a canonical representative type to the large norm type used for centered lifts.
pub trait CanonicalNormEmbedding:
    Clone + core::fmt::Debug + PartialEq + Eq + PartialOrd + Ord
{
    type Norm: LargeNormValue;

    fn centered_abs_norm(value: &Self, modulus: &Self) -> Self::Norm;
}

fn centered_abs_u64(x: u64, modulus: u64) -> u64 {
    if modulus == 0 {
        x.min(x.wrapping_neg())
    } else {
        debug_assert!(
            x <= modulus,
            "centered norm expects a canonical representative below the modulus"
        );
        x.min(modulus.wrapping_sub(x))
    }
}

fn centered_abs_u128(x: u128, modulus: u128) -> u128 {
    if modulus == 0 {
        x.min(x.wrapping_neg())
    } else {
        debug_assert!(
            x <= modulus,
            "centered norm expects a canonical representative below the modulus"
        );
        x.min(modulus.wrapping_sub(x))
    }
}

fn biguint_sqrt_ceil<const N: usize>(value: &BigUint<N>) -> BigUint<N> {
    if value.bits() <= 1 {
        return *value;
    }

    let mut lo = BigUint::<N>::one();
    let mut hi = *value;
    while lo < hi {
        let (delta, borrow) = hi.sub_with_borrow(&lo);
        debug_assert!(!borrow, "sqrt search requires ordered bounds");
        let half = delta.shr_bits(1);
        let (mid, carry) = lo.add_with_carry(&half);
        debug_assert!(!carry, "midpoint must fit the norm representation");

        let (mid_sq, mid_sq_hi) = mid.widening_mul(&mid);
        if !mid_sq_hi.is_zero() || mid_sq > *value {
            hi = mid;
        } else {
            let (next, overflow) = mid.add_small(1);
            if overflow {
                return BigUint::<N>::MAX;
            }
            lo = next;
        }
    }
    lo
}

impl LargeNormValue for u128 {
    fn zero() -> Self {
        0
    }

    fn from_u64(value: u64) -> Self {
        value as u128
    }

    fn add(lhs: &Self, rhs: &Self) -> Self {
        lhs.saturating_add(*rhs)
    }

    fn scale_u64(value: &Self, scalar: u64) -> Self {
        value.saturating_mul(scalar as u128)
    }

    fn square(value: &Self) -> Self {
        value.saturating_mul(*value)
    }

    fn sqrt_ceil(value: &Self) -> Self {
        int_sqrt_ceil(*value)
    }
}

impl<const N: usize> LargeNormValue for BigUint<N> {
    fn zero() -> Self {
        BigUint::ZERO
    }

    fn from_u64(value: u64) -> Self {
        BigUint::from_u64(value)
    }

    fn add(lhs: &Self, rhs: &Self) -> Self {
        let (sum, carry) = lhs.add_with_carry(rhs);
        if carry { BigUint::MAX } else { sum }
    }

    fn scale_u64(value: &Self, scalar: u64) -> Self {
        let (product, carry) = value.mul_by_limb(scalar);
        if carry != 0 { BigUint::MAX } else { product }
    }

    fn square(value: &Self) -> Self {
        let (lo, hi) = value.widening_mul(value);
        if hi.is_zero() { lo } else { BigUint::MAX }
    }

    fn sqrt_ceil(value: &Self) -> Self {
        biguint_sqrt_ceil(value)
    }
}

impl CanonicalNormEmbedding for u64 {
    type Norm = u128;

    fn centered_abs_norm(value: &Self, modulus: &Self) -> Self::Norm {
        centered_abs_u64(*value, *modulus) as u128
    }
}

impl CanonicalNormEmbedding for u128 {
    type Norm = u128;

    fn centered_abs_norm(value: &Self, modulus: &Self) -> Self::Norm {
        centered_abs_u128(*value, *modulus)
    }
}

macro_rules! impl_biguint_norm_embedding {
    ($canonical_limbs:literal, $norm_limbs:literal) => {
        impl CanonicalNormEmbedding for BigUint<$canonical_limbs> {
            type Norm = BigUint<$norm_limbs>;

            fn centered_abs_norm(value: &Self, modulus: &Self) -> Self::Norm {
                let (negative, borrow) = modulus.sub_with_borrow(value);
                debug_assert!(
                    !borrow,
                    "canonical representative must be below the modulus"
                );
                let centered = if value <= &negative { *value } else { negative };
                centered.zero_extend::<$norm_limbs>()
            }
        }
    };
}

impl_biguint_norm_embedding!(2, 4);
impl_biguint_norm_embedding!(3, 6);
impl_biguint_norm_embedding!(4, 8);
impl_biguint_norm_embedding!(6, 12);

/// A ring element with exact large-centered norms.
pub trait LargeNormedRing: Ring {
    /// The exact norm representation used by this backend.
    type Norm: LargeNormValue;

    /// Exact squared `L2` norm in the large companion representation.
    fn l2_norm_sq_large(&self) -> Self::Norm;

    /// Exact `L∞` norm in the large companion representation.
    fn linf_norm_large(&self) -> Self::Norm;
}

/// A bound object that can validate exact norms for vectors over a ring backend.
pub trait VectorNormBound<R: Ring>:
    CanonicalSerialize + CanonicalDeserialize + Valid + Clone + PartialEq + Eq
{
    /// Return whether the supplied vector satisfies this bound.
    fn check_vector(&self, v: &RingVec<R>) -> bool;
}

impl<R: IntegerRing<Uint = u64>> NormedRing for R {
    fn l2_norm_sq(&self) -> u128 {
        let a = centered_abs_u64(self.to_u64(), R::modulus()) as u128;
        a.saturating_mul(a)
    }

    fn linf_norm(&self) -> u64 {
        centered_abs_u64(self.to_u64(), R::modulus())
    }
}

impl<R, C> LargeNormedRing for R
where
    R: LargeCanonicalRing<Canonical = C>,
    C: CanonicalNormEmbedding,
{
    // Word-sized backends such as PrimeField<Q> can implement both NormedRing and
    // LargeNormedRing. Callers should prefer NormedRing on hot word-sized paths and use
    // LargeNormedRing when exact large canonical values are required generically.
    type Norm = C::Norm;

    fn l2_norm_sq_large(&self) -> Self::Norm {
        let centered = self.linf_norm_large();
        Self::Norm::square(&centered)
    }

    fn linf_norm_large(&self) -> Self::Norm {
        C::centered_abs_norm(&self.to_canonical(), &R::modulus_canonical())
    }
}

impl<R, const N: usize> NormedRing for CyclotomicPolyRing<R, N>
where
    R: NegacyclicMulRing<N, Uint = u64>,
{
    fn l2_norm_sq(&self) -> u128 {
        self.coeffs()
            .iter()
            .map(NormedRing::l2_norm_sq)
            .fold(0u128, u128::saturating_add)
    }

    fn linf_norm(&self) -> u64 {
        self.coeffs()
            .iter()
            .map(NormedRing::linf_norm)
            .max()
            .unwrap_or(0)
    }
}

impl<R, const N: usize> LargeNormedRing for CyclotomicPolyRing<R, N>
where
    R: NegacyclicMulRing<N> + LargeNormedRing,
{
    type Norm = R::Norm;

    fn l2_norm_sq_large(&self) -> Self::Norm {
        self.coeffs()
            .iter()
            .map(LargeNormedRing::l2_norm_sq_large)
            .fold(Self::Norm::zero(), |acc, value| {
                Self::Norm::add(&acc, &value)
            })
    }

    fn linf_norm_large(&self) -> Self::Norm {
        self.coeffs()
            .iter()
            .map(LargeNormedRing::linf_norm_large)
            .max()
            .unwrap_or_else(Self::Norm::zero)
    }
}

/// Exact norms of a vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NormStats {
    /// Sum of squared entry norms.
    pub l2_sq: u128,
    /// Maximum `L∞` norm across entries.
    pub linf: u64,
}

impl NormStats {
    /// Compute norms for a lattice vector.
    pub fn compute<R: NormedRing>(v: &RingVec<R>) -> Self {
        let mut l2_sq = 0u128;
        let mut linf = 0u64;
        for entry in v.entries() {
            l2_sq = l2_sq.saturating_add(entry.l2_norm_sq());
            linf = linf.max(entry.linf_norm());
        }
        Self { l2_sq, linf }
    }
}

/// Exact large norms of a vector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LargeNormStats<T> {
    /// Sum of squared entry norms.
    pub l2_sq: T,
    /// Maximum `L∞` norm across entries.
    pub linf: T,
}

impl<T: LargeNormValue> LargeNormStats<T> {
    /// Compute exact large norms for a lattice vector.
    pub fn compute<R>(v: &RingVec<R>) -> Self
    where
        R: LargeNormedRing<Norm = T>,
    {
        let mut l2_sq = T::zero();
        let mut linf = T::zero();
        for entry in v.entries() {
            l2_sq = T::add(&l2_sq, &entry.l2_norm_sq_large());
            linf = linf.max(entry.linf_norm_large());
        }
        Self { l2_sq, linf }
    }
}

/// Upper bounds used to accept or reject witnesses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NormBound {
    /// Maximum squared `L2` norm.
    pub max_l2_sq: u128,
    /// Maximum `L∞` norm.
    pub max_linf: u64,
}

impl CanonicalSerialize for NormBound {
    fn serialized_size(&self) -> usize {
        24
    }

    fn serialize_into(&self, buf: &mut alloc::vec::Vec<u8>) -> Result<(), SerializationError> {
        buf.extend_from_slice(&self.max_l2_sq.to_le_bytes());
        buf.extend_from_slice(&self.max_linf.to_le_bytes());
        Ok(())
    }
}

impl CanonicalDeserialize for NormBound {
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        if data.len() < 24 {
            return Err(SerializationError::UnexpectedEnd);
        }
        let max_l2_sq = u128::from_le_bytes(data[..16].try_into().unwrap());
        let max_linf = u64::from_le_bytes(data[16..24].try_into().unwrap());
        Ok((
            Self {
                max_l2_sq,
                max_linf,
            },
            24,
        ))
    }
}

impl Valid for NormBound {
    fn is_valid(&self) -> bool {
        true
    }
}

impl NormBound {
    /// Build a bound from exact norms.
    pub fn from_stats(stats: &NormStats) -> Self {
        Self {
            max_l2_sq: stats.l2_sq,
            max_linf: stats.linf,
        }
    }

    /// Check whether a vector satisfies this bound.
    pub fn check<R: NormedRing>(&self, v: &RingVec<R>) -> bool {
        let stats = NormStats::compute(v);
        stats.l2_sq <= self.max_l2_sq && stats.linf <= self.max_linf
    }

    /// Compose bounds for vector addition.
    pub fn compose_add(a: &Self, b: &Self) -> Self {
        let linf = a.max_linf.saturating_add(b.max_linf);
        let l2_a = int_sqrt_ceil(a.max_l2_sq);
        let l2_b = int_sqrt_ceil(b.max_l2_sq);
        let l2_sq = l2_a.saturating_add(l2_b);
        Self {
            max_l2_sq: l2_sq.saturating_mul(l2_sq),
            max_linf: linf,
        }
    }

    /// Scale a bound by a non-negative scalar.
    pub fn scale(&self, s: u64) -> Self {
        let s = s as u128;
        Self {
            max_l2_sq: self.max_l2_sq.saturating_mul(s).saturating_mul(s),
            max_linf: self.max_linf.saturating_mul(s as u64),
        }
    }
}

impl<R: NormedRing> VectorNormBound<R> for NormBound {
    fn check_vector(&self, v: &RingVec<R>) -> bool {
        self.check(v)
    }
}

/// Large-modulus companion to [`NormBound`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LargeNormBound<T> {
    /// Maximum squared `L2` norm.
    pub max_l2_sq: T,
    /// Maximum `L∞` norm.
    pub max_linf: T,
}

impl<T> CanonicalSerialize for LargeNormBound<T>
where
    T: CanonicalSerialize,
{
    fn serialized_size(&self) -> usize {
        self.max_l2_sq.serialized_size() + self.max_linf.serialized_size()
    }

    fn serialize_into(&self, buf: &mut alloc::vec::Vec<u8>) -> Result<(), SerializationError> {
        self.max_l2_sq.serialize_into(buf)?;
        self.max_linf.serialize_into(buf)?;
        Ok(())
    }
}

impl<T> CanonicalDeserialize for LargeNormBound<T>
where
    T: CanonicalDeserialize,
{
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        let (max_l2_sq, used_l2) = T::deserialize(data)?;
        let (max_linf, used_linf) = T::deserialize(&data[used_l2..])?;
        Ok((
            Self {
                max_l2_sq,
                max_linf,
            },
            used_l2 + used_linf,
        ))
    }
}

impl<T> Valid for LargeNormBound<T>
where
    T: Valid,
{
    fn is_valid(&self) -> bool {
        self.max_l2_sq.is_valid() && self.max_linf.is_valid()
    }
}

impl<T: LargeNormValue> LargeNormBound<T> {
    /// Build a bound from exact large norms.
    pub fn from_stats(stats: &LargeNormStats<T>) -> Self {
        Self {
            max_l2_sq: stats.l2_sq.clone(),
            max_linf: stats.linf.clone(),
        }
    }

    /// Check whether a vector satisfies this large-modulus bound.
    pub fn check<R>(&self, v: &RingVec<R>) -> bool
    where
        R: LargeNormedRing<Norm = T>,
    {
        let stats = LargeNormStats::compute(v);
        stats.l2_sq <= self.max_l2_sq && stats.linf <= self.max_linf
    }

    /// Compose bounds for vector addition.
    pub fn compose_add(a: &Self, b: &Self) -> Self {
        let linf = T::add(&a.max_linf, &b.max_linf);
        let l2_a = T::sqrt_ceil(&a.max_l2_sq);
        let l2_b = T::sqrt_ceil(&b.max_l2_sq);
        let l2 = T::add(&l2_a, &l2_b);
        Self {
            max_l2_sq: T::square(&l2),
            max_linf: linf,
        }
    }

    /// Scale a bound by a non-negative scalar.
    pub fn scale(&self, scalar: u64) -> Self {
        Self {
            max_l2_sq: T::scale_u64(&T::scale_u64(&self.max_l2_sq, scalar), scalar),
            max_linf: T::scale_u64(&self.max_linf, scalar),
        }
    }
}

impl<R, T> VectorNormBound<R> for LargeNormBound<T>
where
    R: LargeNormedRing<Norm = T>,
    T: LargeNormValue + CanonicalSerialize + CanonicalDeserialize + Valid,
{
    fn check_vector(&self, v: &RingVec<R>) -> bool {
        self.check(v)
    }
}

fn int_sqrt_ceil(n: u128) -> u128 {
    if n <= 1 {
        return n;
    }
    let mut lo = 1u128;
    let mut hi = n;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if mid.saturating_mul(mid) < n {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arith::LargeCanonicalRing;
    use crate::arith::bigint::BigUint;
    use crate::arith::large_prime::Bn254Fr;
    use crate::arith::large_rns::Rns3V0;
    use crate::arith::prime::PrimeField;
    use crate::arith::ring::IntegerRing;
    use crate::arith::z2k::Z2K;
    use crate::lattice::types::RingVec;
    use crate::poly::ring::CyclotomicPolyRing;
    use grid_serialize::{CanonicalDeserialize, CanonicalSerialize};

    type F17 = PrimeField<17>;
    type Z16 = Z2K<4>;
    type Poly8 = CyclotomicPolyRing<F17, 8>;

    #[test]
    fn test_normed_ring_for_coefficients() {
        let a = F17::from_u64(14);
        assert_eq!(a.linf_norm(), 3);
        assert_eq!(a.l2_norm_sq(), 9);

        let b = Z16::from_u64(15);
        assert_eq!(b.linf_norm(), 1);
        assert_eq!(b.l2_norm_sq(), 1);
    }

    #[test]
    fn test_normed_ring_for_cyclotomic() {
        let poly = Poly8::from_array([
            F17::from_u64(1),
            F17::from_u64(16),
            F17::from_u64(3),
            F17::from_u64(0),
            F17::from_u64(0),
            F17::from_u64(0),
            F17::from_u64(0),
            F17::from_u64(0),
        ]);
        assert_eq!(poly.linf_norm(), 3);
        assert_eq!(poly.l2_norm_sq(), 11);
    }

    #[test]
    fn test_norm_stats_and_bounds() {
        let vec = RingVec::new(vec![F17::from_u64(1), F17::from_u64(14), F17::from_u64(0)]);
        let stats = NormStats::compute(&vec);
        assert_eq!(stats.l2_sq, 10);
        assert_eq!(stats.linf, 3);

        let bound = NormBound::from_stats(&stats);
        assert!(bound.check(&vec));
        let stricter = NormBound {
            max_l2_sq: 9,
            max_linf: 2,
        };
        assert!(!stricter.check(&vec));
    }

    #[test]
    fn test_bound_compose_and_scale() {
        let a = NormBound {
            max_l2_sq: 9,
            max_linf: 3,
        };
        let b = NormBound {
            max_l2_sq: 16,
            max_linf: 4,
        };
        let composed = NormBound::compose_add(&a, &b);
        assert_eq!(composed.max_linf, 7);
        assert!(composed.max_l2_sq >= 49);

        let scaled = a.scale(2);
        assert_eq!(scaled.max_l2_sq, 36);
        assert_eq!(scaled.max_linf, 6);
    }

    #[test]
    fn test_norm_bound_serialize_round_trip() {
        let bound = NormBound {
            max_l2_sq: 123,
            max_linf: 7,
        };
        let bytes = bound.serialize().unwrap();
        let decoded = NormBound::deserialize_exact(&bytes).unwrap();
        assert_eq!(decoded, bound);
        assert!(decoded.is_valid());
    }

    #[test]
    fn test_large_norm_stats_for_large_prime_backend() {
        let modulus = Bn254Fr::modulus_canonical();
        let (modulus_minus_one, borrow) = modulus.sub_small(1);
        assert!(!borrow);

        let vec = RingVec::new(vec![
            Bn254Fr::from_u64(5),
            Bn254Fr::from_canonical(&modulus_minus_one),
        ]);
        let stats = LargeNormStats::compute(&vec);
        assert_eq!(stats.linf, BigUint::<8>::from_u64(5));
        assert_eq!(stats.l2_sq, BigUint::<8>::from_u64(26));

        let bound = LargeNormBound::from_stats(&stats);
        assert!(bound.check(&vec));

        let composed = LargeNormBound::compose_add(&bound, &bound);
        assert_eq!(composed.max_linf, BigUint::<8>::from_u64(10));
        assert!(composed.max_l2_sq >= BigUint::<8>::from_u64(104));
    }

    #[test]
    fn test_large_norm_stats_for_large_rns_backend() {
        let modulus = Rns3V0::modulus_canonical();
        let (modulus_minus_three, borrow) = modulus.sub_small(3);
        assert!(!borrow);

        let vec = RingVec::new(vec![
            Rns3V0::from_u64(7),
            Rns3V0::from_canonical(&modulus_minus_three),
        ]);
        let stats = LargeNormStats::compute(&vec);
        assert_eq!(stats.linf, BigUint::<6>::from_u64(7));
        assert_eq!(stats.l2_sq, BigUint::<6>::from_u64(58));

        let bound = LargeNormBound::from_stats(&stats);
        assert!(bound.check(&vec));
        assert_eq!(bound.scale(2).max_linf, BigUint::<6>::from_u64(14));
    }

    #[test]
    fn test_large_norm_bound_serialize_round_trip() {
        let bound = LargeNormBound {
            max_l2_sq: BigUint::<8>::from_u64(123),
            max_linf: BigUint::<8>::from_u64(7),
        };
        let bytes = bound.serialize().unwrap();
        let decoded = LargeNormBound::<BigUint<8>>::deserialize_exact(&bytes).unwrap();
        assert_eq!(decoded, bound);
        assert!(decoded.is_valid());
    }
}
