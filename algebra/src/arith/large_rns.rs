//! Fixed-profile pointer-free large-RNS backend.

use alloc::sync::Arc;
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::bigint::BigUint;
use super::composite::CompositeRing;
use super::large_modulus::{LargeCanonicalRing, LargeRnsProfile};
use super::large_rns_profiles::Rns3V0Profile;
use super::ntt::{NTTRing, NttError, NttPlan, cached_ntt_plan};
use super::ring::{IntegerRing, Ring};
use super::rns::{RnsBasis, mod_inv_u64};
use grid_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError};

/// Fixed-profile residue-domain ring element.
#[repr(transparent)]
pub struct LargeRns<P, const LIMBS: usize>
where
    P: LargeRnsProfile<LIMBS>,
{
    residues: [u64; LIMBS],
    _profile: PhantomData<P>,
}

/// First reviewed 3-limb large-RNS shipping backend.
pub type Rns3V0 = LargeRns<Rns3V0Profile, 3>;

impl<P, const LIMBS: usize> Copy for LargeRns<P, LIMBS> where P: LargeRnsProfile<LIMBS> {}

impl<P, const LIMBS: usize> Clone for LargeRns<P, LIMBS>
where
    P: LargeRnsProfile<LIMBS>,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<P, const LIMBS: usize> PartialEq for LargeRns<P, LIMBS>
where
    P: LargeRnsProfile<LIMBS>,
{
    fn eq(&self, other: &Self) -> bool {
        self.residues == other.residues
    }
}

impl<P, const LIMBS: usize> Eq for LargeRns<P, LIMBS> where P: LargeRnsProfile<LIMBS> {}

impl<P, const LIMBS: usize> LargeRns<P, LIMBS>
where
    P: LargeRnsProfile<LIMBS>,
{
    #[inline(always)]
    fn ensure_profile_valid() {
        let () = P::PROFILE_VALID;
    }

    #[inline(always)]
    fn from_residues_unchecked(residues: [u64; LIMBS]) -> Self {
        Self::ensure_profile_valid();
        Self {
            residues,
            _profile: PhantomData,
        }
    }

    /// Return the raw residues in profile order.
    pub fn residues(&self) -> &[u64; LIMBS] {
        &self.residues
    }

    /// Validate and construct from raw residues.
    pub fn try_from_residues(residues: [u64; LIMBS]) -> Result<Self, SerializationError> {
        for (residue, modulus) in residues.iter().zip(P::MODULI.iter()) {
            if residue >= modulus {
                return Err(SerializationError::InvalidData(alloc::format!(
                    "residue {residue} >= modulus {modulus}"
                )));
            }
        }
        Ok(Self::from_residues_unchecked(residues))
    }

    /// Convert into the dynamic-basis `CompositeRing` utility representation.
    pub fn to_composite(&self) -> CompositeRing {
        CompositeRing::from_residues_with_basis(
            self.residues.to_vec(),
            Arc::new(RnsBasis::new(P::MODULI.to_vec())),
        )
        .expect("large-RNS residues are always canonical for the profile basis")
    }

    /// Convert from the dynamic-basis `CompositeRing` utility representation.
    pub fn try_from_composite(value: &CompositeRing) -> Result<Self, SerializationError> {
        if value.basis().primes.as_slice() != P::MODULI.as_slice() {
            return Err(SerializationError::InvalidData(
                "CompositeRing basis does not match the fixed large-RNS profile".into(),
            ));
        }

        let residues: [u64; LIMBS] = value
            .residues()
            .try_into()
            .map_err(|_| SerializationError::InvalidData("wrong residue count".into()))?;
        Self::try_from_residues(residues)
    }

    fn reduce_biguint_mod_component<const N: usize>(value: &BigUint<N>, modulus: u64) -> u64 {
        let radix_mod = ((1u128 << 64) % modulus as u128) as u64;
        let mut acc = 0u64;
        for &limb in value.limbs.iter().rev() {
            acc =
                (((acc as u128) * (radix_mod as u128) + (limb as u128)) % (modulus as u128)) as u64;
        }
        acc
    }

    fn decompose_canonical<const N: usize>(value: &BigUint<N>) -> [u64; LIMBS] {
        core::array::from_fn(|i| Self::reduce_biguint_mod_component(value, P::MODULI[i]))
    }

    fn reconstruct_canonical(&self) -> BigUint<LIMBS> {
        let mut coeffs = [0u64; LIMBS];
        if LIMBS == 0 {
            return BigUint::ZERO;
        }
        coeffs[0] = self.residues[0];

        for i in 1..LIMBS {
            let mut u = self.residues[i] as u128;
            let p_i = P::MODULI[i] as u128;
            for (j, coeff) in coeffs.iter().enumerate().take(i) {
                let coeff_mod = (*coeff as u128) % p_i;
                if u >= coeff_mod {
                    u = (u - coeff_mod) % p_i;
                } else {
                    u = p_i - ((coeff_mod - u) % p_i);
                }
                u = (u * (P::GARNER_INVERSES[i][j] as u128)) % p_i;
            }
            coeffs[i] = u as u64;
        }

        let mut result = BigUint::<LIMBS>::ZERO;
        for (i, &coeff) in coeffs.iter().enumerate() {
            let product = BigUint::<LIMBS> {
                limbs: P::PREFIX_PRODUCTS[i],
            };
            let (term, carry_term) = product.mul_by_limb(coeff);
            assert_eq!(
                carry_term, 0,
                "fixed large-RNS profile coefficient product does not fit the canonical limb width"
            );
            let (sum, carry_sum) = result.add_with_carry(&term);
            assert!(
                !carry_sum,
                "fixed large-RNS reconstruction overflowed the canonical limb width"
            );
            result = sum;
        }

        result
    }

    #[inline(always)]
    fn primitive_two_adic_root() -> Self {
        Self::from_residues_unchecked(P::TWO_ADIC_ROOT)
    }
}

impl<P, const LIMBS: usize> core::fmt::Debug for LargeRns<P, LIMBS>
where
    P: LargeRnsProfile<LIMBS>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LargeRns")
            .field("residues", &self.residues)
            .field("canonical", &self.to_canonical())
            .finish()
    }
}

impl<P, const LIMBS: usize> core::fmt::Display for LargeRns<P, LIMBS>
where
    P: LargeRnsProfile<LIMBS>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Display::fmt(&self.to_canonical(), f)
    }
}

impl<P, const LIMBS: usize> Add for LargeRns<P, LIMBS>
where
    P: LargeRnsProfile<LIMBS>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let residues = core::array::from_fn(|i| {
            let modulus = P::MODULI[i] as u128;
            let sum = self.residues[i] as u128 + rhs.residues[i] as u128;
            if sum >= modulus {
                (sum - modulus) as u64
            } else {
                sum as u64
            }
        });
        Self::from_residues_unchecked(residues)
    }
}

impl<P, const LIMBS: usize> Add<&Self> for LargeRns<P, LIMBS>
where
    P: LargeRnsProfile<LIMBS>,
{
    type Output = Self;

    fn add(self, rhs: &Self) -> Self::Output {
        self + *rhs
    }
}

impl<P, const LIMBS: usize> AddAssign for LargeRns<P, LIMBS>
where
    P: LargeRnsProfile<LIMBS>,
{
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<P, const LIMBS: usize> AddAssign<&Self> for LargeRns<P, LIMBS>
where
    P: LargeRnsProfile<LIMBS>,
{
    fn add_assign(&mut self, rhs: &Self) {
        *self = *self + *rhs;
    }
}

impl<P, const LIMBS: usize> Sub for LargeRns<P, LIMBS>
where
    P: LargeRnsProfile<LIMBS>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let residues = core::array::from_fn(|i| {
            if self.residues[i] >= rhs.residues[i] {
                self.residues[i] - rhs.residues[i]
            } else {
                P::MODULI[i] - rhs.residues[i] + self.residues[i]
            }
        });
        Self::from_residues_unchecked(residues)
    }
}

impl<P, const LIMBS: usize> Sub<&Self> for LargeRns<P, LIMBS>
where
    P: LargeRnsProfile<LIMBS>,
{
    type Output = Self;

    fn sub(self, rhs: &Self) -> Self::Output {
        self - *rhs
    }
}

impl<P, const LIMBS: usize> SubAssign for LargeRns<P, LIMBS>
where
    P: LargeRnsProfile<LIMBS>,
{
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<P, const LIMBS: usize> SubAssign<&Self> for LargeRns<P, LIMBS>
where
    P: LargeRnsProfile<LIMBS>,
{
    fn sub_assign(&mut self, rhs: &Self) {
        *self = *self - *rhs;
    }
}

impl<P, const LIMBS: usize> Mul for LargeRns<P, LIMBS>
where
    P: LargeRnsProfile<LIMBS>,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let residues = core::array::from_fn(|i| {
            ((self.residues[i] as u128 * rhs.residues[i] as u128) % P::MODULI[i] as u128) as u64
        });
        Self::from_residues_unchecked(residues)
    }
}

impl<P, const LIMBS: usize> Mul<&Self> for LargeRns<P, LIMBS>
where
    P: LargeRnsProfile<LIMBS>,
{
    type Output = Self;

    fn mul(self, rhs: &Self) -> Self::Output {
        self * *rhs
    }
}

impl<P, const LIMBS: usize> MulAssign for LargeRns<P, LIMBS>
where
    P: LargeRnsProfile<LIMBS>,
{
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<P, const LIMBS: usize> MulAssign<&Self> for LargeRns<P, LIMBS>
where
    P: LargeRnsProfile<LIMBS>,
{
    fn mul_assign(&mut self, rhs: &Self) {
        *self = *self * *rhs;
    }
}

impl<P, const LIMBS: usize> Add<Self> for &LargeRns<P, LIMBS>
where
    P: LargeRnsProfile<LIMBS>,
{
    type Output = LargeRns<P, LIMBS>;

    fn add(self, rhs: Self) -> Self::Output {
        *self + *rhs
    }
}

impl<P, const LIMBS: usize> Add<LargeRns<P, LIMBS>> for &LargeRns<P, LIMBS>
where
    P: LargeRnsProfile<LIMBS>,
{
    type Output = LargeRns<P, LIMBS>;

    fn add(self, rhs: LargeRns<P, LIMBS>) -> Self::Output {
        *self + rhs
    }
}

impl<P, const LIMBS: usize> Sub<Self> for &LargeRns<P, LIMBS>
where
    P: LargeRnsProfile<LIMBS>,
{
    type Output = LargeRns<P, LIMBS>;

    fn sub(self, rhs: Self) -> Self::Output {
        *self - *rhs
    }
}

impl<P, const LIMBS: usize> Sub<LargeRns<P, LIMBS>> for &LargeRns<P, LIMBS>
where
    P: LargeRnsProfile<LIMBS>,
{
    type Output = LargeRns<P, LIMBS>;

    fn sub(self, rhs: LargeRns<P, LIMBS>) -> Self::Output {
        *self - rhs
    }
}

impl<P, const LIMBS: usize> Mul<Self> for &LargeRns<P, LIMBS>
where
    P: LargeRnsProfile<LIMBS>,
{
    type Output = LargeRns<P, LIMBS>;

    fn mul(self, rhs: Self) -> Self::Output {
        *self * *rhs
    }
}

impl<P, const LIMBS: usize> Mul<LargeRns<P, LIMBS>> for &LargeRns<P, LIMBS>
where
    P: LargeRnsProfile<LIMBS>,
{
    type Output = LargeRns<P, LIMBS>;

    fn mul(self, rhs: LargeRns<P, LIMBS>) -> Self::Output {
        *self * rhs
    }
}

impl<P, const LIMBS: usize> Neg for LargeRns<P, LIMBS>
where
    P: LargeRnsProfile<LIMBS>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        let residues = core::array::from_fn(|i| {
            if self.residues[i] == 0 {
                0
            } else {
                P::MODULI[i] - self.residues[i]
            }
        });
        Self::from_residues_unchecked(residues)
    }
}

impl<P, const LIMBS: usize> Ring for LargeRns<P, LIMBS>
where
    P: LargeRnsProfile<LIMBS>,
{
    fn zero() -> Self {
        Self::from_residues_unchecked([0; LIMBS])
    }

    fn one() -> Self {
        Self::from_small_u64(1)
    }

    fn add_assign_slice(dst: &mut [Self], src: &[Self]) {
        assert_eq!(dst.len(), src.len(), "slice lengths must match");
        for (lhs, rhs) in dst.iter_mut().zip(src.iter()) {
            *lhs += *rhs;
        }
    }

    fn sub_assign_slice(dst: &mut [Self], src: &[Self]) {
        assert_eq!(dst.len(), src.len(), "slice lengths must match");
        for (lhs, rhs) in dst.iter_mut().zip(src.iter()) {
            *lhs -= *rhs;
        }
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

impl<P, const LIMBS: usize> IntegerRing for LargeRns<P, LIMBS>
where
    P: LargeRnsProfile<LIMBS>,
{
    type Uint = BigUint<LIMBS>;

    fn modulus() -> Self::Uint {
        BigUint { limbs: P::MODULUS }
    }

    fn from_u64(val: u64) -> Self {
        Self::from_small_u64(val)
    }

    fn to_u64(&self) -> u64 {
        self.try_to_u64()
            .expect("large-RNS element does not fit in u64")
    }

    fn lossy_l2_value(&self) -> f64 {
        let v = self.to_canonical().lossy_l2_value();
        let m = Self::modulus_canonical().lossy_l2_value();
        let half = m * 0.5;
        if v > half { v - m } else { v }
    }

    fn reduce(&self) -> Self {
        *self
    }
}

impl<P, const LIMBS: usize> LargeCanonicalRing for LargeRns<P, LIMBS>
where
    P: LargeRnsProfile<LIMBS>,
{
    type Canonical = BigUint<LIMBS>;

    fn modulus_canonical() -> Self::Canonical {
        BigUint { limbs: P::MODULUS }
    }

    fn from_small_u64(value: u64) -> Self {
        let residues = core::array::from_fn(|i| value % P::MODULI[i]);
        Self::from_residues_unchecked(residues)
    }

    fn from_canonical(value: &Self::Canonical) -> Self {
        Self::from_residues_unchecked(Self::decompose_canonical(value))
    }

    fn to_canonical(&self) -> Self::Canonical {
        self.reconstruct_canonical()
    }

    fn try_to_u64(&self) -> Option<u64> {
        self.to_canonical().try_to_u64()
    }

    fn try_to_u128(&self) -> Option<u128> {
        self.to_canonical().try_to_u128()
    }
}

impl<P, const LIMBS: usize> NTTRing for LargeRns<P, LIMBS>
where
    P: LargeRnsProfile<LIMBS> + Send + Sync + 'static,
{
    fn root_of_unity(n: usize) -> Option<Self> {
        if n == 0 || !n.is_power_of_two() {
            return None;
        }
        if n == 1 {
            return Some(Self::one());
        }

        let max_ntt_size = Self::max_ntt_size();
        if n > max_ntt_size {
            return None;
        }

        let mut root = Self::primitive_two_adic_root();
        let mut order = max_ntt_size;
        while order > n {
            root = root.square();
            order >>= 1;
        }

        Some(root)
    }

    fn inv_root_of_unity(n: usize) -> Option<Self> {
        let root = Self::root_of_unity(n)?;
        let residues = core::array::from_fn(|i| mod_inv_u64(root.residues[i], P::MODULI[i]));
        Some(Self::from_residues_unchecked(residues))
    }

    fn inverse_ntt_scale(n: usize) -> Option<Self> {
        if n == 0 {
            return None;
        }
        let residues = core::array::from_fn(|i| {
            let residue = (n as u128 % P::MODULI[i] as u128) as u64;
            mod_inv_u64(residue, P::MODULI[i])
        });
        Some(Self::from_residues_unchecked(residues))
    }

    fn max_ntt_size() -> usize {
        1usize
            .checked_shl(P::TWO_ADICITY as u32)
            .expect("large-RNS profile two-adicity exceeds usize width")
    }

    fn ntt_forward_in_place(coeffs: &mut [Self]) -> Result<(), NttError> {
        let plan = cached_ntt_plan::<Self>(coeffs.len())?;
        Self::ntt_forward_with_plan(coeffs, plan.as_ref())
    }

    fn ntt_forward_with_plan(coeffs: &mut [Self], plan: &NttPlan<Self>) -> Result<(), NttError> {
        super::ntt::scalar_ntt_forward_with_plan(coeffs, plan)
    }

    fn ntt_inverse_in_place(evals: &mut [Self]) -> Result<(), NttError> {
        let plan = cached_ntt_plan::<Self>(evals.len())?;
        Self::ntt_inverse_with_plan(evals, plan.as_ref())
    }

    fn ntt_inverse_with_plan(evals: &mut [Self], plan: &NttPlan<Self>) -> Result<(), NttError> {
        super::ntt::scalar_ntt_inverse_with_plan(evals, plan)
    }
}

impl<P, const LIMBS: usize> CanonicalSerialize for LargeRns<P, LIMBS>
where
    P: LargeRnsProfile<LIMBS>,
{
    fn serialized_size(&self) -> usize {
        LIMBS * 8
    }

    fn serialize_into(&self, buf: &mut alloc::vec::Vec<u8>) -> Result<(), SerializationError> {
        self.to_canonical().serialize_into(buf)
    }
}

impl<P, const LIMBS: usize> CanonicalDeserialize for LargeRns<P, LIMBS>
where
    P: LargeRnsProfile<LIMBS>,
{
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        let (value, consumed) = BigUint::<LIMBS>::deserialize(data)?;
        if value >= Self::modulus_canonical() {
            return Err(SerializationError::InvalidData(alloc::format!(
                "value {} >= modulus {}",
                value,
                Self::modulus_canonical()
            )));
        }
        Ok((Self::from_canonical(&value), consumed))
    }
}

impl<P, const LIMBS: usize> grid_serialize::Valid for LargeRns<P, LIMBS>
where
    P: LargeRnsProfile<LIMBS>,
{
    fn is_valid(&self) -> bool {
        self.residues
            .iter()
            .zip(P::MODULI.iter())
            .all(|(residue, modulus)| residue < modulus)
    }
}

impl<P, const LIMBS: usize> grid_std::UniformRand for LargeRns<P, LIMBS>
where
    P: LargeRnsProfile<LIMBS>,
{
    fn rand<R: grid_std::rand::RngExt + ?Sized>(rng: &mut R) -> Self {
        let residues = core::array::from_fn(|i| {
            let modulus = P::MODULI[i];
            let reject = (u64::MAX % modulus).wrapping_add(1) % modulus;
            let upper = u64::MAX - reject;
            loop {
                let sample: u64 = rng.random();
                if sample <= upper {
                    break sample % modulus;
                }
            }
        });
        Self::from_residues_unchecked(residues)
    }
}

#[cfg(test)]
mod tests {
    use alloc::sync::Arc;

    use super::Rns3V0;
    use crate::arith::LargeCanonicalRing;
    use crate::arith::bigint::BigUint;
    use crate::arith::ring::{self, IntegerRing};
    use grid_serialize::{CanonicalDeserialize, CanonicalSerialize, Valid};

    fn exercise_large_rns_profile<R, const LIMBS: usize>()
    where
        R: IntegerRing<Uint = BigUint<LIMBS>>
            + LargeCanonicalRing<Canonical = BigUint<LIMBS>>
            + CanonicalSerialize
            + CanonicalDeserialize
            + Valid,
    {
        ring::tests::test_ring_axioms(R::from_u64(3), R::from_u64(5), R::from_u64(7));
        ring::tests::test_integer_ring::<R>(19);

        let a = R::from_u64(1234);
        let b = R::from_u64(5678);
        assert_eq!((a.clone() + &b).try_to_u128(), Some(6912));
        assert_eq!(
            (a.clone() * &b).try_to_u128(),
            Some(1234u128.wrapping_mul(5678u128))
        );

        let modulus = R::modulus_canonical();
        let (modulus_minus_one, borrow) = modulus.sub_small(1);
        assert!(!borrow);
        let (modulus_plus_five, carry) = modulus.add_small(5);
        assert!(!carry);

        assert!(
            R::from_canonical(&modulus_minus_one)
                .try_to_u128()
                .is_none()
        );
        assert_eq!(R::from_canonical(&modulus_plus_five).try_to_u64(), Some(5));

        let bytes = a.serialize().unwrap();
        assert_eq!(bytes.len(), LIMBS * 8);
        assert_eq!(bytes, a.to_canonical().serialize().unwrap());
        let (decoded, consumed) = R::deserialize(&bytes).unwrap();
        assert_eq!(consumed, LIMBS * 8);
        assert_eq!(decoded, a);
        assert!(decoded.is_valid());
    }

    #[test]
    fn test_rns3_v0_large_rns_backend() {
        exercise_large_rns_profile::<Rns3V0, 3>();
    }

    #[test]
    fn test_large_rns_composite_interop_round_trip() {
        let value = Rns3V0::from_u64(123456789);
        let dynamic = value.to_composite();
        assert_eq!(dynamic.to_canonical::<3>(), value.to_canonical());
        assert_eq!(Rns3V0::try_from_composite(&dynamic).unwrap(), value);
    }

    #[test]
    fn test_large_rns_try_from_composite_rejects_basis_mismatch() {
        let wrong = crate::arith::composite::CompositeRing::from_u64_with_basis(
            1,
            Arc::new(crate::arith::rns::RnsBasis::new(vec![7, 11, 13])),
        );
        assert!(Rns3V0::try_from_composite(&wrong).is_err());
    }

    #[test]
    fn test_large_rns_deserialize_rejects_value_at_modulus() {
        let bytes = Rns3V0::modulus_canonical().serialize().unwrap();
        assert!(Rns3V0::deserialize(&bytes).is_err());
    }

    #[test]
    fn test_large_rns_slice_hooks_match_scalar_ops() {
        let lhs = [
            Rns3V0::from_u64(1),
            Rns3V0::from_u64(2),
            Rns3V0::from_u64(3),
            Rns3V0::from_u64(4),
            Rns3V0::from_u64(5),
        ];
        let rhs = [
            Rns3V0::from_u64(9),
            Rns3V0::from_u64(8),
            Rns3V0::from_u64(7),
            Rns3V0::from_u64(6),
            Rns3V0::from_u64(5),
        ];

        let mut add_actual = lhs;
        let mut sub_actual = lhs;
        <Rns3V0 as crate::arith::ring::Ring>::add_assign_slice(&mut add_actual, &rhs);
        <Rns3V0 as crate::arith::ring::Ring>::sub_assign_slice(&mut sub_actual, &rhs);

        let add_expected = core::array::from_fn(|i| lhs[i] + rhs[i]);
        let sub_expected = core::array::from_fn(|i| lhs[i] - rhs[i]);

        assert_eq!(add_actual, add_expected);
        assert_eq!(sub_actual, sub_expected);
    }
}
