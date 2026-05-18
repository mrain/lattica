//! Fixed-limb Montgomery large-prime backend.

use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::bigint::BigUint;
use super::large_modulus::{LargeCanonicalRing, LargePrimeProfile};
use super::large_prime_profiles::{
    Bls12_381FqProfile, Bls12_381FrProfile, Bn254FqProfile, Bn254FrProfile,
};
use super::ntt::{NTTRing, NttError, NttPlan, cached_ntt_plan};
use super::ring::{Field, IntegerRing, Ring};
use grid_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError};

/// A fixed-limb Montgomery prime field element.
#[repr(transparent)]
pub struct LargePrimeField<P, const LIMBS: usize>
where
    P: LargePrimeProfile<LIMBS>,
{
    limbs: [u64; LIMBS],
    _profile: PhantomData<P>,
}

impl<P, const LIMBS: usize> Copy for LargePrimeField<P, LIMBS> where P: LargePrimeProfile<LIMBS> {}

impl<P, const LIMBS: usize> Clone for LargePrimeField<P, LIMBS>
where
    P: LargePrimeProfile<LIMBS>,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<P, const LIMBS: usize> PartialEq for LargePrimeField<P, LIMBS>
where
    P: LargePrimeProfile<LIMBS>,
{
    fn eq(&self, other: &Self) -> bool {
        self.limbs == other.limbs
    }
}

impl<P, const LIMBS: usize> Eq for LargePrimeField<P, LIMBS> where P: LargePrimeProfile<LIMBS> {}

/// BN254 scalar field over the large-prime backend.
pub type Bn254Fr = LargePrimeField<Bn254FrProfile, 4>;
/// BN254 base field over the large-prime backend.
pub type Bn254Fq = LargePrimeField<Bn254FqProfile, 4>;
/// BLS12-381 scalar field over the large-prime backend.
pub type Bls12_381Fr = LargePrimeField<Bls12_381FrProfile, 4>;
/// BLS12-381 base field over the large-prime backend.
pub type Bls12_381Fq = LargePrimeField<Bls12_381FqProfile, 6>;

impl<P, const LIMBS: usize> LargePrimeField<P, LIMBS>
where
    P: LargePrimeProfile<LIMBS>,
{
    #[inline(always)]
    const fn zero_limbs() -> [u64; LIMBS] {
        [0; LIMBS]
    }

    #[inline(always)]
    fn from_raw_montgomery(limbs: [u64; LIMBS]) -> Self {
        Self {
            limbs,
            _profile: PhantomData,
        }
    }

    #[inline(always)]
    fn canonical_biguint(limbs: [u64; LIMBS]) -> BigUint<LIMBS> {
        BigUint { limbs }
    }

    #[inline(always)]
    fn modulus_biguint() -> BigUint<LIMBS> {
        Self::canonical_biguint(P::MODULUS)
    }

    #[inline(always)]
    fn two_adic_root_biguint() -> BigUint<LIMBS> {
        Self::canonical_biguint(P::TWO_ADIC_ROOT)
    }

    #[inline(always)]
    fn barrett_mu_biguint() -> BigUint<LIMBS> {
        Self::canonical_biguint(P::BARRETT_MU)
    }

    #[inline(always)]
    fn compare_limbs(lhs: &[u64; LIMBS], rhs: &[u64; LIMBS]) -> core::cmp::Ordering {
        for i in (0..LIMBS).rev() {
            match lhs[i].cmp(&rhs[i]) {
                core::cmp::Ordering::Equal => continue,
                ord => return ord,
            }
        }
        core::cmp::Ordering::Equal
    }

    #[inline(always)]
    fn is_canonical_limbs(limbs: &[u64; LIMBS]) -> bool {
        Self::compare_limbs(limbs, &P::MODULUS) == core::cmp::Ordering::Less
    }

    #[inline(always)]
    fn adc(a: u64, b: u64, carry: &mut u64) -> u64 {
        let tmp = (a as u128) + (b as u128) + (*carry as u128);
        *carry = (tmp >> 64) as u64;
        tmp as u64
    }

    #[inline(always)]
    fn mac_with_carry(a: u64, b: u64, c: u64, carry: &mut u64) -> u64 {
        let tmp = (a as u128) + (b as u128) * (c as u128) + (*carry as u128);
        *carry = (tmp >> 64) as u64;
        tmp as u64
    }

    #[inline(always)]
    fn mac_discard(a: u64, b: u64, c: u64, carry: &mut u64) {
        let tmp = (a as u128) + (b as u128) * (c as u128);
        *carry = (tmp >> 64) as u64;
    }

    #[inline(always)]
    fn add_limbs(lhs: &[u64; LIMBS], rhs: &[u64; LIMBS]) -> ([u64; LIMBS], bool) {
        let mut out = [0; LIMBS];
        let mut carry = 0;
        for i in 0..LIMBS {
            out[i] = Self::adc(lhs[i], rhs[i], &mut carry);
        }
        (out, carry != 0)
    }

    #[inline(always)]
    fn sub_limbs(lhs: &[u64; LIMBS], rhs: &[u64; LIMBS]) -> ([u64; LIMBS], bool) {
        let mut out = [0; LIMBS];
        let mut borrow = 0u64;
        for i in 0..LIMBS {
            let (diff1, borrow1) = lhs[i].overflowing_sub(rhs[i]);
            let (diff2, borrow2) = diff1.overflowing_sub(borrow);
            out[i] = diff2;
            borrow = (borrow1 as u64) + (borrow2 as u64);
        }
        (out, borrow != 0)
    }

    #[inline(always)]
    fn add_modulus_if_borrow(mut limbs: [u64; LIMBS], borrow: bool) -> [u64; LIMBS] {
        if borrow {
            let mut carry = 0u64;
            for (dst, modulus) in limbs.iter_mut().zip(P::MODULUS.iter()) {
                *dst = Self::adc(*dst, *modulus, &mut carry);
            }
        }
        limbs
    }

    #[inline(always)]
    fn sub_modulus_once(limbs: &[u64; LIMBS]) -> [u64; LIMBS] {
        let (reduced, borrow) = Self::sub_limbs(limbs, &P::MODULUS);
        assert!(
            !borrow,
            "subtracting modulus requires a non-negative operand"
        );
        reduced
    }

    #[inline(always)]
    fn cond_subtract_modulus(limbs: [u64; LIMBS], carry: bool) -> [u64; LIMBS] {
        if carry || !Self::is_canonical_limbs(&limbs) {
            Self::sub_modulus_once(&limbs)
        } else {
            limbs
        }
    }

    fn mul_buffers(lhs: &[u64; LIMBS], rhs: &[u64; LIMBS]) -> ([u64; LIMBS], [u64; LIMBS]) {
        let mut lo = [0u64; LIMBS];
        let mut hi = [0u64; LIMBS];

        for (i, &lhs_limb) in lhs.iter().enumerate() {
            let mut carry = 0u64;
            for (j, &rhs_limb) in rhs.iter().enumerate() {
                let k = i + j;
                if k >= LIMBS {
                    hi[k - LIMBS] =
                        Self::mac_with_carry(hi[k - LIMBS], lhs_limb, rhs_limb, &mut carry);
                } else {
                    lo[k] = Self::mac_with_carry(lo[k], lhs_limb, rhs_limb, &mut carry);
                }
            }
            hi[i] = carry;
        }

        (lo, hi)
    }

    fn montgomery_reduce(mut lo: [u64; LIMBS], mut hi: [u64; LIMBS]) -> (bool, [u64; LIMBS]) {
        let mut carry2 = 0u64;

        for i in 0..LIMBS {
            let tmp = lo[i].wrapping_mul(P::MONT_NEG_INV);
            let mut carry = 0u64;

            Self::mac_discard(lo[i], tmp, P::MODULUS[0], &mut carry);
            for j in 1..LIMBS {
                let k = i + j;
                if k >= LIMBS {
                    hi[k - LIMBS] =
                        Self::mac_with_carry(hi[k - LIMBS], tmp, P::MODULUS[j], &mut carry);
                } else {
                    lo[k] = Self::mac_with_carry(lo[k], tmp, P::MODULUS[j], &mut carry);
                }
            }

            hi[i] = Self::adc(hi[i], carry, &mut carry2);
        }

        (carry2 != 0, hi)
    }

    #[inline(always)]
    fn montgomery_mul_limbs(lhs: &[u64; LIMBS], rhs: &[u64; LIMBS]) -> [u64; LIMBS] {
        let (lo, hi) = Self::mul_buffers(lhs, rhs);
        let (carry, limbs) = Self::montgomery_reduce(lo, hi);
        Self::cond_subtract_modulus(limbs, carry)
    }

    #[inline(always)]
    fn montgomery_to_canonical_limbs(limbs: &[u64; LIMBS]) -> [u64; LIMBS] {
        let (carry, reduced) = Self::montgomery_reduce(*limbs, Self::zero_limbs());
        Self::cond_subtract_modulus(reduced, carry)
    }

    fn reduce_canonical(value: &BigUint<LIMBS>) -> BigUint<LIMBS> {
        if Self::is_canonical_limbs(&value.limbs) {
            return *value;
        }

        let modulus = Self::modulus_biguint();
        let (_, quotient) = value.widening_mul(&Self::barrett_mu_biguint());
        let (quotient_times_modulus, quotient_times_modulus_hi) = modulus.widening_mul(&quotient);
        assert!(
            quotient_times_modulus_hi.is_zero(),
            "Barrett quotient must keep q * modulus within the canonical limb width"
        );

        let (reduced, borrow) = value.sub_with_borrow(&quotient_times_modulus);
        assert!(
            !borrow,
            "Barrett quotient must not overshoot the true canonical quotient"
        );

        let reduced = reduced.sub_if_ge(&modulus).sub_if_ge(&modulus);
        assert!(
            reduced < modulus,
            "Barrett reduction correction must produce a canonical representative"
        );
        reduced
    }

    #[cfg(test)]
    #[inline(always)]
    fn modulus_minus_two() -> BigUint<LIMBS> {
        let (minus_two, borrow) = Self::modulus_biguint().sub_small(2);
        debug_assert!(!borrow, "prime modulus must be larger than two");
        minus_two
    }

    #[cfg(test)]
    fn pow_bigint(&self, exp: &BigUint<LIMBS>) -> Self {
        let mut result = Self::one();
        for i in (0..LIMBS).rev() {
            let limb = exp.limbs[i];
            for bit in (0..64).rev() {
                result = result.square();
                if ((limb >> bit) & 1) == 1 {
                    result *= self;
                }
            }
        }
        result
    }

    #[inline(always)]
    fn is_even_biguint(value: &BigUint<LIMBS>) -> bool {
        (value.limbs[0] & 1) == 0
    }

    #[inline(always)]
    fn sub_mod_canonical(
        lhs: &BigUint<LIMBS>,
        rhs: &BigUint<LIMBS>,
        modulus: &BigUint<LIMBS>,
    ) -> BigUint<LIMBS> {
        let (diff, borrow) = lhs.sub_with_borrow(rhs);
        if !borrow {
            diff
        } else {
            let (rhs_minus_lhs, rhs_borrow) = rhs.sub_with_borrow(lhs);
            assert!(
                !rhs_borrow,
                "modular subtraction requires ordered difference"
            );
            let (wrapped, modulus_borrow) = modulus.sub_with_borrow(&rhs_minus_lhs);
            assert!(
                !modulus_borrow,
                "modular subtraction must stay within the modulus"
            );
            wrapped
        }
    }

    #[inline(always)]
    fn half_with_high_bit(value: &BigUint<LIMBS>, high_bit: bool) -> BigUint<LIMBS> {
        let mut out = [0u64; LIMBS];
        let mut incoming = high_bit as u64;
        for i in (0..LIMBS).rev() {
            let next_incoming = value.limbs[i] & 1;
            out[i] = (value.limbs[i] >> 1) | (incoming << 63);
            incoming = next_incoming;
        }
        Self::canonical_biguint(out)
    }

    #[inline(always)]
    fn half_mod_canonical(value: &BigUint<LIMBS>, modulus: &BigUint<LIMBS>) -> BigUint<LIMBS> {
        if Self::is_even_biguint(value) {
            value.shr_bits(1)
        } else {
            let (sum, carry) = value.add_with_carry(modulus);
            Self::half_with_high_bit(&sum, carry)
        }
    }

    fn invert_canonical(value: &BigUint<LIMBS>) -> BigUint<LIMBS> {
        let modulus = Self::modulus_biguint();
        let mut u = *value;
        let mut v = modulus;
        let mut x1 = BigUint::<LIMBS>::one();
        let mut x2 = BigUint::<LIMBS>::ZERO;

        while u != BigUint::<LIMBS>::one() && v != BigUint::<LIMBS>::one() {
            while Self::is_even_biguint(&u) {
                u = u.shr_bits(1);
                x1 = Self::half_mod_canonical(&x1, &modulus);
            }
            while Self::is_even_biguint(&v) {
                v = v.shr_bits(1);
                x2 = Self::half_mod_canonical(&x2, &modulus);
            }

            if u >= v {
                let (next_u, borrow) = u.sub_with_borrow(&v);
                assert!(!borrow, "binary inverse requires ordered subtraction");
                u = next_u;
                x1 = Self::sub_mod_canonical(&x1, &x2, &modulus);
            } else {
                let (next_v, borrow) = v.sub_with_borrow(&u);
                assert!(!borrow, "binary inverse requires ordered subtraction");
                v = next_v;
                x2 = Self::sub_mod_canonical(&x2, &x1, &modulus);
            }
        }

        if u == BigUint::<LIMBS>::one() { x1 } else { x2 }
    }

    #[inline(always)]
    fn primitive_two_adic_root() -> Self {
        Self::from_canonical(&Self::two_adic_root_biguint())
    }
}

impl<P, const LIMBS: usize> core::fmt::Debug for LargePrimeField<P, LIMBS>
where
    P: LargePrimeProfile<LIMBS>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "LargePrimeField({:?})", self.to_canonical())
    }
}

impl<P, const LIMBS: usize> core::fmt::Display for LargePrimeField<P, LIMBS>
where
    P: LargePrimeProfile<LIMBS>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Display::fmt(&self.to_canonical(), f)
    }
}

impl<P, const LIMBS: usize> Add for LargePrimeField<P, LIMBS>
where
    P: LargePrimeProfile<LIMBS>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let (sum, carry) = Self::add_limbs(&self.limbs, &rhs.limbs);
        Self::from_raw_montgomery(Self::cond_subtract_modulus(sum, carry))
    }
}

impl<P, const LIMBS: usize> Add<&Self> for LargePrimeField<P, LIMBS>
where
    P: LargePrimeProfile<LIMBS>,
{
    type Output = Self;

    fn add(self, rhs: &Self) -> Self::Output {
        self + *rhs
    }
}

impl<P, const LIMBS: usize> AddAssign for LargePrimeField<P, LIMBS>
where
    P: LargePrimeProfile<LIMBS>,
{
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<P, const LIMBS: usize> AddAssign<&Self> for LargePrimeField<P, LIMBS>
where
    P: LargePrimeProfile<LIMBS>,
{
    fn add_assign(&mut self, rhs: &Self) {
        *self = *self + *rhs;
    }
}

impl<P, const LIMBS: usize> Sub for LargePrimeField<P, LIMBS>
where
    P: LargePrimeProfile<LIMBS>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let (diff, borrow) = Self::sub_limbs(&self.limbs, &rhs.limbs);
        Self::from_raw_montgomery(Self::add_modulus_if_borrow(diff, borrow))
    }
}

impl<P, const LIMBS: usize> Sub<&Self> for LargePrimeField<P, LIMBS>
where
    P: LargePrimeProfile<LIMBS>,
{
    type Output = Self;

    fn sub(self, rhs: &Self) -> Self::Output {
        self - *rhs
    }
}

impl<P, const LIMBS: usize> SubAssign for LargePrimeField<P, LIMBS>
where
    P: LargePrimeProfile<LIMBS>,
{
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<P, const LIMBS: usize> SubAssign<&Self> for LargePrimeField<P, LIMBS>
where
    P: LargePrimeProfile<LIMBS>,
{
    fn sub_assign(&mut self, rhs: &Self) {
        *self = *self - *rhs;
    }
}

impl<P, const LIMBS: usize> Mul for LargePrimeField<P, LIMBS>
where
    P: LargePrimeProfile<LIMBS>,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::from_raw_montgomery(Self::montgomery_mul_limbs(&self.limbs, &rhs.limbs))
    }
}

impl<P, const LIMBS: usize> Mul<&Self> for LargePrimeField<P, LIMBS>
where
    P: LargePrimeProfile<LIMBS>,
{
    type Output = Self;

    fn mul(self, rhs: &Self) -> Self::Output {
        self * *rhs
    }
}

impl<P, const LIMBS: usize> MulAssign for LargePrimeField<P, LIMBS>
where
    P: LargePrimeProfile<LIMBS>,
{
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<P, const LIMBS: usize> MulAssign<&Self> for LargePrimeField<P, LIMBS>
where
    P: LargePrimeProfile<LIMBS>,
{
    fn mul_assign(&mut self, rhs: &Self) {
        *self = *self * *rhs;
    }
}

impl<P, const LIMBS: usize> Add<Self> for &LargePrimeField<P, LIMBS>
where
    P: LargePrimeProfile<LIMBS>,
{
    type Output = LargePrimeField<P, LIMBS>;

    fn add(self, rhs: Self) -> Self::Output {
        *self + *rhs
    }
}

impl<P, const LIMBS: usize> Add<LargePrimeField<P, LIMBS>> for &LargePrimeField<P, LIMBS>
where
    P: LargePrimeProfile<LIMBS>,
{
    type Output = LargePrimeField<P, LIMBS>;

    fn add(self, rhs: LargePrimeField<P, LIMBS>) -> Self::Output {
        *self + rhs
    }
}

impl<P, const LIMBS: usize> Sub<Self> for &LargePrimeField<P, LIMBS>
where
    P: LargePrimeProfile<LIMBS>,
{
    type Output = LargePrimeField<P, LIMBS>;

    fn sub(self, rhs: Self) -> Self::Output {
        *self - *rhs
    }
}

impl<P, const LIMBS: usize> Sub<LargePrimeField<P, LIMBS>> for &LargePrimeField<P, LIMBS>
where
    P: LargePrimeProfile<LIMBS>,
{
    type Output = LargePrimeField<P, LIMBS>;

    fn sub(self, rhs: LargePrimeField<P, LIMBS>) -> Self::Output {
        *self - rhs
    }
}

impl<P, const LIMBS: usize> Mul<Self> for &LargePrimeField<P, LIMBS>
where
    P: LargePrimeProfile<LIMBS>,
{
    type Output = LargePrimeField<P, LIMBS>;

    fn mul(self, rhs: Self) -> Self::Output {
        *self * *rhs
    }
}

impl<P, const LIMBS: usize> Mul<LargePrimeField<P, LIMBS>> for &LargePrimeField<P, LIMBS>
where
    P: LargePrimeProfile<LIMBS>,
{
    type Output = LargePrimeField<P, LIMBS>;

    fn mul(self, rhs: LargePrimeField<P, LIMBS>) -> Self::Output {
        *self * rhs
    }
}

impl<P, const LIMBS: usize> Neg for LargePrimeField<P, LIMBS>
where
    P: LargePrimeProfile<LIMBS>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        if self.is_zero() {
            self
        } else {
            let (diff, borrow) = Self::sub_limbs(&P::MODULUS, &self.limbs);
            assert!(
                !borrow,
                "canonical Montgomery limb must be below the modulus"
            );
            Self::from_raw_montgomery(diff)
        }
    }
}

impl<P, const LIMBS: usize> Ring for LargePrimeField<P, LIMBS>
where
    P: LargePrimeProfile<LIMBS>,
{
    fn zero() -> Self {
        Self::from_raw_montgomery(Self::zero_limbs())
    }

    fn one() -> Self {
        Self::from_canonical(&BigUint::one())
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

    fn add_assign_scaled_slice(dst: &mut [Self], src: &[Self], scalar: &Self) {
        assert_eq!(dst.len(), src.len(), "slice lengths must match");
        const CHUNK: usize = 64;
        if dst.len() <= CHUNK {
            for (dst_value, src_value) in dst.iter_mut().zip(src.iter()) {
                *dst_value += *src_value * scalar;
            }
            return;
        }
        for (dst_chunk, src_chunk) in dst.chunks_mut(CHUNK).zip(src.chunks(CHUNK)) {
            let mut products = [Self::zero(); CHUNK];
            let len = src_chunk.len();
            products[..len].copy_from_slice(src_chunk);
            Self::scalar_mul_slice(&mut products[..len], scalar);
            Self::add_assign_slice(dst_chunk, &products[..len]);
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

impl<P, const LIMBS: usize> IntegerRing for LargePrimeField<P, LIMBS>
where
    P: LargePrimeProfile<LIMBS>,
{
    type Uint = BigUint<LIMBS>;

    fn modulus() -> Self::Uint {
        Self::modulus_biguint()
    }

    fn from_u64(val: u64) -> Self {
        Self::from_small_u64(val)
    }

    fn to_u64(&self) -> u64 {
        self.try_to_u64()
            .expect("large-prime element does not fit in u64")
    }

    fn lossy_l2_value(&self) -> f64 {
        let v = self.to_canonical().lossy_l2_value();
        let p = Self::modulus_biguint().lossy_l2_value();
        let half = p * 0.5;
        if v > half { v - p } else { v }
    }

    fn reduce(&self) -> Self {
        *self
    }
}

impl<P, const LIMBS: usize> Field for LargePrimeField<P, LIMBS>
where
    P: LargePrimeProfile<LIMBS>,
{
    fn inv(&self) -> Self {
        assert!(!self.is_zero(), "cannot invert zero");
        // Montgomery inverse: self.limbs = x*R mod p.
        // invert_canonical treats the limbs as a canonical integer, computing
        // (x*R)^{-1} mod p = x^{-1}*R^{-1} mod p.
        // from_canonical(c) produces c*R mod p (Montgomery form), so:
        //   from_canonical(x^{-1}*R^{-1}) = x^{-1}*R^{-2} mod p (Montgomery).
        // Montgomery multiply by MONT_R2 (= R^2 mod p as Montgomery = R^3 mod p raw):
        //   montgomery_mul(x^{-1}*R^{-2}, R^3) = x^{-1}*R^{-2}*R^3*R^{-1} mod p
        //                                      = x^{-1} mod p (Montgomery form).
        let inverse_of_raw = Self::invert_canonical(&Self::canonical_biguint(self.limbs));
        Self::from_canonical(&inverse_of_raw) * Self::from_raw_montgomery(P::MONT_R2)
    }
}

impl<P, const LIMBS: usize> LargeCanonicalRing for LargePrimeField<P, LIMBS>
where
    P: LargePrimeProfile<LIMBS>,
{
    type Canonical = BigUint<LIMBS>;

    fn modulus_canonical() -> Self::Canonical {
        Self::modulus_biguint()
    }

    fn from_small_u64(value: u64) -> Self {
        let mut canonical = [0u64; LIMBS];
        canonical[0] = value;
        Self::from_canonical(&Self::canonical_biguint(canonical))
    }

    fn from_canonical(value: &Self::Canonical) -> Self {
        let reduced = Self::reduce_canonical(value);
        let limbs = Self::montgomery_mul_limbs(&reduced.limbs, &P::MONT_R2);
        Self::from_raw_montgomery(limbs)
    }

    fn to_canonical(&self) -> Self::Canonical {
        Self::canonical_biguint(Self::montgomery_to_canonical_limbs(&self.limbs))
    }

    fn try_to_u64(&self) -> Option<u64> {
        self.to_canonical().try_to_u64()
    }

    fn try_to_u128(&self) -> Option<u128> {
        self.to_canonical().try_to_u128()
    }
}

impl<P, const LIMBS: usize> NTTRing for LargePrimeField<P, LIMBS>
where
    P: LargePrimeProfile<LIMBS> + Send + Sync + 'static,
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

        debug_assert_eq!(root.pow(n as u64), Self::one());
        debug_assert!(n == 1 || root.pow((n / 2) as u64) != Self::one());
        Some(root)
    }

    fn inv_root_of_unity(n: usize) -> Option<Self> {
        Self::root_of_unity(n).map(|root| root.inv())
    }

    fn inverse_ntt_scale(n: usize) -> Option<Self> {
        let n = u64::try_from(n).ok()?;
        Some(Self::from_u64(n).inv())
    }

    fn max_ntt_size() -> usize {
        1usize
            .checked_shl(P::TWO_ADICITY as u32)
            .expect("large-prime profile two-adicity exceeds usize width")
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

impl<P, const LIMBS: usize> CanonicalSerialize for LargePrimeField<P, LIMBS>
where
    P: LargePrimeProfile<LIMBS>,
{
    fn serialized_size(&self) -> usize {
        LIMBS * 8
    }

    fn serialize_into(&self, buf: &mut alloc::vec::Vec<u8>) -> Result<(), SerializationError> {
        self.to_canonical().serialize_into(buf)
    }
}

impl<P, const LIMBS: usize> CanonicalDeserialize for LargePrimeField<P, LIMBS>
where
    P: LargePrimeProfile<LIMBS>,
{
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        let (value, consumed) = BigUint::<LIMBS>::deserialize(data)?;
        if !Self::is_canonical_limbs(&value.limbs) {
            return Err(SerializationError::InvalidData(alloc::format!(
                "value {} >= modulus {}",
                value,
                Self::modulus_biguint()
            )));
        }
        Ok((Self::from_canonical(&value), consumed))
    }
}

impl<P, const LIMBS: usize> grid_serialize::Valid for LargePrimeField<P, LIMBS>
where
    P: LargePrimeProfile<LIMBS>,
{
    fn is_valid(&self) -> bool {
        Self::is_canonical_limbs(&self.limbs)
    }
}

impl<P, const LIMBS: usize> grid_std::UniformRand for LargePrimeField<P, LIMBS>
where
    P: LargePrimeProfile<LIMBS>,
{
    fn rand<R: grid_std::rand::RngExt + ?Sized>(rng: &mut R) -> Self {
        let modulus = Self::modulus_biguint();
        let modulus_bits = modulus.bits() as usize;
        let highest_limb = (modulus_bits - 1) / 64;
        let top_bits = modulus_bits % 64;
        let top_mask = if top_bits == 0 {
            u64::MAX
        } else {
            (1u64 << top_bits) - 1
        };
        loop {
            let mut limbs = [0u64; LIMBS];
            for (i, limb) in limbs.iter_mut().enumerate() {
                if i > highest_limb {
                    *limb = 0;
                    continue;
                }
                let sample: u64 = rng.random();
                *limb = if i == highest_limb {
                    sample & top_mask
                } else {
                    sample
                };
            }
            if Self::is_canonical_limbs(&limbs) {
                return Self::from_canonical(&Self::canonical_biguint(limbs));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Bn254Fr;
    use super::LargePrimeField;
    use crate::arith::LargeCanonicalRing;
    use crate::arith::bigint::BigUint;
    use crate::arith::large_modulus::LargePrimeProfile;
    use crate::arith::large_prime_profiles::{
        Bls12_381FqProfile, Bls12_381FrProfile, Bn254FqProfile, Bn254FrProfile,
    };
    use crate::arith::ring::{self, Field, IntegerRing, Ring};
    use grid_serialize::{CanonicalDeserialize, CanonicalSerialize, Valid};
    use grid_std::rand::RngExt;

    fn naive_reduce<const LIMBS: usize>(
        mut value: BigUint<LIMBS>,
        modulus: &BigUint<LIMBS>,
    ) -> BigUint<LIMBS> {
        while value >= *modulus {
            let (next, borrow) = value.sub_with_borrow(modulus);
            assert!(!borrow);
            value = next;
        }
        value
    }

    fn exercise_large_prime_profile<P, const LIMBS: usize>()
    where
        P: LargePrimeProfile<LIMBS>,
        LargePrimeField<P, LIMBS>: Field
            + LargeCanonicalRing<Canonical = BigUint<LIMBS>>
            + CanonicalSerialize
            + CanonicalDeserialize
            + Valid,
    {
        type F<P0, const L0: usize> = LargePrimeField<P0, L0>;

        ring::tests::test_ring_axioms(
            F::<P, LIMBS>::from_u64(3),
            F::<P, LIMBS>::from_u64(5),
            F::<P, LIMBS>::from_u64(7),
        );
        ring::tests::test_integer_ring::<F<P, LIMBS>>(19);
        ring::tests::test_field_axioms(F::<P, LIMBS>::from_u64(7));

        let a = F::<P, LIMBS>::from_u64(1234);
        let b = F::<P, LIMBS>::from_u64(5678);
        assert_eq!((a + b).to_canonical(), BigUint::<LIMBS>::from_u64(6912));
        assert_eq!(
            (a * b).to_canonical(),
            BigUint::<LIMBS>::from_u128(1234u128 * 5678u128)
        );

        let modulus = F::<P, LIMBS>::modulus_canonical();
        let (modulus_minus_one, borrow) = modulus.sub_small(1);
        assert!(!borrow);
        let (modulus_plus_five, carry) = modulus.add_small(5);
        assert!(!carry);
        let max = BigUint {
            limbs: [u64::MAX; LIMBS],
        };
        let reduced_max = F::<P, LIMBS>::from_canonical(&max).to_canonical();
        assert_eq!(reduced_max, naive_reduce(max, &modulus));

        let mut rng = grid_std::test_rng();
        for _ in 0..256 {
            let mut limbs = [0u64; LIMBS];
            for limb in &mut limbs {
                *limb = rng.random();
            }
            let value = BigUint { limbs };
            let reduced = F::<P, LIMBS>::from_canonical(&value).to_canonical();
            assert_eq!(reduced, naive_reduce(value, &modulus));
        }

        let minus_one = F::<P, LIMBS>::from_canonical(&modulus_minus_one);
        assert_eq!(
            (minus_one + F::<P, LIMBS>::from_u64(2)).to_canonical(),
            BigUint::<LIMBS>::one()
        );
        assert_eq!(
            (minus_one * minus_one).to_canonical(),
            BigUint::<LIMBS>::one()
        );
        assert_eq!(
            F::<P, LIMBS>::from_canonical(&modulus_plus_five).to_canonical(),
            BigUint::<LIMBS>::from_u64(5)
        );

        let seven = F::<P, LIMBS>::from_u64(7);
        assert_eq!(
            (seven * seven.inv()).to_canonical(),
            BigUint::<LIMBS>::one()
        );
        assert_eq!(
            seven.inv(),
            seven.pow_bigint(&F::<P, LIMBS>::modulus_minus_two())
        );

        let bytes = minus_one.serialize().unwrap();
        assert_eq!(bytes.len(), LIMBS * 8);
        let (decoded, consumed) = F::<P, LIMBS>::deserialize(&bytes).unwrap();
        assert_eq!(consumed, LIMBS * 8);
        assert_eq!(decoded, minus_one);
        assert!(decoded.is_valid());

        let modulus_bytes = modulus.serialize().unwrap();
        assert!(F::<P, LIMBS>::deserialize(&modulus_bytes).is_err());
    }

    #[test]
    fn test_bn254_fr_large_prime_backend() {
        exercise_large_prime_profile::<Bn254FrProfile, 4>();
    }

    #[test]
    fn test_bn254_fq_large_prime_backend() {
        exercise_large_prime_profile::<Bn254FqProfile, 4>();
    }

    #[test]
    fn test_bls12_381_fr_large_prime_backend() {
        exercise_large_prime_profile::<Bls12_381FrProfile, 4>();
    }

    #[test]
    fn test_bls12_381_fq_large_prime_backend() {
        exercise_large_prime_profile::<Bls12_381FqProfile, 6>();
    }

    #[test]
    fn test_large_prime_slice_hooks_match_scalar_ops() {
        let lhs = [
            Bn254Fr::from_u64(1),
            Bn254Fr::from_u64(7),
            Bn254Fr::from_u64(11),
            Bn254Fr::from_u64(19),
            Bn254Fr::from_u64(23),
        ];
        let rhs = [
            Bn254Fr::from_u64(2),
            Bn254Fr::from_u64(9),
            Bn254Fr::from_u64(13),
            Bn254Fr::from_u64(17),
            Bn254Fr::from_u64(29),
        ];
        let scalar = Bn254Fr::from_u64(31);

        let mut add_hook = lhs;
        let mut sub_hook = lhs;
        let mut scaled_add_hook = lhs;
        let add_scalar = core::array::from_fn::<Bn254Fr, 5, _>(|i| lhs[i] + rhs[i]);
        let sub_scalar = core::array::from_fn::<Bn254Fr, 5, _>(|i| lhs[i] - rhs[i]);
        let scaled_add_scalar = core::array::from_fn::<Bn254Fr, 5, _>(|i| lhs[i] + rhs[i] * scalar);

        Bn254Fr::add_assign_slice(&mut add_hook, &rhs);
        Bn254Fr::sub_assign_slice(&mut sub_hook, &rhs);
        Bn254Fr::add_assign_scaled_slice(&mut scaled_add_hook, &rhs, &scalar);

        assert_eq!(add_hook, add_scalar);
        assert_eq!(sub_hook, sub_scalar);
        assert_eq!(scaled_add_hook, scaled_add_scalar);
    }
}
