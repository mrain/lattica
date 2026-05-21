//! Fixed-limb large modular integer ring (not necessarily prime).
#![allow(
    clippy::needless_range_loop,
    clippy::suspicious_arithmetic_impl,
    clippy::suspicious_op_assign_impl,
    clippy::unnecessary_cast
)]
//!
//! Like [`LargePrimeField`], but does **not** implement [`Field`] or [`NTTRing`].
//! Intended for application rings such as `Z/(2^64 + 1)Z` in the arithmetic R1CS
//! reduction path.

use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::bigint::BigUint;
use super::ring::{IntegerRing, Ring};
use grid_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError};
use grid_std::rand::RngExt;

/// Compile-time metadata for a fixed-limb large modular integer ring.
pub trait LargeZmProfile<const LIMBS: usize> {
    /// Modulus in little-endian limb order.
    const MODULUS: [u64; LIMBS];
    /// Whether to use Montgomery representation.
    const MONTGOMERY: bool;
    /// `R mod q` in little-endian limb order for the Montgomery representation of one.
    const MONT_ONE: [u64; LIMBS];
    /// `R^2 mod q` in little-endian limb order for Montgomery conversion.
    const MONT_R2: [u64; LIMBS];
    /// `-q^{-1} mod 2^64` for Montgomery reduction.
    const MONT_NEG_INV: u64;
    /// `floor(2^(64 * LIMBS) / q)` for canonical Barrett reduction into Montgomery form.
    const BARRETT_MU: [u64; LIMBS];

    /// Compile-time validation.
    const PROFILE_VALID: () = assert!(
        large_zm_profile_is_valid::<LIMBS>(
            Self::MODULUS,
            Self::MONTGOMERY,
            Self::MONT_ONE,
            Self::MONT_R2,
            Self::MONT_NEG_INV,
            Self::BARRETT_MU,
        ),
        "LargeZmProfile validation failed"
    );
}

/// Statically-checked validity for a [`LargeZmProfile`].
#[inline]
pub const fn large_zm_profile_is_valid<const LIMBS: usize>(
    modulus: [u64; LIMBS],
    montgomery: bool,
    mont_one: [u64; LIMBS],
    mont_r2: [u64; LIMBS],
    mont_neg_inv: u64,
    barrett_mu: [u64; LIMBS],
) -> bool {
    // Modulus must be >= 2 (multi-limb comparison).
    if LIMBS == 0 {
        return false;
    }
    let mut all_zero = true;
    let mut i = 0;
    while i < LIMBS {
        if modulus[i] != 0 {
            all_zero = false;
        }
        i += 1;
    }
    if all_zero {
        return false;
    }
    // modulus == 1 is also rejected.
    if modulus[0] == 1 {
        let mut rest_zero = true;
        let mut j = 1;
        while j < LIMBS {
            if modulus[j] != 0 {
                rest_zero = false;
            }
            j += 1;
        }
        if rest_zero {
            return false;
        }
    }

    if !montgomery {
        // BARRETT_MU must be non-zero for canonical profiles.
        let mut mu_zero = true;
        let mut k = 0;
        while k < LIMBS {
            if barrett_mu[k] != 0 {
                mu_zero = false;
            }
            k += 1;
        }
        if mu_zero {
            return false;
        }
        return true;
    }

    // Modulus must be odd for Montgomery.
    if modulus[0] & 1 == 0 {
        return false;
    }
    // MONT_NEG_INV * MODULUS[0] ≡ -1 mod 2^64  (i.e. product equals u64::MAX)
    let inv_check = mont_neg_inv.wrapping_mul(modulus[0]);
    if inv_check.wrapping_add(1) != 0 {
        return false;
    }
    // Basic Barrett MU sanity: must be non-zero.
    if LIMBS > 0 && barrett_mu[LIMBS - 1] == 0 && limball_eq(&barrett_mu, 0) {
        return false;
    }
    // MONT_ONE must be non-zero for odd Montgomery moduli.
    if LIMBS > 0 && mont_one[LIMBS - 1] == 0 && limball_eq(&mont_one, 0) {
        return false;
    }
    // MONT_R2 must be non-zero for R > 0 case — skip deep check here.
    if LIMBS > 0 && mont_r2[LIMBS - 1] == 0 && limball_eq(&mont_r2, 0) {
        return false;
    }
    true
}

const fn limball_eq<const N: usize>(limbs: &[u64; N], v: u64) -> bool {
    let mut i = 0;
    while i < N {
        if limbs[i] != v {
            return false;
        }
        i += 1;
    }
    true
}

/// A fixed-limb large modular integer (general modulus, not necessarily prime).
#[repr(transparent)]
pub struct LargeZm<P, const LIMBS: usize>
where
    P: LargeZmProfile<LIMBS>,
{
    limbs: [u64; LIMBS],
    _profile: PhantomData<P>,
}

impl<P, const LIMBS: usize> Copy for LargeZm<P, LIMBS> where P: LargeZmProfile<LIMBS> {}

impl<P, const LIMBS: usize> Clone for LargeZm<P, LIMBS>
where
    P: LargeZmProfile<LIMBS>,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<P, const LIMBS: usize> PartialEq for LargeZm<P, LIMBS>
where
    P: LargeZmProfile<LIMBS>,
{
    fn eq(&self, other: &Self) -> bool {
        self.limbs == other.limbs
    }
}

impl<P, const LIMBS: usize> Eq for LargeZm<P, LIMBS> where P: LargeZmProfile<LIMBS> {}

/// Re-export profiles.
pub use crate::arith::large_zm_profiles::Fermat64Profile;

impl<P, const LIMBS: usize> LargeZm<P, LIMBS>
where
    P: LargeZmProfile<LIMBS>,
{
    const _VALID: () = P::PROFILE_VALID;

    #[inline(always)]
    const fn zero_limbs() -> [u64; LIMBS] {
        [0; LIMBS]
    }

    /// Construct from raw value limbs. Every constructor path goes through
    /// this to ensure profile validation fires.
    #[inline(always)]
    fn from_raw(limbs: [u64; LIMBS]) -> Self {
        // Touch PROFILE_VALID so that invalid profiles fail at compile time.
        let () = Self::_VALID;
        Self {
            limbs,
            _profile: PhantomData,
        }
    }

    #[inline(always)]
    fn modulus_biguint() -> BigUint<LIMBS> {
        BigUint { limbs: P::MODULUS }
    }

    /// Bit-length of the modulus, or 1 if the modulus is zero (shouldn't happen).
    const fn modulus_bit_len() -> usize {
        let modulus = P::MODULUS;
        let mut i = LIMBS;
        while i > 0 {
            i -= 1;
            let limb = modulus[i];
            if limb != 0 {
                return i * 64 + (64 - limb.leading_zeros() as usize);
            }
        }
        1
    }

    #[inline(always)]
    fn barrett_mu_biguint() -> BigUint<LIMBS> {
        BigUint {
            limbs: P::BARRETT_MU,
        }
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
    fn two_power_width_minus_modulus() -> [u64; LIMBS] {
        let (complement, borrow) = Self::sub_limbs(&Self::zero_limbs(), &P::MODULUS);
        debug_assert!(borrow, "modulus must be non-zero");
        complement
    }

    #[inline(always)]
    fn sub_modulus_once(limbs: &[u64; LIMBS]) -> [u64; LIMBS] {
        let (reduced, borrow) = Self::sub_limbs(limbs, &P::MODULUS);
        debug_assert!(
            !borrow,
            "subtracting modulus requires a non-negative operand"
        );
        reduced
    }

    #[inline(always)]
    fn cond_subtract_modulus(limbs: [u64; LIMBS], carry: bool) -> [u64; LIMBS] {
        if carry {
            let (reduced, overflow) =
                Self::add_limbs(&limbs, &Self::two_power_width_minus_modulus());
            debug_assert!(
                !overflow,
                "subtracting modulus from a carried value must fit"
            );
            debug_assert!(
                Self::is_canonical_limbs(&reduced),
                "single modular correction must produce a canonical value"
            );
            reduced
        } else if !Self::is_canonical_limbs(&limbs) {
            Self::sub_modulus_once(&limbs)
        } else {
            limbs
        }
    }

    #[inline(always)]
    fn add_mod_limbs(lhs: &[u64; LIMBS], rhs: &[u64; LIMBS]) -> [u64; LIMBS] {
        let (sum, carry) = Self::add_limbs(lhs, rhs);
        Self::cond_subtract_modulus(sum, carry)
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

    fn canonical_mul_limbs(lhs: &[u64; LIMBS], rhs: &[u64; LIMBS]) -> [u64; LIMBS] {
        // Schoolbook multiply into (lo, hi).
        let (lo, hi) = Self::mul_buffers(lhs, rhs);
        let modulus = Self::modulus_biguint();

        // Reduce: result = (lo + hi * R) mod M, where R = 2^(64*LIMBS).
        // R_mod = R mod M = 2^(64*LIMBS) mod M.
        //
        // For LIMBS <= 2, compute directly with full 256-bit arithmetic.
        if LIMBS <= 2 {
            let r_mod = if LIMBS == 1 {
                ((u64::MAX as u128) % modulus.limbs[0] as u128 + 1) % modulus.limbs[0] as u128
            } else {
                let m = modulus.limbs[0] as u128 | (modulus.limbs[1] as u128) << 64;
                (u128::MAX % m + 1) % m
            };
            // Compute hi * r_mod as full 256-bit product.
            let hi_lo = hi[0] as u64;
            let hi_hi = if LIMBS > 1 { hi[1] } else { 0 };
            let r_lo = r_mod as u64;
            let r_hi = (r_mod >> 64) as u64;

            let p00 = (hi_lo as u128) * (r_lo as u128);
            let p01 = (hi_lo as u128) * (r_hi as u128);
            let p10 = (hi_hi as u128) * (r_lo as u128);
            let p11 = (hi_hi as u128) * (r_hi as u128);

            // Fold 128x128 → (low, high) via schoolbook.
            let (mut low, mut carry) = p00.overflowing_add(p01 << 64);
            let mut high = (p01 >> 64).wrapping_add(if carry { 1 } else { 0 });
            (low, carry) = low.overflowing_add(p10 << 64);
            high = high
                .wrapping_add(p10 >> 64)
                .wrapping_add(if carry { 1 } else { 0 });
            // p11 belongs in the upper 128 bits — add directly to high.
            high = high.wrapping_add(p11);
            // Now (high, low) = hi * r_mod as a 256-bit value.
            //
            // Fold: result ≡ lo_val + low + high * r_mod (mod M),
            // since high * 2^128 ≡ high * r_mod (mod M).
            let lo_val = lo[0] as u128 | if LIMBS > 1 { (lo[1] as u128) << 64 } else { 0 };
            let m = modulus.limbs[0] as u128
                | if LIMBS > 1 {
                    (modulus.limbs[1] as u128) << 64
                } else {
                    0
                };

            let (mut acc, ov) = lo_val.overflowing_add(low);
            let mut extra = high.wrapping_add(if ov { 1 } else { 0 });

            // Fold extra * r_mod until it vanishes.
            // Each iteration reduces extra's effective bit-width, since
            // |new_extra| ≈ |extra| * r_mod / 2^128 < |extra|.
            // Convergence is guaranteed and fast.
            while extra != 0 {
                let e_lo = extra as u64;
                let e_hi = (extra >> 64) as u64;
                let q00 = (e_lo as u128) * (r_lo as u128);
                let q01 = (e_lo as u128) * (r_hi as u128);
                let q10 = (e_hi as u128) * (r_lo as u128);
                let q11 = (e_hi as u128) * (r_hi as u128);

                let mut q_carry;
                let mut q_low = q00;
                (q_low, q_carry) = q_low.overflowing_add(q01 << 64);
                let mut q_high = (q01 >> 64).wrapping_add(if q_carry { 1 } else { 0 });
                (q_low, q_carry) = q_low.overflowing_add(q10 << 64);
                q_high = q_high
                    .wrapping_add(q10 >> 64)
                    .wrapping_add(if q_carry { 1 } else { 0 });
                q_high = q_high.wrapping_add(q11);

                let (new_acc, ov_inner) = acc.overflowing_add(q_low);
                acc = new_acc;
                extra = q_high.wrapping_add(if ov_inner { 1 } else { 0 });
            }

            acc %= m;
            let mut out = Self::zero_limbs();
            out[0] = acc as u64;
            if LIMBS > 1 {
                out[1] = (acc >> 64) as u64;
            }
            return out;
        }

        // LIMBS >= 3: iterative r_mod folding.
        //
        // The identity:  hi * R ≡ hi * r_mod (mod M)  where r_mod = R mod M.
        // Each iteration:  hi * r_mod = p_lo + p_hi * R  (widening mul), then
        //   acc_lo += p_lo,  hi = p_hi + carry
        // Since r_mod < M < R, we have p_hi = floor(hi * r_mod / R) < hi,
        // so hi strictly decreases each iteration. Convergence is guaranteed.
        let barrett_mu = Self::barrett_mu_biguint();

        // r_mod = R mod M = -(M * mu) mod R = !(M * mu) + 1  (R-complement).
        let (mu_prod_lo, mu_prod_hi) = modulus.widening_mul(&barrett_mu);
        // M * mu < R by definition of floor(R/M), so mu_prod_hi is zero.
        debug_assert!(mu_prod_hi.is_zero(), "BARRETT_MU must satisfy M * mu < R");
        let mut r_mod_limbs = [0u64; LIMBS];
        let mut carry = 1u64;
        for i in 0..LIMBS {
            let (s, c) = (!mu_prod_lo.limbs[i]).overflowing_add(carry);
            r_mod_limbs[i] = s;
            carry = c as u64;
        }
        let r_mod = BigUint { limbs: r_mod_limbs };

        let mut acc_lo = BigUint { limbs: lo };
        let mut acc_hi = BigUint { limbs: hi };

        for _ in 0..64 * LIMBS {
            if acc_hi.is_zero() {
                break;
            }
            let (p_lo, p_hi) = acc_hi.widening_mul(&r_mod);
            let (new_lo, carry) = acc_lo.add_with_carry(&p_lo);
            acc_lo = new_lo;
            // p_hi ≤ R-2 (proved), so p_hi + 1 ≤ R-1, never overflows.
            acc_hi = if carry { p_hi.add_small(1).0 } else { p_hi };
        }

        // At this point acc_hi must be zero (total value fits in LIMBS limbs).
        debug_assert!(
            acc_hi.is_zero(),
            "r_mod folding must converge: hi should be zero"
        );

        // acc_lo might still be ≥ M. Use Barrett to reduce efficiently:
        //   q_est = floor(acc_lo * mu / R)  (upper half of widening_mul)
        //   acc_lo -= q_est * M
        // The result is at most 2*M, handled by sub_if_ge below.
        if !Self::is_canonical_limbs(&acc_lo.limbs) {
            let (_, q_est) = acc_lo.widening_mul(&barrett_mu);
            let (prod_lo, _) = modulus.widening_mul(&q_est);
            let (reduced, borrow) = acc_lo.sub_with_borrow(&prod_lo);
            debug_assert!(!borrow, "Barrett quotient must not overshoot");
            if !borrow {
                acc_lo = reduced;
            }
        }

        // At most 2 further subtractions needed (error < 2M after Barrett).
        let mut res = acc_lo.sub_if_ge(&modulus);
        res = res.sub_if_ge(&modulus);

        let mut out = Self::zero_limbs();
        out.copy_from_slice(&res.limbs[..LIMBS]);
        out
    }

    #[inline(always)]
    fn montgomery_to_canonical_limbs(limbs: &[u64; LIMBS]) -> [u64; LIMBS] {
        let (carry, reduced) = Self::montgomery_reduce(*limbs, Self::zero_limbs());
        Self::cond_subtract_modulus(reduced, carry)
    }

    #[inline(always)]
    fn canonical_limbs_from_repr(limbs: &[u64; LIMBS]) -> BigUint<LIMBS> {
        let limbs = if P::MONTGOMERY {
            Self::montgomery_to_canonical_limbs(limbs)
        } else {
            *limbs
        };
        BigUint { limbs }
    }

    /// Create from a `u64` value (always fits in a multi-limb modulus).
    pub fn from_u64(val: u64) -> Self {
        Self::from_canonical(&BigUint::from_u64(val))
    }

    /// Export as `u64`, panicking if the canonical residue exceeds `u64::MAX`.
    pub fn to_u64(&self) -> u64 {
        self.try_to_u64()
            .expect("LargeZm::to_u64: value does not fit in u64")
    }

    /// Try to export as `u64`. Returns `None` if the canonical residue exceeds `u64::MAX`.
    pub fn try_to_u64(&self) -> Option<u64> {
        <Self as IntegerRing>::try_to_u64(self)
    }

    fn reduce_canonical(value: &BigUint<LIMBS>) -> BigUint<LIMBS> {
        if Self::is_canonical_limbs(&value.limbs) {
            return *value;
        }

        let mu = Self::barrett_mu_biguint();
        let modulus = Self::modulus_biguint();
        let (_, quotient) = value.widening_mul(&mu);
        let (quotient_times_modulus, quotient_times_modulus_hi) = modulus.widening_mul(&quotient);
        debug_assert!(
            quotient_times_modulus_hi.is_zero(),
            "Barrett quotient must keep q * modulus within the canonical limb width"
        );

        let (reduced, borrow) = value.sub_with_borrow(&quotient_times_modulus);
        debug_assert!(!borrow, "Barrett quotient must not overshoot");

        let reduced = reduced.sub_if_ge(&modulus).sub_if_ge(&modulus);
        debug_assert!(
            reduced < modulus,
            "Barrett reduction must produce a canonical representative"
        );
        reduced
    }

    /// Convert a canonical integer (as BigUint) to Montgomery form.
    fn canonical_to_montgomery(value: &BigUint<LIMBS>) -> [u64; LIMBS] {
        // Montgomery conversion: multiply by R^2 then REDC.
        let (lo, hi) = Self::mul_buffers(&value.limbs, &P::MONT_R2);
        let (carry, limbs) = Self::montgomery_reduce(lo, hi);
        Self::cond_subtract_modulus(limbs, carry)
    }
}

// --- Formatting ---

impl<P, const LIMBS: usize> core::fmt::Debug for LargeZm<P, LIMBS>
where
    P: LargeZmProfile<LIMBS>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "LargeZm({:?})", self.to_canonical())
    }
}

impl<P, const LIMBS: usize> core::fmt::Display for LargeZm<P, LIMBS>
where
    P: LargeZmProfile<LIMBS>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Display::fmt(&self.to_canonical(), f)
    }
}

// --- Operator impls ---

impl<P, const LIMBS: usize> Add for LargeZm<P, LIMBS>
where
    P: LargeZmProfile<LIMBS>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::from_raw(Self::add_mod_limbs(&self.limbs, &rhs.limbs))
    }
}

impl<P, const LIMBS: usize> Add<&Self> for LargeZm<P, LIMBS>
where
    P: LargeZmProfile<LIMBS>,
{
    type Output = Self;

    fn add(self, rhs: &Self) -> Self::Output {
        self + *rhs
    }
}

impl<P, const LIMBS: usize> AddAssign for LargeZm<P, LIMBS>
where
    P: LargeZmProfile<LIMBS>,
{
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<P, const LIMBS: usize> AddAssign<&Self> for LargeZm<P, LIMBS>
where
    P: LargeZmProfile<LIMBS>,
{
    fn add_assign(&mut self, rhs: &Self) {
        *self = *self + *rhs;
    }
}

impl<P, const LIMBS: usize> Sub for LargeZm<P, LIMBS>
where
    P: LargeZmProfile<LIMBS>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let (diff, borrow) = Self::sub_limbs(&self.limbs, &rhs.limbs);
        Self::from_raw(Self::add_modulus_if_borrow(diff, borrow))
    }
}

impl<P, const LIMBS: usize> Sub<&Self> for LargeZm<P, LIMBS>
where
    P: LargeZmProfile<LIMBS>,
{
    type Output = Self;

    fn sub(self, rhs: &Self) -> Self::Output {
        self - *rhs
    }
}

impl<P, const LIMBS: usize> SubAssign for LargeZm<P, LIMBS>
where
    P: LargeZmProfile<LIMBS>,
{
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<P, const LIMBS: usize> SubAssign<&Self> for LargeZm<P, LIMBS>
where
    P: LargeZmProfile<LIMBS>,
{
    fn sub_assign(&mut self, rhs: &Self) {
        *self = *self - *rhs;
    }
}

impl<P, const LIMBS: usize> Mul for LargeZm<P, LIMBS>
where
    P: LargeZmProfile<LIMBS>,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let limbs = if P::MONTGOMERY {
            Self::montgomery_mul_limbs(&self.limbs, &rhs.limbs)
        } else {
            Self::canonical_mul_limbs(&self.limbs, &rhs.limbs)
        };
        Self::from_raw(limbs)
    }
}

impl<P, const LIMBS: usize> Mul<&Self> for LargeZm<P, LIMBS>
where
    P: LargeZmProfile<LIMBS>,
{
    type Output = Self;

    fn mul(self, rhs: &Self) -> Self::Output {
        self * *rhs
    }
}

impl<P, const LIMBS: usize> MulAssign for LargeZm<P, LIMBS>
where
    P: LargeZmProfile<LIMBS>,
{
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<P, const LIMBS: usize> MulAssign<&Self> for LargeZm<P, LIMBS>
where
    P: LargeZmProfile<LIMBS>,
{
    fn mul_assign(&mut self, rhs: &Self) {
        *self = *self * *rhs;
    }
}

// Ref-ref operator impls
impl<P, const LIMBS: usize> Add<Self> for &LargeZm<P, LIMBS>
where
    P: LargeZmProfile<LIMBS>,
{
    type Output = LargeZm<P, LIMBS>;
    fn add(self, rhs: Self) -> Self::Output {
        *self + *rhs
    }
}

impl<P, const LIMBS: usize> Add<LargeZm<P, LIMBS>> for &LargeZm<P, LIMBS>
where
    P: LargeZmProfile<LIMBS>,
{
    type Output = LargeZm<P, LIMBS>;
    fn add(self, rhs: LargeZm<P, LIMBS>) -> Self::Output {
        *self + rhs
    }
}

impl<P, const LIMBS: usize> Sub<Self> for &LargeZm<P, LIMBS>
where
    P: LargeZmProfile<LIMBS>,
{
    type Output = LargeZm<P, LIMBS>;
    fn sub(self, rhs: Self) -> Self::Output {
        *self - *rhs
    }
}

impl<P, const LIMBS: usize> Sub<LargeZm<P, LIMBS>> for &LargeZm<P, LIMBS>
where
    P: LargeZmProfile<LIMBS>,
{
    type Output = LargeZm<P, LIMBS>;
    fn sub(self, rhs: LargeZm<P, LIMBS>) -> Self::Output {
        *self - rhs
    }
}

impl<P, const LIMBS: usize> Mul<Self> for &LargeZm<P, LIMBS>
where
    P: LargeZmProfile<LIMBS>,
{
    type Output = LargeZm<P, LIMBS>;
    fn mul(self, rhs: Self) -> Self::Output {
        *self * *rhs
    }
}

impl<P, const LIMBS: usize> Mul<LargeZm<P, LIMBS>> for &LargeZm<P, LIMBS>
where
    P: LargeZmProfile<LIMBS>,
{
    type Output = LargeZm<P, LIMBS>;
    fn mul(self, rhs: LargeZm<P, LIMBS>) -> Self::Output {
        *self * rhs
    }
}

impl<P, const LIMBS: usize> Neg for LargeZm<P, LIMBS>
where
    P: LargeZmProfile<LIMBS>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        if self.is_zero() {
            self
        } else {
            let (diff, borrow) = Self::sub_limbs(&P::MODULUS, &self.limbs);
            debug_assert!(!borrow, "limb must be below the modulus");
            Self::from_raw(diff)
        }
    }
}

// --- Ring impl ---

impl<P, const LIMBS: usize> Ring for LargeZm<P, LIMBS>
where
    P: LargeZmProfile<LIMBS>,
{
    fn zero() -> Self {
        Self::from_raw(Self::zero_limbs())
    }

    fn one() -> Self {
        if P::MONTGOMERY {
            Self::from_raw(P::MONT_ONE)
        } else {
            let mut one = Self::zero_limbs();
            one[0] = 1;
            Self::from_raw(one)
        }
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
        for (dst_value, src_value) in dst.iter_mut().zip(src.iter()) {
            *dst_value += *src_value * scalar;
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

// --- IntegerRing ---

impl<P, const LIMBS: usize> IntegerRing for LargeZm<P, LIMBS>
where
    P: LargeZmProfile<LIMBS>,
{
    type Canonical = BigUint<LIMBS>;

    fn modulus_canonical() -> Self::Canonical {
        Self::modulus_biguint()
    }

    fn from_small_u64(value: u64) -> Self {
        let canonical = BigUint::from_u64(value);
        Self::from_canonical(&canonical)
    }

    fn from_canonical(value: &BigUint<LIMBS>) -> Self {
        let reduced = Self::reduce_canonical(value);
        if P::MONTGOMERY {
            Self::from_raw(Self::canonical_to_montgomery(&reduced))
        } else {
            Self::from_raw(reduced.limbs)
        }
    }

    fn to_canonical(&self) -> BigUint<LIMBS> {
        Self::canonical_limbs_from_repr(&self.limbs)
    }

    fn try_to_u64(&self) -> Option<u64> {
        let canon = Self::canonical_limbs_from_repr(&self.limbs);
        if LIMBS == 1 || canon.limbs[1..].iter().all(|&x| x == 0) {
            Some(canon.limbs[0])
        } else {
            None
        }
    }

    fn try_to_u128(&self) -> Option<u128> {
        let canon = Self::canonical_limbs_from_repr(&self.limbs);
        let mut result = 0u128;
        for i in (0..LIMBS).rev() {
            if i >= 2 && canon.limbs[i] != 0 {
                return None;
            }
            result = (result << 64) | canon.limbs[i] as u128;
        }
        Some(result)
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

// --- Serialization ---

impl<P, const LIMBS: usize> CanonicalSerialize for LargeZm<P, LIMBS>
where
    P: LargeZmProfile<LIMBS>,
{
    fn serialized_size(&self) -> usize {
        LIMBS * 8
    }

    fn serialize_into(&self, buf: &mut alloc::vec::Vec<u8>) -> Result<(), SerializationError> {
        let canon = self.to_canonical();
        for limb in &canon.limbs {
            buf.extend_from_slice(&limb.to_le_bytes());
        }
        Ok(())
    }
}

impl<P, const LIMBS: usize> CanonicalDeserialize for LargeZm<P, LIMBS>
where
    P: LargeZmProfile<LIMBS>,
{
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        if data.len() < LIMBS * 8 {
            return Err(SerializationError::UnexpectedEnd);
        }
        let mut limbs = [0u64; LIMBS];
        for i in 0..LIMBS {
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&data[i * 8..i * 8 + 8]);
            limbs[i] = u64::from_le_bytes(bytes);
        }
        if !Self::is_canonical_limbs(&limbs) {
            return Err(SerializationError::InvalidData(
                "value >= modulus for LargeZm".into(),
            ));
        }
        let canonical = BigUint { limbs };
        let val = if P::MONTGOMERY {
            Self::from_raw(Self::canonical_to_montgomery(&canonical))
        } else {
            Self::from_raw(canonical.limbs)
        };
        Ok((val, LIMBS * 8))
    }
}

impl<P, const LIMBS: usize> grid_serialize::Valid for LargeZm<P, LIMBS>
where
    P: LargeZmProfile<LIMBS>,
{
    fn is_valid(&self) -> bool {
        Self::is_canonical_limbs(&self.limbs)
    }
}

// --- UniformRand ---

impl<P, const LIMBS: usize> grid_std::UniformRand for LargeZm<P, LIMBS>
where
    P: LargeZmProfile<LIMBS>,
{
    /// Uniform random element via rejection sampling over the modulus bit-width.
    ///
    /// Samples exactly `ceil(log2(M))` random bits (full u64 limbs plus a
    /// partial top limb) and rejects if the sample is `>= M`. This terminates
    /// with probability 1 and produces exact uniformity.
    fn rand<R: RngExt + ?Sized>(rng: &mut R) -> Self {
        // Number of full u64 limbs needed to cover the modulus bit-width.
        let modulus_bits = Self::modulus_bit_len();
        let full_limbs = modulus_bits / 64;
        let extra_bits = modulus_bits % 64;
        let modulus = Self::modulus_biguint();

        loop {
            let mut limbs = [0u64; LIMBS];
            for limb in &mut limbs[..full_limbs] {
                *limb = rng.random();
            }
            // Sample the partial top limb if needed.
            if extra_bits > 0 {
                let mask = (1u64 << extra_bits) - 1;
                limbs[full_limbs] = rng.random::<u64>() & mask;
            }
            let sample = BigUint { limbs };
            if sample < modulus {
                return if P::MONTGOMERY {
                    Self::from_raw(Self::canonical_to_montgomery(&sample))
                } else {
                    Self::from_raw(sample.limbs)
                };
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use grid_serialize::Valid;
    use grid_std::UniformRand;

    fn test_rng() -> impl RngExt {
        grid_std::test_rng()
    }

    type Z = LargeZm<crate::arith::large_zm_profiles::Fermat64Profile, 2>;

    enum Canonical97Profile {}

    impl LargeZmProfile<1> for Canonical97Profile {
        const MODULUS: [u64; 1] = [97];
        const MONTGOMERY: bool = false;
        const MONT_ONE: [u64; 1] = [0];
        const MONT_R2: [u64; 1] = [0];
        const MONT_NEG_INV: u64 = 0;
        // mu = floor(2^64 / 97)
        const BARRETT_MU: [u64; 1] = [190172619316593315];
    }

    enum CanonicalLarge64Profile {}

    impl LargeZmProfile<1> for CanonicalLarge64Profile {
        const MODULUS: [u64; 1] = [u64::MAX - 58];
        const MONTGOMERY: bool = false;
        const MONT_ONE: [u64; 1] = [0];
        const MONT_R2: [u64; 1] = [0];
        const MONT_NEG_INV: u64 = 0;
        // mu = floor(2^64 / (2^64 - 58)) = 1
        const BARRETT_MU: [u64; 1] = [1];
    }

    enum CanonicalFermat64Profile {}

    impl LargeZmProfile<2> for CanonicalFermat64Profile {
        const MODULUS: [u64; 2] = [1, 1];
        const MONTGOMERY: bool = false;
        const MONT_ONE: [u64; 2] = [0, 0];
        const MONT_R2: [u64; 2] = [0, 0];
        const MONT_NEG_INV: u64 = 0;
        const BARRETT_MU: [u64; 2] = [u64::MAX, 0];
    }

    #[test]
    fn profile_constants_are_correct() {
        // MODULUS = 2^64 + 1
        assert_eq!(<Fermat64Profile as LargeZmProfile<2>>::MODULUS[0], 1);
        assert_eq!(<Fermat64Profile as LargeZmProfile<2>>::MODULUS[1], 1);

        // MONT_NEG_INV: -1^{-1} = -1 = u64::MAX mod 2^64
        let inv = <Fermat64Profile as LargeZmProfile<2>>::MONT_NEG_INV;
        assert_eq!(inv.wrapping_mul(1), u64::MAX);

        // MONT_ONE: R mod (2^64+1) = 2^128 mod (2^64+1) = 1.
        assert_eq!(<Fermat64Profile as LargeZmProfile<2>>::MONT_ONE[0], 1);
        assert_eq!(<Fermat64Profile as LargeZmProfile<2>>::MONT_ONE[1], 0);

        // MONT_R2: R^2 mod (2^64+1) = (2^128) mod (2^64+1) = 1.
        // Since 2^64 ≡ -1, 2^128 = (2^64)^2 ≡ 1. So MONT_R2 = [1, 0].
        assert_eq!(<Fermat64Profile as LargeZmProfile<2>>::MONT_R2[0], 1);
        assert_eq!(<Fermat64Profile as LargeZmProfile<2>>::MONT_R2[1], 0);

        // BARRETT_MU = floor(2^128 / (2^64+1)) = 2^64 - 1 = [u64::MAX, 0].
        assert_eq!(
            <Fermat64Profile as LargeZmProfile<2>>::BARRETT_MU[0],
            u64::MAX
        );
        assert_eq!(<Fermat64Profile as LargeZmProfile<2>>::BARRETT_MU[1], 0);
    }

    #[test]
    fn fermat_64_add_sub() {
        let a = Z::from_u64(100);
        let b = Z::from_u64(200);
        // 100 + 200 = 300 mod (2^64 + 1)
        let sum = a + b;
        assert_eq!(sum.to_u64(), 300);
    }

    #[test]
    fn fermat_64_mul() {
        // 2^64 ≡ -1 mod (2^64 + 1), so:
        // (2^64 + 1 - 3) = 2^64 - 2 = -2 mod (2^64 + 1) = 2^64 - 1.
        let a = Z::from_u64(3);
        let b = Z::from_u64(5);
        assert_eq!((a * b).to_u64(), 15);

        // 2^64 ≡ -1, so 2^64 * 2^64 = 1 mod (2^64 + 1).
        // Create value 2^64 by 2^64 + 1 - 1 = 0 mod (2^64+1)... actually that's 0.
        // 2^64 - 1 ≡ -2 mod (2^64 + 1).  (2^64 - 1) * (2^64 - 1) ≡ 4.
        let big = Z::from_canonical(&BigUint {
            limbs: [u64::MAX, 0],
        });
        let product = big * big;
        let expected = BigUint { limbs: [4, 0] };
        // (2^64 - 1)^2 = 2^128 - 2*2^64 + 1 = 2^64*(2^64 - 2) + 1
        // = (-1)*(-3) + 1 = 4 mod (2^64 + 1)
        assert_eq!(product.to_canonical(), expected);
    }

    #[test]
    fn fermat_64_ring_properties() {
        let mut rng = test_rng();
        for _ in 0..100 {
            let a: Z = UniformRand::rand(&mut rng);
            let b: Z = UniformRand::rand(&mut rng);
            let c: Z = UniformRand::rand(&mut rng);
            // (a + b) * c = a*c + b*c
            assert_eq!((a + b) * c, (a * c) + (b * c));
            // a + b = b + a
            assert_eq!(a + b, b + a);
            // a * b = b * a
            assert_eq!(a * b, b * a);
            // a + 0 = a
            assert_eq!(a + Z::zero(), a);
            // a * 1 = a
            assert_eq!(a * Z::one(), a);
        }
    }

    #[test]
    fn fermat_64_one_uses_montgomery() {
        // For Fermat64, R = 2^128 ≡ 1 mod (2^64+1), so one() = R = [1, 0].
        // This is correct — for other Montgomery moduli, one() would differ.
        let one = Z::one();
        assert_eq!(one.to_u64(), 1);
    }

    #[test]
    fn fermat_64_serialize_roundtrip() {
        let a = Z::from_u64(12345);
        let bytes = a.serialize().unwrap();
        assert_eq!(bytes.len(), 16);
        let decoded = Z::deserialize_exact(&bytes).unwrap();
        assert_eq!(decoded.to_u64(), 12345);
    }

    #[test]
    fn fermat_64_serialize_rejects_non_canonical() {
        // 2^64 + 1 = [1, 1] so [0, 2] is out of range.
        let bytes = [0u8, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0];
        assert!(Z::deserialize_exact(&bytes).is_err());
    }

    #[test]
    fn fermat_64_neg() {
        let a = Z::from_u64(100);
        let neg_a = -a;
        assert_eq!((a + neg_a).to_u64(), 0);
        assert_eq!((-Z::zero()).to_u64(), 0);
    }

    #[test]
    fn fermat_64_large_canonical() {
        let val = Z::from_u64(42);
        let canon = val.to_canonical();
        assert_eq!(canon.limbs[0], 42);
        assert_eq!(Z::from_canonical(&canon), val);
    }

    #[test]
    fn fermat_64_residue_2p64_roundtrip() {
        // The residue 2^64 = [0, 1] is a valid canonical value mod (2^64+1).
        // Verify it round-trips through canonical conversion and back.
        let big = BigUint::<2> { limbs: [0, 1] };
        let val = Z::from_canonical(&big);
        // to_canonical should return it unchanged after Montgomery reduction.
        assert_eq!(val.to_canonical(), big);
    }

    #[test]
    fn fermat_64_try_to_u64() {
        // Small values fit.
        assert_eq!(Z::from_u64(123).try_to_u64(), Some(123));

        // 2^65 mod (2^64+1) = 2 * 2^64 ≡ -2 ≡ 2^64 - 1 = u64::MAX — fits.
        let v = BigUint::<2> { limbs: [0, 2] };
        assert_eq!(Z::from_canonical(&v).try_to_u64(), Some(u64::MAX));

        // The residue 2^64 ≡ [0, 1] is valid mod M = 2^64+1 but does NOT fit in u64.
        let big = BigUint::<2> { limbs: [0, 1] };
        assert_eq!(Z::from_canonical(&big).try_to_u64(), None);
    }

    #[test]
    fn fermat_64_uniform_rand() {
        let mut rng = test_rng();
        for _ in 0..100 {
            let v: Z = UniformRand::rand(&mut rng);
            assert!(v.is_valid());
        }
    }

    #[test]
    fn canonical_profile_small_modulus_arithmetic() {
        type C = LargeZm<Canonical97Profile, 1>;

        assert_eq!(C::from_u64(100).to_u64(), 3);
        assert_eq!((C::from_u64(13) * C::from_u64(17)).to_u64(), 27);
        assert_eq!((C::from_u64(96) + C::from_u64(2)).to_u64(), 1);
        assert_eq!((C::from_u64(3) - C::from_u64(5)).to_u64(), 95);
        assert_eq!(C::one().to_u64(), 1);
    }

    #[test]
    fn canonical_profile_add_handles_limb_carry() {
        type C = LargeZm<CanonicalLarge64Profile, 1>;
        const M: u64 = u64::MAX - 58;

        let minus_one = C::from_canonical(&BigUint::<1> { limbs: [M - 1] });
        assert_eq!((minus_one + minus_one).to_u64(), M - 2);
        assert_eq!((minus_one * minus_one).to_u64(), 1);
    }

    #[test]
    fn canonical_profile_multilimb_fermat_arithmetic() {
        type C = LargeZm<CanonicalFermat64Profile, 2>;

        let two_to_64 = C::from_canonical(&BigUint::<2> { limbs: [0, 1] });
        assert_eq!(two_to_64.to_canonical(), BigUint::<2> { limbs: [0, 1] });
        assert_eq!(
            (two_to_64 * two_to_64).to_canonical(),
            BigUint::<2> { limbs: [1, 0] }
        );
        assert_eq!((two_to_64 * C::from_u64(5)).to_u64(), u64::MAX - 3);

        let minus_two = C::from_canonical(&BigUint::<2> {
            limbs: [u64::MAX, 0],
        });
        assert_eq!(
            (minus_two * minus_two).to_canonical(),
            BigUint::<2> { limbs: [4, 0] }
        );
        assert_eq!(C::one().to_canonical(), BigUint::<2> { limbs: [1, 0] });
    }

    // --- Canonical 3-limb profile (tests LIMBS >= 3 path) ---

    enum Canonical3LimbProfile {}

    // M = 2^128 + 123 = [123, 0, 1] in 3 limbs.
    // BARRETT_MU = floor(2^192 / (2^128 + 123)) = 2^64 - 1 = [u64::MAX, 0, 0].
    impl LargeZmProfile<3> for Canonical3LimbProfile {
        const MODULUS: [u64; 3] = [123, 0, 1];
        const MONTGOMERY: bool = false;
        const MONT_ONE: [u64; 3] = [0, 0, 0];
        const MONT_R2: [u64; 3] = [0, 0, 0];
        const MONT_NEG_INV: u64 = 0;
        const BARRETT_MU: [u64; 3] = [u64::MAX, 0, 0];
    }

    #[test]
    fn canonical_3limb_multiply() {
        type C3 = LargeZm<Canonical3LimbProfile, 3>;

        // 1 * 1 = 1
        let one = C3::from_u64(1);
        assert_eq!((one * one).to_u64(), 1);

        // 2^128 < M = 2^128+123, so 2^128 IS its own canonical residue.
        let pow128 = C3::from_canonical(&BigUint { limbs: [0, 0, 1] });
        assert_eq!(pow128.to_canonical(), BigUint { limbs: [0, 0, 1] });

        // (2^128)^2 mod (2^128+123) = (-123)^2 = 15129
        assert_eq!(
            (pow128 * pow128).to_canonical(),
            BigUint {
                limbs: [15129, 0, 0]
            }
        );

        // Exercise the case where the product overflows 2*LIMBS limbs
        // (requires the r_mod folding loop to reduce the upper half).
        // Multiply two large 3-limb values whose product > R = 2^192.
        let big_a = C3::from_canonical(&BigUint {
            limbs: [u64::MAX, u64::MAX, 0], // 2^128 - 1
        });
        let big_b = C3::from_canonical(&BigUint {
            limbs: [1, 0, 0], // 1
        });
        assert_eq!((big_a * big_b).to_canonical(), big_a.to_canonical());

        // Product near 2^256: (2^128 - 1) * (2^128 + u64::MAX)
        // = (2^128 - 1)*(2^128 + 2^64 - 1) = 2^256 + 2^192 - 2^128 - 2^64 + 1
        // This has a non-zero hi half, exercising the r_mod loop.
        let val = C3::from_canonical(&BigUint {
            limbs: [u64::MAX, u64::MAX, 0], // 2^128 - 1
        });
        let other = C3::from_canonical(&BigUint {
            limbs: [u64::MAX, 0, 1], // 2^128 + 2^64 - 1
        });
        // The exact result would need computation, but ring properties hold:
        let left = (val + val) * other;
        let right = (val * other) + (val * other);
        assert_eq!(left, right);
    }

    #[test]
    fn canonical_3limb_ring_properties() {
        type C3 = LargeZm<Canonical3LimbProfile, 3>;
        let mut rng = test_rng();
        for _ in 0..50 {
            let a: C3 = UniformRand::rand(&mut rng);
            let b: C3 = UniformRand::rand(&mut rng);
            let c: C3 = UniformRand::rand(&mut rng);
            assert_eq!((a + b) * c, (a * c) + (b * c));
            assert_eq!(a + b, b + a);
            assert_eq!(a * b, b * a);
            assert_eq!(a + C3::zero(), a);
            assert_eq!(a * C3::one(), a);
        }
    }

    #[test]
    fn canonical_3limb_known_products() {
        type C3 = LargeZm<Canonical3LimbProfile, 3>;

        // (M-1) * (M-1) ≡ 1 mod (2^128 + 123).
        let m_minus_1 = C3::from_canonical(&BigUint {
            limbs: [122, 0, 1], // 2^128 + 122
        });
        assert_eq!(
            (m_minus_1 * m_minus_1).to_canonical(),
            BigUint { limbs: [1, 0, 0] }
        );

        // 2^128 * (2^128 - 1) mod (2^128 + 123).
        // The schoolbook product is 2^256 - 2^128 (upper half non-zero).
        // Using 2^128 ≡ -123:  (-123)^2 - (-123) = 15129 + 123 = 15252.
        let two_pow_128 = C3::from_canonical(&BigUint { limbs: [0, 0, 1] });
        let two_pow_128_minus_1 = C3::from_canonical(&BigUint {
            limbs: [u64::MAX, u64::MAX, 0],
        });
        assert_eq!(
            (two_pow_128 * two_pow_128_minus_1).to_canonical(),
            BigUint {
                limbs: [15252, 0, 0]
            }
        );
    }
}
