//! `PrimeModulus` — `Z_q` where `q` is an odd prime that fits in a single `u64`.
//!
//! Uses Montgomery multiplication for efficient modular arithmetic.
//! Implements [`Ring`], [`IntegerRing`], [`Field`], and [`NTTRing`].

use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::large_modulus::LargeCanonicalRing;
use super::ntt::{NTTRing, NttError, NttPlan, cached_ntt_plan};
use super::ring::{Field, IntegerRing, Ring};
use crate::simd::montgomery_prime;
use grid_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError};

/// Goldilocks prime: `2^64 - 2^32 + 1`.
pub const GOLDILOCKS_MODULUS: u64 = 0xffff_ffff_0000_0001;

/// Module sealing the [`PrimeFieldLimb`] trait so only the four primitive unsigned integers
/// can be used as limb types.
mod limb_sealed {
    pub trait Sealed {}
    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
}

/// Abstraction over the limb (word) type used as the Montgomery backend for [`PrimeField`].
///
/// Only implemented for `u8`, `u16`, `u32`, and `u64`.
pub trait PrimeFieldLimb:
    limb_sealed::Sealed
    + Copy
    + PartialEq
    + Eq
    + PartialOrd
    + Ord
    + core::fmt::Debug
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::AddAssign
    + core::ops::SubAssign
    + core::ops::MulAssign
    + Send
    + Sync
    + 'static
{
    /// The next-wider unsigned integer (`u16` for `u8`, `u32` for `u16`, `u64` for `u32`, `u128` for `u64`).
    type Wide: Copy + 'static;

    /// Number of bits in this limb type.
    const BITS: u32;
    const ZERO: Self;
    const ONE: Self;
    const MAX: Self;

    fn from_u64(v: u64) -> Self;

    fn to_u64(self) -> u64;

    fn to_wide(self) -> Self::Wide;

    fn from_wide_truncate(w: Self::Wide) -> Self;

    fn wrapping_add(self, rhs: Self) -> Self;

    fn wrapping_sub(self, rhs: Self) -> Self;

    fn wrapping_mul(self, rhs: Self) -> Self;

    fn overflowing_add(self, rhs: Self) -> (Self, bool);

    fn wide_add(a: Self::Wide, b: Self::Wide) -> Self::Wide;

    fn wide_mul(a: Self::Wide, b: Self::Wide) -> Self::Wide;

    fn wide_shr(w: Self::Wide, shift: u32) -> Self::Wide;

    fn wide_from_u64(v: u64) -> Self::Wide;

    fn wrapping_neg(self) -> Self;

    fn mod_limb(self, modulus: Self) -> Self;

    fn leading_zeros(self) -> u32;
}

macro_rules! impl_prime_field_limb {
    ($ty:ty, $wide:ty, $bits:expr) => {
        impl PrimeFieldLimb for $ty {
            type Wide = $wide;
            const BITS: u32 = $bits;
            const ZERO: Self = 0;
            const ONE: Self = 1;
            const MAX: Self = <$ty>::MAX;

            #[inline(always)]
            fn from_u64(v: u64) -> Self {
                v as $ty
            }
            #[inline(always)]
            fn to_u64(self) -> u64 {
                self as u64
            }
            #[inline(always)]
            fn to_wide(self) -> Self::Wide {
                self as $wide
            }
            #[inline(always)]
            fn from_wide_truncate(w: Self::Wide) -> Self {
                w as $ty
            }
            #[inline(always)]
            fn wrapping_add(self, rhs: Self) -> Self {
                <$ty>::wrapping_add(self, rhs)
            }
            #[inline(always)]
            fn wrapping_sub(self, rhs: Self) -> Self {
                <$ty>::wrapping_sub(self, rhs)
            }
            #[inline(always)]
            fn wrapping_mul(self, rhs: Self) -> Self {
                <$ty>::wrapping_mul(self, rhs)
            }
            #[inline(always)]
            fn overflowing_add(self, rhs: Self) -> (Self, bool) {
                <$ty>::overflowing_add(self, rhs)
            }
            #[inline(always)]
            fn wide_add(a: Self::Wide, b: Self::Wide) -> Self::Wide {
                a + b
            }
            #[inline(always)]
            fn wide_mul(a: Self::Wide, b: Self::Wide) -> Self::Wide {
                a * b
            }
            #[inline(always)]
            fn wide_shr(w: Self::Wide, shift: u32) -> Self::Wide {
                w >> shift
            }
            #[inline(always)]
            fn wide_from_u64(v: u64) -> Self::Wide {
                v as $wide
            }
            #[inline(always)]
            fn wrapping_neg(self) -> Self {
                self.wrapping_neg()
            }
            #[inline(always)]
            fn mod_limb(self, modulus: Self) -> Self {
                if self < modulus { self } else { self % modulus }
            }
            #[inline(always)]
            fn leading_zeros(self) -> u32 {
                <$ty>::leading_zeros(self)
            }
        }
    };
}

impl_prime_field_limb!(u64, u128, 64);
impl_prime_field_limb!(u32, u64, 32);
impl_prime_field_limb!(u16, u32, 16);
impl_prime_field_limb!(u8, u16, 8);

/// A modular integer in Montgomery form for a prime modulus.
///
/// The value is stored as `a * R mod q` where `R = 2^{L::BITS}`.
/// This allows multiplication without explicit division.
///
/// `Q` is the prime modulus (must be an odd prime that fits in the limb type).
/// `L` is the limb type (defaults to `u64`).
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PrimeField<const Q: u64, L: PrimeFieldLimb = u64> {
    /// Value in Montgomery form: `val = a * R mod Q` where `R = 2^{L::BITS}`.
    val: L,
}

impl<const Q: u64, L: PrimeFieldLimb> PrimeField<Q, L> {
    /// The validated modulus for this field.
    pub const MODULUS: u64 = {
        assert!(
            Q >= 3 && Q % 2 == 1 && is_prime_const(Q),
            "PrimeField modulus must be an odd prime"
        );
        // Verify Q fits in the limb type. For L=u64 (BITS=64), Q: u64 always fits.
        // For smaller limbs, check that Q < 2^BITS.
        if L::BITS < 64 {
            assert!(
                Q < (1u64 << L::BITS),
                "PrimeField modulus must fit in the limb type"
            );
        }
        Q
    };

    #[inline]
    fn modulus_limb() -> L {
        L::from_u64(Q)
    }

    /// Minimum bytes needed to represent values in [0, Q).
    const SERIALIZED_BYTES: usize = {
        let bits = 64 - (Q - 1).leading_zeros() as usize;
        bits.div_ceil(8)
    };

    /// R = 2^{L::BITS} mod Q (Montgomery constant), computed as u64.
    const R_MOD_Q_U64: u64 = {
        if L::BITS < 64 {
            (1u64 << L::BITS) % Q
        } else {
            (u64::MAX % Q + 1) % Q
        }
    };

    #[inline]
    fn r_mod_q_limb() -> L {
        L::from_u64(Self::R_MOD_Q_U64)
    }

    /// R^2 mod Q — used to convert to Montgomery form.
    const R2_MOD_Q_U64: u64 = {
        let r = Self::R_MOD_Q_U64 as u128;
        let r2 = (r * r) % (Q as u128);
        r2 as u64
    };

    /// Q_INV = -Q^{-1} mod 2^64 (Montgomery reduction constant).
    pub(crate) const Q_INV_U64: u64 = {
        let q = Q;
        let mut inv: u64 = 1;
        let mut i = 0;
        while i < 6 {
            inv = inv.wrapping_mul(2u64.wrapping_sub(q.wrapping_mul(inv)));
            i += 1;
        }
        inv.wrapping_neg()
    };

    #[inline]
    fn q_inv_limb() -> L {
        // For limb types smaller than u64, the inverse only needs the low BITS bits.
        let inv = if L::BITS < 64 {
            Self::Q_INV_U64 & ((1u64 << L::BITS).wrapping_sub(1))
        } else {
            Self::Q_INV_U64
        };
        L::from_u64(inv)
    }

    #[inline(always)]
    fn check_modulus() {
        let _ = Self::MODULUS;
    }

    #[inline(always)]
    pub(crate) fn add_raw_words(lhs: L, rhs: L) -> L {
        let q = Self::modulus_limb();
        // Fast path: if Q < 2^{BITS-1}, sum can't overflow the limb.
        if Q < (1u64 << (L::BITS - 1)) {
            let sum = lhs.wrapping_add(rhs);
            if sum >= q {
                return sum.wrapping_sub(q);
            }
            return sum;
        }

        let (sum, carry) = lhs.overflowing_add(rhs);
        if carry || sum >= q {
            sum.wrapping_sub(q)
        } else {
            sum
        }
    }

    #[inline(always)]
    pub(crate) fn sub_raw_words(lhs: L, rhs: L) -> L {
        let q = Self::modulus_limb();
        if lhs >= rhs {
            lhs - rhs
        } else {
            q.wrapping_sub(rhs).wrapping_add(lhs)
        }
    }

    /// Montgomery reduction: given `t < Q * R`, compute `t * R^{-1} mod Q`.
    #[inline(always)]
    fn montgomery_reduce(t: L::Wide) -> L {
        let q_limb = Self::modulus_limb();
        if Q >= (1u64 << (L::BITS - 1)) {
            return Self::montgomery_reduce_wide(t);
        }

        let m = L::from_wide_truncate(t).wrapping_mul(Self::q_inv_limb());
        let t_plus_mq = L::wide_add(t, L::wide_mul(L::to_wide(m), L::to_wide(q_limb)));
        let result = L::from_wide_truncate(L::wide_shr(t_plus_mq, L::BITS));
        if result >= q_limb {
            result.wrapping_sub(q_limb)
        } else {
            result
        }
    }

    #[inline(always)]
    fn montgomery_reduce_wide(t: L::Wide) -> L {
        let q_limb = Self::modulus_limb();
        let q_inv = Self::q_inv_limb();
        let t_lo = L::from_wide_truncate(t);
        let m = t_lo.wrapping_mul(q_inv);
        let product = L::wide_mul(L::to_wide(m), L::to_wide(q_limb));
        let bits = L::BITS;

        // Split t and product into low/high limbs
        let t_hi = L::from_wide_truncate(L::wide_shr(t, bits));
        let prod_lo = L::from_wide_truncate(product);
        let prod_hi = L::from_wide_truncate(L::wide_shr(product, bits));

        let (_sum_lo, carry_lo) = t_lo.overflowing_add(prod_lo);
        let (sum_hi, carry_hi_a) = t_hi.overflowing_add(prod_hi);
        let (sum_hi, carry_hi_b) = sum_hi.overflowing_add(L::from_u64(carry_lo as u64));
        let overflow = carry_hi_a || carry_hi_b;

        if overflow {
            sum_hi.wrapping_add(q_limb.wrapping_neg())
        } else if sum_hi >= q_limb {
            sum_hi.wrapping_sub(q_limb)
        } else {
            sum_hi
        }
    }

    /// Convert a normal integer (as a limb) to Montgomery form.
    #[inline]
    pub fn to_montgomery(val: L) -> Self {
        Self::check_modulus();
        let q = Self::modulus_limb();
        let val = val.mod_limb(q);
        let t = L::wide_mul(L::to_wide(val), L::wide_from_u64(Self::R2_MOD_Q_U64));
        Self {
            val: Self::montgomery_reduce(t),
        }
    }

    /// Convert from Montgomery form back to a limb value.
    #[inline]
    pub fn from_montgomery(self) -> L {
        Self::check_modulus();
        Self::montgomery_reduce(L::to_wide(self.val))
    }

    /// Create directly from a value already in Montgomery form (unchecked).
    #[inline]
    pub fn from_raw(val: L) -> Self {
        Self::check_modulus();
        Self { val }
    }

    /// Get the raw Montgomery form value.
    #[inline]
    pub const fn raw(&self) -> L {
        self.val
    }

    #[cfg(any(target_arch = "aarch64", test))]
    #[inline]
    pub(crate) fn montgomery_reduce_word(word: L) -> L {
        Self::montgomery_reduce(L::to_wide(word))
    }

    #[inline]
    pub(crate) fn mul_raw_words(lhs: L, rhs: L) -> L {
        Self::montgomery_reduce(L::wide_mul(L::to_wide(lhs), L::to_wide(rhs)))
    }

    pub(crate) fn values_mut(slice: &mut [Self]) -> &mut [L] {
        // SAFETY: `PrimeField<Q, L>` is `#[repr(transparent)]` over `L`.
        unsafe { core::slice::from_raw_parts_mut(slice.as_mut_ptr().cast::<L>(), slice.len()) }
    }

    pub(crate) fn values(slice: &[Self]) -> &[L] {
        // SAFETY: `PrimeField<Q, L>` is `#[repr(transparent)]` over `L`.
        unsafe { core::slice::from_raw_parts(slice.as_ptr().cast::<L>(), slice.len()) }
    }
}

impl<const Q: u64, L: PrimeFieldLimb> core::fmt::Debug for PrimeField<Q, L> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "PrimeField<{}>({}", Q, self.from_montgomery().to_u64())?;
        write!(f, ")")
    }
}

impl<const Q: u64, L: PrimeFieldLimb> core::fmt::Display for PrimeField<Q, L> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.from_montgomery().to_u64())
    }
}

// --- Operator impls ---

impl<const Q: u64, L: PrimeFieldLimb> Add for PrimeField<Q, L> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::check_modulus();
        Self {
            val: Self::add_raw_words(self.val, rhs.val),
        }
    }
}

impl<const Q: u64, L: PrimeFieldLimb> Add<&Self> for PrimeField<Q, L> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: &Self) -> Self {
        self + *rhs
    }
}

impl<const Q: u64, L: PrimeFieldLimb> AddAssign for PrimeField<Q, L> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const Q: u64, L: PrimeFieldLimb> AddAssign<&Self> for PrimeField<Q, L> {
    #[inline]
    fn add_assign(&mut self, rhs: &Self) {
        *self = *self + *rhs;
    }
}

impl<const Q: u64, L: PrimeFieldLimb> Sub for PrimeField<Q, L> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::check_modulus();
        Self {
            val: Self::sub_raw_words(self.val, rhs.val),
        }
    }
}

impl<const Q: u64, L: PrimeFieldLimb> Sub<&Self> for PrimeField<Q, L> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: &Self) -> Self {
        self - *rhs
    }
}

impl<const Q: u64, L: PrimeFieldLimb> SubAssign for PrimeField<Q, L> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const Q: u64, L: PrimeFieldLimb> SubAssign<&Self> for PrimeField<Q, L> {
    #[inline]
    fn sub_assign(&mut self, rhs: &Self) {
        *self = *self - *rhs;
    }
}

impl<const Q: u64, L: PrimeFieldLimb> Mul for PrimeField<Q, L> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::check_modulus();
        Self {
            val: Self::mul_raw_words(self.val, rhs.val),
        }
    }
}

impl<const Q: u64, L: PrimeFieldLimb> Mul<&Self> for PrimeField<Q, L> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: &Self) -> Self {
        self * *rhs
    }
}

impl<const Q: u64, L: PrimeFieldLimb> MulAssign for PrimeField<Q, L> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const Q: u64, L: PrimeFieldLimb> MulAssign<&Self> for PrimeField<Q, L> {
    #[inline]
    fn mul_assign(&mut self, rhs: &Self) {
        *self = *self * *rhs;
    }
}

impl<const Q: u64, L: PrimeFieldLimb> Add<Self> for &PrimeField<Q, L> {
    type Output = PrimeField<Q, L>;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        *self + *rhs
    }
}

impl<const Q: u64, L: PrimeFieldLimb> Add<PrimeField<Q, L>> for &PrimeField<Q, L> {
    type Output = PrimeField<Q, L>;
    #[inline]
    fn add(self, rhs: PrimeField<Q, L>) -> Self::Output {
        *self + rhs
    }
}

impl<const Q: u64, L: PrimeFieldLimb> Sub<Self> for &PrimeField<Q, L> {
    type Output = PrimeField<Q, L>;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        *self - *rhs
    }
}

impl<const Q: u64, L: PrimeFieldLimb> Sub<PrimeField<Q, L>> for &PrimeField<Q, L> {
    type Output = PrimeField<Q, L>;
    #[inline]
    fn sub(self, rhs: PrimeField<Q, L>) -> Self::Output {
        *self - rhs
    }
}

impl<const Q: u64, L: PrimeFieldLimb> Mul<Self> for &PrimeField<Q, L> {
    type Output = PrimeField<Q, L>;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        *self * *rhs
    }
}

impl<const Q: u64, L: PrimeFieldLimb> Mul<PrimeField<Q, L>> for &PrimeField<Q, L> {
    type Output = PrimeField<Q, L>;
    #[inline]
    fn mul(self, rhs: PrimeField<Q, L>) -> Self::Output {
        *self * rhs
    }
}

impl<const Q: u64, L: PrimeFieldLimb> Neg for PrimeField<Q, L> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::check_modulus();
        if self.val == L::ZERO {
            self
        } else {
            Self {
                val: Self::modulus_limb().wrapping_sub(self.val),
            }
        }
    }
}

// --- Trait impls ---

impl<const Q: u64, L: PrimeFieldLimb> Ring for PrimeField<Q, L> {
    #[inline]
    fn zero() -> Self {
        Self::check_modulus();
        Self { val: L::ZERO }
    }

    #[inline]
    fn one() -> Self {
        Self::check_modulus();
        Self {
            val: Self::r_mod_q_limb(),
        }
    }

    fn add_assign_slice(dst: &mut [Self], src: &[Self]) {
        montgomery_prime::add_assign(dst, src);
    }

    fn sub_assign_slice(dst: &mut [Self], src: &[Self]) {
        montgomery_prime::sub_assign(dst, src);
    }

    fn scalar_mul_slice(dst: &mut [Self], scalar: &Self) {
        montgomery_prime::scalar_mul(dst, scalar);
    }

    fn pointwise_mul_assign_slice(dst: &mut [Self], rhs: &[Self]) {
        montgomery_prime::pointwise_mul_assign(dst, rhs);
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

    fn dot_product(lhs: &[Self], rhs: &[Self]) -> Self {
        assert_eq!(lhs.len(), rhs.len(), "slice lengths must match");
        const CHUNK: usize = 64;
        if lhs.len() <= CHUNK {
            return lhs
                .iter()
                .zip(rhs.iter())
                .fold(Self::zero(), |mut acc, (lhs, rhs)| {
                    acc += *lhs * rhs;
                    acc
                });
        }
        let mut acc = Self::zero();
        for (lhs_chunk, rhs_chunk) in lhs.chunks(CHUNK).zip(rhs.chunks(CHUNK)) {
            let mut products = [Self::zero(); CHUNK];
            let len = lhs_chunk.len();
            products[..len].copy_from_slice(lhs_chunk);
            Self::pointwise_mul_assign_slice(&mut products[..len], rhs_chunk);
            acc = products[..len].iter().fold(acc, |mut acc, value| {
                acc += *value;
                acc
            });
        }
        acc
    }

    #[inline]
    fn add_ref(a: &Self, b: &Self) -> Self {
        a + b
    }

    #[inline]
    fn sub_ref(a: &Self, b: &Self) -> Self {
        a - b
    }

    #[inline]
    fn mul_ref(a: &Self, b: &Self) -> Self {
        a * b
    }
}

impl<const Q: u64, L: PrimeFieldLimb> IntegerRing for PrimeField<Q, L> {
    type Uint = u64;

    #[inline]
    fn modulus() -> u64 {
        Self::check_modulus();
        Self::MODULUS
    }

    #[inline]
    fn from_u64(val: u64) -> Self {
        Self::to_montgomery(L::from_u64(val % Q))
    }

    #[inline]
    fn to_u64(&self) -> u64 {
        self.from_montgomery().to_u64()
    }

    #[inline]
    fn lossy_l2_value(&self) -> f64 {
        let v = self.from_montgomery().to_u64() as f64;
        let q = Self::modulus() as f64;
        let half = q * 0.5;
        if v > half { v - q } else { v }
    }

    #[inline]
    fn reduce(&self) -> Self {
        Self::check_modulus();
        *self
    }
}

impl<const Q: u64, L: PrimeFieldLimb> Field for PrimeField<Q, L> {
    fn inv(&self) -> Self {
        Self::check_modulus();
        assert!(!self.is_zero(), "cannot invert zero");
        self.pow(Q - 2)
    }
}

impl<const Q: u64, L: PrimeFieldLimb> LargeCanonicalRing for PrimeField<Q, L> {
    type Canonical = u64;

    #[inline]
    fn modulus_canonical() -> Self::Canonical {
        Self::MODULUS
    }

    #[inline]
    fn from_small_u64(value: u64) -> Self {
        Self::from_u64(value)
    }

    #[inline]
    fn from_canonical(value: &Self::Canonical) -> Self {
        Self::from_u64(*value)
    }

    #[inline]
    fn to_canonical(&self) -> Self::Canonical {
        self.to_u64()
    }

    #[inline]
    fn try_to_u64(&self) -> Option<u64> {
        Some(self.to_u64())
    }

    #[inline]
    fn try_to_u128(&self) -> Option<u128> {
        Some(self.to_u64() as u128)
    }
}

impl<const Q: u64, L: PrimeFieldLimb> NTTRing for PrimeField<Q, L> {
    fn root_of_unity(n: usize) -> Option<Self> {
        Self::check_modulus();
        if !(Q - 1).is_multiple_of(n as u64) {
            return None;
        }

        let g = Self::find_generator()?;
        let exp = (Q - 1) / (n as u64);
        let root = g.pow(exp);

        if root.pow(n as u64) != Self::one() {
            return None;
        }
        if n > 1 && root.pow((n / 2) as u64) == Self::one() {
            return None;
        }

        Some(root)
    }

    fn inv_root_of_unity(n: usize) -> Option<Self> {
        Self::check_modulus();
        Self::root_of_unity(n).map(|r| r.inv())
    }

    fn inverse_ntt_scale(n: usize) -> Option<Self> {
        Self::check_modulus();
        let n = u64::try_from(n).ok()?;
        Some(Self::from_u64(n).inv())
    }

    fn max_ntt_size() -> usize {
        Self::check_modulus();
        let mut v = Q - 1;
        let mut max = 1usize;
        while v.is_multiple_of(2) && max <= usize::MAX / 2 {
            v >>= 1;
            max <<= 1;
        }
        max
    }

    fn ntt_forward_in_place(coeffs: &mut [Self]) -> Result<(), NttError>
    where
        Self: Field,
    {
        let plan = cached_ntt_plan::<Self>(coeffs.len())?;
        Self::ntt_forward_with_plan(coeffs, plan.as_ref())
    }

    fn ntt_forward_with_plan(coeffs: &mut [Self], plan: &NttPlan<Self>) -> Result<(), NttError>
    where
        Self: Field,
    {
        montgomery_prime::ntt_forward_with_plan(coeffs, plan)
    }

    fn ntt_inverse_in_place(evals: &mut [Self]) -> Result<(), NttError>
    where
        Self: Field,
    {
        let plan = cached_ntt_plan::<Self>(evals.len())?;
        Self::ntt_inverse_with_plan(evals, plan.as_ref())
    }

    fn ntt_inverse_with_plan(evals: &mut [Self], plan: &NttPlan<Self>) -> Result<(), NttError>
    where
        Self: Field,
    {
        montgomery_prime::ntt_inverse_with_plan(evals, plan)
    }
}

impl<const Q: u64, L: PrimeFieldLimb> PrimeField<Q, L> {
    /// Find a generator of the multiplicative group Z_q*.
    ///
    /// Uses trial from small values. Returns `None` if Q is not prime
    /// (but this type assumes Q is prime).
    fn find_generator() -> Option<Self> {
        Self::check_modulus();
        if Q == GOLDILOCKS_MODULUS {
            return Some(Self::from_u64(7));
        }

        // Factor Q - 1
        let mut factors = [0u64; 64];
        let mut num_factors = 0;
        let mut n = Q - 1;
        let mut d = 2u64;
        while d <= n / d {
            if n.is_multiple_of(d) {
                factors[num_factors] = d;
                num_factors += 1;
                while n.is_multiple_of(d) {
                    n /= d;
                }
            }
            d += 1;
        }
        if n > 1 {
            factors[num_factors] = n;
            num_factors += 1;
        }

        // Trial generators
        for g_val in 2..Q {
            let g = Self::from_u64(g_val);
            let mut is_generator = true;
            for factor in factors.iter().take(num_factors) {
                let exp = (Q - 1) / factor;
                if g.pow(exp) == Self::one() {
                    is_generator = false;
                    break;
                }
            }
            if is_generator {
                return Some(g);
            }
        }
        None
    }
}

impl<const Q: u64, L: PrimeFieldLimb> CanonicalSerialize for PrimeField<Q, L> {
    fn serialized_size(&self) -> usize {
        Self::check_modulus();
        Self::SERIALIZED_BYTES
    }

    fn serialize_into(&self, buf: &mut alloc::vec::Vec<u8>) -> Result<(), SerializationError> {
        Self::check_modulus();
        let val: u64 = self.from_montgomery().to_u64();
        let bytes = val.to_le_bytes();
        buf.extend_from_slice(&bytes[..Self::SERIALIZED_BYTES]);
        Ok(())
    }
}

impl<const Q: u64, L: PrimeFieldLimb> CanonicalDeserialize for PrimeField<Q, L> {
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        Self::check_modulus();
        if data.len() < Self::SERIALIZED_BYTES {
            return Err(SerializationError::UnexpectedEnd);
        }
        let mut bytes = [0u8; 8];
        bytes[..Self::SERIALIZED_BYTES].copy_from_slice(&data[..Self::SERIALIZED_BYTES]);
        let val = u64::from_le_bytes(bytes);
        if val >= Q {
            return Err(SerializationError::InvalidData(alloc::format!(
                "value {val} >= modulus {Q}"
            )));
        }
        Ok((Self::from_u64(val), Self::SERIALIZED_BYTES))
    }
}

impl<const Q: u64, L: PrimeFieldLimb> grid_serialize::Valid for PrimeField<Q, L> {
    fn is_valid(&self) -> bool {
        Self::check_modulus();
        self.val < Self::modulus_limb()
    }
}

impl<const Q: u64, L: PrimeFieldLimb> grid_std::UniformRand for PrimeField<Q, L> {
    fn rand<R: grid_std::rand::RngExt + ?Sized>(rng: &mut R) -> Self {
        Self::check_modulus();
        // Rejection sampling in u64, then convert to field element
        let reject = (u64::MAX % Q).wrapping_add(1) % Q;
        let upper = u64::MAX - reject;
        loop {
            let sample: u64 = rng.random();
            if sample <= upper {
                return Self::from_u64(sample % Q);
            }
        }
    }
}

const fn is_prime_const(n: u64) -> bool {
    const SMALL_PRIMES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    const MILLER_RABIN_BASES: [u64; 7] = [2, 325, 9_375, 28_178, 450_775, 9_780_504, 1_795_265_022];

    if n < 2 {
        return false;
    }

    let mut i = 0usize;
    while i < SMALL_PRIMES.len() {
        let p = SMALL_PRIMES[i];
        if n == p {
            return true;
        }
        if n.is_multiple_of(p) {
            return false;
        }
        i += 1;
    }

    let mut d = n - 1;
    let mut s = 0u32;
    while d.is_multiple_of(2) {
        d >>= 1;
        s += 1;
    }

    let mut base_idx = 0usize;
    while base_idx < MILLER_RABIN_BASES.len() {
        let a = MILLER_RABIN_BASES[base_idx] % n;
        base_idx += 1;
        if a == 0 {
            continue;
        }

        let mut x = pow_mod_const(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }

        let mut witness = true;
        let mut round = 1u32;
        while round < s {
            x = mul_mod_const(x, x, n);
            if x == n - 1 {
                witness = false;
                break;
            }
            round += 1;
        }
        if witness {
            return false;
        }
    }

    true
}

const fn mul_mod_const(lhs: u64, rhs: u64, modulus: u64) -> u64 {
    (((lhs as u128) * (rhs as u128)) % (modulus as u128)) as u64
}

const fn pow_mod_const(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut acc = 1u64;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = mul_mod_const(acc, base, modulus);
        }
        exp >>= 1;
        if exp > 0 {
            base = mul_mod_const(base, base, modulus);
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arith::ring::tests::{test_field_axioms, test_integer_ring, test_ring_axioms};
    use crate::simd::montgomery_prime::MontgomeryPrimeSimd;
    use core::hint::black_box;
    use core::mem::{align_of, size_of};
    use grid_serialize::Valid;
    use grid_std::UniformRand;

    // A small NTT-friendly prime: 17 (17 - 1 = 16 = 2^4)
    type F17 = PrimeField<17>;
    // A common lattice-friendly prime: 12289 (12288 = 2^12 * 3)
    type F12289 = PrimeField<12289>;
    // ML-KEM-style modulus that fits the widened multiply / NTT band.
    type F3329 = PrimeField<3329>;
    // Representative prime just above 2^32 to exercise add/sub-only widening.
    type F4294967311 = PrimeField<4294967311>;
    // Representative > 2^32 modulus with large power-of-two NTT support.
    type F184683593729 = PrimeField<184683593729>;
    // Representative prime above the AVX2 add/sub band but still valid for PrimeField.
    type F9223372036854775783 = PrimeField<9223372036854775783>;
    // Goldilocks high-band prime.
    type FGoldilocks = PrimeField<GOLDILOCKS_MODULUS>;

    fn chi_square_within_critical(observed: &[usize], expected: f64, critical: f64) -> bool {
        let statistic = observed
            .iter()
            .map(|&count| {
                let diff = count as f64 - expected;
                (diff * diff) / expected
            })
            .sum::<f64>();
        statistic < critical
    }

    #[test]
    fn test_montgomery_constants() {
        // Verify R mod Q: for u64 limb, R = 2^64, R mod Q = (u64::MAX % Q + 1) % Q
        assert_ne!(F17::R_MOD_Q_U64, 0);

        // Verify Q_INV: Q * Q_INV ≡ -1 (mod 2^64)
        let product = (17u64).wrapping_mul(F17::Q_INV_U64);
        assert_eq!(product, u64::MAX); // -1 mod 2^64
    }

    #[test]
    fn test_repr_transparent_layout() {
        assert_eq!(size_of::<F17>(), size_of::<u64>());
        assert_eq!(align_of::<F17>(), align_of::<u64>());
    }

    #[test]
    fn test_to_from_montgomery() {
        for val in 0..17u64 {
            let m = F17::to_montgomery(val);
            let back = m.from_montgomery();
            assert_eq!(back, val, "round-trip failed for {val}");
        }
    }

    #[test]
    fn test_goldilocks_to_from_montgomery_and_raw_contract() {
        let samples = [
            0u64,
            1,
            2,
            17,
            (1u64 << 32) - 1,
            1u64 << 32,
            GOLDILOCKS_MODULUS - 2,
            GOLDILOCKS_MODULUS - 1,
        ];

        for value in samples {
            let mont = FGoldilocks::to_montgomery(value);
            assert_eq!(mont.to_u64(), value);
            assert!(mont.raw() < GOLDILOCKS_MODULUS);
            assert!(mont.is_valid());

            let restored = FGoldilocks::from_raw(mont.raw());
            assert_eq!(restored, mont);
            assert_eq!(restored.to_u64(), value);
        }

        let invalid = FGoldilocks::from_raw(GOLDILOCKS_MODULUS);
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_add_sub() {
        let a = F17::from_u64(13);
        let b = F17::from_u64(10);
        assert_eq!((a + b).to_u64(), 6); // (13 + 10) mod 17 = 6
        assert_eq!((a - b).to_u64(), 3); // (13 - 10) mod 17 = 3
        assert_eq!((b - a).to_u64(), 14); // (10 - 13) mod 17 = -3 mod 17 = 14
    }

    #[test]
    fn test_goldilocks_add_sub_mul_near_modulus() {
        let pairs = [
            (GOLDILOCKS_MODULUS - 1, GOLDILOCKS_MODULUS - 2),
            ((1u64 << 32) - 1, (1u64 << 32) + 7),
            (123_456_789, 987_654_321),
            (GOLDILOCKS_MODULUS - (1u64 << 32), 1u64 << 32),
        ];

        for (lhs, rhs) in pairs {
            let a = FGoldilocks::from_u64(lhs);
            let b = FGoldilocks::from_u64(rhs);
            let expected_add =
                (((lhs as u128) + (rhs as u128)) % (GOLDILOCKS_MODULUS as u128)) as u64;
            let expected_sub = (((lhs as u128) + (GOLDILOCKS_MODULUS as u128) - (rhs as u128))
                % (GOLDILOCKS_MODULUS as u128)) as u64;
            let expected_mul =
                (((lhs as u128) * (rhs as u128)) % (GOLDILOCKS_MODULUS as u128)) as u64;

            assert_eq!((a + b).to_u64(), expected_add);
            assert_eq!((a - b).to_u64(), expected_sub);
            assert_eq!((a * b).to_u64(), expected_mul);
        }
    }

    #[test]
    fn test_slice_hooks_match_scalar_ops() {
        let lhs = [
            F12289::from_u64(1),
            F12289::from_u64(7),
            F12289::from_u64(12_000),
            F12289::from_u64(3),
            F12289::from_u64(11),
        ];
        let rhs = [
            F12289::from_u64(2),
            F12289::from_u64(9),
            F12289::from_u64(1_111),
            F12289::from_u64(14),
            F12289::from_u64(6),
        ];

        let mut add_hook = lhs;
        let mut sub_hook = lhs;
        let add_scalar = core::array::from_fn::<F12289, 5, _>(|i| lhs[i] + rhs[i]);
        let sub_scalar = core::array::from_fn::<F12289, 5, _>(|i| lhs[i] - rhs[i]);

        F12289::add_assign_slice(&mut add_hook, &rhs);
        F12289::sub_assign_slice(&mut sub_hook, &rhs);

        assert_eq!(add_hook, add_scalar);
        assert_eq!(sub_hook, sub_scalar);
    }

    #[test]
    fn test_simd_qualification_bands() {
        assert!(black_box(
            <F17 as MontgomeryPrimeSimd>::AVX2_ADD_SUB_QUALIFIED
        ));
        assert!(black_box(
            <F17 as MontgomeryPrimeSimd>::NEON_ADD_SUB_QUALIFIED
        ));
        assert!(black_box(
            <F17 as MontgomeryPrimeSimd>::AVX2_MONTGOMERY_QUALIFIED
        ));
        assert!(!black_box(
            <F17 as MontgomeryPrimeSimd>::NEON_MONTGOMERY_QUALIFIED
        ));

        assert!(black_box(
            <F3329 as MontgomeryPrimeSimd>::AVX2_ADD_SUB_QUALIFIED
        ));
        assert!(black_box(
            <F3329 as MontgomeryPrimeSimd>::AVX2_MONTGOMERY_QUALIFIED
        ));
        assert!(black_box(
            <F3329 as MontgomeryPrimeSimd>::AVX2_NTT_QUALIFIED
        ));

        assert!(black_box(
            <F4294967311 as MontgomeryPrimeSimd>::AVX2_ADD_SUB_QUALIFIED
        ));
        assert!(black_box(
            <F4294967311 as MontgomeryPrimeSimd>::NEON_ADD_SUB_QUALIFIED
        ));
        assert!(black_box(
            <F4294967311 as MontgomeryPrimeSimd>::AVX2_MONTGOMERY_QUALIFIED
        ));
        assert!(!black_box(
            <F4294967311 as MontgomeryPrimeSimd>::NEON_MONTGOMERY_QUALIFIED
        ));
        assert!(black_box(
            <F4294967311 as MontgomeryPrimeSimd>::AVX2_NTT_QUALIFIED
        ));
        assert!(!black_box(
            <F4294967311 as MontgomeryPrimeSimd>::NEON_NTT_QUALIFIED
        ));

        assert!(!black_box(
            <F9223372036854775783 as MontgomeryPrimeSimd>::AVX2_ADD_SUB_QUALIFIED
        ));
        assert!(black_box(
            <F9223372036854775783 as MontgomeryPrimeSimd>::NEON_ADD_SUB_QUALIFIED
        ));
        assert!(black_box(
            <F9223372036854775783 as MontgomeryPrimeSimd>::AVX2_MONTGOMERY_QUALIFIED
        ));
        assert!(!black_box(
            <F9223372036854775783 as MontgomeryPrimeSimd>::NEON_MONTGOMERY_QUALIFIED
        ));
        assert!(black_box(
            <F9223372036854775783 as MontgomeryPrimeSimd>::AVX2_NTT_QUALIFIED
        ));
        assert!(!black_box(
            <F9223372036854775783 as MontgomeryPrimeSimd>::NEON_NTT_QUALIFIED
        ));

        assert!(!black_box(
            <FGoldilocks as MontgomeryPrimeSimd>::AVX2_ADD_SUB_QUALIFIED
        ));
        assert!(!black_box(
            <FGoldilocks as MontgomeryPrimeSimd>::NEON_ADD_SUB_QUALIFIED
        ));
        assert!(black_box(
            <FGoldilocks as MontgomeryPrimeSimd>::AVX2_MONTGOMERY_QUALIFIED
        ));
        assert!(!black_box(
            <FGoldilocks as MontgomeryPrimeSimd>::NEON_MONTGOMERY_QUALIFIED
        ));
        assert!(black_box(
            <FGoldilocks as MontgomeryPrimeSimd>::AVX2_NTT_QUALIFIED
        ));
        assert!(!black_box(
            <FGoldilocks as MontgomeryPrimeSimd>::NEON_NTT_QUALIFIED
        ));
    }

    #[test]
    fn test_slice_hooks_match_scalar_ops_3329() {
        let lhs = [
            F3329::from_u64(1),
            F3329::from_u64(7),
            F3329::from_u64(3_000),
            F3329::from_u64(3),
            F3329::from_u64(11),
        ];
        let rhs = [
            F3329::from_u64(2),
            F3329::from_u64(9),
            F3329::from_u64(1_111),
            F3329::from_u64(14),
            F3329::from_u64(6),
        ];
        let scalar = F3329::from_u64(19);

        let mut add_hook = lhs;
        let mut sub_hook = lhs;
        let mut mul_hook = lhs;
        let mut scalar_mul_hook = lhs;
        let mut scaled_add_hook = lhs;
        let add_scalar = core::array::from_fn::<F3329, 5, _>(|i| lhs[i] + rhs[i]);
        let sub_scalar = core::array::from_fn::<F3329, 5, _>(|i| lhs[i] - rhs[i]);
        let mul_scalar = core::array::from_fn::<F3329, 5, _>(|i| lhs[i] * rhs[i]);
        let scalar_mul_scalar = core::array::from_fn::<F3329, 5, _>(|i| lhs[i] * scalar);
        let scaled_add_scalar = core::array::from_fn::<F3329, 5, _>(|i| lhs[i] + rhs[i] * scalar);

        F3329::add_assign_slice(&mut add_hook, &rhs);
        F3329::sub_assign_slice(&mut sub_hook, &rhs);
        F3329::pointwise_mul_assign_slice(&mut mul_hook, &rhs);
        F3329::scalar_mul_slice(&mut scalar_mul_hook, &scalar);
        F3329::add_assign_scaled_slice(&mut scaled_add_hook, &rhs, &scalar);

        assert_eq!(add_hook, add_scalar);
        assert_eq!(sub_hook, sub_scalar);
        assert_eq!(mul_hook, mul_scalar);
        assert_eq!(scalar_mul_hook, scalar_mul_scalar);
        assert_eq!(scaled_add_hook, scaled_add_scalar);
    }

    #[test]
    fn test_slice_hooks_match_scalar_ops_above_2pow32() {
        let lhs = [
            F4294967311::from_u64(1),
            F4294967311::from_u64(7),
            F4294967311::from_u64(4_000_000_000),
            F4294967311::from_u64(3),
            F4294967311::from_u64(11),
        ];
        let rhs = [
            F4294967311::from_u64(2),
            F4294967311::from_u64(9),
            F4294967311::from_u64(111_111_111),
            F4294967311::from_u64(14),
            F4294967311::from_u64(6),
        ];

        let mut add_hook = lhs;
        let mut sub_hook = lhs;
        let add_scalar = core::array::from_fn::<F4294967311, 5, _>(|i| lhs[i] + rhs[i]);
        let sub_scalar = core::array::from_fn::<F4294967311, 5, _>(|i| lhs[i] - rhs[i]);

        F4294967311::add_assign_slice(&mut add_hook, &rhs);
        F4294967311::sub_assign_slice(&mut sub_hook, &rhs);

        assert_eq!(add_hook, add_scalar);
        assert_eq!(sub_hook, sub_scalar);
    }

    #[test]
    fn test_slice_hooks_match_scalar_ops_goldilocks() {
        let lhs = [
            FGoldilocks::from_u64(1),
            FGoldilocks::from_u64((1u64 << 32) - 1),
            FGoldilocks::from_u64(GOLDILOCKS_MODULUS - 2),
            FGoldilocks::from_u64(123_456_789),
            FGoldilocks::from_u64(9_876_543_210),
        ];
        let rhs = [
            FGoldilocks::from_u64(2),
            FGoldilocks::from_u64((1u64 << 32) + 7),
            FGoldilocks::from_u64(GOLDILOCKS_MODULUS - 3),
            FGoldilocks::from_u64(987_654_321),
            FGoldilocks::from_u64(77_777_777),
        ];
        let scalar = FGoldilocks::from_u64((1u64 << 32) + 13);

        let mut add_hook = lhs;
        let mut sub_hook = lhs;
        let mut mul_hook = lhs;
        let mut scalar_mul_hook = lhs;
        let add_scalar = core::array::from_fn::<FGoldilocks, 5, _>(|i| lhs[i] + rhs[i]);
        let sub_scalar = core::array::from_fn::<FGoldilocks, 5, _>(|i| lhs[i] - rhs[i]);
        let mul_scalar = core::array::from_fn::<FGoldilocks, 5, _>(|i| lhs[i] * rhs[i]);
        let scalar_mul_scalar = core::array::from_fn::<FGoldilocks, 5, _>(|i| lhs[i] * scalar);

        FGoldilocks::add_assign_slice(&mut add_hook, &rhs);
        FGoldilocks::sub_assign_slice(&mut sub_hook, &rhs);
        FGoldilocks::pointwise_mul_assign_slice(&mut mul_hook, &rhs);
        FGoldilocks::scalar_mul_slice(&mut scalar_mul_hook, &scalar);

        assert_eq!(add_hook, add_scalar);
        assert_eq!(sub_hook, sub_scalar);
        assert_eq!(mul_hook, mul_scalar);
        assert_eq!(scalar_mul_hook, scalar_mul_scalar);
    }

    #[test]
    fn test_mul_hooks_match_scalar_ops() {
        let lhs = [
            F12289::from_u64(1),
            F12289::from_u64(7),
            F12289::from_u64(12_000),
            F12289::from_u64(3),
            F12289::from_u64(11),
        ];
        let rhs = [
            F12289::from_u64(2),
            F12289::from_u64(9),
            F12289::from_u64(1_111),
            F12289::from_u64(14),
            F12289::from_u64(6),
        ];
        let scalar = F12289::from_u64(19);

        let mut mul_hook = lhs;
        let mut scalar_mul_hook = lhs;
        let mul_scalar = core::array::from_fn::<F12289, 5, _>(|i| lhs[i] * rhs[i]);
        let scalar_mul_scalar = core::array::from_fn::<F12289, 5, _>(|i| lhs[i] * scalar);

        F12289::pointwise_mul_assign_slice(&mut mul_hook, &rhs);
        F12289::scalar_mul_slice(&mut scalar_mul_hook, &scalar);

        assert_eq!(mul_hook, mul_scalar);
        assert_eq!(scalar_mul_hook, scalar_mul_scalar);
    }

    #[test]
    fn test_mul_hooks_match_scalar_ops_8380417() {
        type F8380417 = PrimeField<8380417>;

        let lhs = [
            F8380417::from_u64(1),
            F8380417::from_u64(7),
            F8380417::from_u64(8_000_000),
            F8380417::from_u64(3),
            F8380417::from_u64(11),
        ];
        let rhs = [
            F8380417::from_u64(2),
            F8380417::from_u64(9),
            F8380417::from_u64(111_111),
            F8380417::from_u64(14),
            F8380417::from_u64(6),
        ];
        let scalar = F8380417::from_u64(19);

        let mut mul_hook = lhs;
        let mut scalar_mul_hook = lhs;
        let mul_scalar = core::array::from_fn::<F8380417, 5, _>(|i| lhs[i] * rhs[i]);
        let scalar_mul_scalar = core::array::from_fn::<F8380417, 5, _>(|i| lhs[i] * scalar);

        F8380417::pointwise_mul_assign_slice(&mut mul_hook, &rhs);
        F8380417::scalar_mul_slice(&mut scalar_mul_hook, &scalar);

        assert_eq!(mul_hook, mul_scalar);
        assert_eq!(scalar_mul_hook, scalar_mul_scalar);
    }

    #[test]
    fn test_mul() {
        let a = F17::from_u64(5);
        let b = F17::from_u64(4);
        assert_eq!((a * b).to_u64(), 3); // (5 * 4) mod 17 = 20 mod 17 = 3
    }

    #[test]
    fn test_inv() {
        // 5^(-1) mod 17: 5 * 7 = 35 = 2*17 + 1, so inv(5) = 7
        let a = F17::from_u64(5);
        let inv_a = a.inv();
        assert_eq!(inv_a.to_u64(), 7);
        assert_eq!((a * inv_a).to_u64(), 1);
    }

    #[test]
    fn test_inv_all_nonzero() {
        for val in 1..17u64 {
            let a = F17::from_u64(val);
            let inv = a.inv();
            assert_eq!(
                (a * inv).to_u64(),
                1,
                "inv failed for {val}: got {}",
                inv.to_u64()
            );
        }
    }

    #[test]
    fn test_ring_axioms_f17() {
        let a = F17::from_u64(3);
        let b = F17::from_u64(7);
        let c = F17::from_u64(11);
        test_ring_axioms(a, b, c);
    }

    #[test]
    fn test_integer_ring_f17() {
        test_integer_ring::<F17>(10);
    }

    #[test]
    fn test_field_axioms_f17() {
        for val in 0..17u64 {
            test_field_axioms(F17::from_u64(val));
        }
    }

    #[test]
    fn test_ring_axioms_f12289() {
        let mut rng = grid_std::test_rng();
        let a = F12289::rand(&mut rng);
        let b = F12289::rand(&mut rng);
        let c = F12289::rand(&mut rng);
        test_ring_axioms(a, b, c);
    }

    #[test]
    fn test_ntt_ring() {
        // F17: Q-1 = 16 = 2^4, so max NTT size = 16
        assert_eq!(F17::max_ntt_size(), 16);
        assert!(F17::supports_ntt(2));
        assert!(F17::supports_ntt(4));
        assert!(F17::supports_ntt(8));
        assert!(F17::supports_ntt(16));
        assert!(!F17::supports_ntt(32));

        // Verify root of unity
        let root = F17::root_of_unity(4).unwrap();
        let r4 = root.pow(4);
        assert_eq!(r4.to_u64(), 1, "root^4 should be 1");
        assert_ne!(root.pow(2).to_u64(), 1, "root^2 should not be 1");
    }

    #[test]
    fn test_ntt_ring_12289() {
        // F12289: Q-1 = 12288 = 2^12 * 3
        assert_eq!(F12289::max_ntt_size(), 4096); // 2^12
        assert!(F12289::supports_ntt(4096));
        assert!(!F12289::supports_ntt(8192));
    }

    #[test]
    fn test_ntt_round_trip_3329() {
        let input = [
            F3329::from_u64(1),
            F3329::from_u64(2),
            F3329::from_u64(3),
            F3329::from_u64(4),
            F3329::from_u64(5),
            F3329::from_u64(6),
            F3329::from_u64(7),
            F3329::from_u64(8),
        ];
        let mut evals = input;

        F3329::ntt_forward_in_place(&mut evals).unwrap();
        F3329::ntt_inverse_in_place(&mut evals).unwrap();

        assert_eq!(evals, input);
    }

    #[test]
    fn test_ntt_round_trip_above_2pow32() {
        let mut rng = grid_std::test_rng();
        let original: Vec<F184683593729> =
            (0..256).map(|_| F184683593729::rand(&mut rng)).collect();
        let mut values = original.clone();

        F184683593729::ntt_forward_in_place(&mut values).unwrap();
        F184683593729::ntt_inverse_in_place(&mut values).unwrap();

        assert_eq!(values, original);
    }

    #[test]
    fn test_goldilocks_ntt_round_trip_512() {
        let mut rng = grid_std::test_rng();
        let original: Vec<FGoldilocks> = (0..512).map(|_| FGoldilocks::rand(&mut rng)).collect();
        let mut transformed = original.clone();

        FGoldilocks::ntt_forward_in_place(&mut transformed).unwrap();
        FGoldilocks::ntt_inverse_in_place(&mut transformed).unwrap();

        assert_eq!(transformed, original);
    }

    #[test]
    fn test_pow() {
        let a = F17::from_u64(3);
        assert_eq!(a.pow(0).to_u64(), 1);
        assert_eq!(a.pow(1).to_u64(), 3);
        assert_eq!(a.pow(2).to_u64(), 9);
        assert_eq!(a.pow(3).to_u64(), 10); // 27 mod 17 = 10
        assert_eq!(a.pow(16).to_u64(), 1); // Fermat's little theorem
    }

    #[test]
    fn test_uniform_rand() {
        let mut rng = grid_std::test_rng();
        let mut counts = [0usize; 17];
        for _ in 0..17_000 {
            let a = F17::rand(&mut rng);
            let value = a.to_u64();
            assert!(value < 17);
            counts[value as usize] += 1;
        }
        assert!(chi_square_within_critical(&counts, 1_000.0, 40.79));
    }

    #[test]
    fn test_serialize_round_trip() {
        use grid_serialize::{CanonicalDeserialize, CanonicalSerialize};
        let ser_bytes = F17::SERIALIZED_BYTES;
        for val in 0..17u64 {
            let a = F17::from_u64(val);
            let bytes = a.serialize().unwrap();
            assert_eq!(bytes.len(), ser_bytes);
            let (b, consumed) = F17::deserialize(&bytes).unwrap();
            assert_eq!(consumed, ser_bytes);
            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_deserialize_rejects_out_of_range_value() {
        let err = F17::deserialize(&17u64.to_le_bytes()).unwrap_err();
        assert!(matches!(err, SerializationError::InvalidData(_)));
    }

    #[test]
    fn test_prime_field_large_canonical_ring_round_trip() {
        use super::LargeCanonicalRing;

        let value = F17::from_small_u64(19);
        assert_eq!(value.to_canonical(), 2);
        assert_eq!(F17::modulus_canonical(), 17);
        assert_eq!(F17::from_canonical(&20).to_canonical(), 3);
        assert_eq!(value.try_to_u64(), Some(2));
        assert_eq!(value.try_to_u128(), Some(2));
    }

    // --- Narrow-limb backend tests ---

    type F251u8 = PrimeField<251, u8>;
    type F12289u16 = PrimeField<12289, u16>;
    type F65521u16 = PrimeField<65521, u16>;
    type F4294967291u32 = PrimeField<4294967291, u32>;

    #[test]
    fn test_narrow_u8_ring_mul() {
        // 251 is prime and fits in u8. Test basic ring operations.
        let a = F251u8::from_u64(200);
        let b = F251u8::from_u64(100);
        assert_eq!((a + b).to_u64(), 49); // 300 mod 251 = 49
        assert_eq!((a - b).to_u64(), 100); // 100 mod 251
        assert_eq!((a * b).to_u64(), (200 * 100) % 251);
        assert_eq!(a.inv().to_u64(), {
            // 200^(-1) mod 251
            let mut inv = 0u64;
            for i in 1..251 {
                if (200 * i) % 251 == 1 {
                    inv = i;
                    break;
                }
            }
            inv
        });
        let zero = F251u8::zero();
        assert!(zero.is_zero());
        let one = F251u8::one();
        assert_eq!(one.to_u64(), 1);
        assert_eq!((a * one).to_u64(), a.to_u64());
    }

    #[test]
    fn test_narrow_u16_from_u64_reduction() {
        // from_u64(65536) must reduce mod 12289, NOT truncate to u16 first.
        // 65536 mod 12289 = 65536 - 5*12289 = 65536 - 61445 = 4091
        let a = F12289u16::from_u64(65536);
        assert_eq!(a.to_u64(), 65536 % 12289);

        // from_u64(0x1_0000) = 65536 — same test, hex literal
        let b = F12289u16::from_u64(0x1_0000);
        assert_eq!(b.to_u64(), 65536 % 12289);

        // from_u64 of a value that wraps u16 multiple times: 3 * 65536 + 42
        let c = F12289u16::from_u64(3 * 65536 + 42);
        assert_eq!(c.to_u64(), (3 * 65536 + 42) % 12289);

        // from_u64 of a value safely within u16 range should still be correct
        let d = F12289u16::from_u64(42);
        assert_eq!(d.to_u64(), 42);
    }

    #[test]
    fn test_narrow_u16_high_modulus_mul() {
        // 65521 is the largest prime < 2^16. Test multiplication near the modulus boundary.
        let a = F65521u16::from_u64(65520);
        let b = F65521u16::from_u64(65520);
        let expected = ((65520u128 * 65520) % 65521) as u64;
        assert_eq!((a * b).to_u64(), expected);

        // Near-wrapping add: 65520 + 65520 mod 65521 = 131040 mod 65521 = 65519
        assert_eq!((a + a).to_u64(), 65519);

        // Check inversion works for high values
        let inv = a.inv();
        assert_eq!((a * inv).to_u64(), 1);

        // High value from_u64 that wraps u16
        let c = F65521u16::from_u64(131042); // 2 * 65521
        assert_eq!(c.to_u64(), 0);
    }

    #[test]
    fn test_narrow_u32_from_u64_boundary() {
        // 4294967291 is prime and close to 2^32. Test from_u64 with values near 2^32.
        let q = 4294967291u64;
        let a = F4294967291u32::from_u64((1u64 << 32) + 12345);
        assert_eq!(a.to_u64(), ((1u64 << 32) + 12345) % q);

        // Value exactly at 2^32
        let b = F4294967291u32::from_u64(1u64 << 32);
        assert_eq!(b.to_u64(), (1u64 << 32) % q);

        // Value that wraps u32 multiple times
        let big = (1u64 << 33) + (1u64 << 32) + 789;
        let c = F4294967291u32::from_u64(big);
        assert_eq!(c.to_u64(), big % q);

        // Basic ring operations with this high modulus
        let x = F4294967291u32::from_u64(q - 1);
        let y = F4294967291u32::from_u64(2);
        assert_eq!((x + y).to_u64(), 1); // (q-1 + 2) mod q = 1
        assert_eq!((x * y).to_u64(), (2 * (q - 1)) % q);
    }
}
