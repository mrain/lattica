//! `GF(2)` — the binary field.
//!
//! Addition = XOR, multiplication = AND, negation = identity.
//! No modulus, no carries, no Montgomery form.
//!
//! The following lint fires because Add/Sub use `^` (XOR)
//! and Mul uses `&` (AND), which is correct for characteristic 2.
#![allow(clippy::suspicious_arithmetic_impl, clippy::suspicious_op_assign_impl)]

use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::ring::{IntegerRing, Ring};
use crate::simd::dispatch::{Backend, selected_backend};
use grid_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError, Valid};
use grid_std::rand::RngExt;

/// An element of `GF(2)`, stored as 0 or 1.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct GF2(u8);

impl GF2 {
    pub const fn new(val: u8) -> Self {
        GF2(val & 1)
    }

    #[inline]
    fn as_u8_ptr(slice: &[Self]) -> *const u8 {
        slice.as_ptr().cast::<u8>()
    }

    #[inline]
    fn as_mut_u8_ptr(slice: &mut [Self]) -> *mut u8 {
        slice.as_mut_ptr().cast::<u8>()
    }
}

// --- Formatting ---

impl core::fmt::Debug for GF2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "GF2({})", self.0)
    }
}

impl core::fmt::Display for GF2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// --- Operator impls ---

impl Add for GF2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        GF2(self.0 ^ rhs.0)
    }
}

impl Add<&Self> for GF2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: &Self) -> Self {
        GF2(self.0 ^ rhs.0)
    }
}

impl AddAssign for GF2 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

impl AddAssign<&Self> for GF2 {
    #[inline]
    fn add_assign(&mut self, rhs: &Self) {
        self.0 ^= rhs.0;
    }
}

impl Sub for GF2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        GF2(self.0 ^ rhs.0)
    }
}

impl Sub<&Self> for GF2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: &Self) -> Self {
        GF2(self.0 ^ rhs.0)
    }
}

impl SubAssign for GF2 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

impl SubAssign<&Self> for GF2 {
    #[inline]
    fn sub_assign(&mut self, rhs: &Self) {
        self.0 ^= rhs.0;
    }
}

impl Mul for GF2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        GF2(self.0 & rhs.0)
    }
}

impl Mul<&Self> for GF2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: &Self) -> Self {
        GF2(self.0 & rhs.0)
    }
}

impl MulAssign for GF2 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl MulAssign<&Self> for GF2 {
    #[inline]
    fn mul_assign(&mut self, rhs: &Self) {
        self.0 &= rhs.0;
    }
}

// Ref-ref operator impls
impl Add<Self> for &GF2 {
    type Output = GF2;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        GF2(self.0 ^ rhs.0)
    }
}

impl Add<GF2> for &GF2 {
    type Output = GF2;
    #[inline]
    fn add(self, rhs: GF2) -> Self::Output {
        GF2(self.0 ^ rhs.0)
    }
}

impl Sub<Self> for &GF2 {
    type Output = GF2;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        GF2(self.0 ^ rhs.0)
    }
}

impl Sub<GF2> for &GF2 {
    type Output = GF2;
    #[inline]
    fn sub(self, rhs: GF2) -> Self::Output {
        GF2(self.0 ^ rhs.0)
    }
}

impl Mul<Self> for &GF2 {
    type Output = GF2;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        GF2(self.0 & rhs.0)
    }
}

impl Mul<GF2> for &GF2 {
    type Output = GF2;
    #[inline]
    fn mul(self, rhs: GF2) -> Self::Output {
        GF2(self.0 & rhs.0)
    }
}

impl Neg for GF2 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self
    }
}

// --- Ring impl ---

impl Ring for GF2 {
    #[inline]
    fn zero() -> Self {
        GF2(0)
    }

    #[inline]
    fn one() -> Self {
        GF2(1)
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0 == 0
    }

    #[inline]
    fn is_one(&self) -> bool {
        self.0 == 1
    }

    #[inline]
    fn double(&self) -> Self {
        GF2(0)
    }

    #[inline]
    fn square(&self) -> Self {
        *self
    }

    #[inline]
    fn add_ref(a: &Self, b: &Self) -> Self {
        GF2(a.0 ^ b.0)
    }

    #[inline]
    fn sub_ref(a: &Self, b: &Self) -> Self {
        GF2(a.0 ^ b.0)
    }

    #[inline]
    fn mul_ref(a: &Self, b: &Self) -> Self {
        GF2(a.0 & b.0)
    }

    fn add_assign_slice(dst: &mut [Self], src: &[Self]) {
        assert_eq!(dst.len(), src.len(), "slice lengths must match");
        #[cfg(target_arch = "x86_64")]
        if matches!(selected_backend(), Backend::Avx2) {
            // SAFETY: backend selection enables AVX2 only when detected.
            unsafe {
                crate::simd::avx2::gf2::add_assign(
                    Self::as_mut_u8_ptr(dst),
                    Self::as_u8_ptr(src),
                    dst.len(),
                );
            }
            return;
        }
        #[cfg(target_arch = "aarch64")]
        if matches!(selected_backend(), Backend::Neon) {
            // SAFETY: backend selection enables NEON only when detected.
            unsafe {
                crate::simd::aarch64::gf2::add_assign(
                    Self::as_mut_u8_ptr(dst),
                    Self::as_u8_ptr(src),
                    dst.len(),
                );
            }
            return;
        }
        for (d, s) in dst.iter_mut().zip(src.iter()) {
            d.0 ^= s.0;
        }
    }

    fn sub_assign_slice(dst: &mut [Self], src: &[Self]) {
        // Same as add_assign_slice in char 2.
        Self::add_assign_slice(dst, src);
    }

    fn scalar_mul_slice(dst: &mut [Self], scalar: &Self) {
        if scalar.is_zero() {
            dst.fill(GF2(0));
        }
        // else scalar = 1, no-op
    }

    fn pointwise_mul_assign_slice(dst: &mut [Self], rhs: &[Self]) {
        assert_eq!(dst.len(), rhs.len(), "slice lengths must match");
        #[cfg(target_arch = "x86_64")]
        if matches!(selected_backend(), Backend::Avx2) {
            // SAFETY: backend selection enables AVX2 only when detected.
            unsafe {
                crate::simd::avx2::gf2::pointwise_mul_assign(
                    Self::as_mut_u8_ptr(dst),
                    Self::as_u8_ptr(rhs),
                    dst.len(),
                );
            }
            return;
        }
        #[cfg(target_arch = "aarch64")]
        if matches!(selected_backend(), Backend::Neon) {
            // SAFETY: backend selection enables NEON only when detected.
            unsafe {
                crate::simd::aarch64::gf2::pointwise_mul_assign(
                    Self::as_mut_u8_ptr(dst),
                    Self::as_u8_ptr(rhs),
                    dst.len(),
                );
            }
            return;
        }
        for (d, r) in dst.iter_mut().zip(rhs.iter()) {
            d.0 &= r.0;
        }
    }

    fn dot_product(lhs: &[Self], rhs: &[Self]) -> Self {
        assert_eq!(lhs.len(), rhs.len(), "slice lengths must match");
        #[cfg(target_arch = "x86_64")]
        if matches!(selected_backend(), Backend::Avx2) {
            // SAFETY: backend selection enables AVX2 only when detected.
            unsafe {
                let result = crate::simd::avx2::gf2::dot_product(
                    Self::as_u8_ptr(lhs),
                    Self::as_u8_ptr(rhs),
                    lhs.len(),
                );
                return GF2((result & 1) as u8);
            }
        }
        #[cfg(target_arch = "aarch64")]
        if matches!(selected_backend(), Backend::Neon) {
            // SAFETY: backend selection enables NEON only when detected.
            unsafe {
                let result = crate::simd::aarch64::gf2::dot_product(
                    Self::as_u8_ptr(lhs),
                    Self::as_u8_ptr(rhs),
                    lhs.len(),
                );
                return GF2((result & 1) as u8);
            }
        }
        let mut acc: u64 = 0;
        for (l, r) in lhs.iter().zip(rhs.iter()) {
            acc += (l.0 & r.0) as u64;
        }
        GF2((acc & 1) as u8)
    }

    fn add_assign_scaled_slice(dst: &mut [Self], src: &[Self], scalar: &Self) {
        if scalar.is_zero() {
            return;
        }
        // scalar = 1, same as add_assign_slice
        Self::add_assign_slice(dst, src);
    }
}

// --- IntegerRing impl ---

impl IntegerRing for GF2 {
    type Uint = u64;

    #[inline]
    fn modulus() -> u64 {
        2
    }

    #[inline]
    fn from_u64(val: u64) -> Self {
        GF2::new(val as u8)
    }

    #[inline]
    fn to_u64(&self) -> u64 {
        self.0 as u64
    }

    #[inline]
    fn lossy_l2_value(&self) -> f64 {
        if self.0 == 0 { 0.0 } else { 1.0 }
    }

    #[inline]
    fn reduce(&self) -> Self {
        *self
    }
}

// --- Field impl ---

impl super::ring::Field for GF2 {
    #[inline]
    fn inv(&self) -> Self {
        assert!(!self.is_zero(), "division by zero in GF(2)");
        GF2(1)
    }
}

// --- Serialization ---

impl CanonicalSerialize for GF2 {
    fn serialized_size(&self) -> usize {
        1
    }

    fn serialize_into(&self, buf: &mut alloc::vec::Vec<u8>) -> Result<(), SerializationError> {
        buf.push(self.0);
        Ok(())
    }
}

impl CanonicalDeserialize for GF2 {
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        if data.is_empty() {
            return Err(SerializationError::UnexpectedEnd);
        }
        if data[0] > 1 {
            return Err(SerializationError::InvalidData(
                "GF2 value out of range".into(),
            ));
        }
        Ok((GF2(data[0]), 1))
    }
}

impl Valid for GF2 {
    #[inline]
    fn is_valid(&self) -> bool {
        true
    }
}

// --- Random ---

impl grid_std::UniformRand for GF2 {
    fn rand<R: RngExt + ?Sized>(rng: &mut R) -> Self {
        GF2::new(rng.random::<u64>() as u8)
    }
}

// --- LargeCanonicalRing ---

impl crate::arith::large_modulus::LargeCanonicalRing for GF2 {
    type Canonical = u64;

    fn modulus_canonical() -> u64 {
        2
    }

    fn from_small_u64(value: u64) -> Self {
        GF2::new(value as u8)
    }

    fn from_canonical(value: &u64) -> Self {
        GF2::new(*value as u8)
    }

    fn to_canonical(&self) -> u64 {
        self.0 as u64
    }

    fn try_to_u64(&self) -> Option<u64> {
        Some(self.0 as u64)
    }

    fn try_to_u128(&self) -> Option<u128> {
        Some(self.0 as u128)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use grid_std::UniformRand;

    fn test_rng() -> impl RngExt {
        grid_std::test_rng()
    }

    #[test]
    fn add_sub_identity() {
        let zero = GF2::zero();
        let one = GF2::one();
        assert_eq!(one + zero, one);
        assert_eq!(one + one, zero);
        assert_eq!(one - zero, one);
        assert_eq!(one - one, zero);
        assert_eq!(zero - one, one); // -1 = 1 in char 2
    }

    #[test]
    fn mul_identity() {
        let zero = GF2::zero();
        let one = GF2::one();
        assert_eq!(one * zero, zero);
        assert_eq!(zero * one, zero);
        assert_eq!(one * one, one);
    }

    #[test]
    fn neg_is_identity() {
        assert_eq!(-GF2(0), GF2(0));
        assert_eq!(-GF2(1), GF2(1));
    }

    #[test]
    fn double_is_zero() {
        assert_eq!(GF2(0).double(), GF2(0));
        assert_eq!(GF2(1).double(), GF2(0));
    }

    #[test]
    fn square_is_identity() {
        assert_eq!(GF2(0).square(), GF2(0));
        assert_eq!(GF2(1).square(), GF2(1));
    }

    #[test]
    fn field_inv() {
        use crate::arith::ring::Field;
        assert_eq!(GF2(1).inv(), GF2(1));
    }

    #[test]
    #[should_panic(expected = "division by zero")]
    fn field_inv_zero_panics() {
        use crate::arith::ring::Field;
        let _ = GF2(0).inv();
    }

    #[test]
    fn integer_ring_modulus() {
        assert_eq!(GF2::modulus(), 2);
    }

    #[test]
    fn from_u64_reduces() {
        assert_eq!(GF2::from_u64(0), GF2(0));
        assert_eq!(GF2::from_u64(1), GF2(1));
        assert_eq!(GF2::from_u64(2), GF2(0));
        assert_eq!(GF2::from_u64(3), GF2(1));
        assert_eq!(GF2::from_u64(100), GF2(0));
    }

    #[test]
    fn to_u64() {
        assert_eq!(GF2(0).to_u64(), 0);
        assert_eq!(GF2(1).to_u64(), 1);
    }

    #[test]
    fn lossy_l2_value() {
        assert_eq!(GF2(0).lossy_l2_value(), 0.0);
        assert_eq!(GF2(1).lossy_l2_value(), 1.0);
    }

    #[test]
    fn slice_add() {
        let mut dst = vec![GF2(0), GF2(1), GF2(0), GF2(1), GF2(1), GF2(0)];
        let src = vec![GF2(1), GF2(1), GF2(0), GF2(0), GF2(1), GF2(0)];
        GF2::add_assign_slice(&mut dst, &src);
        assert_eq!(dst, vec![GF2(1), GF2(0), GF2(0), GF2(1), GF2(0), GF2(0)]);
    }

    #[test]
    fn slice_add_long() {
        let n = 1024;
        let mut dst = vec![GF2(0); n];
        let src = vec![GF2(1); n];
        GF2::add_assign_slice(&mut dst, &src);
        assert!(dst.iter().all(|x| x.0 == 1));
    }

    #[test]
    fn slice_sub_equals_add() {
        let n = 256;
        let mut a = (0..n).map(|i| GF2::new(i as u8)).collect::<Vec<_>>();
        let b = (0..n).map(|i| GF2::new((i * 3) as u8)).collect::<Vec<_>>();
        let mut a_sub = a.clone();
        GF2::add_assign_slice(&mut a, &b);
        GF2::sub_assign_slice(&mut a_sub, &b);
        assert_eq!(a, a_sub);
    }

    #[test]
    fn slice_mul() {
        let mut dst = vec![GF2(1), GF2(0), GF2(1), GF2(1), GF2(0)];
        let rhs = vec![GF2(1), GF2(1), GF2(0), GF2(1), GF2(1)];
        GF2::pointwise_mul_assign_slice(&mut dst, &rhs);
        assert_eq!(dst, vec![GF2(1), GF2(0), GF2(0), GF2(1), GF2(0)]);
    }

    #[test]
    fn slice_mul_long() {
        let n = 1024;
        let mut dst = vec![GF2(1); n];
        let rhs = vec![GF2(1); n];
        GF2::pointwise_mul_assign_slice(&mut dst, &rhs);
        assert!(dst.iter().all(|x| x.0 == 1));
    }

    #[test]
    fn scalar_mul_zero() {
        let mut dst = vec![GF2(1), GF2(0), GF2(1), GF2(1)];
        GF2::scalar_mul_slice(&mut dst, &GF2(0));
        assert!(dst.iter().all(|x| x.0 == 0));
    }

    #[test]
    fn scalar_mul_one_noop() {
        let orig = vec![GF2(1), GF2(0), GF2(1), GF2(0)];
        let mut dst = orig.clone();
        GF2::scalar_mul_slice(&mut dst, &GF2(1));
        assert_eq!(dst, orig);
    }

    #[test]
    fn dot_product_small() {
        let a = vec![GF2(1), GF2(0), GF2(1)];
        let b = vec![GF2(1), GF2(1), GF2(0)];
        // <a,b> = 1*1 + 0*1 + 1*0 = 1 (mod 2)
        assert_eq!(GF2::dot_product(&a, &b), GF2(1));
    }

    #[test]
    fn dot_product_orthogonal() {
        let a = vec![GF2(1), GF2(1), GF2(1), GF2(1)];
        let b = vec![GF2(1), GF2(1), GF2(1), GF2(1)];
        // 1*1 + 1*1 + 1*1 + 1*1 = 4 = 0 mod 2
        assert_eq!(GF2::dot_product(&a, &b), GF2(0));
    }

    #[test]
    fn dot_product_long() {
        let n = 2048;
        let a = vec![GF2(1); n];
        let b = vec![GF2(1); n];
        // 2048 ones = 2048 mod 2 = 0
        assert_eq!(GF2::dot_product(&a, &b), GF2(0));
    }

    #[test]
    fn dot_product_long_odd() {
        let n = 2049;
        let a = vec![GF2(1); n];
        let b = vec![GF2(1); n];
        assert_eq!(GF2::dot_product(&a, &b), GF2(1));
    }

    #[test]
    fn add_assign_scaled_zero() {
        let mut dst = vec![GF2(1), GF2(0), GF2(1)];
        let src = vec![GF2(1), GF2(1), GF2(0)];
        let orig = dst.clone();
        GF2::add_assign_scaled_slice(&mut dst, &src, &GF2(0));
        assert_eq!(dst, orig);
    }

    #[test]
    fn add_assign_scaled_one() {
        let mut dst = vec![GF2(0), GF2(1), GF2(0)];
        let src = vec![GF2(1), GF2(0), GF2(1)];
        GF2::add_assign_scaled_slice(&mut dst, &src, &GF2(1));
        assert_eq!(dst, vec![GF2(1), GF2(1), GF2(1)]);
    }

    #[test]
    fn serialize_roundtrip() {
        let val = GF2(1);
        let bytes = val.serialize().unwrap();
        assert_eq!(bytes.len(), 1);
        assert_eq!(bytes[0], 1);
        let decoded = GF2::deserialize_exact(&bytes).unwrap();
        assert_eq!(decoded, val);
    }

    #[test]
    fn serialize_zero() {
        let bytes = GF2(0).serialize().unwrap();
        assert_eq!(bytes, vec![0]);
    }

    #[test]
    fn deserialize_rejects_non_canonical() {
        // byte values > 1 are rejected
        for v in [2u8, 3, 255] {
            let bytes = vec![v];
            assert!(
                GF2::deserialize_exact(&bytes).is_err(),
                "byte {v} should be rejected"
            );
        }
    }

    #[test]
    fn uniform_rand_produces_valid() {
        let mut rng = test_rng();
        for _ in 0..1000 {
            let v: GF2 = UniformRand::rand(&mut rng);
            assert!(v.0 == 0 || v.0 == 1);
        }
    }

    #[test]
    fn ref_ref_operators() {
        #![allow(clippy::op_ref)]

        let a = GF2(1);
        let b = GF2(1);
        assert_eq!(&a + &b, GF2(0));
        assert_eq!(&a + b, GF2(0));
        assert_eq!(&a * &b, GF2(1));
        assert_eq!(&a * b, GF2(1));
    }

    #[test]
    fn large_canonical_roundtrip() {
        use crate::arith::large_modulus::LargeCanonicalRing;
        let val = GF2(1);
        let canon = val.to_canonical();
        assert_eq!(canon, 1);
        assert_eq!(GF2::from_canonical(&canon), val);
        assert_eq!(GF2::from_canonical(&2u64), GF2(0));
    }
}
