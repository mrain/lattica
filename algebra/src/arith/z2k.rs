//! `Z2K` — `Z_{2^k}` where the modulus is a power of two.
//!
//! Uses bitwise masking for reduction. Does NOT support NTT (no primitive roots of unity).
//! Used in some FHE schemes and gadget decompositions.

use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::ring::{IntegerRing, Ring};
use crate::simd::dispatch::{Backend, selected_backend};
use grid_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError};

/// A modular integer in `Z_{2^K}`.
///
/// All arithmetic is performed modulo `2^K` using bitwise masking.
/// `K` must be between 1 and 64 (inclusive).
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Z2K<const K: u32> {
    val: u64,
}

impl<const K: u32> Z2K<K> {
    /// The bitmask: `2^K - 1`.
    const MASK: u64 = {
        assert!(K >= 1 && K <= 64, "K must be between 1 and 64");
        if K == 64 { u64::MAX } else { (1u64 << K) - 1 }
    };

    /// Minimum bytes needed to represent values in [0, 2^K).
    const SERIALIZED_BYTES: usize = (K as usize).div_ceil(8);

    /// Create from a raw value (will be masked).
    #[inline]
    pub const fn new(val: u64) -> Self {
        Self {
            val: val & Self::MASK,
        }
    }

    /// Get the underlying value.
    #[inline]
    pub const fn value(&self) -> u64 {
        self.val
    }

    fn values_mut(slice: &mut [Self]) -> &mut [u64] {
        // SAFETY: `Z2K<K>` is `#[repr(transparent)]` over `u64`.
        unsafe { core::slice::from_raw_parts_mut(slice.as_mut_ptr().cast::<u64>(), slice.len()) }
    }

    fn values(slice: &[Self]) -> &[u64] {
        // SAFETY: `Z2K<K>` is `#[repr(transparent)]` over `u64`.
        unsafe { core::slice::from_raw_parts(slice.as_ptr().cast::<u64>(), slice.len()) }
    }
}

impl<const K: u32> core::fmt::Debug for Z2K<K> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Z_2^{}({})", K, self.val)
    }
}

impl<const K: u32> core::fmt::Display for Z2K<K> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.val)
    }
}

// --- Operator impls ---

impl<const K: u32> Add for Z2K<K> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.val.wrapping_add(rhs.val))
    }
}

impl<const K: u32> Add<&Self> for Z2K<K> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: &Self) -> Self {
        self + *rhs
    }
}

impl<const K: u32> AddAssign for Z2K<K> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const K: u32> AddAssign<&Self> for Z2K<K> {
    #[inline]
    fn add_assign(&mut self, rhs: &Self) {
        *self = *self + *rhs;
    }
}

impl<const K: u32> Sub for Z2K<K> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.val.wrapping_sub(rhs.val))
    }
}

impl<const K: u32> Sub<&Self> for Z2K<K> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: &Self) -> Self {
        self - *rhs
    }
}

impl<const K: u32> SubAssign for Z2K<K> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const K: u32> SubAssign<&Self> for Z2K<K> {
    #[inline]
    fn sub_assign(&mut self, rhs: &Self) {
        *self = *self - *rhs;
    }
}

impl<const K: u32> Mul for Z2K<K> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::new(self.val.wrapping_mul(rhs.val))
    }
}

impl<const K: u32> Mul<&Self> for Z2K<K> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: &Self) -> Self {
        self * *rhs
    }
}

impl<const K: u32> MulAssign for Z2K<K> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const K: u32> MulAssign<&Self> for Z2K<K> {
    #[inline]
    fn mul_assign(&mut self, rhs: &Self) {
        *self = *self * *rhs;
    }
}

impl<const K: u32> Add<Self> for &Z2K<K> {
    type Output = Z2K<K>;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        *self + *rhs
    }
}

impl<const K: u32> Add<Z2K<K>> for &Z2K<K> {
    type Output = Z2K<K>;
    #[inline]
    fn add(self, rhs: Z2K<K>) -> Self::Output {
        *self + rhs
    }
}

impl<const K: u32> Sub<Self> for &Z2K<K> {
    type Output = Z2K<K>;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        *self - *rhs
    }
}

impl<const K: u32> Sub<Z2K<K>> for &Z2K<K> {
    type Output = Z2K<K>;
    #[inline]
    fn sub(self, rhs: Z2K<K>) -> Self::Output {
        *self - rhs
    }
}

impl<const K: u32> Mul<Self> for &Z2K<K> {
    type Output = Z2K<K>;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        *self * *rhs
    }
}

impl<const K: u32> Mul<Z2K<K>> for &Z2K<K> {
    type Output = Z2K<K>;
    #[inline]
    fn mul(self, rhs: Z2K<K>) -> Self::Output {
        *self * rhs
    }
}

impl<const K: u32> Neg for Z2K<K> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        if self.val == 0 {
            self
        } else {
            Self::new(self.val.wrapping_neg())
        }
    }
}

// --- Trait impls ---

impl<const K: u32> Ring for Z2K<K> {
    #[inline]
    fn zero() -> Self {
        Self { val: 0 }
    }

    #[inline]
    fn one() -> Self {
        Self { val: 1 }
    }

    fn add_assign_slice(dst: &mut [Self], src: &[Self]) {
        assert_eq!(dst.len(), src.len(), "slice lengths must match");
        #[cfg(target_arch = "x86_64")]
        if matches!(selected_backend(), Backend::Avx2) {
            // SAFETY: backend selection only enables this path when AVX2 is available.
            unsafe {
                crate::simd::avx2::u64_arith::add_assign_u64_masked(
                    Self::values_mut(dst),
                    Self::values(src),
                    Self::MASK,
                );
            }
            return;
        }
        #[cfg(target_arch = "aarch64")]
        if matches!(selected_backend(), Backend::Neon) {
            // SAFETY: backend selection only enables this path when NEON is available.
            unsafe {
                crate::simd::aarch64::add_assign_u64_masked(
                    Self::values_mut(dst),
                    Self::values(src),
                    Self::MASK,
                );
            }
            return;
        }
        for (lhs, rhs) in dst.iter_mut().zip(src.iter()) {
            *lhs += *rhs;
        }
    }

    fn sub_assign_slice(dst: &mut [Self], src: &[Self]) {
        assert_eq!(dst.len(), src.len(), "slice lengths must match");
        #[cfg(target_arch = "x86_64")]
        if matches!(selected_backend(), Backend::Avx2) {
            // SAFETY: backend selection only enables this path when AVX2 is available.
            unsafe {
                crate::simd::avx2::u64_arith::sub_assign_u64_masked(
                    Self::values_mut(dst),
                    Self::values(src),
                    Self::MASK,
                );
            }
            return;
        }
        #[cfg(target_arch = "aarch64")]
        if matches!(selected_backend(), Backend::Neon) {
            // SAFETY: backend selection only enables this path when NEON is available.
            unsafe {
                crate::simd::aarch64::sub_assign_u64_masked(
                    Self::values_mut(dst),
                    Self::values(src),
                    Self::MASK,
                );
            }
            return;
        }
        for (lhs, rhs) in dst.iter_mut().zip(src.iter()) {
            *lhs -= *rhs;
        }
    }

    fn scalar_mul_slice(dst: &mut [Self], scalar: &Self) {
        #[cfg(target_arch = "x86_64")]
        if K <= 32 && matches!(selected_backend(), Backend::Avx2) {
            // SAFETY: backend selection only enables this path when AVX2 is available.
            unsafe {
                crate::simd::avx2::u64_arith::scalar_mul_u64_low32_masked(
                    Self::values_mut(dst),
                    scalar.value(),
                    Self::MASK,
                );
            }
            return;
        }
        #[cfg(target_arch = "aarch64")]
        if K <= 32 && matches!(selected_backend(), Backend::Neon) {
            // SAFETY: backend selection only enables this path when NEON is available.
            unsafe {
                crate::simd::aarch64::scalar_mul_u64_low32_masked(
                    Self::values_mut(dst),
                    scalar.value(),
                    Self::MASK,
                );
            }
            return;
        }
        for value in dst.iter_mut() {
            *value *= *scalar;
        }
    }

    fn pointwise_mul_assign_slice(dst: &mut [Self], rhs: &[Self]) {
        assert_eq!(dst.len(), rhs.len(), "slice lengths must match");
        #[cfg(target_arch = "x86_64")]
        if K <= 32 && matches!(selected_backend(), Backend::Avx2) {
            // SAFETY: backend selection only enables this path when AVX2 is available.
            unsafe {
                crate::simd::avx2::u64_arith::mul_assign_u64_low32_masked(
                    Self::values_mut(dst),
                    Self::values(rhs),
                    Self::MASK,
                );
            }
            return;
        }
        #[cfg(target_arch = "aarch64")]
        if K <= 32 && matches!(selected_backend(), Backend::Neon) {
            // SAFETY: backend selection only enables this path when NEON is available.
            unsafe {
                crate::simd::aarch64::mul_assign_u64_low32_masked(
                    Self::values_mut(dst),
                    Self::values(rhs),
                    Self::MASK,
                );
            }
            return;
        }
        for (lhs, rhs) in dst.iter_mut().zip(rhs.iter()) {
            *lhs *= *rhs;
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

impl<const K: u32> IntegerRing for Z2K<K> {
    type Canonical = u64;

    #[inline]
    fn modulus_canonical() -> u64 {
        if K == 64 {
            0 // 2^64 doesn't fit in u64.
        } else {
            1u64 << K
        }
    }

    #[inline]
    fn from_small_u64(val: u64) -> Self {
        Self::new(val)
    }

    #[inline]
    fn from_canonical(value: &u64) -> Self {
        Self::new(*value)
    }

    #[inline]
    fn to_canonical(&self) -> u64 {
        self.val
    }

    #[inline]
    fn try_to_u64(&self) -> Option<u64> {
        Some(self.val)
    }

    #[inline]
    fn try_to_u128(&self) -> Option<u128> {
        Some(self.val as u128)
    }

    #[inline]
    fn lossy_l2_value(&self) -> f64 {
        // Signed wrap-around: values in `[2^(K-1), 2^K)` → `[-2^(K-1), 0)`
        if K == 64 {
            self.val as i64 as f64
        } else {
            let half: u64 = 1 << (K - 1);
            if self.val >= half {
                (self.val as i128 - (1u128 << K) as i128) as f64
            } else {
                self.val as f64
            }
        }
    }

    #[inline]
    fn reduce(&self) -> Self {
        *self // Already masked
    }
}

impl<const K: u32> CanonicalSerialize for Z2K<K> {
    fn serialized_size(&self) -> usize {
        Self::SERIALIZED_BYTES
    }

    fn serialize_into(&self, buf: &mut alloc::vec::Vec<u8>) -> Result<(), SerializationError> {
        let bytes = self.val.to_le_bytes();
        buf.extend_from_slice(&bytes[..Self::SERIALIZED_BYTES]);
        Ok(())
    }
}

impl<const K: u32> CanonicalDeserialize for Z2K<K> {
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        if data.len() < Self::SERIALIZED_BYTES {
            return Err(SerializationError::UnexpectedEnd);
        }
        let mut bytes = [0u8; 8];
        bytes[..Self::SERIALIZED_BYTES].copy_from_slice(&data[..Self::SERIALIZED_BYTES]);
        let raw = u64::from_le_bytes(bytes);
        if raw != (raw & Self::MASK) {
            return Err(SerializationError::InvalidData(alloc::format!(
                "value {raw} is not canonical for modulus 2^{K}"
            )));
        }
        Ok((Self::new(raw), Self::SERIALIZED_BYTES))
    }
}

impl<const K: u32> grid_serialize::Valid for Z2K<K> {
    fn is_valid(&self) -> bool {
        self.val == self.val & Self::MASK
    }
}

impl<const K: u32> grid_std::UniformRand for Z2K<K> {
    fn rand<R: grid_std::rand::RngExt + ?Sized>(rng: &mut R) -> Self {
        let val: u64 = rng.random();
        Self::new(val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arith::ring::tests::test_ring_axioms;
    use core::mem::{align_of, size_of};

    type Z16 = Z2K<4>; // Z_{2^4} = Z_16
    type Z256 = Z2K<8>; // Z_{2^8} = Z_256
    type Z2_64 = Z2K<64>;

    #[test]
    fn test_basic_ops() {
        let a = Z16::from_u64(13);
        let b = Z16::from_u64(10);
        assert_eq!((a + b).to_u64(), 7); // (13 + 10) mod 16 = 7
        assert_eq!((a - b).to_u64(), 3); // (13 - 10) mod 16 = 3
        assert_eq!((b - a).to_u64(), 13); // (10 - 13) mod 16 = -3 mod 16 = 13
        assert_eq!((a * b).to_u64(), 2); // (13 * 10) mod 16 = 130 mod 16 = 2
    }

    #[test]
    fn test_neg() {
        assert_eq!((-Z16::from_u64(0)).to_u64(), 0);
        assert_eq!((-Z16::from_u64(1)).to_u64(), 15);
        assert_eq!((-Z16::from_u64(5)).to_u64(), 11);
    }

    #[test]
    fn test_masking() {
        // Values above 16 should be masked
        assert_eq!(Z16::from_u64(16).to_u64(), 0);
        assert_eq!(Z16::from_u64(17).to_u64(), 1);
        assert_eq!(Z16::from_u64(255).to_u64(), 15);
    }

    #[test]
    fn test_repr_transparent_layout() {
        assert_eq!(size_of::<Z16>(), size_of::<u64>());
        assert_eq!(align_of::<Z16>(), align_of::<u64>());
    }

    #[test]
    fn test_slice_hooks_match_scalar_ops() {
        let lhs = [
            Z16::from_u64(1),
            Z16::from_u64(7),
            Z16::from_u64(15),
            Z16::from_u64(3),
            Z16::from_u64(11),
        ];
        let rhs = [
            Z16::from_u64(2),
            Z16::from_u64(9),
            Z16::from_u64(1),
            Z16::from_u64(14),
            Z16::from_u64(6),
        ];

        let mut add_hook = lhs;
        let mut sub_hook = lhs;
        let add_scalar = core::array::from_fn::<Z16, 5, _>(|i| lhs[i] + rhs[i]);
        let sub_scalar = core::array::from_fn::<Z16, 5, _>(|i| lhs[i] - rhs[i]);

        Z16::add_assign_slice(&mut add_hook, &rhs);
        Z16::sub_assign_slice(&mut sub_hook, &rhs);

        assert_eq!(add_hook, add_scalar);
        assert_eq!(sub_hook, sub_scalar);
    }

    #[test]
    fn test_mul_hooks_match_scalar_ops() {
        let lhs = [
            Z16::from_u64(1),
            Z16::from_u64(7),
            Z16::from_u64(15),
            Z16::from_u64(3),
            Z16::from_u64(11),
        ];
        let rhs = [
            Z16::from_u64(2),
            Z16::from_u64(9),
            Z16::from_u64(1),
            Z16::from_u64(14),
            Z16::from_u64(6),
        ];
        let scalar = Z16::from_u64(5);

        let mut mul_hook = lhs;
        let mut scalar_mul_hook = lhs;
        let mut scaled_add_hook = lhs;
        let mul_scalar = core::array::from_fn::<Z16, 5, _>(|i| lhs[i] * rhs[i]);
        let scalar_mul_scalar = core::array::from_fn::<Z16, 5, _>(|i| lhs[i] * scalar);
        let scaled_add_scalar = core::array::from_fn::<Z16, 5, _>(|i| lhs[i] + rhs[i] * scalar);

        Z16::pointwise_mul_assign_slice(&mut mul_hook, &rhs);
        Z16::scalar_mul_slice(&mut scalar_mul_hook, &scalar);
        Z16::add_assign_scaled_slice(&mut scaled_add_hook, &rhs, &scalar);

        assert_eq!(mul_hook, mul_scalar);
        assert_eq!(scalar_mul_hook, scalar_mul_scalar);
        assert_eq!(scaled_add_hook, scaled_add_scalar);
    }

    #[test]
    fn test_dot_product_hook_matches_scalar_ops() {
        let lhs = [
            Z16::from_u64(1),
            Z16::from_u64(7),
            Z16::from_u64(15),
            Z16::from_u64(3),
            Z16::from_u64(11),
        ];
        let rhs = [
            Z16::from_u64(2),
            Z16::from_u64(9),
            Z16::from_u64(1),
            Z16::from_u64(14),
            Z16::from_u64(6),
        ];

        let dot_scalar = lhs
            .iter()
            .zip(rhs.iter())
            .fold(Z16::zero(), |mut acc, (lhs, rhs)| {
                acc += *lhs * rhs;
                acc
            });

        assert_eq!(Z16::dot_product(&lhs, &rhs), dot_scalar);
    }

    #[test]
    fn test_ring_axioms_z16() {
        let a = Z16::from_u64(3);
        let b = Z16::from_u64(7);
        let c = Z16::from_u64(11);
        test_ring_axioms(a, b, c);
    }

    #[test]
    fn test_ring_axioms_z256() {
        let a = Z256::from_u64(37);
        let b = Z256::from_u64(191);
        let c = Z256::from_u64(100);
        test_ring_axioms(a, b, c);
    }

    #[test]
    fn test_reduce_idempotent() {
        let a = Z16::from_u64(13);
        assert_eq!(a.reduce(), a);
    }

    #[test]
    fn test_uniform_rand() {
        let mut rng = grid_std::test_rng();
        for _ in 0..100 {
            let a = <Z16 as grid_std::UniformRand>::rand(&mut rng);
            assert!(a.to_u64() < 16);
        }
    }
    #[test]
    fn test_serialize_round_trip() {
        use grid_serialize::{CanonicalDeserialize, CanonicalSerialize};
        for val in 0..16u64 {
            let a = Z16::from_u64(val);
            let bytes = a.serialize().unwrap();
            let (b, _) = Z16::deserialize(&bytes).unwrap();
            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_deserialize_rejects_too_short() {
        // Less than SERIALIZED_BYTES (Z16=Z2K<4> needs 1 byte)
        let err = Z16::deserialize(&[]).unwrap_err();
        assert!(matches!(err, SerializationError::UnexpectedEnd));
    }

    #[test]
    fn test_power_of_two_integer_ring_round_trip() {
        let value = Z16::from_small_u64(31);
        assert_eq!(value.to_canonical(), 15);
        assert_eq!(Z16::modulus_canonical(), 16);
        assert_eq!(Z16::from_canonical(&18).to_canonical(), 2);
        assert_eq!(value.try_to_u64(), Some(15));
        assert_eq!(value.try_to_u128(), Some(15));
    }

    #[test]
    fn test_power_of_two_integer_ring_handles_2_to_64_modulus() {
        // `IntegerRing::Canonical = u64` for small rings; `0` is the existing sentinel for
        // the modulus 2^64, which cannot be represented as a `u64`.
        assert_eq!(Z2_64::modulus_canonical(), 0);

        let canonical = 7u64;
        let reduced = Z2_64::from_canonical(&canonical);
        assert_eq!(reduced.to_canonical(), 7);
    }
}
