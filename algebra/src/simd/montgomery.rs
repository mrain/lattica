//! Internal SIMD qualification and dispatch for Montgomery-backed prime fields.

use crate::arith::ntt::{
    NttError, NttPlan, bit_reverse_permute, scalar_ntt_forward_with_plan,
    scalar_ntt_inverse_with_plan,
};
use crate::arith::prime::PrimeField;
use crate::arith::limb::UintLimb;
use crate::simd::dispatch::{Backend, selected_backend};

const AVX2_ADD_SUB_MAX_Q: u64 = 1u64 << 62;

mod sealed {
    pub trait Sealed {}
}

impl<const Q: u64, L: UintLimb> sealed::Sealed for PrimeField<Q, L> {}

/// Internal capability boundary for SIMD-qualified Montgomery prime fields.
#[allow(dead_code)]
pub(crate) trait MontgomeryUintSimd: sealed::Sealed + Sized {
    /// The limb type backing this field.
    type Limb: UintLimb;

    /// Whether AVX2 add/sub is sound for this modulus.
    const AVX2_ADD_SUB_QUALIFIED: bool;
    /// Whether NEON add/sub is sound for this modulus.
    const NEON_ADD_SUB_QUALIFIED: bool;
    /// Whether AVX2 Montgomery multiply is sound for this modulus.
    const AVX2_MONTGOMERY_QUALIFIED: bool;
    /// Whether NEON Montgomery multiply is sound for this modulus.
    const NEON_MONTGOMERY_QUALIFIED: bool;
    /// Whether AVX2 NTT butterflies are sound for this modulus.
    const AVX2_NTT_QUALIFIED: bool;
    /// Whether NEON NTT butterflies are sound for this modulus.
    const NEON_NTT_QUALIFIED: bool;

    /// Borrow the raw Montgomery limbs behind a prime-field slice.
    fn values(slice: &[Self]) -> &[Self::Limb];
    /// Borrow the raw Montgomery limbs behind a mutable prime-field slice.
    fn values_mut(slice: &mut [Self]) -> &mut [Self::Limb];
}

impl<const Q: u64, L: UintLimb> MontgomeryUintSimd for PrimeField<Q, L> {
    type Limb = L;

    // SIMD qualification: all limb sizes are AVX2-qualified.
    const AVX2_ADD_SUB_QUALIFIED: bool = (L::BITS == 64 && Q <= AVX2_ADD_SUB_MAX_Q)
        || L::BITS == 16
        || L::BITS == 8
        || L::BITS == 32;
    const NEON_ADD_SUB_QUALIFIED: bool =
        (L::BITS == 64 && Q < (1u64 << 63)) || L::BITS == 16 || L::BITS == 8 || L::BITS == 32;
    const AVX2_MONTGOMERY_QUALIFIED: bool =
        L::BITS == 64 || L::BITS == 16 || L::BITS == 8 || L::BITS == 32;
    const NEON_MONTGOMERY_QUALIFIED: bool =
        L::BITS == 16 || L::BITS == 8 || L::BITS == 32;
    const AVX2_NTT_QUALIFIED: bool = Self::AVX2_MONTGOMERY_QUALIFIED;
    const NEON_NTT_QUALIFIED: bool = Self::NEON_MONTGOMERY_QUALIFIED;

    #[inline]
    fn values(slice: &[Self]) -> &[L] {
        PrimeField::<Q, L>::values(slice)
    }

    #[inline]
    fn values_mut(slice: &mut [Self]) -> &mut [L] {
        PrimeField::<Q, L>::values_mut(slice)
    }
}

/// SAFETY: Caller must ensure L::BITS == 64 (i.e. L = u64) before calling.
#[inline(always)]
unsafe fn as_u64_slice<L: UintLimb>(s: &[L]) -> &[u64] {
    // When L::BITS == 64, the sealed trait guarantees L = u64.
    unsafe { core::slice::from_raw_parts(s.as_ptr().cast::<u64>(), s.len()) }
}

/// SAFETY: Caller must ensure L::BITS == 64 (i.e. L = u64) before calling.
#[inline(always)]
unsafe fn as_u64_slice_mut<L: UintLimb>(s: &mut [L]) -> &mut [u64] {
    unsafe { core::slice::from_raw_parts_mut(s.as_mut_ptr().cast::<u64>(), s.len()) }
}

/// SAFETY: Caller must ensure L::BITS == 16 (i.e. L = u16) before calling.
#[inline(always)]
unsafe fn as_u16_slice<L: UintLimb>(s: &[L]) -> &[u16] {
    unsafe { core::slice::from_raw_parts(s.as_ptr().cast::<u16>(), s.len()) }
}

/// SAFETY: Caller must ensure L::BITS == 16 (i.e. L = u16) before calling.
#[inline(always)]
unsafe fn as_u16_slice_mut<L: UintLimb>(s: &mut [L]) -> &mut [u16] {
    unsafe { core::slice::from_raw_parts_mut(s.as_mut_ptr().cast::<u16>(), s.len()) }
}

/// SAFETY: Caller must ensure L::BITS == 8 (i.e. L = u8) before calling.
#[inline(always)]
unsafe fn as_u8_slice<L: UintLimb>(s: &[L]) -> &[u8] {
    unsafe { core::slice::from_raw_parts(s.as_ptr().cast::<u8>(), s.len()) }
}

/// SAFETY: Caller must ensure L::BITS == 8 (i.e. L = u8) before calling.
#[inline(always)]
unsafe fn as_u8_slice_mut<L: UintLimb>(s: &mut [L]) -> &mut [u8] {
    unsafe { core::slice::from_raw_parts_mut(s.as_mut_ptr().cast::<u8>(), s.len()) }
}

/// SAFETY: Caller must ensure L::BITS == 32 (i.e. L = u32) before calling.
#[inline(always)]
unsafe fn as_u32_slice<L: UintLimb>(s: &[L]) -> &[u32] {
    unsafe { core::slice::from_raw_parts(s.as_ptr().cast::<u32>(), s.len()) }
}

/// SAFETY: Caller must ensure L::BITS == 32 (i.e. L = u32) before calling.
#[inline(always)]
unsafe fn as_u32_slice_mut<L: UintLimb>(s: &mut [L]) -> &mut [u32] {
    unsafe { core::slice::from_raw_parts_mut(s.as_mut_ptr().cast::<u32>(), s.len()) }
}

// ---------------------------------------------------------------------------
// Dispatch functions
// ---------------------------------------------------------------------------

#[inline]
pub(crate) fn add_assign<const Q: u64, L: UintLimb>(
    dst: &mut [PrimeField<Q, L>],
    src: &[PrimeField<Q, L>],
) {
    assert_eq!(dst.len(), src.len(), "slice lengths must match");
    let backend = selected_backend();

    #[cfg(target_arch = "x86_64")]
    if matches!(backend, Backend::Avx2)
        && L::BITS == 16
        && <PrimeField<Q, L> as MontgomeryUintSimd>::AVX2_ADD_SUB_QUALIFIED
    {
        unsafe {
            crate::simd::avx2::u16_arith::add_assign_prime_u16(
                as_u16_slice_mut(<PrimeField<Q, L> as MontgomeryUintSimd>::values_mut(dst)),
                as_u16_slice(<PrimeField<Q, L> as MontgomeryUintSimd>::values(src)),
                Q,
            );
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if matches!(backend, Backend::Avx2)
        && L::BITS == 8
        && <PrimeField<Q, L> as MontgomeryUintSimd>::AVX2_ADD_SUB_QUALIFIED
    {
        unsafe {
            crate::simd::avx2::u8_arith::add_assign_prime_u8(
                as_u8_slice_mut(<PrimeField<Q, L> as MontgomeryUintSimd>::values_mut(dst)),
                as_u8_slice(<PrimeField<Q, L> as MontgomeryUintSimd>::values(src)),
                Q,
            );
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if matches!(backend, Backend::Avx2)
        && L::BITS == 32
        && <PrimeField<Q, L> as MontgomeryUintSimd>::AVX2_ADD_SUB_QUALIFIED
    {
        unsafe {
            crate::simd::avx2::u32_arith::add_assign_prime_u32(
                as_u32_slice_mut(<PrimeField<Q, L> as MontgomeryUintSimd>::values_mut(dst)),
                as_u32_slice(<PrimeField<Q, L> as MontgomeryUintSimd>::values(src)),
                Q,
            );
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if matches!(backend, Backend::Avx2)
        && L::BITS == 64
        && <PrimeField<Q, L> as MontgomeryUintSimd>::AVX2_ADD_SUB_QUALIFIED
    {
        unsafe {
            crate::simd::avx2::u64_arith::add_assign_prime_u64(
                as_u64_slice_mut(<PrimeField<Q, L> as MontgomeryUintSimd>::values_mut(dst)),
                as_u64_slice(<PrimeField<Q, L> as MontgomeryUintSimd>::values(src)),
                Q,
            );
        }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    if matches!(backend, Backend::Neon)
        && L::BITS == 16
        && <PrimeField<Q, L> as MontgomeryUintSimd>::NEON_ADD_SUB_QUALIFIED
    {
        unsafe {
            crate::simd::aarch64::u16_arith::add_assign_prime_u16(
                as_u16_slice_mut(<PrimeField<Q, L> as MontgomeryUintSimd>::values_mut(dst)),
                as_u16_slice(<PrimeField<Q, L> as MontgomeryUintSimd>::values(src)),
                Q,
            );
        }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    if matches!(backend, Backend::Neon)
        && L::BITS == 8
        && <PrimeField<Q, L> as MontgomeryUintSimd>::NEON_ADD_SUB_QUALIFIED
    {
        unsafe {
            crate::simd::aarch64::u8_arith::add_assign_prime_u8(
                as_u8_slice_mut(<PrimeField<Q, L> as MontgomeryUintSimd>::values_mut(dst)),
                as_u8_slice(<PrimeField<Q, L> as MontgomeryUintSimd>::values(src)),
                Q,
            );
        }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    if matches!(backend, Backend::Neon)
        && L::BITS == 32
        && <PrimeField<Q, L> as MontgomeryUintSimd>::NEON_ADD_SUB_QUALIFIED
    {
        unsafe {
            crate::simd::aarch64::u32_arith::add_assign_prime_u32(
                as_u32_slice_mut(<PrimeField<Q, L> as MontgomeryUintSimd>::values_mut(dst)),
                as_u32_slice(<PrimeField<Q, L> as MontgomeryUintSimd>::values(src)),
                Q,
            );
        }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    if matches!(backend, Backend::Neon)
        && <PrimeField<Q, L> as MontgomeryUintSimd>::NEON_ADD_SUB_QUALIFIED
    {
        unsafe {
            crate::simd::aarch64::add_assign_prime_u64(
                as_u64_slice_mut(<PrimeField<Q, L> as MontgomeryUintSimd>::values_mut(dst)),
                as_u64_slice(<PrimeField<Q, L> as MontgomeryUintSimd>::values(src)),
                Q,
            );
        }
        return;
    }

    for (lhs, rhs) in dst.iter_mut().zip(src.iter()) {
        *lhs += *rhs;
    }
}

#[inline]
pub(crate) fn sub_assign<const Q: u64, L: UintLimb>(
    dst: &mut [PrimeField<Q, L>],
    src: &[PrimeField<Q, L>],
) {
    assert_eq!(dst.len(), src.len(), "slice lengths must match");
    let backend = selected_backend();

    #[cfg(target_arch = "x86_64")]
    if matches!(backend, Backend::Avx2)
        && L::BITS == 16
        && <PrimeField<Q, L> as MontgomeryUintSimd>::AVX2_ADD_SUB_QUALIFIED
    {
        unsafe {
            crate::simd::avx2::u16_arith::sub_assign_prime_u16(
                as_u16_slice_mut(<PrimeField<Q, L> as MontgomeryUintSimd>::values_mut(dst)),
                as_u16_slice(<PrimeField<Q, L> as MontgomeryUintSimd>::values(src)),
                Q,
            );
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if matches!(backend, Backend::Avx2)
        && L::BITS == 8
        && <PrimeField<Q, L> as MontgomeryUintSimd>::AVX2_ADD_SUB_QUALIFIED
    {
        unsafe {
            crate::simd::avx2::u8_arith::sub_assign_prime_u8(
                as_u8_slice_mut(<PrimeField<Q, L> as MontgomeryUintSimd>::values_mut(dst)),
                as_u8_slice(<PrimeField<Q, L> as MontgomeryUintSimd>::values(src)),
                Q,
            );
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if matches!(backend, Backend::Avx2)
        && L::BITS == 32
        && <PrimeField<Q, L> as MontgomeryUintSimd>::AVX2_ADD_SUB_QUALIFIED
    {
        unsafe {
            crate::simd::avx2::u32_arith::sub_assign_prime_u32(
                as_u32_slice_mut(<PrimeField<Q, L> as MontgomeryUintSimd>::values_mut(dst)),
                as_u32_slice(<PrimeField<Q, L> as MontgomeryUintSimd>::values(src)),
                Q,
            );
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if matches!(backend, Backend::Avx2)
        && L::BITS == 64
        && <PrimeField<Q, L> as MontgomeryUintSimd>::AVX2_ADD_SUB_QUALIFIED
    {
        unsafe {
            crate::simd::avx2::u64_arith::sub_assign_prime_u64(
                as_u64_slice_mut(<PrimeField<Q, L> as MontgomeryUintSimd>::values_mut(dst)),
                as_u64_slice(<PrimeField<Q, L> as MontgomeryUintSimd>::values(src)),
                Q,
            );
        }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    if matches!(backend, Backend::Neon)
        && L::BITS == 16
        && <PrimeField<Q, L> as MontgomeryUintSimd>::NEON_ADD_SUB_QUALIFIED
    {
        unsafe {
            crate::simd::aarch64::u16_arith::sub_assign_prime_u16(
                as_u16_slice_mut(<PrimeField<Q, L> as MontgomeryUintSimd>::values_mut(dst)),
                as_u16_slice(<PrimeField<Q, L> as MontgomeryUintSimd>::values(src)),
                Q,
            );
        }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    if matches!(backend, Backend::Neon)
        && L::BITS == 8
        && <PrimeField<Q, L> as MontgomeryUintSimd>::NEON_ADD_SUB_QUALIFIED
    {
        unsafe {
            crate::simd::aarch64::u8_arith::sub_assign_prime_u8(
                as_u8_slice_mut(<PrimeField<Q, L> as MontgomeryUintSimd>::values_mut(dst)),
                as_u8_slice(<PrimeField<Q, L> as MontgomeryUintSimd>::values(src)),
                Q,
            );
        }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    if matches!(backend, Backend::Neon)
        && L::BITS == 32
        && <PrimeField<Q, L> as MontgomeryUintSimd>::NEON_ADD_SUB_QUALIFIED
    {
        unsafe {
            crate::simd::aarch64::u32_arith::sub_assign_prime_u32(
                as_u32_slice_mut(<PrimeField<Q, L> as MontgomeryUintSimd>::values_mut(dst)),
                as_u32_slice(<PrimeField<Q, L> as MontgomeryUintSimd>::values(src)),
                Q,
            );
        }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    if matches!(backend, Backend::Neon)
        && <PrimeField<Q, L> as MontgomeryUintSimd>::NEON_ADD_SUB_QUALIFIED
    {
        unsafe {
            crate::simd::aarch64::sub_assign_prime_u64(
                as_u64_slice_mut(<PrimeField<Q, L> as MontgomeryUintSimd>::values_mut(dst)),
                as_u64_slice(<PrimeField<Q, L> as MontgomeryUintSimd>::values(src)),
                Q,
            );
        }
        return;
    }

    for (lhs, rhs) in dst.iter_mut().zip(src.iter()) {
        *lhs -= *rhs;
    }
}

#[inline]
pub(crate) fn scalar_mul<const Q: u64, L: UintLimb>(
    dst: &mut [PrimeField<Q, L>],
    scalar: &PrimeField<Q, L>,
) {
    let backend = selected_backend();

    #[cfg(target_arch = "x86_64")]
    if matches!(backend, Backend::Avx2)
        && L::BITS == 16
        && <PrimeField<Q, L> as MontgomeryUintSimd>::AVX2_MONTGOMERY_QUALIFIED
    {
        unsafe {
            crate::simd::avx2::u16_arith::scalar_mul_prime_montgomery_u16::<Q>(
                as_u16_slice_mut(<PrimeField<Q, L> as MontgomeryUintSimd>::values_mut(dst)),
                scalar.raw().to_u64(),
            );
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if matches!(backend, Backend::Avx2)
        && L::BITS == 8
        && <PrimeField<Q, L> as MontgomeryUintSimd>::AVX2_MONTGOMERY_QUALIFIED
    {
        unsafe {
            crate::simd::avx2::u8_arith::scalar_mul_prime_montgomery_u8::<Q>(
                as_u8_slice_mut(<PrimeField<Q, L> as MontgomeryUintSimd>::values_mut(dst)),
                scalar.raw().to_u64(),
            );
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if matches!(backend, Backend::Avx2)
        && L::BITS == 32
        && <PrimeField<Q, L> as MontgomeryUintSimd>::AVX2_MONTGOMERY_QUALIFIED
    {
        unsafe {
            crate::simd::avx2::u32_arith::scalar_mul_prime_montgomery_u32::<Q>(
                as_u32_slice_mut(<PrimeField<Q, L> as MontgomeryUintSimd>::values_mut(dst)),
                scalar.raw().to_u64(),
            );
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if matches!(backend, Backend::Avx2)
        && L::BITS == 64
        && <PrimeField<Q, L> as MontgomeryUintSimd>::AVX2_MONTGOMERY_QUALIFIED
    {
        unsafe {
            crate::simd::avx2::u64_arith::scalar_mul_prime_montgomery_u64::<Q>(
                as_u64_slice_mut(<PrimeField<Q, L> as MontgomeryUintSimd>::values_mut(dst)),
                scalar.raw().to_u64(),
            );
        }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    if matches!(backend, Backend::Neon)
        && L::BITS == 16
        && <PrimeField<Q, L> as MontgomeryUintSimd>::NEON_MONTGOMERY_QUALIFIED
    {
        unsafe {
            crate::simd::aarch64::u16_arith::scalar_mul_prime_montgomery_u16::<Q>(
                as_u16_slice_mut(<PrimeField<Q, L> as MontgomeryUintSimd>::values_mut(dst)),
                scalar.raw().to_u64(),
            );
        }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    if matches!(backend, Backend::Neon)
        && L::BITS == 8
        && <PrimeField<Q, L> as MontgomeryUintSimd>::NEON_MONTGOMERY_QUALIFIED
    {
        unsafe {
            crate::simd::aarch64::u8_arith::scalar_mul_prime_montgomery_u8::<Q>(
                as_u8_slice_mut(<PrimeField<Q, L> as MontgomeryUintSimd>::values_mut(dst)),
                scalar.raw().to_u64(),
            );
        }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    if matches!(backend, Backend::Neon)
        && L::BITS == 32
        && <PrimeField<Q, L> as MontgomeryUintSimd>::NEON_MONTGOMERY_QUALIFIED
    {
        unsafe {
            crate::simd::aarch64::u32_arith::scalar_mul_prime_montgomery_u32::<Q>(
                as_u32_slice_mut(<PrimeField<Q, L> as MontgomeryUintSimd>::values_mut(dst)),
                scalar.raw().to_u64(),
            );
        }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    if matches!(backend, Backend::Neon)
        && <PrimeField<Q, L> as MontgomeryUintSimd>::NEON_MONTGOMERY_QUALIFIED
    {
        unsafe {
            crate::simd::aarch64::scalar_mul_prime_montgomery_u64::<Q>(
                as_u64_slice_mut(<PrimeField<Q, L> as MontgomeryUintSimd>::values_mut(dst)),
                scalar.raw().to_u64(),
            );
        }
        return;
    }

    for value in dst.iter_mut() {
        *value *= *scalar;
    }
}

#[inline]
pub(crate) fn pointwise_mul_assign<const Q: u64, L: UintLimb>(
    dst: &mut [PrimeField<Q, L>],
    rhs: &[PrimeField<Q, L>],
) {
    assert_eq!(dst.len(), rhs.len(), "slice lengths must match");
    let backend = selected_backend();

    #[cfg(target_arch = "x86_64")]
    if matches!(backend, Backend::Avx2)
        && L::BITS == 16
        && <PrimeField<Q, L> as MontgomeryUintSimd>::AVX2_MONTGOMERY_QUALIFIED
    {
        unsafe {
            crate::simd::avx2::u16_arith::mul_assign_prime_montgomery_u16::<Q>(
                as_u16_slice_mut(<PrimeField<Q, L> as MontgomeryUintSimd>::values_mut(dst)),
                as_u16_slice(<PrimeField<Q, L> as MontgomeryUintSimd>::values(rhs)),
            );
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if matches!(backend, Backend::Avx2)
        && L::BITS == 8
        && <PrimeField<Q, L> as MontgomeryUintSimd>::AVX2_MONTGOMERY_QUALIFIED
    {
        unsafe {
            crate::simd::avx2::u8_arith::mul_assign_prime_montgomery_u8::<Q>(
                as_u8_slice_mut(<PrimeField<Q, L> as MontgomeryUintSimd>::values_mut(dst)),
                as_u8_slice(<PrimeField<Q, L> as MontgomeryUintSimd>::values(rhs)),
            );
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if matches!(backend, Backend::Avx2)
        && L::BITS == 32
        && <PrimeField<Q, L> as MontgomeryUintSimd>::AVX2_MONTGOMERY_QUALIFIED
    {
        unsafe {
            crate::simd::avx2::u32_arith::mul_assign_prime_montgomery_u32::<Q>(
                as_u32_slice_mut(<PrimeField<Q, L> as MontgomeryUintSimd>::values_mut(dst)),
                as_u32_slice(<PrimeField<Q, L> as MontgomeryUintSimd>::values(rhs)),
            );
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if matches!(backend, Backend::Avx2)
        && L::BITS == 64
        && <PrimeField<Q, L> as MontgomeryUintSimd>::AVX2_MONTGOMERY_QUALIFIED
    {
        unsafe {
            crate::simd::avx2::u64_arith::mul_assign_prime_montgomery_u64::<Q>(
                as_u64_slice_mut(<PrimeField<Q, L> as MontgomeryUintSimd>::values_mut(dst)),
                as_u64_slice(<PrimeField<Q, L> as MontgomeryUintSimd>::values(rhs)),
            );
        }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    if matches!(backend, Backend::Neon)
        && L::BITS == 16
        && <PrimeField<Q, L> as MontgomeryUintSimd>::NEON_MONTGOMERY_QUALIFIED
    {
        unsafe {
            crate::simd::aarch64::u16_arith::mul_assign_prime_montgomery_u16::<Q>(
                as_u16_slice_mut(<PrimeField<Q, L> as MontgomeryUintSimd>::values_mut(dst)),
                as_u16_slice(<PrimeField<Q, L> as MontgomeryUintSimd>::values(rhs)),
            );
        }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    if matches!(backend, Backend::Neon)
        && L::BITS == 8
        && <PrimeField<Q, L> as MontgomeryUintSimd>::NEON_MONTGOMERY_QUALIFIED
    {
        unsafe {
            crate::simd::aarch64::u8_arith::mul_assign_prime_montgomery_u8::<Q>(
                as_u8_slice_mut(<PrimeField<Q, L> as MontgomeryUintSimd>::values_mut(dst)),
                as_u8_slice(<PrimeField<Q, L> as MontgomeryUintSimd>::values(rhs)),
            );
        }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    if matches!(backend, Backend::Neon)
        && L::BITS == 32
        && <PrimeField<Q, L> as MontgomeryUintSimd>::NEON_MONTGOMERY_QUALIFIED
    {
        unsafe {
            crate::simd::aarch64::u32_arith::mul_assign_prime_montgomery_u32::<Q>(
                as_u32_slice_mut(<PrimeField<Q, L> as MontgomeryUintSimd>::values_mut(dst)),
                as_u32_slice(<PrimeField<Q, L> as MontgomeryUintSimd>::values(rhs)),
            );
        }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    if matches!(backend, Backend::Neon)
        && <PrimeField<Q, L> as MontgomeryUintSimd>::NEON_MONTGOMERY_QUALIFIED
    {
        unsafe {
            crate::simd::aarch64::mul_assign_prime_montgomery_u64::<Q>(
                as_u64_slice_mut(<PrimeField<Q, L> as MontgomeryUintSimd>::values_mut(dst)),
                as_u64_slice(<PrimeField<Q, L> as MontgomeryUintSimd>::values(rhs)),
            );
        }
        return;
    }

    for (lhs, rhs) in dst.iter_mut().zip(rhs.iter()) {
        *lhs *= *rhs;
    }
}

#[inline]
pub(crate) fn ntt_forward_with_plan<const Q: u64, L: UintLimb>(
    coeffs: &mut [PrimeField<Q, L>],
    plan: &NttPlan<PrimeField<Q, L>>,
) -> Result<(), NttError> {
    let backend = selected_backend();

    #[cfg(target_arch = "x86_64")]
    if matches!(backend, Backend::Avx2)
        && <PrimeField<Q, L> as MontgomeryUintSimd>::AVX2_NTT_QUALIFIED
    {
        return avx2_ntt_forward_with_plan(coeffs, plan);
    }

    #[cfg(target_arch = "aarch64")]
    if matches!(backend, Backend::Neon)
        && <PrimeField<Q, L> as MontgomeryUintSimd>::NEON_NTT_QUALIFIED
    {
        return neon_ntt_forward_with_plan(coeffs, plan);
    }

    scalar_ntt_forward_with_plan(coeffs, plan)
}

#[inline]
pub(crate) fn ntt_inverse_with_plan<const Q: u64, L: UintLimb>(
    evals: &mut [PrimeField<Q, L>],
    plan: &NttPlan<PrimeField<Q, L>>,
) -> Result<(), NttError> {
    let backend = selected_backend();

    #[cfg(target_arch = "x86_64")]
    if matches!(backend, Backend::Avx2)
        && <PrimeField<Q, L> as MontgomeryUintSimd>::AVX2_NTT_QUALIFIED
    {
        return avx2_ntt_inverse_with_plan(evals, plan);
    }

    #[cfg(target_arch = "aarch64")]
    if matches!(backend, Backend::Neon)
        && <PrimeField<Q, L> as MontgomeryUintSimd>::NEON_NTT_QUALIFIED
    {
        return neon_ntt_inverse_with_plan(evals, plan);
    }

    scalar_ntt_inverse_with_plan(evals, plan)
}

// ---------------------------------------------------------------------------
// Platform-specific NTT helpers
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
fn avx2_ntt_forward_with_plan<const Q: u64, L: UintLimb>(
    coeffs: &mut [PrimeField<Q, L>],
    plan: &NttPlan<PrimeField<Q, L>>,
) -> Result<(), NttError> {
    let n = coeffs.len();
    plan.validate_len(n)?;

    bit_reverse_permute(coeffs);

    let values = <PrimeField<Q, L> as MontgomeryUintSimd>::values_mut(coeffs);
    let mut len = 2usize;
    for twiddles in plan.forward_stage_twiddles() {
        let half = len / 2;
        let twiddles = <PrimeField<Q, L> as MontgomeryUintSimd>::values(twiddles);
        for start in (0..n).step_by(len) {
            let (even, odd) = values[start..start + len].split_at_mut(half);
            if L::BITS == 16 {
                unsafe {
                    crate::simd::avx2::u16_arith::butterfly_forward_prime_montgomery_u16::<Q>(
                        as_u16_slice_mut(even),
                        as_u16_slice_mut(odd),
                        as_u16_slice(twiddles),
                    );
                }
            } else if L::BITS == 8 {
                unsafe {
                    crate::simd::avx2::u8_arith::butterfly_forward_prime_montgomery_u8::<Q>(
                        as_u8_slice_mut(even),
                        as_u8_slice_mut(odd),
                        as_u8_slice(twiddles),
                    );
                }
            } else if L::BITS == 32 {
                unsafe {
                    crate::simd::avx2::u32_arith::butterfly_forward_prime_montgomery_u32::<Q>(
                        as_u32_slice_mut(even),
                        as_u32_slice_mut(odd),
                        as_u32_slice(twiddles),
                    );
                }
            } else {
                unsafe {
                    crate::simd::avx2::u64_arith::butterfly_forward_prime_montgomery_u64::<Q>(
                        as_u64_slice_mut(even),
                        as_u64_slice_mut(odd),
                        as_u64_slice(twiddles),
                    );
                }
            }
        }
        len <<= 1;
    }

    Ok(())
}

#[cfg(target_arch = "aarch64")]
fn neon_ntt_forward_with_plan<const Q: u64, L: UintLimb>(
    coeffs: &mut [PrimeField<Q, L>],
    plan: &NttPlan<PrimeField<Q, L>>,
) -> Result<(), NttError> {
    let n = coeffs.len();
    plan.validate_len(n)?;

    bit_reverse_permute(coeffs);

    let values = <PrimeField<Q, L> as MontgomeryUintSimd>::values_mut(coeffs);
    let mut len = 2usize;
    for twiddles in plan.forward_stage_twiddles() {
        let half = len / 2;
        let twiddles = <PrimeField<Q, L> as MontgomeryUintSimd>::values(twiddles);
        for start in (0..n).step_by(len) {
            let (even, odd) = values[start..start + len].split_at_mut(half);
            if L::BITS == 16 {
                unsafe {
                    crate::simd::aarch64::u16_arith::butterfly_forward_prime_montgomery_u16::<Q>(
                        as_u16_slice_mut(even),
                        as_u16_slice_mut(odd),
                        as_u16_slice(twiddles),
                    );
                }
            } else if L::BITS == 8 {
                unsafe {
                    crate::simd::aarch64::u8_arith::butterfly_forward_prime_montgomery_u8::<Q>(
                        as_u8_slice_mut(even),
                        as_u8_slice_mut(odd),
                        as_u8_slice(twiddles),
                    );
                }
            } else if L::BITS == 32 {
                unsafe {
                    crate::simd::aarch64::u32_arith::butterfly_forward_prime_montgomery_u32::<Q>(
                        as_u32_slice_mut(even),
                        as_u32_slice_mut(odd),
                        as_u32_slice(twiddles),
                    );
                }
            } else {
                unsafe {
                    crate::simd::aarch64::butterfly_forward_prime_montgomery_u64::<Q>(
                        as_u64_slice_mut(even),
                        as_u64_slice_mut(odd),
                        as_u64_slice(twiddles),
                    );
                }
            }
        }
        len <<= 1;
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
fn avx2_ntt_inverse_with_plan<const Q: u64, L: UintLimb>(
    evals: &mut [PrimeField<Q, L>],
    plan: &NttPlan<PrimeField<Q, L>>,
) -> Result<(), NttError> {
    let n = evals.len();
    plan.validate_len(n)?;

    let values = <PrimeField<Q, L> as MontgomeryUintSimd>::values_mut(evals);
    let inverse_scale: u64 = plan.inverse_scale().raw().to_u64();
    let mut len = n;
    for twiddles in plan.inverse_stage_twiddles() {
        let half = len / 2;
        let final_stage = len == 2;
        let twiddles = <PrimeField<Q, L> as MontgomeryUintSimd>::values(twiddles);
        for start in (0..n).step_by(len) {
            let (even, odd) = values[start..start + len].split_at_mut(half);
            if L::BITS == 16 {
                unsafe {
                    crate::simd::avx2::u16_arith::butterfly_inverse_prime_montgomery_u16::<Q>(
                        as_u16_slice_mut(even),
                        as_u16_slice_mut(odd),
                        as_u16_slice(twiddles),
                    );
                }
                if final_stage {
                    unsafe {
                        crate::simd::avx2::u16_arith::scalar_mul_prime_montgomery_u16::<Q>(
                            as_u16_slice_mut(&mut values[start..start + len]),
                            inverse_scale,
                        );
                    }
                }
            } else if L::BITS == 8 {
                unsafe {
                    crate::simd::avx2::u8_arith::butterfly_inverse_prime_montgomery_u8::<Q>(
                        as_u8_slice_mut(even),
                        as_u8_slice_mut(odd),
                        as_u8_slice(twiddles),
                    );
                }
                if final_stage {
                    unsafe {
                        crate::simd::avx2::u8_arith::scalar_mul_prime_montgomery_u8::<Q>(
                            as_u8_slice_mut(&mut values[start..start + len]),
                            inverse_scale,
                        );
                    }
                }
            } else if L::BITS == 32 {
                unsafe {
                    crate::simd::avx2::u32_arith::butterfly_inverse_prime_montgomery_u32::<Q>(
                        as_u32_slice_mut(even),
                        as_u32_slice_mut(odd),
                        as_u32_slice(twiddles),
                    );
                }
                if final_stage {
                    unsafe {
                        crate::simd::avx2::u32_arith::scalar_mul_prime_montgomery_u32::<Q>(
                            as_u32_slice_mut(&mut values[start..start + len]),
                            inverse_scale,
                        );
                    }
                }
            } else {
                unsafe {
                    crate::simd::avx2::u64_arith::butterfly_inverse_prime_montgomery_u64::<Q>(
                        as_u64_slice_mut(even),
                        as_u64_slice_mut(odd),
                        as_u64_slice(twiddles),
                    );
                }
                if final_stage {
                    unsafe {
                        crate::simd::avx2::u64_arith::scalar_mul_prime_montgomery_u64::<Q>(
                            as_u64_slice_mut(&mut values[start..start + len]),
                            inverse_scale,
                        );
                    }
                }
            }
        }
        len >>= 1;
    }

    bit_reverse_permute(evals);

    Ok(())
}

#[cfg(target_arch = "aarch64")]
fn neon_ntt_inverse_with_plan<const Q: u64, L: UintLimb>(
    evals: &mut [PrimeField<Q, L>],
    plan: &NttPlan<PrimeField<Q, L>>,
) -> Result<(), NttError> {
    let n = evals.len();
    plan.validate_len(n)?;

    let values = <PrimeField<Q, L> as MontgomeryUintSimd>::values_mut(evals);
    let inverse_scale: u64 = plan.inverse_scale().raw().to_u64();
    let mut len = n;
    for twiddles in plan.inverse_stage_twiddles() {
        let half = len / 2;
        let final_stage = len == 2;
        let twiddles = <PrimeField<Q, L> as MontgomeryUintSimd>::values(twiddles);
        for start in (0..n).step_by(len) {
            let (even, odd) = values[start..start + len].split_at_mut(half);
            if L::BITS == 16 {
                unsafe {
                    crate::simd::aarch64::u16_arith::butterfly_inverse_prime_montgomery_u16::<Q>(
                        as_u16_slice_mut(even),
                        as_u16_slice_mut(odd),
                        as_u16_slice(twiddles),
                    );
                }
                if final_stage {
                    unsafe {
                        crate::simd::aarch64::u16_arith::scalar_mul_prime_montgomery_u16::<Q>(
                            as_u16_slice_mut(&mut values[start..start + len]),
                            inverse_scale,
                        );
                    }
                }
            } else if L::BITS == 8 {
                unsafe {
                    crate::simd::aarch64::u8_arith::butterfly_inverse_prime_montgomery_u8::<Q>(
                        as_u8_slice_mut(even),
                        as_u8_slice_mut(odd),
                        as_u8_slice(twiddles),
                    );
                }
                if final_stage {
                    unsafe {
                        crate::simd::aarch64::u8_arith::scalar_mul_prime_montgomery_u8::<Q>(
                            as_u8_slice_mut(&mut values[start..start + len]),
                            inverse_scale,
                        );
                    }
                }
            } else if L::BITS == 32 {
                unsafe {
                    crate::simd::aarch64::u32_arith::butterfly_inverse_prime_montgomery_u32::<Q>(
                        as_u32_slice_mut(even),
                        as_u32_slice_mut(odd),
                        as_u32_slice(twiddles),
                    );
                }
                if final_stage {
                    unsafe {
                        crate::simd::aarch64::u32_arith::scalar_mul_prime_montgomery_u32::<Q>(
                            as_u32_slice_mut(&mut values[start..start + len]),
                            inverse_scale,
                        );
                    }
                }
            } else {
                unsafe {
                    crate::simd::aarch64::butterfly_inverse_prime_montgomery_u64::<Q>(
                        as_u64_slice_mut(even),
                        as_u64_slice_mut(odd),
                        as_u64_slice(twiddles),
                    );
                }
                if final_stage {
                    unsafe {
                        crate::simd::aarch64::scalar_mul_prime_montgomery_u64::<Q>(
                            as_u64_slice_mut(&mut values[start..start + len]),
                            inverse_scale,
                        );
                    }
                }
            }
        }
        len >>= 1;
    }

    bit_reverse_permute(evals);

    Ok(())
}
