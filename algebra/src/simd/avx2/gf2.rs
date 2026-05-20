//! GF(2) AVX2 SIMD kernels.
//!
//! 32 lanes of u8 per `__m256i`. Addition = XOR, multiplication = AND.
//! Dot product uses nibble-popcount LUT + `_mm256_sad_epu8` for horizontal sum.
#![allow(unsafe_op_in_unsafe_fn)]

use core::arch::x86_64::{
    __m256i, _mm256_add_epi8, _mm256_add_epi64, _mm256_and_si256, _mm256_loadu_si256,
    _mm256_sad_epu8, _mm256_set1_epi8, _mm256_setr_epi8, _mm256_setzero_si256, _mm256_shuffle_epi8,
    _mm256_srli_epi16, _mm256_storeu_si256, _mm256_xor_si256,
};

/// `dst[i] ^= src[i]` for `i in 0..len`.
///
/// # Safety
/// `dst` and `src` must be valid for reads/writes of `len` bytes.
/// The caller must ensure AVX2 is available on the current CPU.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn add_assign(dst: *mut u8, src: *const u8, len: usize) {
    let mut i = 0;
    while i + 32 <= len {
        let v = unsafe { _mm256_loadu_si256(dst.add(i) as *const __m256i) };
        let w = unsafe { _mm256_loadu_si256(src.add(i) as *const __m256i) };
        unsafe { _mm256_storeu_si256(dst.add(i) as *mut __m256i, _mm256_xor_si256(v, w)) };
        i += 32;
    }
    for j in i..len {
        unsafe {
            *dst.add(j) ^= *src.add(j);
        }
    }
}

/// `dst[i] &= src[i]` for `i in 0..len`.
///
/// # Safety
/// `dst` and `src` must be valid for reads/writes of `len` bytes.
/// The caller must ensure AVX2 is available on the current CPU.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn pointwise_mul_assign(dst: *mut u8, src: *const u8, len: usize) {
    let mut i = 0;
    while i + 32 <= len {
        let v = unsafe { _mm256_loadu_si256(dst.add(i) as *const __m256i) };
        let w = unsafe { _mm256_loadu_si256(src.add(i) as *const __m256i) };
        unsafe { _mm256_storeu_si256(dst.add(i) as *mut __m256i, _mm256_and_si256(v, w)) };
        i += 32;
    }
    for j in i..len {
        unsafe {
            *dst.add(j) &= *src.add(j);
        }
    }
}

/// Sum of `(lhs[i] & rhs[i])` over GF(2) for `i in 0..len`, returned as `u64` (caller takes mod 2).
///
/// # Safety
/// `lhs` and `rhs` must be valid for reads of `len` bytes.
/// The caller must ensure AVX2 is available on the current CPU.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn dot_product(lhs: *const u8, rhs: *const u8, len: usize) -> u64 {
    // Nibble popcount LUT: popcount of 0..15
    let lut = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3,
        3, 4,
    );
    let nibble_mask = _mm256_set1_epi8(0x0f);
    let zero = _mm256_setzero_si256();
    let mut acc = zero;

    let mut i = 0;
    while i + 32 <= len {
        let v = unsafe { _mm256_loadu_si256(lhs.add(i) as *const __m256i) };
        let w = unsafe { _mm256_loadu_si256(rhs.add(i) as *const __m256i) };
        let mul = _mm256_and_si256(v, w);
        let lo = _mm256_and_si256(mul, nibble_mask);
        let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(mul), nibble_mask);
        let pc = _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo), _mm256_shuffle_epi8(lut, hi));
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(pc, zero));
        i += 32;
    }

    // Horizontal sum: store accumulator into a [u64; 4] scratch array and sum
    let mut acc_buf = [0u64; 4];
    unsafe { _mm256_storeu_si256(acc_buf.as_mut_ptr() as *mut __m256i, acc); }
    let mut result = acc_buf.iter().sum::<u64>();

    // Scalar tail
    for j in i..len {
        unsafe {
            let bit = (*lhs.add(j) & *rhs.add(j)) as u64;
            result += bit;
        }
    }
    result
}
