//! GF(2) NEON SIMD kernels.
//!
//! 16 lanes per `uint8x16_t`. Addition = XOR (veorq_u8), multiplication = AND (vandq_u8).
//! Dot product uses nibble-popcount LUT via `vqtbl1q_u8` + pairwise reduction.
#![allow(unsafe_op_in_unsafe_fn)]

use core::arch::aarch64::{
    uint8x16_t, vaddq_u8, vaddvq_u8, vandq_u8, vdupq_n_u8, veorq_u8, vld1q_u8, vqtbl1q_u8,
    vshrq_n_u8, vst1q_u8,
};

/// `dst[i] ^= src[i]` for `i in 0..len`.
///
/// # Safety
/// `dst` and `src` must be valid for reads/writes of `len` bytes.
/// The caller must ensure NEON is available on the current CPU.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn add_assign(dst: *mut u8, src: *const u8, len: usize) {
    let mut i = 0;
    while i + 16 <= len {
        let v = vld1q_u8(dst.add(i));
        let w = vld1q_u8(src.add(i));
        vst1q_u8(dst.add(i), veorq_u8(v, w));
        i += 16;
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
/// The caller must ensure NEON is available on the current CPU.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn pointwise_mul_assign(dst: *mut u8, src: *const u8, len: usize) {
    let mut i = 0;
    while i + 16 <= len {
        let v = vld1q_u8(dst.add(i));
        let w = vld1q_u8(src.add(i));
        vst1q_u8(dst.add(i), vandq_u8(v, w));
        i += 16;
    }
    for j in i..len {
        unsafe {
            *dst.add(j) &= *src.add(j);
        }
    }
}

/// Sum of `(lhs[i] & rhs[i])` for `i in 0..len`, returned as `u64` (caller takes mod 2).
///
/// # Safety
/// `lhs` and `rhs` must be valid for reads of `len` bytes.
/// The caller must ensure NEON is available on the current CPU.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn dot_product(lhs: *const u8, rhs: *const u8, len: usize) -> u64 {
    // Nibble popcount LUT
    let lut: uint8x16_t = vld1q_u8([0u8, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4].as_ptr());
    let nibble_mask = vdupq_n_u8(0x0f);
    let mut result: u64 = 0;

    let mut i = 0;
    while i + 16 <= len {
        let v = vld1q_u8(lhs.add(i));
        let w = vld1q_u8(rhs.add(i));
        let mul = vandq_u8(v, w);
        let lo = vandq_u8(mul, nibble_mask);
        let hi = vshrq_n_u8::<4>(mul);
        let hi = vandq_u8(hi, nibble_mask);
        let pc = vaddq_u8(vqtbl1q_u8(lut, lo), vqtbl1q_u8(lut, hi));
        result += vaddvq_u8(pc) as u64;
        i += 16;
    }

    for j in i..len {
        unsafe {
            result += (*lhs.add(j) & *rhs.add(j)) as u64;
        }
    }
    result
}
