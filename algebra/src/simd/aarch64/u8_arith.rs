//! NEON SIMD backend for u8 prime-field arithmetic.
//!
//! 16 lanes per `uint8x16_t`.  Montgomery mul works in u16 space via
//! radix-256 reduction (all intermediates fit in u16, no carry chain).
#![allow(unsafe_op_in_unsafe_fn)]

use crate::arith::prime::PrimeField;

use core::arch::aarch64::{
    uint16x8_t, vaddq_u8, vaddq_u16, vandq_u8, vandq_u16, vcgtq_u8, vcgtq_u16, vcombine_u8,
    vdupq_n_u8, vdupq_n_u16, vget_high_u8, vget_low_u8, vld1q_u8, vmovl_u8, vmovn_u16, vmulq_u16,
    vorrq_u8, vshrq_n_u16, vst1q_u8, vsubq_u8, vsubq_u16,
};

// ---------------------------------------------------------------------------
// Montgomery multiplication kernel (radix-256, in u16 space)
// ---------------------------------------------------------------------------

/// Montgomery multiply on 8 u16 lanes using radix R=256.
///
/// All inputs and outputs are < 256, so products fit in u16 without overflow.
#[target_feature(enable = "neon")]
unsafe fn montgomery_mul_u8_radix_neon<const Q: u64>(
    lhs: uint16x8_t,
    rhs: uint16x8_t,
    q_vec: uint16x8_t,
    q_inv_vec: uint16x8_t,
    threshold_vec: uint16x8_t,
) -> uint16x8_t {
    let mask8 = vdupq_n_u16(0xFF);

    // t = a * b  (< 65536, fits in u16)
    let t_full = vmulq_u16(lhs, rhs);
    let t_lo8 = vandq_u16(t_full, mask8);
    let t_hi8 = vshrq_n_u16::<8>(t_full);

    // m = t_lo8 * q_inv mod 256
    let q_inv_8 = vandq_u16(q_inv_vec, mask8);
    let m = vandq_u16(vmulq_u16(t_lo8, q_inv_8), mask8);

    // mq = m * Q  (< 65536)
    let mq_full = vmulq_u16(m, q_vec);
    let mq_lo8 = vandq_u16(mq_full, mask8);
    let mq_hi8 = vshrq_n_u16::<8>(mq_full);

    // sum = t + mq
    // sum_lo8 = t_lo8 + mq_lo8 is at most 510, (sum_lo8 >> 8) ∈ {0, 1}
    let sum_lo8 = vaddq_u16(t_lo8, mq_lo8);
    let result = vaddq_u16(vaddq_u16(t_hi8, mq_hi8), vshrq_n_u16::<8>(sum_lo8));

    // result < 2·Q ⇒ single conditional subtract suffices
    let reduce = vcgtq_u16(result, threshold_vec);
    vsubq_u16(result, vandq_u16(reduce, q_vec))
}

// ---------------------------------------------------------------------------
// Batch Montgomery mul (16 u8 lanes → two u16 halves per iteration)
// ---------------------------------------------------------------------------

#[target_feature(enable = "neon")]
unsafe fn mul_assign_prime_montgomery_u8_neon<const Q: u64>(
    dst: *mut u8,
    src: *const u8,
    len: usize,
) {
    let q_vec = vdupq_n_u16(Q as u16);
    let q_inv_vec = vdupq_n_u16(PrimeField::<Q>::Q_INV_U64 as u16);
    let threshold_vec = vdupq_n_u16((Q - 1) as u16);

    let mut i = 0usize;
    while i + 16 <= len {
        let a = vld1q_u8(dst.add(i));
        let b = vld1q_u8(src.add(i));

        // Unpack low 8 → u16
        let a_lo = vmovl_u8(vget_low_u8(a));
        let b_lo = vmovl_u8(vget_low_u8(b));
        let res_lo = montgomery_mul_u8_radix_neon::<Q>(a_lo, b_lo, q_vec, q_inv_vec, threshold_vec);

        // Unpack high 8 → u16
        let a_hi = vmovl_u8(vget_high_u8(a));
        let b_hi = vmovl_u8(vget_high_u8(b));
        let res_hi = montgomery_mul_u8_radix_neon::<Q>(a_hi, b_hi, q_vec, q_inv_vec, threshold_vec);

        // Pack back to u8 (results are < Q < 256, so truncation is safe)
        let packed_lo = vmovn_u16(res_lo);
        let packed_hi = vmovn_u16(res_hi);
        let result = vcombine_u8(packed_lo, packed_hi);

        vst1q_u8(dst.add(i), result);
        i += 16;
    }
    for j in i..len {
        let prod = PrimeField::<Q, u8>::mul_raw_words(*dst.add(j), *src.add(j));
        *dst.add(j) = prod;
    }
}

#[target_feature(enable = "neon")]
unsafe fn scalar_mul_prime_montgomery_u8_neon<const Q: u64>(dst: *mut u8, len: usize, scalar: u8) {
    let q_vec = vdupq_n_u16(Q as u16);
    let q_inv_vec = vdupq_n_u16(PrimeField::<Q>::Q_INV_U64 as u16);
    let threshold_vec = vdupq_n_u16((Q - 1) as u16);
    let scalar_vec = vdupq_n_u16(scalar as u16);

    let mut i = 0usize;
    while i + 16 <= len {
        let a = vld1q_u8(dst.add(i));

        let a_lo = vmovl_u8(vget_low_u8(a));
        let res_lo =
            montgomery_mul_u8_radix_neon::<Q>(a_lo, scalar_vec, q_vec, q_inv_vec, threshold_vec);

        let a_hi = vmovl_u8(vget_high_u8(a));
        let res_hi =
            montgomery_mul_u8_radix_neon::<Q>(a_hi, scalar_vec, q_vec, q_inv_vec, threshold_vec);

        let packed_lo = vmovn_u16(res_lo);
        let packed_hi = vmovn_u16(res_hi);
        let result = vcombine_u8(packed_lo, packed_hi);

        vst1q_u8(dst.add(i), result);
        i += 16;
    }
    for j in i..len {
        let prod = PrimeField::<Q, u8>::mul_raw_words(*dst.add(j), scalar);
        *dst.add(j) = prod;
    }
}

// ---------------------------------------------------------------------------
// Modular add / sub (16-lane u8, native unsigned compare)
// ---------------------------------------------------------------------------

#[target_feature(enable = "neon")]
unsafe fn add_assign_prime_u8_neon(dst: *mut u8, src: *const u8, len: usize, modulus: u64) {
    let q_vec = vdupq_n_u8(modulus as u8);
    let threshold_vec = vdupq_n_u8((modulus - 1) as u8);
    let small = modulus < (1u64 << 7);

    let mut i = 0usize;
    while i + 16 <= len {
        let lhs = vld1q_u8(dst.add(i));
        let rhs = vld1q_u8(src.add(i));
        let sum = vaddq_u8(lhs, rhs);
        if small {
            let reduce = vcgtq_u8(sum, threshold_vec);
            let result = vsubq_u8(sum, vandq_u8(reduce, q_vec));
            vst1q_u8(dst.add(i), result);
        } else {
            let carry = vcgtq_u8(lhs, sum);
            let ge_q = vcgtq_u8(sum, threshold_vec);
            let reduce = vorrq_u8(carry, ge_q);
            let result = vsubq_u8(sum, vandq_u8(reduce, q_vec));
            vst1q_u8(dst.add(i), result);
        }
        i += 16;
    }
    let q = modulus as u8;
    if small {
        for j in i..len {
            let s = (*dst.add(j)).wrapping_add(*src.add(j));
            *dst.add(j) = if s >= q { s.wrapping_sub(q) } else { s };
        }
    } else {
        for j in i..len {
            let (s, carry) = (*dst.add(j)).overflowing_add(*src.add(j));
            *dst.add(j) = if carry || s >= q {
                s.wrapping_sub(q)
            } else {
                s
            };
        }
    }
}

#[target_feature(enable = "neon")]
unsafe fn sub_assign_prime_u8_neon(dst: *mut u8, src: *const u8, len: usize, modulus: u64) {
    let q_vec = vdupq_n_u8(modulus as u8);

    let mut i = 0usize;
    while i + 16 <= len {
        let lhs = vld1q_u8(dst.add(i));
        let rhs = vld1q_u8(src.add(i));
        let borrow = vcgtq_u8(rhs, lhs);
        let diff = vsubq_u8(lhs, rhs);
        let result = vaddq_u8(diff, vandq_u8(borrow, q_vec));
        vst1q_u8(dst.add(i), result);
        i += 16;
    }
    let q = modulus as u8;
    for j in i..len {
        let lhs = *dst.add(j);
        let rhs = *src.add(j);
        *dst.add(j) = if lhs >= rhs {
            lhs.wrapping_sub(rhs)
        } else {
            q.wrapping_sub(rhs).wrapping_add(lhs)
        };
    }
}

// ---------------------------------------------------------------------------
// NTT butterflies
// ---------------------------------------------------------------------------

#[target_feature(enable = "neon")]
unsafe fn butterfly_forward_prime_montgomery_u8_neon<const Q: u64>(
    even: *mut u8,
    odd: *mut u8,
    twiddles: *const u8,
    len: usize,
) {
    let q_vec16 = vdupq_n_u16(Q as u16);
    let q_inv_vec = vdupq_n_u16(PrimeField::<Q>::Q_INV_U64 as u16);
    let threshold_vec16 = vdupq_n_u16((Q - 1) as u16);
    let small = Q < (1u64 << 7);

    let q_vec = vdupq_n_u8(Q as u8);
    let threshold_vec = vdupq_n_u8((Q - 1) as u8);

    let mut i = 0usize;
    while i + 16 <= len {
        let even_val = vld1q_u8(even.add(i));
        let odd_val = vld1q_u8(odd.add(i));
        let tw = vld1q_u8(twiddles.add(i));

        // temp = odd * twiddle (unpack to u16, mul, pack back)
        let odd_lo = vmovl_u8(vget_low_u8(odd_val));
        let tw_lo = vmovl_u8(vget_low_u8(tw));
        let temp_lo =
            montgomery_mul_u8_radix_neon::<Q>(odd_lo, tw_lo, q_vec16, q_inv_vec, threshold_vec16);
        let odd_hi = vmovl_u8(vget_high_u8(odd_val));
        let tw_hi = vmovl_u8(vget_high_u8(tw));
        let temp_hi =
            montgomery_mul_u8_radix_neon::<Q>(odd_hi, tw_hi, q_vec16, q_inv_vec, threshold_vec16);
        let temp_lo8 = vmovn_u16(temp_lo);
        let temp_hi8 = vmovn_u16(temp_hi);
        let temp = vcombine_u8(temp_lo8, temp_hi8);

        // even' = even + temp
        let sum = vaddq_u8(even_val, temp);
        let even_new = if small {
            let reduce = vcgtq_u8(sum, threshold_vec);
            vsubq_u8(sum, vandq_u8(reduce, q_vec))
        } else {
            let carry = vcgtq_u8(even_val, sum);
            let ge_q = vcgtq_u8(sum, threshold_vec);
            let reduce = vorrq_u8(carry, ge_q);
            vsubq_u8(sum, vandq_u8(reduce, q_vec))
        };

        // odd' = even - temp
        let borrow = vcgtq_u8(temp, even_val);
        let diff = vsubq_u8(even_val, temp);
        let odd_new = vaddq_u8(diff, vandq_u8(borrow, q_vec));

        vst1q_u8(even.add(i), even_new);
        vst1q_u8(odd.add(i), odd_new);
        i += 16;
    }
    for j in i..len {
        let even_val = *even.add(j);
        let odd_val = *odd.add(j);
        let tw = *twiddles.add(j);
        let temp = PrimeField::<Q, u8>::mul_raw_words(odd_val, tw);
        *even.add(j) = PrimeField::<Q, u8>::add_raw_words(even_val, temp);
        *odd.add(j) = PrimeField::<Q, u8>::sub_raw_words(even_val, temp);
    }
}

#[target_feature(enable = "neon")]
unsafe fn butterfly_inverse_prime_montgomery_u8_neon<const Q: u64>(
    even: *mut u8,
    odd: *mut u8,
    twiddles: *const u8,
    len: usize,
) {
    let q_vec16 = vdupq_n_u16(Q as u16);
    let q_inv_vec = vdupq_n_u16(PrimeField::<Q>::Q_INV_U64 as u16);
    let threshold_vec16 = vdupq_n_u16((Q - 1) as u16);
    let small = Q < (1u64 << 7);

    let q_vec = vdupq_n_u8(Q as u8);
    let threshold_vec = vdupq_n_u8((Q - 1) as u8);

    let mut i = 0usize;
    while i + 16 <= len {
        let even_val = vld1q_u8(even.add(i));
        let odd_val = vld1q_u8(odd.add(i));
        let tw = vld1q_u8(twiddles.add(i));

        // sum = even + odd
        let sum = vaddq_u8(even_val, odd_val);
        let even_new = if small {
            let reduce = vcgtq_u8(sum, threshold_vec);
            vsubq_u8(sum, vandq_u8(reduce, q_vec))
        } else {
            let carry = vcgtq_u8(even_val, sum);
            let ge_q = vcgtq_u8(sum, threshold_vec);
            let reduce = vorrq_u8(carry, ge_q);
            vsubq_u8(sum, vandq_u8(reduce, q_vec))
        };

        // diff = even - odd
        let borrow = vcgtq_u8(odd_val, even_val);
        let diff = vsubq_u8(even_val, odd_val);
        let diff_mod = vaddq_u8(diff, vandq_u8(borrow, q_vec));

        // odd' = diff * twiddle
        let diff_lo = vmovl_u8(vget_low_u8(diff_mod));
        let tw_lo = vmovl_u8(vget_low_u8(tw));
        let odd_lo =
            montgomery_mul_u8_radix_neon::<Q>(diff_lo, tw_lo, q_vec16, q_inv_vec, threshold_vec16);
        let diff_hi = vmovl_u8(vget_high_u8(diff_mod));
        let tw_hi = vmovl_u8(vget_high_u8(tw));
        let odd_hi =
            montgomery_mul_u8_radix_neon::<Q>(diff_hi, tw_hi, q_vec16, q_inv_vec, threshold_vec16);
        let odd_lo8 = vmovn_u16(odd_lo);
        let odd_hi8 = vmovn_u16(odd_hi);
        let odd_new = vcombine_u8(odd_lo8, odd_hi8);

        vst1q_u8(even.add(i), even_new);
        vst1q_u8(odd.add(i), odd_new);
        i += 16;
    }
    for j in i..len {
        let even_val = *even.add(j);
        let odd_val = *odd.add(j);
        let tw = *twiddles.add(j);
        let sum = PrimeField::<Q, u8>::add_raw_words(even_val, odd_val);
        let diff = PrimeField::<Q, u8>::sub_raw_words(even_val, odd_val);
        *even.add(j) = sum;
        *odd.add(j) = PrimeField::<Q, u8>::mul_raw_words(diff, tw);
    }
}

// ---------------------------------------------------------------------------
// Public wrappers
// ---------------------------------------------------------------------------

#[inline]
pub(crate) unsafe fn add_assign_prime_u8(dst: &mut [u8], src: &[u8], modulus: u64) {
    debug_assert_eq!(dst.len(), src.len());
    add_assign_prime_u8_neon(dst.as_mut_ptr(), src.as_ptr(), dst.len(), modulus);
}

#[inline]
pub(crate) unsafe fn sub_assign_prime_u8(dst: &mut [u8], src: &[u8], modulus: u64) {
    debug_assert_eq!(dst.len(), src.len());
    sub_assign_prime_u8_neon(dst.as_mut_ptr(), src.as_ptr(), dst.len(), modulus);
}

#[inline]
pub(crate) unsafe fn mul_assign_prime_montgomery_u8<const Q: u64>(dst: &mut [u8], src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    mul_assign_prime_montgomery_u8_neon::<Q>(dst.as_mut_ptr(), src.as_ptr(), dst.len());
}

#[inline]
pub(crate) unsafe fn scalar_mul_prime_montgomery_u8<const Q: u64>(dst: &mut [u8], scalar: u64) {
    scalar_mul_prime_montgomery_u8_neon::<Q>(dst.as_mut_ptr(), dst.len(), scalar as u8);
}

#[inline]
pub(crate) unsafe fn butterfly_forward_prime_montgomery_u8<const Q: u64>(
    even: &mut [u8],
    odd: &mut [u8],
    twiddles: &[u8],
) {
    debug_assert_eq!(even.len(), odd.len());
    debug_assert_eq!(even.len(), twiddles.len());
    butterfly_forward_prime_montgomery_u8_neon::<Q>(
        even.as_mut_ptr(),
        odd.as_mut_ptr(),
        twiddles.as_ptr(),
        even.len(),
    );
}

#[inline]
pub(crate) unsafe fn butterfly_inverse_prime_montgomery_u8<const Q: u64>(
    even: &mut [u8],
    odd: &mut [u8],
    twiddles: &[u8],
) {
    debug_assert_eq!(even.len(), odd.len());
    debug_assert_eq!(even.len(), twiddles.len());
    butterfly_inverse_prime_montgomery_u8_neon::<Q>(
        even.as_mut_ptr(),
        odd.as_mut_ptr(),
        twiddles.as_ptr(),
        even.len(),
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::super::u64_arith::neon_available;
    use super::*;
    use crate::arith::prime::PrimeField;
    use crate::arith::ring::IntegerRing;

    type F127u8 = PrimeField<127, u8>;
    type F251u8 = PrimeField<251, u8>;

    const N: usize = 34; // one full SIMD iter (16) × 2 + 2 scalar tail

    fn skip_if_no_neon() -> bool {
        !neon_available()
    }

    #[test]
    fn test_u8_small_add_sub_matches_scalar() {
        if skip_if_no_neon() {
            return;
        }

        let modulus = 127u64;
        let q = modulus as u8;
        let lhs: [u8; N] = core::array::from_fn(|i| (i as u64 * 13 + 1) as u8 % q);
        let rhs: [u8; N] = core::array::from_fn(|i| (i as u64 * 7 + 5) as u8 % q);

        let mut simd_add = lhs;
        let mut simd_sub = lhs;
        let scalar_add: [u8; N] = core::array::from_fn(|i| {
            let s = lhs[i] + rhs[i];
            if s >= q { s - q } else { s }
        });
        let scalar_sub: [u8; N] = core::array::from_fn(|i| {
            if lhs[i] >= rhs[i] {
                lhs[i] - rhs[i]
            } else {
                lhs[i] + q - rhs[i]
            }
        });

        unsafe {
            add_assign_prime_u8(&mut simd_add, &rhs, modulus);
            sub_assign_prime_u8(&mut simd_sub, &rhs, modulus);
        }

        assert_eq!(simd_add, scalar_add);
        assert_eq!(simd_sub, scalar_sub);
    }

    #[test]
    fn test_u8_small_mul_matches_scalar() {
        if skip_if_no_neon() {
            return;
        }

        let lhs: [u8; N] = core::array::from_fn(|i| F127u8::from_u64((i * 13 + 1) as u64).raw());
        let rhs: [u8; N] = core::array::from_fn(|i| F127u8::from_u64((i * 7 + 5) as u64).raw());

        let mut simd_mul = lhs;
        let scalar_mul: [u8; N] = core::array::from_fn(|i| F127u8::mul_raw_words(lhs[i], rhs[i]));

        unsafe {
            mul_assign_prime_montgomery_u8::<127>(&mut simd_mul, &rhs);
        }

        assert_eq!(simd_mul, scalar_mul);
    }

    #[test]
    fn test_u8_small_butterfly_matches_scalar() {
        if skip_if_no_neon() {
            return;
        }

        let mut fwd_even: [u8; N] =
            core::array::from_fn(|i| F127u8::from_u64((i * 13 + 10) as u64).raw());
        let mut fwd_odd: [u8; N] =
            core::array::from_fn(|i| F127u8::from_u64((i * 7 + 3) as u64).raw());
        let twiddles: [u8; N] =
            core::array::from_fn(|i| F127u8::from_u64((i * 5 + 1) as u64).raw());

        let mut s_e = fwd_even;
        let mut s_o = fwd_odd;
        for i in 0..N {
            let u = s_e[i];
            let t = F127u8::mul_raw_words(s_o[i], twiddles[i]);
            s_e[i] = F127u8::add_raw_words(u, t);
            s_o[i] = F127u8::sub_raw_words(u, t);
        }

        unsafe {
            butterfly_forward_prime_montgomery_u8::<127>(&mut fwd_even, &mut fwd_odd, &twiddles);
        }

        assert_eq!(fwd_even, s_e);
        assert_eq!(fwd_odd, s_o);
    }

    #[test]
    fn test_u8_small_inverse_butterfly_matches_scalar() {
        if skip_if_no_neon() {
            return;
        }

        let mut inv_even: [u8; N] =
            core::array::from_fn(|i| F127u8::from_u64((i * 13 + 10) as u64).raw());
        let mut inv_odd: [u8; N] =
            core::array::from_fn(|i| F127u8::from_u64((i * 7 + 3) as u64).raw());
        let twiddles: [u8; N] =
            core::array::from_fn(|i| F127u8::from_u64((i * 5 + 1) as u64).raw());

        let mut s_e = inv_even;
        let mut s_o = inv_odd;
        for i in 0..N {
            let sum = F127u8::add_raw_words(s_e[i], s_o[i]);
            let diff = F127u8::sub_raw_words(s_e[i], s_o[i]);
            s_e[i] = sum;
            s_o[i] = F127u8::mul_raw_words(diff, twiddles[i]);
        }

        unsafe {
            butterfly_inverse_prime_montgomery_u8::<127>(&mut inv_even, &mut inv_odd, &twiddles);
        }

        assert_eq!(inv_even, s_e);
        assert_eq!(inv_odd, s_o);
    }

    #[test]
    fn test_u8_small_ntt_round_trip() {
        if skip_if_no_neon() {
            return;
        }
        // 193 is prime, 193-1 = 192 = 64*3, supports n=64 NTT
        use crate::arith::ntt::NttPlan;
        use alloc::vec::Vec;

        type F193u8 = PrimeField<193, u8>;
        let n = 64;
        let original: Vec<F193u8> = (0..n)
            .map(|i| F193u8::from_u64((i as u64 * 13 + 1) % 193))
            .collect();

        let plan = NttPlan::<F193u8>::build(n).unwrap();
        let mut values = original.clone();
        crate::simd::montgomery::ntt_forward_with_plan(&mut values, &plan).unwrap();
        crate::simd::montgomery::ntt_inverse_with_plan(&mut values, &plan).unwrap();

        assert_eq!(values, original);
    }

    // --- Large modulus (Q >= 128) tests ---

    #[test]
    fn test_u8_large_add_sub_matches_scalar() {
        if skip_if_no_neon() {
            return;
        }

        // 251 > 128, exercises carry detection
        let modulus = 251u64;
        let q = modulus as u8;
        let lhs: [u8; N] = core::array::from_fn(|i| (i as u64 * 50 + 1) as u8 % q);
        let rhs: [u8; N] = core::array::from_fn(|i| (i as u64 * 70 + 5) as u8 % q);

        let mut simd_add = lhs;
        let mut simd_sub = lhs;
        let scalar_add: [u8; N] = core::array::from_fn(|i| {
            let (s, carry) = lhs[i].overflowing_add(rhs[i]);
            if carry || s >= q {
                s.wrapping_sub(q)
            } else {
                s
            }
        });
        let scalar_sub: [u8; N] = core::array::from_fn(|i| {
            if lhs[i] >= rhs[i] {
                lhs[i] - rhs[i]
            } else {
                q.wrapping_sub(rhs[i]).wrapping_add(lhs[i])
            }
        });

        unsafe {
            add_assign_prime_u8(&mut simd_add, &rhs, modulus);
            sub_assign_prime_u8(&mut simd_sub, &rhs, modulus);
        }

        assert_eq!(simd_add, scalar_add);
        assert_eq!(simd_sub, scalar_sub);
    }

    #[test]
    fn test_u8_large_mul_matches_scalar() {
        if skip_if_no_neon() {
            return;
        }

        let lhs: [u8; N] = core::array::from_fn(|i| F251u8::from_u64((i * 50 + 1) as u64).raw());
        let rhs: [u8; N] = core::array::from_fn(|i| F251u8::from_u64((i * 70 + 5) as u64).raw());

        let mut simd_mul = lhs;
        let scalar_mul: [u8; N] = core::array::from_fn(|i| F251u8::mul_raw_words(lhs[i], rhs[i]));

        unsafe {
            mul_assign_prime_montgomery_u8::<251>(&mut simd_mul, &rhs);
        }

        assert_eq!(simd_mul, scalar_mul);
    }

    #[test]
    fn test_u8_large_butterfly_matches_scalar() {
        if skip_if_no_neon() {
            return;
        }

        let mut fwd_even: [u8; N] =
            core::array::from_fn(|i| F251u8::from_u64((i as u64 * 50 + 1) % 251).raw());
        let mut fwd_odd: [u8; N] =
            core::array::from_fn(|i| F251u8::from_u64((i as u64 * 70 + 5) % 251).raw());
        let twiddles: [u8; N] =
            core::array::from_fn(|i| F251u8::from_u64((i as u64 * 20 + 3) % 251).raw());

        let mut s_e = fwd_even;
        let mut s_o = fwd_odd;
        for i in 0..N {
            let u = s_e[i];
            let t = F251u8::mul_raw_words(s_o[i], twiddles[i]);
            s_e[i] = F251u8::add_raw_words(u, t);
            s_o[i] = F251u8::sub_raw_words(u, t);
        }

        unsafe {
            butterfly_forward_prime_montgomery_u8::<251>(&mut fwd_even, &mut fwd_odd, &twiddles);
        }

        assert_eq!(fwd_even, s_e);
        assert_eq!(fwd_odd, s_o);
    }

    #[test]
    fn test_u8_large_inverse_butterfly_matches_scalar() {
        if skip_if_no_neon() {
            return;
        }

        let mut inv_even: [u8; N] =
            core::array::from_fn(|i| F251u8::from_u64((i as u64 * 40 + 7) % 251).raw());
        let mut inv_odd: [u8; N] =
            core::array::from_fn(|i| F251u8::from_u64((i as u64 * 60 + 11) % 251).raw());
        let twiddles: [u8; N] =
            core::array::from_fn(|i| F251u8::from_u64((i as u64 * 15 + 1) % 251).raw());

        let mut s_e = inv_even;
        let mut s_o = inv_odd;
        for i in 0..N {
            let sum = F251u8::add_raw_words(s_e[i], s_o[i]);
            let diff = F251u8::sub_raw_words(s_e[i], s_o[i]);
            s_e[i] = sum;
            s_o[i] = F251u8::mul_raw_words(diff, twiddles[i]);
        }

        unsafe {
            butterfly_inverse_prime_montgomery_u8::<251>(&mut inv_even, &mut inv_odd, &twiddles);
        }

        assert_eq!(inv_even, s_e);
        assert_eq!(inv_odd, s_o);
    }
}
