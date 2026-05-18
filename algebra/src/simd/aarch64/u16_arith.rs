//! NEON SIMD backend for u16 prime-field arithmetic.
//!
//! 8 lanes per `uint16x8_t`.  Widening multiply via `vmull_u16` /
//! `vmull_high_u16` (16×16→32), then narrow back to u16 for carry-chain
//! reduction.  NEON has no `vmulhi_u16`, so the kernel differs from AVX2.
#![allow(unsafe_op_in_unsafe_fn)]

use crate::arith::prime::PrimeField;

use core::arch::aarch64::{
    uint16x8_t, vaddq_u16, vandq_u16, vcgtq_u16, vcombine_u16, vdupq_n_u16, vget_high_u16,
    vget_low_u16, vld1q_u16, vmovn_u32, vmull_u16, vmulq_u16, vorrq_u16, vshrn_n_u32, vst1q_u16,
    vsubq_u16,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Full 8-lane u16×u16 → (lo:8×u16, hi:8×u16).
///
/// Uses two `vmull_u16` calls (low + high halves), then extracts lower 16 bits
/// and upper 16 bits from each 32-bit product.
#[target_feature(enable = "neon")]
unsafe fn widen_mul_u16_full_neon(a: uint16x8_t, b: uint16x8_t) -> (uint16x8_t, uint16x8_t) {
    // Full u32 products for lanes 0-3 and 4-7
    let prod_lo = vmull_u16(vget_low_u16(a), vget_low_u16(b));
    let prod_hi = vmull_u16(vget_high_u16(a), vget_high_u16(b));

    // Lower 16 bits: truncate
    let lo_lo = vmovn_u32(prod_lo);
    let lo_hi = vmovn_u32(prod_hi);
    let lo = vcombine_u16(lo_lo, lo_hi);

    // Upper 16 bits: shift right 16, narrow
    let hi_lo = vshrn_n_u32::<16>(prod_lo);
    let hi_hi = vshrn_n_u32::<16>(prod_hi);
    let hi = vcombine_u16(hi_lo, hi_hi);

    (lo, hi)
}

// ---------------------------------------------------------------------------
// Montgomery multiplication kernel
// ---------------------------------------------------------------------------

#[target_feature(enable = "neon")]
unsafe fn montgomery_mul_u16_neon<const Q: u64>(lhs: uint16x8_t, rhs: uint16x8_t) -> uint16x8_t {
    let q_vec = vdupq_n_u16(Q as u16);
    let q_inv_vec = vdupq_n_u16(PrimeField::<Q>::Q_INV_U64 as u16);
    let threshold_vec = vdupq_n_u16((Q - 1) as u16);
    let ones = vdupq_n_u16(1);
    let small = Q < (1u64 << 15);

    let (t_lo, t_hi) = widen_mul_u16_full_neon(lhs, rhs);

    // m = t_lo * Q_INV (mod 2^16)
    let m = vmulq_u16(t_lo, q_inv_vec);

    // mq = m * Q
    let (mq_lo, mq_hi) = widen_mul_u16_full_neon(m, q_vec);

    // sum = t + mq (emulated u32 addition with carry tracking)
    let sum_lo = vaddq_u16(t_lo, mq_lo);
    let carry_lo = vcgtq_u16(t_lo, sum_lo); // unsigned: true where overflow
    let sum_hi_a = vaddq_u16(t_hi, mq_hi);
    let sum_hi = vaddq_u16(sum_hi_a, vandq_u16(carry_lo, ones));

    if small {
        let reduce = vcgtq_u16(sum_hi, threshold_vec);
        vsubq_u16(sum_hi, vandq_u16(reduce, q_vec))
    } else {
        let carry_hi_a = vcgtq_u16(t_hi, sum_hi_a);
        let carry_hi_b = vcgtq_u16(sum_hi_a, sum_hi);
        let overflow = vorrq_u16(carry_hi_a, carry_hi_b);
        let ge_q = vcgtq_u16(sum_hi, threshold_vec);
        let reduce = vorrq_u16(overflow, ge_q);
        vsubq_u16(sum_hi, vandq_u16(reduce, q_vec))
    }
}

// ---------------------------------------------------------------------------
// Batch Montgomery multiply
// ---------------------------------------------------------------------------

#[target_feature(enable = "neon")]
unsafe fn mul_assign_prime_montgomery_u16_neon<const Q: u64>(
    dst: *mut u16,
    src: *const u16,
    len: usize,
) {
    let mut i = 0usize;
    while i + 8 <= len {
        let lhs = vld1q_u16(dst.add(i));
        let rhs = vld1q_u16(src.add(i));
        let result = montgomery_mul_u16_neon::<Q>(lhs, rhs);
        vst1q_u16(dst.add(i), result);
        i += 8;
    }
    for j in i..len {
        let prod = PrimeField::<Q, u16>::mul_raw_words(*dst.add(j), *src.add(j));
        *dst.add(j) = prod;
    }
}

#[target_feature(enable = "neon")]
unsafe fn scalar_mul_prime_montgomery_u16_neon<const Q: u64>(
    dst: *mut u16,
    len: usize,
    scalar: u16,
) {
    let scalar_vec = vdupq_n_u16(scalar);
    let mut i = 0usize;
    while i + 8 <= len {
        let lhs = vld1q_u16(dst.add(i));
        let result = montgomery_mul_u16_neon::<Q>(lhs, scalar_vec);
        vst1q_u16(dst.add(i), result);
        i += 8;
    }
    for j in i..len {
        let prod = PrimeField::<Q, u16>::mul_raw_words(*dst.add(j), scalar);
        *dst.add(j) = prod;
    }
}

// ---------------------------------------------------------------------------
// Modular add / sub (8-lane u16)
// ---------------------------------------------------------------------------

#[target_feature(enable = "neon")]
unsafe fn add_assign_prime_u16_neon(dst: *mut u16, src: *const u16, len: usize, modulus: u64) {
    let q_vec = vdupq_n_u16(modulus as u16);
    let threshold_vec = vdupq_n_u16((modulus - 1) as u16);
    let small = modulus < (1u64 << 15);

    let mut i = 0usize;
    while i + 8 <= len {
        let lhs = vld1q_u16(dst.add(i));
        let rhs = vld1q_u16(src.add(i));
        let sum = vaddq_u16(lhs, rhs);
        if small {
            let reduce = vcgtq_u16(sum, threshold_vec);
            let result = vsubq_u16(sum, vandq_u16(reduce, q_vec));
            vst1q_u16(dst.add(i), result);
        } else {
            let carry = vcgtq_u16(lhs, sum);
            let ge_q = vcgtq_u16(sum, threshold_vec);
            let reduce = vorrq_u16(carry, ge_q);
            let result = vsubq_u16(sum, vandq_u16(reduce, q_vec));
            vst1q_u16(dst.add(i), result);
        }
        i += 8;
    }
    let q = modulus as u16;
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
unsafe fn sub_assign_prime_u16_neon(dst: *mut u16, src: *const u16, len: usize, modulus: u64) {
    let q_vec = vdupq_n_u16(modulus as u16);

    let mut i = 0usize;
    while i + 8 <= len {
        let lhs = vld1q_u16(dst.add(i));
        let rhs = vld1q_u16(src.add(i));
        let borrow = vcgtq_u16(rhs, lhs);
        let diff = vsubq_u16(lhs, rhs);
        let result = vaddq_u16(diff, vandq_u16(borrow, q_vec));
        vst1q_u16(dst.add(i), result);
        i += 8;
    }
    let q = modulus as u16;
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
unsafe fn butterfly_forward_prime_montgomery_u16_neon<const Q: u64>(
    even: *mut u16,
    odd: *mut u16,
    twiddles: *const u16,
    len: usize,
) {
    let q_vec = vdupq_n_u16(Q as u16);
    let threshold_vec = vdupq_n_u16((Q - 1) as u16);
    let small = Q < (1u64 << 15);

    let mut i = 0usize;
    while i + 8 <= len {
        let even_val = vld1q_u16(even.add(i));
        let odd_val = vld1q_u16(odd.add(i));
        let tw = vld1q_u16(twiddles.add(i));

        let temp = montgomery_mul_u16_neon::<Q>(odd_val, tw);

        // even' = even + temp
        let sum = vaddq_u16(even_val, temp);
        let even_new = if small {
            let reduce = vcgtq_u16(sum, threshold_vec);
            vsubq_u16(sum, vandq_u16(reduce, q_vec))
        } else {
            let carry = vcgtq_u16(even_val, sum);
            let ge_q = vcgtq_u16(sum, threshold_vec);
            let reduce = vorrq_u16(carry, ge_q);
            vsubq_u16(sum, vandq_u16(reduce, q_vec))
        };

        // odd' = even - temp
        let borrow = vcgtq_u16(temp, even_val);
        let diff = vsubq_u16(even_val, temp);
        let odd_new = vaddq_u16(diff, vandq_u16(borrow, q_vec));

        vst1q_u16(even.add(i), even_new);
        vst1q_u16(odd.add(i), odd_new);
        i += 8;
    }
    for j in i..len {
        let even_val = *even.add(j);
        let odd_val = *odd.add(j);
        let tw = *twiddles.add(j);
        let temp = PrimeField::<Q, u16>::mul_raw_words(odd_val, tw);
        *even.add(j) = PrimeField::<Q, u16>::add_raw_words(even_val, temp);
        *odd.add(j) = PrimeField::<Q, u16>::sub_raw_words(even_val, temp);
    }
}

#[target_feature(enable = "neon")]
unsafe fn butterfly_inverse_prime_montgomery_u16_neon<const Q: u64>(
    even: *mut u16,
    odd: *mut u16,
    twiddles: *const u16,
    len: usize,
) {
    let q_vec = vdupq_n_u16(Q as u16);
    let threshold_vec = vdupq_n_u16((Q - 1) as u16);
    let small = Q < (1u64 << 15);

    let mut i = 0usize;
    while i + 8 <= len {
        let even_val = vld1q_u16(even.add(i));
        let odd_val = vld1q_u16(odd.add(i));
        let tw = vld1q_u16(twiddles.add(i));

        // sum = even + odd
        let sum = vaddq_u16(even_val, odd_val);
        let even_new = if small {
            let reduce = vcgtq_u16(sum, threshold_vec);
            vsubq_u16(sum, vandq_u16(reduce, q_vec))
        } else {
            let carry = vcgtq_u16(even_val, sum);
            let ge_q = vcgtq_u16(sum, threshold_vec);
            let reduce = vorrq_u16(carry, ge_q);
            vsubq_u16(sum, vandq_u16(reduce, q_vec))
        };

        // diff = even - odd
        let borrow = vcgtq_u16(odd_val, even_val);
        let diff = vsubq_u16(even_val, odd_val);
        let diff_mod = vaddq_u16(diff, vandq_u16(borrow, q_vec));

        // odd' = diff * twiddle
        let odd_new = montgomery_mul_u16_neon::<Q>(diff_mod, tw);

        vst1q_u16(even.add(i), even_new);
        vst1q_u16(odd.add(i), odd_new);
        i += 8;
    }
    for j in i..len {
        let even_val = *even.add(j);
        let odd_val = *odd.add(j);
        let tw = *twiddles.add(j);
        let sum = PrimeField::<Q, u16>::add_raw_words(even_val, odd_val);
        let diff = PrimeField::<Q, u16>::sub_raw_words(even_val, odd_val);
        *even.add(j) = sum;
        *odd.add(j) = PrimeField::<Q, u16>::mul_raw_words(diff, tw);
    }
}

// ---------------------------------------------------------------------------
// Public wrappers
// ---------------------------------------------------------------------------

#[inline]
pub(crate) unsafe fn add_assign_prime_u16(dst: &mut [u16], src: &[u16], modulus: u64) {
    debug_assert_eq!(dst.len(), src.len());
    add_assign_prime_u16_neon(dst.as_mut_ptr(), src.as_ptr(), dst.len(), modulus);
}

#[inline]
pub(crate) unsafe fn sub_assign_prime_u16(dst: &mut [u16], src: &[u16], modulus: u64) {
    debug_assert_eq!(dst.len(), src.len());
    sub_assign_prime_u16_neon(dst.as_mut_ptr(), src.as_ptr(), dst.len(), modulus);
}

#[inline]
pub(crate) unsafe fn mul_assign_prime_montgomery_u16<const Q: u64>(dst: &mut [u16], src: &[u16]) {
    debug_assert_eq!(dst.len(), src.len());
    mul_assign_prime_montgomery_u16_neon::<Q>(dst.as_mut_ptr(), src.as_ptr(), dst.len());
}

#[inline]
pub(crate) unsafe fn scalar_mul_prime_montgomery_u16<const Q: u64>(dst: &mut [u16], scalar: u64) {
    scalar_mul_prime_montgomery_u16_neon::<Q>(dst.as_mut_ptr(), dst.len(), scalar as u16);
}

#[inline]
pub(crate) unsafe fn butterfly_forward_prime_montgomery_u16<const Q: u64>(
    even: &mut [u16],
    odd: &mut [u16],
    twiddles: &[u16],
) {
    debug_assert_eq!(even.len(), odd.len());
    debug_assert_eq!(even.len(), twiddles.len());
    butterfly_forward_prime_montgomery_u16_neon::<Q>(
        even.as_mut_ptr(),
        odd.as_mut_ptr(),
        twiddles.as_ptr(),
        even.len(),
    );
}

#[inline]
pub(crate) unsafe fn butterfly_inverse_prime_montgomery_u16<const Q: u64>(
    even: &mut [u16],
    odd: &mut [u16],
    twiddles: &[u16],
) {
    debug_assert_eq!(even.len(), odd.len());
    debug_assert_eq!(even.len(), twiddles.len());
    butterfly_inverse_prime_montgomery_u16_neon::<Q>(
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

    type F12289u16 = PrimeField<12289, u16>;
    type F65521u16 = PrimeField<65521, u16>;

    const N: usize = 18; // one full SIMD iter (8) × 2 + 2 scalar tail

    fn skip_if_no_neon() -> bool {
        !neon_available()
    }

    // --- Small modulus (Q < 2^15) tests ---

    #[test]
    fn test_u16_small_add_sub_matches_scalar() {
        if skip_if_no_neon() {
            return;
        }

        let modulus = 12289u64;
        let q = modulus as u16;
        let lhs: [u16; N] = core::array::from_fn(|i| (i as u64 * 300 + 1) as u16 % q);
        let rhs: [u16; N] = core::array::from_fn(|i| (i as u64 * 70 + 5) as u16 % q);

        let mut simd_add = lhs;
        let mut simd_sub = lhs;
        let scalar_add: [u16; N] = core::array::from_fn(|i| {
            let s = lhs[i] + rhs[i];
            if s >= q { s - q } else { s }
        });
        let scalar_sub: [u16; N] = core::array::from_fn(|i| {
            if lhs[i] >= rhs[i] {
                lhs[i] - rhs[i]
            } else {
                lhs[i] + q - rhs[i]
            }
        });

        unsafe {
            add_assign_prime_u16(&mut simd_add, &rhs, modulus);
            sub_assign_prime_u16(&mut simd_sub, &rhs, modulus);
        }

        assert_eq!(simd_add, scalar_add);
        assert_eq!(simd_sub, scalar_sub);
    }

    #[test]
    fn test_u16_small_mul_matches_scalar() {
        if skip_if_no_neon() {
            return;
        }

        let lhs: [u16; N] =
            core::array::from_fn(|i| F12289u16::from_u64((i * 300 + 1) as u64).raw());
        let rhs: [u16; N] =
            core::array::from_fn(|i| F12289u16::from_u64((i * 70 + 5) as u64).raw());

        let mut simd_mul = lhs;
        let scalar_mul: [u16; N] =
            core::array::from_fn(|i| F12289u16::mul_raw_words(lhs[i], rhs[i]));

        unsafe {
            mul_assign_prime_montgomery_u16::<12289>(&mut simd_mul, &rhs);
        }

        assert_eq!(simd_mul, scalar_mul);
    }

    #[test]
    fn test_u16_small_butterfly_matches_scalar() {
        if skip_if_no_neon() {
            return;
        }

        let mut fwd_even: [u16; N] =
            core::array::from_fn(|i| F12289u16::from_u64((i * 13 + 10) as u64).raw());
        let mut fwd_odd: [u16; N] =
            core::array::from_fn(|i| F12289u16::from_u64((i * 7 + 3) as u64).raw());
        let twiddles: [u16; N] =
            core::array::from_fn(|i| F12289u16::from_u64((i * 5 + 1) as u64).raw());

        let mut s_e = fwd_even;
        let mut s_o = fwd_odd;
        for i in 0..N {
            let u = s_e[i];
            let t = F12289u16::mul_raw_words(s_o[i], twiddles[i]);
            s_e[i] = F12289u16::add_raw_words(u, t);
            s_o[i] = F12289u16::sub_raw_words(u, t);
        }

        unsafe {
            butterfly_forward_prime_montgomery_u16::<12289>(&mut fwd_even, &mut fwd_odd, &twiddles);
        }

        assert_eq!(fwd_even, s_e);
        assert_eq!(fwd_odd, s_o);
    }

    #[test]
    fn test_u16_small_inverse_butterfly_matches_scalar() {
        if skip_if_no_neon() {
            return;
        }

        let mut inv_even: [u16; N] =
            core::array::from_fn(|i| F12289u16::from_u64((i * 13 + 10) as u64).raw());
        let mut inv_odd: [u16; N] =
            core::array::from_fn(|i| F12289u16::from_u64((i * 7 + 3) as u64).raw());
        let twiddles: [u16; N] =
            core::array::from_fn(|i| F12289u16::from_u64((i * 5 + 1) as u64).raw());

        let mut s_e = inv_even;
        let mut s_o = inv_odd;
        for i in 0..N {
            let sum = F12289u16::add_raw_words(s_e[i], s_o[i]);
            let diff = F12289u16::sub_raw_words(s_e[i], s_o[i]);
            s_e[i] = sum;
            s_o[i] = F12289u16::mul_raw_words(diff, twiddles[i]);
        }

        unsafe {
            butterfly_inverse_prime_montgomery_u16::<12289>(&mut inv_even, &mut inv_odd, &twiddles);
        }

        assert_eq!(inv_even, s_e);
        assert_eq!(inv_odd, s_o);
    }

    #[test]
    fn test_u16_small_ntt_round_trip() {
        if skip_if_no_neon() {
            return;
        }
        use crate::arith::ntt::NttPlan;
        use alloc::vec::Vec;

        let n = 64;
        let original: Vec<F12289u16> = (0..n)
            .map(|i| F12289u16::from_u64((i * 73 + 1) as u64 % 12289))
            .collect();

        let plan = NttPlan::<F12289u16>::build(n).unwrap();
        let mut values = original.clone();
        crate::simd::montgomery_prime::ntt_forward_with_plan(&mut values, &plan).unwrap();
        crate::simd::montgomery_prime::ntt_inverse_with_plan(&mut values, &plan).unwrap();

        assert_eq!(values, original);
    }

    // --- Large modulus (Q >= 2^15) tests ---

    #[test]
    fn test_u16_large_add_sub_matches_scalar() {
        if skip_if_no_neon() {
            return;
        }

        let modulus = 65521u64;
        let q = modulus as u16;
        let lhs: [u16; N] = core::array::from_fn(|i| (i as u64 * 10000 + 1) as u16 % q);
        let rhs: [u16; N] = core::array::from_fn(|i| (i as u64 * 20000 + 5) as u16 % q);

        let mut simd_add = lhs;
        let mut simd_sub = lhs;
        let scalar_add: [u16; N] = core::array::from_fn(|i| {
            let (s, carry) = lhs[i].overflowing_add(rhs[i]);
            if carry || s >= q {
                s.wrapping_sub(q)
            } else {
                s
            }
        });
        let scalar_sub: [u16; N] = core::array::from_fn(|i| {
            if lhs[i] >= rhs[i] {
                lhs[i] - rhs[i]
            } else {
                q.wrapping_sub(rhs[i]).wrapping_add(lhs[i])
            }
        });

        unsafe {
            add_assign_prime_u16(&mut simd_add, &rhs, modulus);
            sub_assign_prime_u16(&mut simd_sub, &rhs, modulus);
        }

        assert_eq!(simd_add, scalar_add);
        assert_eq!(simd_sub, scalar_sub);
    }

    #[test]
    fn test_u16_large_mul_matches_scalar() {
        if skip_if_no_neon() {
            return;
        }

        let lhs: [u16; N] =
            core::array::from_fn(|i| F65521u16::from_u64((i * 10000 + 1) as u64).raw());
        let rhs: [u16; N] =
            core::array::from_fn(|i| F65521u16::from_u64((i * 20000 + 5) as u64).raw());

        let mut simd_mul = lhs;
        let scalar_mul: [u16; N] =
            core::array::from_fn(|i| F65521u16::mul_raw_words(lhs[i], rhs[i]));

        unsafe {
            mul_assign_prime_montgomery_u16::<65521>(&mut simd_mul, &rhs);
        }

        assert_eq!(simd_mul, scalar_mul);
    }

    #[test]
    fn test_u16_large_butterfly_matches_scalar() {
        if skip_if_no_neon() {
            return;
        }

        let mut fwd_even: [u16; N] =
            core::array::from_fn(|i| F65521u16::from_u64((i as u64 * 10000 + 1) % 65521).raw());
        let mut fwd_odd: [u16; N] =
            core::array::from_fn(|i| F65521u16::from_u64((i as u64 * 20000 + 5) % 65521).raw());
        let twiddles: [u16; N] =
            core::array::from_fn(|i| F65521u16::from_u64((i as u64 * 7000 + 3) % 65521).raw());

        let mut s_e = fwd_even;
        let mut s_o = fwd_odd;
        for i in 0..N {
            let u = s_e[i];
            let t = F65521u16::mul_raw_words(s_o[i], twiddles[i]);
            s_e[i] = F65521u16::add_raw_words(u, t);
            s_o[i] = F65521u16::sub_raw_words(u, t);
        }

        unsafe {
            butterfly_forward_prime_montgomery_u16::<65521>(&mut fwd_even, &mut fwd_odd, &twiddles);
        }

        assert_eq!(fwd_even, s_e);
        assert_eq!(fwd_odd, s_o);
    }

    #[test]
    fn test_u16_large_inverse_butterfly_matches_scalar() {
        if skip_if_no_neon() {
            return;
        }

        let mut inv_even: [u16; N] =
            core::array::from_fn(|i| F65521u16::from_u64((i as u64 * 8000 + 7) % 65521).raw());
        let mut inv_odd: [u16; N] =
            core::array::from_fn(|i| F65521u16::from_u64((i as u64 * 15000 + 11) % 65521).raw());
        let twiddles: [u16; N] =
            core::array::from_fn(|i| F65521u16::from_u64((i as u64 * 5000 + 1) % 65521).raw());

        let mut s_e = inv_even;
        let mut s_o = inv_odd;
        for i in 0..N {
            let sum = F65521u16::add_raw_words(s_e[i], s_o[i]);
            let diff = F65521u16::sub_raw_words(s_e[i], s_o[i]);
            s_e[i] = sum;
            s_o[i] = F65521u16::mul_raw_words(diff, twiddles[i]);
        }

        unsafe {
            butterfly_inverse_prime_montgomery_u16::<65521>(&mut inv_even, &mut inv_odd, &twiddles);
        }

        assert_eq!(inv_even, s_e);
        assert_eq!(inv_odd, s_o);
    }
}
