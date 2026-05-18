//! NEON SIMD backend for u32 prime-field arithmetic.
//!
//! 4 lanes per `uint32x4_t`.  Full 32x32 → 64 multiply via `vmull_u32` /
//! `vmull_high_u32`, then narrow back to u32 for carry-chain reduction.
#![allow(unsafe_op_in_unsafe_fn)]

use crate::arith::prime::PrimeField;

use core::arch::aarch64::{
    uint32x4_t, vaddq_u32, vandq_u32, vcgtq_u32, vcombine_u32, vdupq_n_u32, vget_high_u32,
    vget_low_u32, vld1q_u32, vmovn_u64, vmull_u32, vmulq_u32, vorrq_u32, vshrn_n_u64, vst1q_u32,
    vsubq_u32,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Full 4-lane u32×u32 → (lo:4×u32, hi:4×u32).
#[target_feature(enable = "neon")]
unsafe fn widen_mul_u32_full_neon(a: uint32x4_t, b: uint32x4_t) -> (uint32x4_t, uint32x4_t) {
    // Low u64 products for lanes 0-1
    let prod_lo = vmull_u32(vget_low_u32(a), vget_low_u32(b));
    // Low u64 products for lanes 2-3
    let prod_hi = vmull_u32(vget_high_u32(a), vget_high_u32(b));

    // Lower 32 bits of each product: truncate
    let lo_lo = vmovn_u64(prod_lo);
    let lo_hi = vmovn_u64(prod_hi);
    let lo = vcombine_u32(lo_lo, lo_hi);

    // Upper 32 bits: shift right 32, narrow
    let hi_lo = vshrn_n_u64::<32>(prod_lo);
    let hi_hi = vshrn_n_u64::<32>(prod_hi);
    let hi = vcombine_u32(hi_lo, hi_hi);

    (lo, hi)
}

// ---------------------------------------------------------------------------
// Montgomery multiplication kernel
// ---------------------------------------------------------------------------

#[target_feature(enable = "neon")]
unsafe fn montgomery_mul_u32_neon<const Q: u64>(lhs: uint32x4_t, rhs: uint32x4_t) -> uint32x4_t {
    let q_vec = vdupq_n_u32(Q as u32);
    let q_inv_vec = vdupq_n_u32(PrimeField::<Q>::Q_INV_U64 as u32);
    let threshold_vec = vdupq_n_u32((Q - 1) as u32);
    let ones = vdupq_n_u32(1);
    let small = Q < (1u64 << 31);

    let (t_lo, t_hi) = widen_mul_u32_full_neon(lhs, rhs);

    // m = t_lo * Q_INV (mod 2^32)
    let m = vmulq_u32(t_lo, q_inv_vec);

    // mq = m * Q
    let (mq_lo, mq_hi) = widen_mul_u32_full_neon(m, q_vec);

    // sum = t + mq (emulated u64 addition with carry tracking)
    let sum_lo = vaddq_u32(t_lo, mq_lo);
    let carry_lo = vcgtq_u32(t_lo, sum_lo); // unsigned: true where overflow
    let sum_hi_a = vaddq_u32(t_hi, mq_hi);
    let sum_hi = vaddq_u32(sum_hi_a, vandq_u32(carry_lo, ones));

    if small {
        let reduce = vcgtq_u32(sum_hi, threshold_vec);
        vsubq_u32(sum_hi, vandq_u32(reduce, q_vec))
    } else {
        let carry_hi_a = vcgtq_u32(t_hi, sum_hi_a);
        let carry_hi_b = vcgtq_u32(sum_hi_a, sum_hi);
        let overflow = vorrq_u32(carry_hi_a, carry_hi_b);
        let ge_q = vcgtq_u32(sum_hi, threshold_vec);
        let reduce = vorrq_u32(overflow, ge_q);
        vsubq_u32(sum_hi, vandq_u32(reduce, q_vec))
    }
}

// ---------------------------------------------------------------------------
// Batch Montgomery multiply
// ---------------------------------------------------------------------------

#[target_feature(enable = "neon")]
unsafe fn mul_assign_prime_montgomery_u32_neon<const Q: u64>(
    dst: *mut u32,
    src: *const u32,
    len: usize,
) {
    let mut i = 0usize;
    while i + 4 <= len {
        let lhs = vld1q_u32(dst.add(i));
        let rhs = vld1q_u32(src.add(i));
        let result = montgomery_mul_u32_neon::<Q>(lhs, rhs);
        vst1q_u32(dst.add(i), result);
        i += 4;
    }
    for j in i..len {
        let prod = PrimeField::<Q, u32>::mul_raw_words(*dst.add(j), *src.add(j));
        *dst.add(j) = prod;
    }
}

#[target_feature(enable = "neon")]
unsafe fn scalar_mul_prime_montgomery_u32_neon<const Q: u64>(
    dst: *mut u32,
    len: usize,
    scalar: u32,
) {
    let scalar_vec = vdupq_n_u32(scalar);
    let mut i = 0usize;
    while i + 4 <= len {
        let lhs = vld1q_u32(dst.add(i));
        let result = montgomery_mul_u32_neon::<Q>(lhs, scalar_vec);
        vst1q_u32(dst.add(i), result);
        i += 4;
    }
    for j in i..len {
        let prod = PrimeField::<Q, u32>::mul_raw_words(*dst.add(j), scalar);
        *dst.add(j) = prod;
    }
}

// ---------------------------------------------------------------------------
// Modular add / sub (4-lane)
// ---------------------------------------------------------------------------

#[target_feature(enable = "neon")]
unsafe fn add_assign_prime_u32_neon(dst: *mut u32, src: *const u32, len: usize, modulus: u64) {
    let q_vec = vdupq_n_u32(modulus as u32);
    let threshold_vec = vdupq_n_u32((modulus - 1) as u32);
    let small = modulus < (1u64 << 31);

    let mut i = 0usize;
    while i + 4 <= len {
        let lhs = vld1q_u32(dst.add(i));
        let rhs = vld1q_u32(src.add(i));
        let sum = vaddq_u32(lhs, rhs);
        if small {
            let reduce = vcgtq_u32(sum, threshold_vec);
            let result = vsubq_u32(sum, vandq_u32(reduce, q_vec));
            vst1q_u32(dst.add(i), result);
        } else {
            let carry = vcgtq_u32(lhs, sum);
            let ge_q = vcgtq_u32(sum, threshold_vec);
            let reduce = vorrq_u32(carry, ge_q);
            let result = vsubq_u32(sum, vandq_u32(reduce, q_vec));
            vst1q_u32(dst.add(i), result);
        }
        i += 4;
    }
    let q = modulus as u32;
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
unsafe fn sub_assign_prime_u32_neon(dst: *mut u32, src: *const u32, len: usize, modulus: u64) {
    let q_vec = vdupq_n_u32(modulus as u32);

    let mut i = 0usize;
    while i + 4 <= len {
        let lhs = vld1q_u32(dst.add(i));
        let rhs = vld1q_u32(src.add(i));
        let borrow = vcgtq_u32(rhs, lhs);
        let diff = vsubq_u32(lhs, rhs);
        let result = vaddq_u32(diff, vandq_u32(borrow, q_vec));
        vst1q_u32(dst.add(i), result);
        i += 4;
    }
    let q = modulus as u32;
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
unsafe fn butterfly_forward_prime_montgomery_u32_neon<const Q: u64>(
    even: *mut u32,
    odd: *mut u32,
    twiddles: *const u32,
    len: usize,
) {
    let q_vec = vdupq_n_u32(Q as u32);
    let threshold_vec = vdupq_n_u32((Q - 1) as u32);
    let small = Q < (1u64 << 31);

    let mut i = 0usize;
    while i + 4 <= len {
        let even_val = vld1q_u32(even.add(i));
        let odd_val = vld1q_u32(odd.add(i));
        let tw = vld1q_u32(twiddles.add(i));

        let temp = montgomery_mul_u32_neon::<Q>(odd_val, tw);

        // even' = even + temp
        let sum = vaddq_u32(even_val, temp);
        let even_new = if small {
            let reduce = vcgtq_u32(sum, threshold_vec);
            vsubq_u32(sum, vandq_u32(reduce, q_vec))
        } else {
            let carry = vcgtq_u32(even_val, sum);
            let ge_q = vcgtq_u32(sum, threshold_vec);
            let reduce = vorrq_u32(carry, ge_q);
            vsubq_u32(sum, vandq_u32(reduce, q_vec))
        };

        // odd' = even - temp
        let borrow = vcgtq_u32(temp, even_val);
        let diff = vsubq_u32(even_val, temp);
        let odd_new = vaddq_u32(diff, vandq_u32(borrow, q_vec));

        vst1q_u32(even.add(i), even_new);
        vst1q_u32(odd.add(i), odd_new);
        i += 4;
    }
    for j in i..len {
        let even_val = *even.add(j);
        let odd_val = *odd.add(j);
        let tw = *twiddles.add(j);
        let temp = PrimeField::<Q, u32>::mul_raw_words(odd_val, tw);
        *even.add(j) = PrimeField::<Q, u32>::add_raw_words(even_val, temp);
        *odd.add(j) = PrimeField::<Q, u32>::sub_raw_words(even_val, temp);
    }
}

#[target_feature(enable = "neon")]
unsafe fn butterfly_inverse_prime_montgomery_u32_neon<const Q: u64>(
    even: *mut u32,
    odd: *mut u32,
    twiddles: *const u32,
    len: usize,
) {
    let q_vec = vdupq_n_u32(Q as u32);
    let threshold_vec = vdupq_n_u32((Q - 1) as u32);
    let small = Q < (1u64 << 31);

    let mut i = 0usize;
    while i + 4 <= len {
        let even_val = vld1q_u32(even.add(i));
        let odd_val = vld1q_u32(odd.add(i));
        let tw = vld1q_u32(twiddles.add(i));

        // sum = even + odd
        let sum = vaddq_u32(even_val, odd_val);
        let even_new = if small {
            let reduce = vcgtq_u32(sum, threshold_vec);
            vsubq_u32(sum, vandq_u32(reduce, q_vec))
        } else {
            let carry = vcgtq_u32(even_val, sum);
            let ge_q = vcgtq_u32(sum, threshold_vec);
            let reduce = vorrq_u32(carry, ge_q);
            vsubq_u32(sum, vandq_u32(reduce, q_vec))
        };

        // diff = even - odd
        let borrow = vcgtq_u32(odd_val, even_val);
        let diff = vsubq_u32(even_val, odd_val);
        let diff_mod = vaddq_u32(diff, vandq_u32(borrow, q_vec));

        // odd' = diff * twiddle
        let odd_new = montgomery_mul_u32_neon::<Q>(diff_mod, tw);

        vst1q_u32(even.add(i), even_new);
        vst1q_u32(odd.add(i), odd_new);
        i += 4;
    }
    for j in i..len {
        let even_val = *even.add(j);
        let odd_val = *odd.add(j);
        let tw = *twiddles.add(j);
        let sum = PrimeField::<Q, u32>::add_raw_words(even_val, odd_val);
        let diff = PrimeField::<Q, u32>::sub_raw_words(even_val, odd_val);
        *even.add(j) = sum;
        *odd.add(j) = PrimeField::<Q, u32>::mul_raw_words(diff, tw);
    }
}

// ---------------------------------------------------------------------------
// Public wrappers
// ---------------------------------------------------------------------------

#[inline]
pub(crate) unsafe fn add_assign_prime_u32(dst: &mut [u32], src: &[u32], modulus: u64) {
    debug_assert_eq!(dst.len(), src.len());
    add_assign_prime_u32_neon(dst.as_mut_ptr(), src.as_ptr(), dst.len(), modulus);
}

#[inline]
pub(crate) unsafe fn sub_assign_prime_u32(dst: &mut [u32], src: &[u32], modulus: u64) {
    debug_assert_eq!(dst.len(), src.len());
    sub_assign_prime_u32_neon(dst.as_mut_ptr(), src.as_ptr(), dst.len(), modulus);
}

#[inline]
pub(crate) unsafe fn mul_assign_prime_montgomery_u32<const Q: u64>(dst: &mut [u32], src: &[u32]) {
    debug_assert_eq!(dst.len(), src.len());
    mul_assign_prime_montgomery_u32_neon::<Q>(dst.as_mut_ptr(), src.as_ptr(), dst.len());
}

#[inline]
pub(crate) unsafe fn scalar_mul_prime_montgomery_u32<const Q: u64>(dst: &mut [u32], scalar: u64) {
    scalar_mul_prime_montgomery_u32_neon::<Q>(dst.as_mut_ptr(), dst.len(), scalar as u32);
}

#[inline]
pub(crate) unsafe fn butterfly_forward_prime_montgomery_u32<const Q: u64>(
    even: &mut [u32],
    odd: &mut [u32],
    twiddles: &[u32],
) {
    debug_assert_eq!(even.len(), odd.len());
    debug_assert_eq!(even.len(), twiddles.len());
    butterfly_forward_prime_montgomery_u32_neon::<Q>(
        even.as_mut_ptr(),
        odd.as_mut_ptr(),
        twiddles.as_ptr(),
        even.len(),
    );
}

#[inline]
pub(crate) unsafe fn butterfly_inverse_prime_montgomery_u32<const Q: u64>(
    even: &mut [u32],
    odd: &mut [u32],
    twiddles: &[u32],
) {
    debug_assert_eq!(even.len(), odd.len());
    debug_assert_eq!(even.len(), twiddles.len());
    butterfly_inverse_prime_montgomery_u32_neon::<Q>(
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

    type F12289u32 = PrimeField<12289, u32>;
    type F4294967291u32 = PrimeField<4294967291, u32>;

    const N: usize = 10; // one full SIMD iter (4) + 2 scalar tail × multiple

    fn skip_if_no_neon() -> bool {
        !neon_available()
    }

    #[test]
    fn test_u32_small_add_sub_matches_scalar() {
        if skip_if_no_neon() {
            return;
        }

        let modulus = 12289u64;
        let q = modulus as u32;
        let lhs: [u32; N] = core::array::from_fn(|i| (i as u64 * 300 + 1) as u32 % q);
        let rhs: [u32; N] = core::array::from_fn(|i| (i as u64 * 70 + 5) as u32 % q);

        let mut simd_add = lhs;
        let mut simd_sub = lhs;
        let scalar_add: [u32; N] = core::array::from_fn(|i| {
            let s = lhs[i] + rhs[i];
            if s >= q { s - q } else { s }
        });
        let scalar_sub: [u32; N] = core::array::from_fn(|i| {
            if lhs[i] >= rhs[i] {
                lhs[i] - rhs[i]
            } else {
                lhs[i] + q - rhs[i]
            }
        });

        unsafe {
            add_assign_prime_u32(&mut simd_add, &rhs, modulus);
            sub_assign_prime_u32(&mut simd_sub, &rhs, modulus);
        }

        assert_eq!(simd_add, scalar_add);
        assert_eq!(simd_sub, scalar_sub);
    }

    #[test]
    fn test_u32_small_mul_matches_scalar() {
        if skip_if_no_neon() {
            return;
        }

        let lhs: [u32; N] =
            core::array::from_fn(|i| F12289u32::from_u64((i * 300 + 1) as u64).raw());
        let rhs: [u32; N] =
            core::array::from_fn(|i| F12289u32::from_u64((i * 70 + 5) as u64).raw());

        let mut simd_mul = lhs;
        let scalar_mul: [u32; N] =
            core::array::from_fn(|i| F12289u32::mul_raw_words(lhs[i], rhs[i]));

        unsafe {
            mul_assign_prime_montgomery_u32::<12289>(&mut simd_mul, &rhs);
        }

        assert_eq!(simd_mul, scalar_mul);
    }

    #[test]
    fn test_u32_small_butterfly_matches_scalar() {
        if skip_if_no_neon() {
            return;
        }

        let mut fwd_even: [u32; N] =
            core::array::from_fn(|i| F12289u32::from_u64((i * 13 + 10) as u64).raw());
        let mut fwd_odd: [u32; N] =
            core::array::from_fn(|i| F12289u32::from_u64((i * 7 + 3) as u64).raw());
        let twiddles: [u32; N] =
            core::array::from_fn(|i| F12289u32::from_u64((i * 5 + 1) as u64).raw());

        let mut s_e = fwd_even;
        let mut s_o = fwd_odd;
        for i in 0..N {
            let u = s_e[i];
            let t = F12289u32::mul_raw_words(s_o[i], twiddles[i]);
            s_e[i] = F12289u32::add_raw_words(u, t);
            s_o[i] = F12289u32::sub_raw_words(u, t);
        }

        unsafe {
            butterfly_forward_prime_montgomery_u32::<12289>(&mut fwd_even, &mut fwd_odd, &twiddles);
        }

        assert_eq!(fwd_even, s_e);
        assert_eq!(fwd_odd, s_o);
    }

    #[test]
    fn test_u32_small_inverse_butterfly_matches_scalar() {
        if skip_if_no_neon() {
            return;
        }

        let mut inv_even: [u32; N] =
            core::array::from_fn(|i| F12289u32::from_u64((i * 13 + 10) as u64).raw());
        let mut inv_odd: [u32; N] =
            core::array::from_fn(|i| F12289u32::from_u64((i * 7 + 3) as u64).raw());
        let twiddles: [u32; N] =
            core::array::from_fn(|i| F12289u32::from_u64((i * 5 + 1) as u64).raw());

        let mut s_e = inv_even;
        let mut s_o = inv_odd;
        for i in 0..N {
            let sum = F12289u32::add_raw_words(s_e[i], s_o[i]);
            let diff = F12289u32::sub_raw_words(s_e[i], s_o[i]);
            s_e[i] = sum;
            s_o[i] = F12289u32::mul_raw_words(diff, twiddles[i]);
        }

        unsafe {
            butterfly_inverse_prime_montgomery_u32::<12289>(&mut inv_even, &mut inv_odd, &twiddles);
        }

        assert_eq!(inv_even, s_e);
        assert_eq!(inv_odd, s_o);
    }

    #[test]
    fn test_u32_small_ntt_round_trip() {
        if skip_if_no_neon() {
            return;
        }
        use crate::arith::ntt::NttPlan;
        use alloc::vec::Vec;

        let n = 64;
        let original: Vec<F12289u32> = (0..n)
            .map(|i| F12289u32::from_u64((i * 73 + 1) as u64 % 12289))
            .collect();

        let plan = NttPlan::<F12289u32>::build(n).unwrap();
        let mut values = original.clone();
        crate::simd::montgomery_prime::ntt_forward_with_plan(&mut values, &plan).unwrap();
        crate::simd::montgomery_prime::ntt_inverse_with_plan(&mut values, &plan).unwrap();

        assert_eq!(values, original);
    }

    // --- Large modulus (Q >= 2^31) tests ---

    #[test]
    fn test_u32_large_add_sub_matches_scalar() {
        if skip_if_no_neon() {
            return;
        }

        // 4294967291 is prime and > 2^31
        let modulus = 4294967291u64;
        let q = modulus as u32;
        let lhs: [u32; N] = core::array::from_fn(|i| (i as u64 * 500_000_000 + 1) as u32 % q);
        let rhs: [u32; N] = core::array::from_fn(|i| (i as u64 * 800_000_000 + 5) as u32 % q);

        let mut simd_add = lhs;
        let mut simd_sub = lhs;
        let scalar_add: [u32; N] = core::array::from_fn(|i| {
            let (s, carry) = lhs[i].overflowing_add(rhs[i]);
            if carry || s >= q {
                s.wrapping_sub(q)
            } else {
                s
            }
        });
        let scalar_sub: [u32; N] = core::array::from_fn(|i| {
            if lhs[i] >= rhs[i] {
                lhs[i] - rhs[i]
            } else {
                q.wrapping_sub(rhs[i]).wrapping_add(lhs[i])
            }
        });

        unsafe {
            add_assign_prime_u32(&mut simd_add, &rhs, modulus);
            sub_assign_prime_u32(&mut simd_sub, &rhs, modulus);
        }

        assert_eq!(simd_add, scalar_add);
        assert_eq!(simd_sub, scalar_sub);
    }

    #[test]
    fn test_u32_large_mul_matches_scalar() {
        if skip_if_no_neon() {
            return;
        }

        let lhs: [u32; N] =
            core::array::from_fn(|i| F4294967291u32::from_u64((i * 500_000_000 + 1) as u64).raw());
        let rhs: [u32; N] =
            core::array::from_fn(|i| F4294967291u32::from_u64((i * 800_000_000 + 5) as u64).raw());

        let mut simd_mul = lhs;
        let scalar_mul: [u32; N] =
            core::array::from_fn(|i| F4294967291u32::mul_raw_words(lhs[i], rhs[i]));

        unsafe {
            mul_assign_prime_montgomery_u32::<4294967291>(&mut simd_mul, &rhs);
        }

        assert_eq!(simd_mul, scalar_mul);
    }

    #[test]
    fn test_u32_large_butterfly_matches_scalar() {
        if skip_if_no_neon() {
            return;
        }

        let mut fwd_even: [u32; N] = core::array::from_fn(|i| {
            F4294967291u32::from_u64((i as u64 * 100_000_000 + 1) % 4294967291).raw()
        });
        let mut fwd_odd: [u32; N] = core::array::from_fn(|i| {
            F4294967291u32::from_u64((i as u64 * 200_000_000 + 5) % 4294967291).raw()
        });
        let twiddles: [u32; N] = core::array::from_fn(|i| {
            F4294967291u32::from_u64((i as u64 * 70_000_000 + 3) % 4294967291).raw()
        });

        let mut s_e = fwd_even;
        let mut s_o = fwd_odd;
        for i in 0..N {
            let u = s_e[i];
            let t = F4294967291u32::mul_raw_words(s_o[i], twiddles[i]);
            s_e[i] = F4294967291u32::add_raw_words(u, t);
            s_o[i] = F4294967291u32::sub_raw_words(u, t);
        }

        unsafe {
            butterfly_forward_prime_montgomery_u32::<4294967291>(
                &mut fwd_even,
                &mut fwd_odd,
                &twiddles,
            );
        }

        assert_eq!(fwd_even, s_e);
        assert_eq!(fwd_odd, s_o);
    }

    #[test]
    fn test_u32_large_inverse_butterfly_matches_scalar() {
        if skip_if_no_neon() {
            return;
        }

        let mut inv_even: [u32; N] = core::array::from_fn(|i| {
            F4294967291u32::from_u64((i as u64 * 400_000_000 + 7) % 4294967291).raw()
        });
        let mut inv_odd: [u32; N] = core::array::from_fn(|i| {
            F4294967291u32::from_u64((i as u64 * 600_000_000 + 11) % 4294967291).raw()
        });
        let twiddles: [u32; N] = core::array::from_fn(|i| {
            F4294967291u32::from_u64((i as u64 * 150_000_000 + 1) % 4294967291).raw()
        });

        let mut s_e = inv_even;
        let mut s_o = inv_odd;
        for i in 0..N {
            let sum = F4294967291u32::add_raw_words(s_e[i], s_o[i]);
            let diff = F4294967291u32::sub_raw_words(s_e[i], s_o[i]);
            s_e[i] = sum;
            s_o[i] = F4294967291u32::mul_raw_words(diff, twiddles[i]);
        }

        unsafe {
            butterfly_inverse_prime_montgomery_u32::<4294967291>(
                &mut inv_even,
                &mut inv_odd,
                &twiddles,
            );
        }

        assert_eq!(inv_even, s_e);
        assert_eq!(inv_odd, s_o);
    }
}
