//! u32 AVX2 SIMD kernels for Montgomery prime field arithmetic.
//!
//! 8 lanes per `__m256i` for add/sub/compare, but widending multiply only
//! covers 4 lanes per `_mm256_mul_epu32` call → two-pass (even/odd) for full coverage.
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(unused_unsafe)]

use core::arch::x86_64::{
    __m256i, _mm256_add_epi32, _mm256_and_si256, _mm256_cmpgt_epi32, _mm256_loadu_si256,
    _mm256_mul_epu32, _mm256_mullo_epi32, _mm256_or_si256, _mm256_set1_epi32, _mm256_slli_epi64,
    _mm256_srli_epi64, _mm256_storeu_si256, _mm256_sub_epi32, _mm256_xor_si256,
};

use crate::arith::prime::PrimeField;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline(always)]
unsafe fn cmpgt_epu32_avx2(lhs: __m256i, rhs: __m256i) -> __m256i {
    let sign = unsafe { _mm256_set1_epi32(i32::MIN) };
    unsafe { _mm256_cmpgt_epi32(_mm256_xor_si256(lhs, sign), _mm256_xor_si256(rhs, sign)) }
}

/// Full 8-lane u32×u32 → (lo:8×u32, hi:8×u32). Uses two `_mm256_mul_epu32` passes.
#[target_feature(enable = "avx2")]
unsafe fn widen_mul_u32_full(a: __m256i, b: __m256i) -> (__m256i, __m256i) {
    // Low 32 bits: mullo_epi32 covers all 8 lanes.
    let lo = unsafe { _mm256_mullo_epi32(a, b) };

    // High 32 bits via two mul_epu32 passes (each covers 4 even-positioned lanes).
    let t_even = unsafe { _mm256_mul_epu32(a, b) }; // lanes 0,2,4,6 as u64
    let a_odd = unsafe { _mm256_srli_epi64::<32>(a) };
    let b_odd = unsafe { _mm256_srli_epi64::<32>(b) };
    let t_odd = unsafe { _mm256_mul_epu32(a_odd, b_odd) }; // lanes 1,3,5,7

    // Extract high 32 bits from each u64 product.
    // hi_even = [hi0, 0, hi2, 0, hi4, 0, hi6, 0] as u32 lanes
    let hi_even = unsafe { _mm256_srli_epi64::<32>(t_even) };
    // hi_odd  = [hi1, 0, hi3, 0, hi5, 0, hi7, 0] as u32 lanes (low 32b of each u64)
    let hi_odd = unsafe { _mm256_srli_epi64::<32>(t_odd) };

    // Merge: shift odd values up 32 bits per lane, OR with even.
    // hi_odd << 32 = [0, hi1, 0, hi3, 0, hi5, 0, hi7]
    // Then OR: [hi0, hi1, hi2, hi3, hi4, hi5, hi6, hi7]
    let hi_odd_shifted = unsafe { _mm256_slli_epi64::<32>(hi_odd) };
    let hi = unsafe { _mm256_or_si256(hi_even, hi_odd_shifted) };

    (lo, hi)
}

// ---------------------------------------------------------------------------
// Montgomery multiplication kernel
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
unsafe fn montgomery_mul_u32_avx2<const Q: u64>(lhs: __m256i, rhs: __m256i) -> __m256i {
    let q_vec = unsafe { _mm256_set1_epi32(Q as i32) };
    let q_inv_vec = unsafe { _mm256_set1_epi32(PrimeField::<Q>::Q_INV_U64 as u32 as i32) };
    let threshold_vec = unsafe { _mm256_set1_epi32((Q - 1) as i32) };
    let ones = unsafe { _mm256_set1_epi32(1) };
    let small = Q < (1u64 << 31);

    // 1. t = a * b (full 8-lane widen)
    let (t_lo, t_hi) = widen_mul_u32_full(lhs, rhs);

    // 2. m = t_lo * q_inv (mod 2^32)
    let m = unsafe { _mm256_mullo_epi32(t_lo, q_inv_vec) };

    // 3. mq = m * Q
    let (mq_lo, mq_hi) = widen_mul_u32_full(m, q_vec);

    // 4. sum = t + mq (u64 addition across lo/hi u32 halves)
    let sum_lo = unsafe { _mm256_add_epi32(t_lo, mq_lo) };
    let carry_lo = cmpgt_epu32_avx2(t_lo, sum_lo);
    let sum_hi_a = unsafe { _mm256_add_epi32(t_hi, mq_hi) };
    let sum_hi = unsafe { _mm256_add_epi32(sum_hi_a, _mm256_and_si256(carry_lo, ones)) };

    // 5-6. Overflow detection and reduction
    if small {
        let reduce = cmpgt_epu32_avx2(sum_hi, threshold_vec);
        unsafe { _mm256_sub_epi32(sum_hi, _mm256_and_si256(reduce, q_vec)) }
    } else {
        let carry_hi_a = cmpgt_epu32_avx2(t_hi, sum_hi_a);
        let carry_hi_b = cmpgt_epu32_avx2(sum_hi_a, sum_hi);
        let overflow = unsafe { _mm256_or_si256(carry_hi_a, carry_hi_b) };
        let ge_q = cmpgt_epu32_avx2(sum_hi, threshold_vec);
        let reduce = unsafe { _mm256_or_si256(overflow, ge_q) };
        unsafe { _mm256_sub_epi32(sum_hi, _mm256_and_si256(reduce, q_vec)) }
    }
}

// ---------------------------------------------------------------------------
// Batch Montgomery multiply
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
unsafe fn mul_assign_prime_montgomery_u32_avx2<const Q: u64>(
    dst: *mut u32,
    src: *const u32,
    len: usize,
) {
    let mut i = 0usize;
    while i + 8 <= len {
        let lhs = unsafe { _mm256_loadu_si256(dst.add(i).cast::<__m256i>()) };
        let rhs = unsafe { _mm256_loadu_si256(src.add(i).cast::<__m256i>()) };
        let result = montgomery_mul_u32_avx2::<Q>(lhs, rhs);
        unsafe { _mm256_storeu_si256(dst.add(i).cast::<__m256i>(), result) };
        i += 8;
    }
    for j in i..len {
        let prod = PrimeField::<Q, u32>::mul_raw_words(*dst.add(j), *src.add(j));
        *dst.add(j) = prod;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn scalar_mul_prime_montgomery_u32_avx2<const Q: u64>(
    dst: *mut u32,
    len: usize,
    scalar: u32,
) {
    let scalar_vec = unsafe { _mm256_set1_epi32(scalar as i32) };
    let mut i = 0usize;
    while i + 8 <= len {
        let lhs = unsafe { _mm256_loadu_si256(dst.add(i).cast::<__m256i>()) };
        let result = montgomery_mul_u32_avx2::<Q>(lhs, scalar_vec);
        unsafe { _mm256_storeu_si256(dst.add(i).cast::<__m256i>(), result) };
        i += 8;
    }
    for j in i..len {
        let prod = PrimeField::<Q, u32>::mul_raw_words(*dst.add(j), scalar);
        *dst.add(j) = prod;
    }
}

// ---------------------------------------------------------------------------
// Modular add / sub (8-lane)
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
unsafe fn add_assign_prime_u32_avx2(dst: *mut u32, src: *const u32, len: usize, modulus: u64) {
    let q_vec = unsafe { _mm256_set1_epi32(modulus as i32) };
    let threshold_vec = unsafe { _mm256_set1_epi32((modulus - 1) as i32) };
    let small = modulus < (1u64 << 31);

    let mut i = 0usize;
    while i + 8 <= len {
        let lhs = unsafe { _mm256_loadu_si256(dst.add(i).cast::<__m256i>()) };
        let rhs = unsafe { _mm256_loadu_si256(src.add(i).cast::<__m256i>()) };
        let sum = unsafe { _mm256_add_epi32(lhs, rhs) };
        if small {
            let reduce = cmpgt_epu32_avx2(sum, threshold_vec);
            let result = unsafe { _mm256_sub_epi32(sum, _mm256_and_si256(reduce, q_vec)) };
            unsafe { _mm256_storeu_si256(dst.add(i).cast::<__m256i>(), result) };
        } else {
            let carry = cmpgt_epu32_avx2(lhs, sum);
            let ge_q = cmpgt_epu32_avx2(sum, threshold_vec);
            let reduce = unsafe { _mm256_or_si256(carry, ge_q) };
            let result = unsafe { _mm256_sub_epi32(sum, _mm256_and_si256(reduce, q_vec)) };
            unsafe { _mm256_storeu_si256(dst.add(i).cast::<__m256i>(), result) };
        }
        i += 8;
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

#[target_feature(enable = "avx2")]
unsafe fn sub_assign_prime_u32_avx2(dst: *mut u32, src: *const u32, len: usize, modulus: u64) {
    let q_vec = unsafe { _mm256_set1_epi32(modulus as i32) };

    let mut i = 0usize;
    while i + 8 <= len {
        let lhs = unsafe { _mm256_loadu_si256(dst.add(i).cast::<__m256i>()) };
        let rhs = unsafe { _mm256_loadu_si256(src.add(i).cast::<__m256i>()) };
        let borrow = cmpgt_epu32_avx2(rhs, lhs);
        let diff = unsafe { _mm256_sub_epi32(lhs, rhs) };
        let result = unsafe { _mm256_add_epi32(diff, _mm256_and_si256(borrow, q_vec)) };
        unsafe { _mm256_storeu_si256(dst.add(i).cast::<__m256i>(), result) };
        i += 8;
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

#[target_feature(enable = "avx2")]
unsafe fn butterfly_forward_prime_montgomery_u32_avx2<const Q: u64>(
    even: *mut u32,
    odd: *mut u32,
    twiddles: *const u32,
    len: usize,
) {
    let q_vec = unsafe { _mm256_set1_epi32(Q as i32) };
    let threshold_vec = unsafe { _mm256_set1_epi32((Q - 1) as i32) };
    let small = Q < (1u64 << 31);

    let mut i = 0usize;
    while i + 8 <= len {
        let even_val = unsafe { _mm256_loadu_si256(even.add(i).cast::<__m256i>()) };
        let odd_val = unsafe { _mm256_loadu_si256(odd.add(i).cast::<__m256i>()) };
        let tw = unsafe { _mm256_loadu_si256(twiddles.add(i).cast::<__m256i>()) };

        // temp = odd * twiddle
        let temp = montgomery_mul_u32_avx2::<Q>(odd_val, tw);

        // even' = even + temp
        let sum = unsafe { _mm256_add_epi32(even_val, temp) };
        let even_new = if small {
            let reduce = cmpgt_epu32_avx2(sum, threshold_vec);
            unsafe { _mm256_sub_epi32(sum, _mm256_and_si256(reduce, q_vec)) }
        } else {
            let carry = cmpgt_epu32_avx2(even_val, sum);
            let ge_q = cmpgt_epu32_avx2(sum, threshold_vec);
            let reduce = unsafe { _mm256_or_si256(carry, ge_q) };
            unsafe { _mm256_sub_epi32(sum, _mm256_and_si256(reduce, q_vec)) }
        };

        // odd' = even - temp
        let borrow = cmpgt_epu32_avx2(temp, even_val);
        let diff = unsafe { _mm256_sub_epi32(even_val, temp) };
        let odd_new = unsafe { _mm256_add_epi32(diff, _mm256_and_si256(borrow, q_vec)) };

        unsafe { _mm256_storeu_si256(even.add(i).cast::<__m256i>(), even_new) };
        unsafe { _mm256_storeu_si256(odd.add(i).cast::<__m256i>(), odd_new) };
        i += 8;
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

#[target_feature(enable = "avx2")]
unsafe fn butterfly_inverse_prime_montgomery_u32_avx2<const Q: u64>(
    even: *mut u32,
    odd: *mut u32,
    twiddles: *const u32,
    len: usize,
) {
    let q_vec = unsafe { _mm256_set1_epi32(Q as i32) };
    let threshold_vec = unsafe { _mm256_set1_epi32((Q - 1) as i32) };
    let small = Q < (1u64 << 31);

    let mut i = 0usize;
    while i + 8 <= len {
        let even_val = unsafe { _mm256_loadu_si256(even.add(i).cast::<__m256i>()) };
        let odd_val = unsafe { _mm256_loadu_si256(odd.add(i).cast::<__m256i>()) };
        let tw = unsafe { _mm256_loadu_si256(twiddles.add(i).cast::<__m256i>()) };

        // sum = even + odd
        let sum = unsafe { _mm256_add_epi32(even_val, odd_val) };
        let even_new = if small {
            let reduce = cmpgt_epu32_avx2(sum, threshold_vec);
            unsafe { _mm256_sub_epi32(sum, _mm256_and_si256(reduce, q_vec)) }
        } else {
            let carry = cmpgt_epu32_avx2(even_val, sum);
            let ge_q = cmpgt_epu32_avx2(sum, threshold_vec);
            let reduce = unsafe { _mm256_or_si256(carry, ge_q) };
            unsafe { _mm256_sub_epi32(sum, _mm256_and_si256(reduce, q_vec)) }
        };

        // diff = even - odd
        let borrow = cmpgt_epu32_avx2(odd_val, even_val);
        let diff = unsafe { _mm256_sub_epi32(even_val, odd_val) };
        let diff_mod = unsafe { _mm256_add_epi32(diff, _mm256_and_si256(borrow, q_vec)) };

        // odd' = diff * twiddle
        let odd_new = montgomery_mul_u32_avx2::<Q>(diff_mod, tw);

        unsafe { _mm256_storeu_si256(even.add(i).cast::<__m256i>(), even_new) };
        unsafe { _mm256_storeu_si256(odd.add(i).cast::<__m256i>(), odd_new) };
        i += 8;
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
    add_assign_prime_u32_avx2(dst.as_mut_ptr(), src.as_ptr(), dst.len(), modulus);
}

#[inline]
pub(crate) unsafe fn sub_assign_prime_u32(dst: &mut [u32], src: &[u32], modulus: u64) {
    debug_assert_eq!(dst.len(), src.len());
    sub_assign_prime_u32_avx2(dst.as_mut_ptr(), src.as_ptr(), dst.len(), modulus);
}

#[inline]
pub(crate) unsafe fn mul_assign_prime_montgomery_u32<const Q: u64>(dst: &mut [u32], src: &[u32]) {
    debug_assert_eq!(dst.len(), src.len());
    mul_assign_prime_montgomery_u32_avx2::<Q>(dst.as_mut_ptr(), src.as_ptr(), dst.len());
}

#[inline]
pub(crate) unsafe fn scalar_mul_prime_montgomery_u32<const Q: u64>(dst: &mut [u32], scalar: u64) {
    scalar_mul_prime_montgomery_u32_avx2::<Q>(dst.as_mut_ptr(), dst.len(), scalar as u32);
}

#[inline]
pub(crate) unsafe fn butterfly_forward_prime_montgomery_u32<const Q: u64>(
    even: &mut [u32],
    odd: &mut [u32],
    twiddles: &[u32],
) {
    debug_assert_eq!(even.len(), odd.len());
    debug_assert_eq!(even.len(), twiddles.len());
    butterfly_forward_prime_montgomery_u32_avx2::<Q>(
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
    butterfly_inverse_prime_montgomery_u32_avx2::<Q>(
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
    use super::super::u64_arith::avx2_available;
    use super::*;
    use crate::arith::prime::PrimeField;
    use crate::arith::ring::IntegerRing;

    type F12289u32 = PrimeField<12289, u32>;
    type F4294967291u32 = PrimeField<4294967291, u32>;

    const N: usize = 10; // one full SIMD iter (8) + 2 scalar tail

    fn skip_if_no_avx2() -> bool {
        !avx2_available()
    }

    #[test]
    fn test_u32_small_add_sub_matches_scalar() {
        if skip_if_no_avx2() {
            return;
        }
        let modulus = 12289u64;
        let q = modulus as u32;
        let lhs: [u32; N] = core::array::from_fn(|i| ((i as u64 * 700 + 1) % 12289) as u32);
        let rhs: [u32; N] = core::array::from_fn(|i| ((i as u64 * 300 + 5) % 12289) as u32);
        let mut sa = lhs;
        let mut ss = lhs;
        let scalar_add: [u32; N] = core::array::from_fn(|i| {
            let s = lhs[i].wrapping_add(rhs[i]);
            if s >= q { s - q } else { s }
        });
        let scalar_sub: [u32; N] = core::array::from_fn(|i| {
            if lhs[i] >= rhs[i] {
                lhs[i] - rhs[i]
            } else {
                q - rhs[i] + lhs[i]
            }
        });
        unsafe {
            add_assign_prime_u32(&mut sa, &rhs, modulus);
            sub_assign_prime_u32(&mut ss, &rhs, modulus);
        }
        assert_eq!(sa, scalar_add);
        assert_eq!(ss, scalar_sub);
    }

    #[test]
    fn test_u32_small_mul_matches_scalar() {
        if skip_if_no_avx2() {
            return;
        }
        let lhs: [u32; N] =
            core::array::from_fn(|i| F12289u32::from_u64((i as u64 * 700 + 1) % 12289).raw());
        let rhs: [u32; N] =
            core::array::from_fn(|i| F12289u32::from_u64((i as u64 * 300 + 5) % 12289).raw());
        let mut simd = lhs;
        let scalar: [u32; N] = core::array::from_fn(|i| F12289u32::mul_raw_words(lhs[i], rhs[i]));
        unsafe {
            mul_assign_prime_montgomery_u32::<12289>(&mut simd, &rhs);
        }
        assert_eq!(simd, scalar);
    }

    #[test]
    fn test_u32_small_scalar_mul_matches_scalar() {
        if skip_if_no_avx2() {
            return;
        }
        let scalar = F12289u32::from_u64(19);
        let vals: [u32; N] =
            core::array::from_fn(|i| F12289u32::from_u64((i as u64 * 700 + 1) % 12289).raw());
        let mut simd = vals;
        let expected: [u32; N] =
            core::array::from_fn(|i| F12289u32::mul_raw_words(vals[i], scalar.raw()));
        unsafe {
            scalar_mul_prime_montgomery_u32::<12289>(&mut simd, scalar.raw() as u64);
        }
        assert_eq!(simd, expected);
    }

    #[test]
    fn test_u32_small_butterfly_matches_scalar() {
        if skip_if_no_avx2() {
            return;
        }
        let mut fwd_even: [u32; N] =
            core::array::from_fn(|i| F12289u32::from_u64((i as u64 * 700 + 1) % 12289).raw());
        let mut fwd_odd: [u32; N] =
            core::array::from_fn(|i| F12289u32::from_u64((i as u64 * 300 + 5) % 12289).raw());
        let twiddles: [u32; N] =
            core::array::from_fn(|i| F12289u32::from_u64((i as u64 * 113 + 7) % 12289).raw());
        let mut se = fwd_even;
        let mut so = fwd_odd;
        for i in 0..N {
            let t = F12289u32::mul_raw_words(so[i], twiddles[i]);
            let s = F12289u32::add_raw_words(se[i], t);
            let d = F12289u32::sub_raw_words(se[i], t);
            se[i] = s;
            so[i] = d;
        }
        unsafe {
            butterfly_forward_prime_montgomery_u32::<12289>(&mut fwd_even, &mut fwd_odd, &twiddles);
        }
        assert_eq!(fwd_even, se);
        assert_eq!(fwd_odd, so);
    }

    #[test]
    fn test_u32_small_inverse_butterfly_matches_scalar() {
        if skip_if_no_avx2() {
            return;
        }
        let mut inv_even: [u32; N] =
            core::array::from_fn(|i| F12289u32::from_u64((i as u64 * 500 + 100) % 12289).raw());
        let mut inv_odd: [u32; N] =
            core::array::from_fn(|i| F12289u32::from_u64((i as u64 * 70 + 5) % 12289).raw());
        let twiddles: [u32; N] =
            core::array::from_fn(|i| F12289u32::from_u64((i as u64 * 113 + 7) % 12289).raw());
        let mut se = inv_even;
        let mut so = inv_odd;
        for i in 0..N {
            let s = F12289u32::add_raw_words(se[i], so[i]);
            let d = F12289u32::sub_raw_words(se[i], so[i]);
            se[i] = s;
            so[i] = F12289u32::mul_raw_words(d, twiddles[i]);
        }
        unsafe {
            butterfly_inverse_prime_montgomery_u32::<12289>(&mut inv_even, &mut inv_odd, &twiddles);
        }
        assert_eq!(inv_even, se);
        assert_eq!(inv_odd, so);
    }

    #[test]
    fn test_u32_small_ntt_round_trip() {
        if skip_if_no_avx2() {
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

    #[test]
    fn test_u32_large_add_sub_matches_scalar() {
        if skip_if_no_avx2() {
            return;
        }
        // 4294967291 > 2^31, uses carry detection
        let modulus = 4294967291u64;
        let q = modulus as u32;
        let lhs: [u32; N] =
            core::array::from_fn(|i| (i as u64 * 1000000000 + 1).wrapping_rem(modulus) as u32);
        let rhs: [u32; N] =
            core::array::from_fn(|i| (i as u64 * 2000000000 + 5).wrapping_rem(modulus) as u32);
        let mut sa = lhs;
        let mut ss = lhs;
        let scalar_add: [u32; N] = core::array::from_fn(|i| {
            let (s, c) = lhs[i].overflowing_add(rhs[i]);
            if c || s >= q { s.wrapping_sub(q) } else { s }
        });
        let scalar_sub: [u32; N] = core::array::from_fn(|i| {
            if lhs[i] >= rhs[i] {
                lhs[i] - rhs[i]
            } else {
                q.wrapping_sub(rhs[i]).wrapping_add(lhs[i])
            }
        });
        unsafe {
            add_assign_prime_u32(&mut sa, &rhs, modulus);
            sub_assign_prime_u32(&mut ss, &rhs, modulus);
        }
        assert_eq!(sa, scalar_add);
        assert_eq!(ss, scalar_sub);
    }

    #[test]
    fn test_u32_large_mul_matches_scalar() {
        if skip_if_no_avx2() {
            return;
        }
        let lhs: [u32; N] = core::array::from_fn(|i| {
            F4294967291u32::from_u64((i * 1000000000 + 1) as u64 % 4294967291).raw()
        });
        let rhs: [u32; N] = core::array::from_fn(|i| {
            F4294967291u32::from_u64((i * 2000000000 + 5) as u64 % 4294967291).raw()
        });
        let mut simd = lhs;
        let scalar: [u32; N] =
            core::array::from_fn(|i| F4294967291u32::mul_raw_words(lhs[i], rhs[i]));
        unsafe {
            mul_assign_prime_montgomery_u32::<4294967291>(&mut simd, &rhs);
        }
        assert_eq!(simd, scalar);
    }

    #[test]
    fn test_u32_large_butterfly_matches_scalar() {
        if skip_if_no_avx2() {
            return;
        }
        // 4294967291 > 2^31, exercises carry detection in butterfly add path
        let mut fwd_even: [u32; N] = core::array::from_fn(|i| {
            F4294967291u32::from_u64((i as u64 * 1_000_000_000 + 1) % 4_294_967_291).raw()
        });
        let mut fwd_odd: [u32; N] = core::array::from_fn(|i| {
            F4294967291u32::from_u64((i as u64 * 2_000_000_000 + 5) % 4_294_967_291).raw()
        });
        let twiddles: [u32; N] = core::array::from_fn(|i| {
            F4294967291u32::from_u64((i as u64 * 700_000_000 + 3) % 4_294_967_291).raw()
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
            butterfly_forward_prime_montgomery_u32::<4_294_967_291>(
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
        if skip_if_no_avx2() {
            return;
        }
        let mut inv_even: [u32; N] = core::array::from_fn(|i| {
            F4294967291u32::from_u64((i as u64 * 800_000_000 + 7) % 4_294_967_291).raw()
        });
        let mut inv_odd: [u32; N] = core::array::from_fn(|i| {
            F4294967291u32::from_u64((i as u64 * 1_500_000_000 + 11) % 4_294_967_291).raw()
        });
        let twiddles: [u32; N] = core::array::from_fn(|i| {
            F4294967291u32::from_u64((i as u64 * 500_000_000 + 1) % 4_294_967_291).raw()
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
            butterfly_inverse_prime_montgomery_u32::<4_294_967_291>(
                &mut inv_even,
                &mut inv_odd,
                &twiddles,
            );
        }
        assert_eq!(inv_even, s_e);
        assert_eq!(inv_odd, s_o);
    }
}
