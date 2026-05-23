//! u16 AVX2 SIMD kernels for Montgomery prime field arithmetic.
//!
//! 16 lanes of u16 per `__m256i` (vs 4 lanes of u64).
#![allow(unsafe_op_in_unsafe_fn)]

use core::arch::x86_64::{
    __m256i, _mm256_add_epi16, _mm256_and_si256, _mm256_cmpgt_epi16, _mm256_loadu_si256,
    _mm256_mulhi_epu16, _mm256_mullo_epi16, _mm256_or_si256, _mm256_set1_epi16,
    _mm256_storeu_si256, _mm256_sub_epi16, _mm256_xor_si256,
};

use crate::arith::prime::PrimeField;

#[inline(always)]
unsafe fn cmpgt_epu16_avx2(lhs: __m256i, rhs: __m256i) -> __m256i {
    let sign = unsafe { _mm256_set1_epi16(i16::MIN) };
    unsafe { _mm256_cmpgt_epi16(_mm256_xor_si256(lhs, sign), _mm256_xor_si256(rhs, sign)) }
}

// ---------------------------------------------------------------------------
// Montgomery multiplication kernel
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
unsafe fn montgomery_mul_u16_avx2<const Q: u64>(lhs: __m256i, rhs: __m256i) -> __m256i {
    let q_vec = _mm256_set1_epi16(Q as i16);
    let q_inv_vec = _mm256_set1_epi16(PrimeField::<Q>::Q_INV_U64 as u16 as i16);
    let threshold_vec = _mm256_set1_epi16((Q - 1) as i16);
    let ones = _mm256_set1_epi16(1);

    // 1. t = a * b (widening 16x16 -> 32, split lo/hi)
    let t_lo = _mm256_mullo_epi16(lhs, rhs);
    let t_hi = _mm256_mulhi_epu16(lhs, rhs);

    // 2. m = t_lo * q_inv (mod 2^16)
    let m = _mm256_mullo_epi16(t_lo, q_inv_vec);

    // 3. mq = m * Q (widening 16x16 -> 32)
    let mq_lo = _mm256_mullo_epi16(m, q_vec);
    let mq_hi = _mm256_mulhi_epu16(m, q_vec);

    // 4. sum = t + mq (u32 addition across lo/hi u16 halves)
    let sum_lo = _mm256_add_epi16(t_lo, mq_lo);
    let carry_lo = cmpgt_epu16_avx2(t_lo, sum_lo);
    let sum_hi_a = _mm256_add_epi16(t_hi, mq_hi);
    let sum_hi = _mm256_add_epi16(sum_hi_a, _mm256_and_si256(carry_lo, ones));

    // 5-6. Overflow detection and reduction
    if Q < (1u64 << 15) {
        // Small modulus: no overflow possible in the u32 sum
        let reduce = cmpgt_epu16_avx2(sum_hi, threshold_vec);
        _mm256_sub_epi16(sum_hi, _mm256_and_si256(reduce, q_vec))
    } else {
        // Large modulus: check for overflow
        let carry_hi_a = cmpgt_epu16_avx2(t_hi, sum_hi_a);
        let carry_hi_b = cmpgt_epu16_avx2(sum_hi_a, sum_hi);
        let overflow = _mm256_or_si256(carry_hi_a, carry_hi_b);
        let ge_q = cmpgt_epu16_avx2(sum_hi, threshold_vec);
        let reduce = _mm256_or_si256(overflow, ge_q);
        _mm256_sub_epi16(sum_hi, _mm256_and_si256(reduce, q_vec))
    }
}

// ---------------------------------------------------------------------------
// Batch Montgomery multiply
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
unsafe fn mul_assign_prime_montgomery_u16_avx2<const Q: u64>(
    dst: *mut u16,
    src: *const u16,
    len: usize,
) {
    let mut i = 0usize;
    while i + 16 <= len {
        let lhs = _mm256_loadu_si256(dst.add(i).cast::<__m256i>());
        let rhs = _mm256_loadu_si256(src.add(i).cast::<__m256i>());
        let result = montgomery_mul_u16_avx2::<Q>(lhs, rhs);
        _mm256_storeu_si256(dst.add(i).cast::<__m256i>(), result);
        i += 16;
    }
    // Scalar tail
    for j in i..len {
        let prod = PrimeField::<Q, u16>::mul_raw_words(*dst.add(j), *src.add(j));
        *dst.add(j) = prod;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn scalar_mul_prime_montgomery_u16_avx2<const Q: u64>(
    dst: *mut u16,
    len: usize,
    scalar: u16,
) {
    let scalar_vec = _mm256_set1_epi16(scalar as i16);
    let mut i = 0usize;
    while i + 16 <= len {
        let lhs = _mm256_loadu_si256(dst.add(i).cast::<__m256i>());
        let result = montgomery_mul_u16_avx2::<Q>(lhs, scalar_vec);
        _mm256_storeu_si256(dst.add(i).cast::<__m256i>(), result);
        i += 16;
    }
    for j in i..len {
        let prod = PrimeField::<Q, u16>::mul_raw_words(*dst.add(j), scalar);
        *dst.add(j) = prod;
    }
}

// ---------------------------------------------------------------------------
// Modular add / sub
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
unsafe fn add_assign_prime_u16_avx2(dst: *mut u16, src: *const u16, len: usize, modulus: u64) {
    let q_vec = _mm256_set1_epi16(modulus as i16);
    let threshold_vec = _mm256_set1_epi16((modulus - 1) as i16);
    let small = modulus < (1u64 << 15);

    let mut i = 0usize;
    while i + 16 <= len {
        let lhs = _mm256_loadu_si256(dst.add(i).cast::<__m256i>());
        let rhs = _mm256_loadu_si256(src.add(i).cast::<__m256i>());
        let sum = _mm256_add_epi16(lhs, rhs);
        if small {
            let reduce = cmpgt_epu16_avx2(sum, threshold_vec);
            let result = _mm256_sub_epi16(sum, _mm256_and_si256(reduce, q_vec));
            _mm256_storeu_si256(dst.add(i).cast::<__m256i>(), result);
        } else {
            let carry = cmpgt_epu16_avx2(lhs, sum);
            let ge_q = cmpgt_epu16_avx2(sum, threshold_vec);
            let reduce = _mm256_or_si256(carry, ge_q);
            let result = _mm256_sub_epi16(sum, _mm256_and_si256(reduce, q_vec));
            _mm256_storeu_si256(dst.add(i).cast::<__m256i>(), result);
        }
        i += 16;
    }
    // Scalar tail
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

#[target_feature(enable = "avx2")]
unsafe fn sub_assign_prime_u16_avx2(dst: *mut u16, src: *const u16, len: usize, modulus: u64) {
    let q_vec = _mm256_set1_epi16(modulus as i16);

    let mut i = 0usize;
    while i + 16 <= len {
        let lhs = _mm256_loadu_si256(dst.add(i).cast::<__m256i>());
        let rhs = _mm256_loadu_si256(src.add(i).cast::<__m256i>());
        // detect borrow: rhs > lhs
        let borrow = cmpgt_epu16_avx2(rhs, lhs);
        let diff = _mm256_sub_epi16(lhs, rhs);
        let result = _mm256_add_epi16(diff, _mm256_and_si256(borrow, q_vec));
        _mm256_storeu_si256(dst.add(i).cast::<__m256i>(), result);
        i += 16;
    }
    // Scalar tail
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

#[target_feature(enable = "avx2")]
unsafe fn butterfly_forward_prime_montgomery_u16_avx2<const Q: u64>(
    even: *mut u16,
    odd: *mut u16,
    twiddles: *const u16,
    len: usize,
) {
    let q_vec = _mm256_set1_epi16(Q as i16);
    let threshold_vec = _mm256_set1_epi16((Q - 1) as i16);
    let small = Q < (1u64 << 15);

    let mut i = 0usize;
    while i + 16 <= len {
        let even_val = _mm256_loadu_si256(even.add(i).cast::<__m256i>());
        let odd_val = _mm256_loadu_si256(odd.add(i).cast::<__m256i>());
        let tw = _mm256_loadu_si256(twiddles.add(i).cast::<__m256i>());

        // temp = odd * twiddle (Montgomery)
        let temp = montgomery_mul_u16_avx2::<Q>(odd_val, tw);

        // even' = even + temp
        let sum = _mm256_add_epi16(even_val, temp);
        let even_new = if small {
            let reduce = cmpgt_epu16_avx2(sum, threshold_vec);
            _mm256_sub_epi16(sum, _mm256_and_si256(reduce, q_vec))
        } else {
            let carry = cmpgt_epu16_avx2(even_val, sum);
            let ge_q = cmpgt_epu16_avx2(sum, threshold_vec);
            let reduce = _mm256_or_si256(carry, ge_q);
            _mm256_sub_epi16(sum, _mm256_and_si256(reduce, q_vec))
        };

        // odd' = even - temp
        let borrow = cmpgt_epu16_avx2(temp, even_val);
        let diff = _mm256_sub_epi16(even_val, temp);
        let odd_new = _mm256_add_epi16(diff, _mm256_and_si256(borrow, q_vec));

        _mm256_storeu_si256(even.add(i).cast::<__m256i>(), even_new);
        _mm256_storeu_si256(odd.add(i).cast::<__m256i>(), odd_new);
        i += 16;
    }
    // Scalar tail
    for j in i..len {
        let even_val = *even.add(j);
        let odd_val = *odd.add(j);
        let tw = *twiddles.add(j);
        let temp = PrimeField::<Q, u16>::mul_raw_words(odd_val, tw);
        let sum = PrimeField::<Q, u16>::add_raw_words(even_val, temp);
        let diff = PrimeField::<Q, u16>::sub_raw_words(even_val, temp);
        *even.add(j) = sum;
        *odd.add(j) = diff;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn butterfly_inverse_prime_montgomery_u16_avx2<const Q: u64>(
    even: *mut u16,
    odd: *mut u16,
    twiddles: *const u16,
    len: usize,
) {
    let q_vec = _mm256_set1_epi16(Q as i16);
    let threshold_vec = _mm256_set1_epi16((Q - 1) as i16);
    let small = Q < (1u64 << 15);

    let mut i = 0usize;
    while i + 16 <= len {
        let even_val = _mm256_loadu_si256(even.add(i).cast::<__m256i>());
        let odd_val = _mm256_loadu_si256(odd.add(i).cast::<__m256i>());
        let tw = _mm256_loadu_si256(twiddles.add(i).cast::<__m256i>());

        // sum = even + odd
        let sum = _mm256_add_epi16(even_val, odd_val);
        let even_new = if small {
            let reduce = cmpgt_epu16_avx2(sum, threshold_vec);
            _mm256_sub_epi16(sum, _mm256_and_si256(reduce, q_vec))
        } else {
            let carry = cmpgt_epu16_avx2(even_val, sum);
            let ge_q = cmpgt_epu16_avx2(sum, threshold_vec);
            let reduce = _mm256_or_si256(carry, ge_q);
            _mm256_sub_epi16(sum, _mm256_and_si256(reduce, q_vec))
        };

        // diff = even - odd
        let borrow = cmpgt_epu16_avx2(odd_val, even_val);
        let diff = _mm256_sub_epi16(even_val, odd_val);
        let diff_mod = _mm256_add_epi16(diff, _mm256_and_si256(borrow, q_vec));

        // odd' = diff * twiddle (Montgomery)
        let odd_new = montgomery_mul_u16_avx2::<Q>(diff_mod, tw);

        _mm256_storeu_si256(even.add(i).cast::<__m256i>(), even_new);
        _mm256_storeu_si256(odd.add(i).cast::<__m256i>(), odd_new);
        i += 16;
    }
    // Scalar tail
    for j in i..len {
        let even_val = *even.add(j);
        let odd_val = *odd.add(j);
        let tw = *twiddles.add(j);
        let sum = PrimeField::<Q, u16>::add_raw_words(even_val, odd_val);
        let diff = PrimeField::<Q, u16>::sub_raw_words(even_val, odd_val);
        let odd_new = PrimeField::<Q, u16>::mul_raw_words(diff, tw);
        *even.add(j) = sum;
        *odd.add(j) = odd_new;
    }
}

// ---------------------------------------------------------------------------
// Public wrappers
// ---------------------------------------------------------------------------

#[inline]
pub(crate) unsafe fn add_assign_prime_u16(dst: &mut [u16], src: &[u16], modulus: u64) {
    debug_assert_eq!(dst.len(), src.len());
    add_assign_prime_u16_avx2(dst.as_mut_ptr(), src.as_ptr(), dst.len(), modulus);
}

#[inline]
pub(crate) unsafe fn sub_assign_prime_u16(dst: &mut [u16], src: &[u16], modulus: u64) {
    debug_assert_eq!(dst.len(), src.len());
    sub_assign_prime_u16_avx2(dst.as_mut_ptr(), src.as_ptr(), dst.len(), modulus);
}

#[inline]
pub(crate) unsafe fn mul_assign_prime_montgomery_u16<const Q: u64>(dst: &mut [u16], src: &[u16]) {
    debug_assert_eq!(dst.len(), src.len());
    mul_assign_prime_montgomery_u16_avx2::<Q>(dst.as_mut_ptr(), src.as_ptr(), dst.len());
}

#[inline]
pub(crate) unsafe fn scalar_mul_prime_montgomery_u16<const Q: u64>(dst: &mut [u16], scalar: u64) {
    scalar_mul_prime_montgomery_u16_avx2::<Q>(dst.as_mut_ptr(), dst.len(), scalar as u16);
}

#[inline]
pub(crate) unsafe fn butterfly_forward_prime_montgomery_u16<const Q: u64>(
    even: &mut [u16],
    odd: &mut [u16],
    twiddles: &[u16],
) {
    debug_assert_eq!(even.len(), odd.len());
    debug_assert_eq!(even.len(), twiddles.len());
    butterfly_forward_prime_montgomery_u16_avx2::<Q>(
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
    butterfly_inverse_prime_montgomery_u16_avx2::<Q>(
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

    type F12289u16 = PrimeField<12289, u16>;
    type F65521u16 = PrimeField<65521, u16>;

    /// 18 elements — enough for one full SIMD iteration (16 lanes) + 2 scalar tail.
    const N: usize = 18;

    fn skip_if_no_avx2() -> bool {
        !avx2_available()
    }

    // --- Small modulus (Q < 2^15) tests ---

    #[test]
    fn test_u16_small_add_sub_matches_scalar() {
        if skip_if_no_avx2() {
            return;
        }

        let modulus = 12289u64;
        let q = modulus as u16;
        let lhs: [u16; N] = core::array::from_fn(|i| (i as u64 * 700 + 1) as u16 % q);
        let rhs: [u16; N] = core::array::from_fn(|i| (i as u64 * 300 + 5) as u16 % q);

        let mut simd_add = lhs;
        let mut simd_sub = lhs;
        let scalar_add: [u16; N] = core::array::from_fn(|i| {
            let s = lhs[i].wrapping_add(rhs[i]);
            if s >= q { s - q } else { s }
        });
        let scalar_sub: [u16; N] = core::array::from_fn(|i| {
            if lhs[i] >= rhs[i] {
                lhs[i] - rhs[i]
            } else {
                q - rhs[i] + lhs[i]
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
        if skip_if_no_avx2() {
            return;
        }

        let lhs: [u16; N] =
            core::array::from_fn(|i| F12289u16::from_u64((i * 700 + 1) as u64).raw());
        let rhs: [u16; N] =
            core::array::from_fn(|i| F12289u16::from_u64((i * 300 + 5) as u64).raw());

        let mut simd_mul = lhs;
        let scalar_mul: [u16; N] =
            core::array::from_fn(|i| F12289u16::mul_raw_words(lhs[i], rhs[i]));

        unsafe {
            mul_assign_prime_montgomery_u16::<12289>(&mut simd_mul, &rhs);
        }

        assert_eq!(simd_mul, scalar_mul);
    }

    #[test]
    fn test_u16_small_scalar_mul_matches_scalar() {
        if skip_if_no_avx2() {
            return;
        }

        let scalar = F12289u16::from_u64(19);
        let vals: [u16; N] =
            core::array::from_fn(|i| F12289u16::from_u64((i * 700 + 1) as u64).raw());

        let mut simd = vals;
        let expected: [u16; N] =
            core::array::from_fn(|i| F12289u16::mul_raw_words(vals[i], scalar.raw()));

        unsafe {
            scalar_mul_prime_montgomery_u16::<12289>(&mut simd, scalar.raw() as u64);
        }

        assert_eq!(simd, expected);
    }

    #[test]
    fn test_u16_small_butterfly_matches_scalar() {
        if skip_if_no_avx2() {
            return;
        }

        let mut fwd_even: [u16; N] =
            core::array::from_fn(|i| F12289u16::from_u64((i * 700 + 1) as u64).raw());
        let mut fwd_odd: [u16; N] =
            core::array::from_fn(|i| F12289u16::from_u64((i * 300 + 5) as u64).raw());
        let twiddles: [u16; N] =
            core::array::from_fn(|i| F12289u16::from_u64((i * 113 + 7) as u64).raw());

        let mut scalar_even = fwd_even;
        let mut scalar_odd = fwd_odd;
        for i in 0..N {
            let u = scalar_even[i];
            let v = F12289u16::mul_raw_words(scalar_odd[i], twiddles[i]);
            scalar_even[i] = F12289u16::add_raw_words(u, v);
            scalar_odd[i] = F12289u16::sub_raw_words(u, v);
        }

        unsafe {
            butterfly_forward_prime_montgomery_u16::<12289>(&mut fwd_even, &mut fwd_odd, &twiddles);
        }

        assert_eq!(fwd_even, scalar_even);
        assert_eq!(fwd_odd, scalar_odd);
    }

    #[test]
    fn test_u16_small_inverse_butterfly_matches_scalar() {
        if skip_if_no_avx2() {
            return;
        }

        let mut inv_even: [u16; N] =
            core::array::from_fn(|i| F12289u16::from_u64((i * 500 + 100) as u64).raw());
        let mut inv_odd: [u16; N] =
            core::array::from_fn(|i| F12289u16::from_u64((i * 70 + 5) as u64).raw());
        let twiddles: [u16; N] =
            core::array::from_fn(|i| F12289u16::from_u64((i * 113 + 7) as u64).raw());

        let mut scalar_even = inv_even;
        let mut scalar_odd = inv_odd;
        for i in 0..N {
            let u = scalar_even[i];
            let v = scalar_odd[i];
            scalar_even[i] = F12289u16::add_raw_words(u, v);
            let diff = F12289u16::sub_raw_words(u, v);
            scalar_odd[i] = F12289u16::mul_raw_words(diff, twiddles[i]);
        }

        unsafe {
            butterfly_inverse_prime_montgomery_u16::<12289>(&mut inv_even, &mut inv_odd, &twiddles);
        }

        assert_eq!(inv_even, scalar_even);
        assert_eq!(inv_odd, scalar_odd);
    }

    #[test]
    fn test_u16_small_ntt_round_trip() {
        if skip_if_no_avx2() {
            return;
        }

        use crate::arith::ntt::NttPlan;
        use alloc::vec::Vec;

        let n = 64;
        let original: Vec<F12289u16> = (0..n)
            .map(|i| F12289u16::from_u64((i * 73 + 1) as u64 % 12289))
            .collect();

        // Build a plan and run forward NTT via the SIMD path
        let plan = NttPlan::<F12289u16>::build(n).unwrap();
        let mut values = original.clone();
        crate::simd::montgomery::ntt_forward_with_plan(&mut values, &plan).unwrap();
        crate::simd::montgomery::ntt_inverse_with_plan(&mut values, &plan).unwrap();

        assert_eq!(values, original);
    }

    // --- Large modulus (Q >= 2^15) tests ---

    #[test]
    fn test_u16_large_add_sub_matches_scalar() {
        if skip_if_no_avx2() {
            return;
        }

        // 65521 is prime and > 2^15. The add kernel must use carry detection.
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
        if skip_if_no_avx2() {
            return;
        }

        // 65521 uses the wide Montgomery reduction path (Q >= 2^15).
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
        if skip_if_no_avx2() {
            return;
        }
        // 65521 > 2^15, exercises carry detection in butterfly add path
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
        if skip_if_no_avx2() {
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
