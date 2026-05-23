//! u8 AVX2 SIMD kernels for Montgomery prime field arithmetic.
//!
//! 32 lanes of u8 per `__m256i` for add/sub/compare.
//! Widening multiply unpacks u8→u16, processes as u16, then packs back.
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(unused_unsafe)]

use core::arch::x86_64::{
    __m128i, __m256i, _mm_packus_epi16, _mm256_add_epi8, _mm256_add_epi16, _mm256_and_si256,
    _mm256_castsi256_si128, _mm256_cmpgt_epi8, _mm256_cmpgt_epi16, _mm256_cvtepu8_epi16,
    _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_mullo_epi16, _mm256_or_si256,
    _mm256_set1_epi8, _mm256_set1_epi16, _mm256_setr_m128i, _mm256_srli_epi16, _mm256_storeu_si256,
    _mm256_sub_epi8, _mm256_sub_epi16, _mm256_xor_si256,
};

use crate::arith::prime::PrimeField;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline(always)]
unsafe fn cmpgt_epu8_avx2(lhs: __m256i, rhs: __m256i) -> __m256i {
    let sign = unsafe { _mm256_set1_epi8(i8::MIN) };
    unsafe { _mm256_cmpgt_epi8(_mm256_xor_si256(lhs, sign), _mm256_xor_si256(rhs, sign)) }
}

// ---------------------------------------------------------------------------
// Montgomery mul on a 16-lane u16 group (unpacked from u8)
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
/// Montgomery mul for a 16-lane u16 group, using 8-bit radix (R=256).
/// Each lane value is an 8-bit Montgomery residue zero-extended to u16.
///
/// With 8-bit operands the entire computation fits in u16 without overflow:
/// t ≤ 255·255 < 2^16, m ≤ 255, mq ≤ 255·255 < 2^16, and
/// result = t_hi8 + mq_hi8 + (sum_lo8 >> 8) ≤ 255 + 255 + 1 < 2^16.
/// No carry chain is needed — just compare the result against Q.
unsafe fn montgomery_mul_u8_radix<const Q: u64>(
    lhs: __m256i,
    rhs: __m256i,
    q_vec: __m256i,
    q_inv_vec: __m256i,
    threshold_vec: __m256i,
) -> __m256i {
    let mask8 = unsafe { _mm256_set1_epi16(0xFF) };
    let sign16 = unsafe { _mm256_set1_epi16(i16::MIN) };

    // t = a * b
    let t_full = unsafe { _mm256_mullo_epi16(lhs, rhs) };
    let t_lo8 = unsafe { _mm256_and_si256(t_full, mask8) };
    let t_hi8 = unsafe { _mm256_srli_epi16::<8>(t_full) };

    // m = t_lo8 * q_inv mod 256
    let q_inv_8 = unsafe { _mm256_and_si256(q_inv_vec, mask8) };
    let m = unsafe { _mm256_and_si256(_mm256_mullo_epi16(t_lo8, q_inv_8), mask8) };

    // mq = m * Q
    let mq_full = unsafe { _mm256_mullo_epi16(m, q_vec) };
    let mq_lo8 = unsafe { _mm256_and_si256(mq_full, mask8) };
    let mq_hi8 = unsafe { _mm256_srli_epi16::<8>(mq_full) };

    // sum = t + mq → result = sum >> 8 = t_hi8 + mq_hi8 + (sum_lo8 >> 8)
    // sum_lo8 = t_lo8 + mq_lo8 is at most 510, so sum_lo8 >> 8 ∈ {0, 1}.
    // All intermediates fit in u16; no carry chain needed.
    let sum_lo8 = unsafe { _mm256_add_epi16(t_lo8, mq_lo8) };
    let result = unsafe {
        _mm256_add_epi16(
            _mm256_add_epi16(t_hi8, mq_hi8),
            _mm256_srli_epi16::<8>(sum_lo8),
        )
    };

    // result < 2·Q holds, so a single conditional subtract is enough.
    let reduce = unsafe {
        _mm256_cmpgt_epi16(
            _mm256_xor_si256(result, sign16),
            _mm256_xor_si256(threshold_vec, sign16),
        )
    };
    unsafe { _mm256_sub_epi16(result, _mm256_and_si256(reduce, q_vec)) }
}

// ---------------------------------------------------------------------------
// Batch Montgomery mul (32 u8 lanes)
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
unsafe fn mul_assign_prime_montgomery_u8_avx2<const Q: u64>(
    dst: *mut u8,
    src: *const u8,
    len: usize,
) {
    let q_vec = unsafe { _mm256_set1_epi16(Q as i16) };
    let q_inv_vec = unsafe { _mm256_set1_epi16(PrimeField::<Q>::Q_INV_U64 as u16 as i16) };
    let threshold_vec = unsafe { _mm256_set1_epi16((Q - 1) as i16) };

    let mut i = 0usize;
    while i + 32 <= len {
        let a = unsafe { _mm256_loadu_si256(dst.add(i).cast::<__m256i>()) };
        let b = unsafe { _mm256_loadu_si256(src.add(i).cast::<__m256i>()) };

        // Unpack low 16 → u16, multiply, pack
        let a_lo16 = unsafe { _mm256_cvtepu8_epi16(_mm256_castsi256_si128(a)) };
        let b_lo16 = unsafe { _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b)) };
        let res_lo = montgomery_mul_u8_radix::<Q>(a_lo16, b_lo16, q_vec, q_inv_vec, threshold_vec);

        // Unpack high 16 → u16, multiply
        let a_hi16 = unsafe { _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(a) as __m128i) };
        let b_hi16 = unsafe { _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(b) as __m128i) };
        let res_hi = montgomery_mul_u8_radix::<Q>(a_hi16, b_hi16, q_vec, q_inv_vec, threshold_vec);

        // Pack back to u8 (128-bit halves avoid interleave)
        let packed_lo = unsafe {
            _mm_packus_epi16(
                _mm256_castsi256_si128(res_lo),
                _mm256_extracti128_si256::<1>(res_lo) as __m128i,
            )
        };
        let packed_hi = unsafe {
            _mm_packus_epi16(
                _mm256_castsi256_si128(res_hi),
                _mm256_extracti128_si256::<1>(res_hi) as __m128i,
            )
        };
        let result = unsafe { _mm256_setr_m128i(packed_lo, packed_hi) };
        unsafe { _mm256_storeu_si256(dst.add(i).cast::<__m256i>(), result) };
        i += 32;
    }
    // Scalar tail
    for j in i..len {
        let prod = PrimeField::<Q, u8>::mul_raw_words(*dst.add(j), *src.add(j));
        *dst.add(j) = prod;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn scalar_mul_prime_montgomery_u8_avx2<const Q: u64>(dst: *mut u8, len: usize, scalar: u8) {
    let q_vec = unsafe { _mm256_set1_epi16(Q as i16) };
    let q_inv_vec = unsafe { _mm256_set1_epi16(PrimeField::<Q>::Q_INV_U64 as u16 as i16) };
    let threshold_vec = unsafe { _mm256_set1_epi16((Q - 1) as i16) };

    let scalar_vec = unsafe { _mm256_set1_epi16(scalar as i16) };

    let mut i = 0usize;
    while i + 32 <= len {
        let a = unsafe { _mm256_loadu_si256(dst.add(i).cast::<__m256i>()) };

        let a_lo16 = unsafe { _mm256_cvtepu8_epi16(_mm256_castsi256_si128(a)) };
        let res_lo =
            montgomery_mul_u8_radix::<Q>(a_lo16, scalar_vec, q_vec, q_inv_vec, threshold_vec);

        let a_hi16 = unsafe { _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(a) as __m128i) };
        let res_hi =
            montgomery_mul_u8_radix::<Q>(a_hi16, scalar_vec, q_vec, q_inv_vec, threshold_vec);

        let packed_lo = unsafe {
            _mm_packus_epi16(
                _mm256_castsi256_si128(res_lo),
                _mm256_extracti128_si256::<1>(res_lo) as __m128i,
            )
        };
        let packed_hi = unsafe {
            _mm_packus_epi16(
                _mm256_castsi256_si128(res_hi),
                _mm256_extracti128_si256::<1>(res_hi) as __m128i,
            )
        };
        let result = unsafe { _mm256_setr_m128i(packed_lo, packed_hi) };
        unsafe { _mm256_storeu_si256(dst.add(i).cast::<__m256i>(), result) };
        i += 32;
    }
    for j in i..len {
        let prod = PrimeField::<Q, u8>::mul_raw_words(*dst.add(j), scalar);
        *dst.add(j) = prod;
    }
}

// ---------------------------------------------------------------------------
// Modular add / sub (native 32-lane u8)
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
unsafe fn add_assign_prime_u8_avx2(dst: *mut u8, src: *const u8, len: usize, modulus: u64) {
    let q_vec = unsafe { _mm256_set1_epi8(modulus as i8) };
    let threshold_vec = unsafe { _mm256_set1_epi8((modulus - 1) as i8) };
    let small = modulus < (1u64 << 7);

    let mut i = 0usize;
    while i + 32 <= len {
        let lhs = unsafe { _mm256_loadu_si256(dst.add(i).cast::<__m256i>()) };
        let rhs = unsafe { _mm256_loadu_si256(src.add(i).cast::<__m256i>()) };
        let sum = unsafe { _mm256_add_epi8(lhs, rhs) };
        if small {
            let reduce = cmpgt_epu8_avx2(sum, threshold_vec);
            let result = unsafe { _mm256_sub_epi8(sum, _mm256_and_si256(reduce, q_vec)) };
            unsafe { _mm256_storeu_si256(dst.add(i).cast::<__m256i>(), result) };
        } else {
            let carry = cmpgt_epu8_avx2(lhs, sum);
            let ge_q = cmpgt_epu8_avx2(sum, threshold_vec);
            let reduce = unsafe { _mm256_or_si256(carry, ge_q) };
            let result = unsafe { _mm256_sub_epi8(sum, _mm256_and_si256(reduce, q_vec)) };
            unsafe { _mm256_storeu_si256(dst.add(i).cast::<__m256i>(), result) };
        }
        i += 32;
    }
    // Scalar tail
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

#[target_feature(enable = "avx2")]
unsafe fn sub_assign_prime_u8_avx2(dst: *mut u8, src: *const u8, len: usize, modulus: u64) {
    let q_vec = unsafe { _mm256_set1_epi8(modulus as i8) };

    let mut i = 0usize;
    while i + 32 <= len {
        let lhs = unsafe { _mm256_loadu_si256(dst.add(i).cast::<__m256i>()) };
        let rhs = unsafe { _mm256_loadu_si256(src.add(i).cast::<__m256i>()) };
        let borrow = cmpgt_epu8_avx2(rhs, lhs);
        let diff = unsafe { _mm256_sub_epi8(lhs, rhs) };
        let result = unsafe { _mm256_add_epi8(diff, _mm256_and_si256(borrow, q_vec)) };
        unsafe { _mm256_storeu_si256(dst.add(i).cast::<__m256i>(), result) };
        i += 32;
    }
    // Scalar tail
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

#[target_feature(enable = "avx2")]
unsafe fn butterfly_forward_prime_montgomery_u8_avx2<const Q: u64>(
    even: *mut u8,
    odd: *mut u8,
    twiddles: *const u8,
    len: usize,
) {
    let q_vec_16 = unsafe { _mm256_set1_epi16(Q as i16) };
    let q_inv_vec = unsafe { _mm256_set1_epi16(PrimeField::<Q>::Q_INV_U64 as u16 as i16) };
    let threshold_vec_16 = unsafe { _mm256_set1_epi16((Q - 1) as i16) };
    let small = Q < (1u64 << 7);

    let q_vec = unsafe { _mm256_set1_epi8(Q as i8) };
    let threshold_vec = unsafe { _mm256_set1_epi8((Q - 1) as i8) };

    let mut i = 0usize;
    while i + 32 <= len {
        let even_val = unsafe { _mm256_loadu_si256(even.add(i).cast::<__m256i>()) };
        let odd_val = unsafe { _mm256_loadu_si256(odd.add(i).cast::<__m256i>()) };
        let tw = unsafe { _mm256_loadu_si256(twiddles.add(i).cast::<__m256i>()) };

        // temp = odd * twiddle: unpack both halves, mul, repack
        let odd_lo = unsafe { _mm256_cvtepu8_epi16(_mm256_castsi256_si128(odd_val)) };
        let tw_lo = unsafe { _mm256_cvtepu8_epi16(_mm256_castsi256_si128(tw)) };
        let temp_lo =
            montgomery_mul_u8_radix::<Q>(odd_lo, tw_lo, q_vec_16, q_inv_vec, threshold_vec_16);
        let odd_hi =
            unsafe { _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(odd_val) as __m128i) };
        let tw_hi = unsafe { _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(tw) as __m128i) };
        let temp_hi =
            montgomery_mul_u8_radix::<Q>(odd_hi, tw_hi, q_vec_16, q_inv_vec, threshold_vec_16);
        let temp = unsafe {
            _mm256_setr_m128i(
                _mm_packus_epi16(
                    _mm256_castsi256_si128(temp_lo),
                    _mm256_extracti128_si256::<1>(temp_lo) as __m128i,
                ),
                _mm_packus_epi16(
                    _mm256_castsi256_si128(temp_hi),
                    _mm256_extracti128_si256::<1>(temp_hi) as __m128i,
                ),
            )
        };

        // even' = even + temp
        let sum = unsafe { _mm256_add_epi8(even_val, temp) };
        let even_new = if small {
            let reduce = cmpgt_epu8_avx2(sum, threshold_vec);
            unsafe { _mm256_sub_epi8(sum, _mm256_and_si256(reduce, q_vec)) }
        } else {
            let carry = cmpgt_epu8_avx2(even_val, sum);
            let ge_q = cmpgt_epu8_avx2(sum, threshold_vec);
            let reduce = unsafe { _mm256_or_si256(carry, ge_q) };
            unsafe { _mm256_sub_epi8(sum, _mm256_and_si256(reduce, q_vec)) }
        };

        // odd' = even - temp
        let borrow = cmpgt_epu8_avx2(temp, even_val);
        let diff = unsafe { _mm256_sub_epi8(even_val, temp) };
        let odd_new = unsafe { _mm256_add_epi8(diff, _mm256_and_si256(borrow, q_vec)) };

        unsafe { _mm256_storeu_si256(even.add(i).cast::<__m256i>(), even_new) };
        unsafe { _mm256_storeu_si256(odd.add(i).cast::<__m256i>(), odd_new) };
        i += 32;
    }
    // Scalar tail
    for j in i..len {
        let even_val = *even.add(j);
        let odd_val = *odd.add(j);
        let tw = *twiddles.add(j);
        let temp = PrimeField::<Q, u8>::mul_raw_words(odd_val, tw);
        let sum = PrimeField::<Q, u8>::add_raw_words(even_val, temp);
        let diff = PrimeField::<Q, u8>::sub_raw_words(even_val, temp);
        *even.add(j) = sum;
        *odd.add(j) = diff;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn butterfly_inverse_prime_montgomery_u8_avx2<const Q: u64>(
    even: *mut u8,
    odd: *mut u8,
    twiddles: *const u8,
    len: usize,
) {
    let q_vec_16 = unsafe { _mm256_set1_epi16(Q as i16) };
    let q_inv_vec = unsafe { _mm256_set1_epi16(PrimeField::<Q>::Q_INV_U64 as u16 as i16) };
    let threshold_vec_16 = unsafe { _mm256_set1_epi16((Q - 1) as i16) };
    let small = Q < (1u64 << 7);

    let q_vec = unsafe { _mm256_set1_epi8(Q as i8) };
    let threshold_vec = unsafe { _mm256_set1_epi8((Q - 1) as i8) };

    let mut i = 0usize;
    while i + 32 <= len {
        let even_val = unsafe { _mm256_loadu_si256(even.add(i).cast::<__m256i>()) };
        let odd_val = unsafe { _mm256_loadu_si256(odd.add(i).cast::<__m256i>()) };
        let tw = unsafe { _mm256_loadu_si256(twiddles.add(i).cast::<__m256i>()) };

        // sum = even + odd
        let sum = unsafe { _mm256_add_epi8(even_val, odd_val) };
        let even_new = if small {
            let reduce = cmpgt_epu8_avx2(sum, threshold_vec);
            unsafe { _mm256_sub_epi8(sum, _mm256_and_si256(reduce, q_vec)) }
        } else {
            let carry = cmpgt_epu8_avx2(even_val, sum);
            let ge_q = cmpgt_epu8_avx2(sum, threshold_vec);
            let reduce = unsafe { _mm256_or_si256(carry, ge_q) };
            unsafe { _mm256_sub_epi8(sum, _mm256_and_si256(reduce, q_vec)) }
        };

        // diff = even - odd
        let borrow = cmpgt_epu8_avx2(odd_val, even_val);
        let diff = unsafe { _mm256_sub_epi8(even_val, odd_val) };
        let diff_mod = unsafe { _mm256_add_epi8(diff, _mm256_and_si256(borrow, q_vec)) };

        // odd' = diff * twiddle
        let diff_lo = unsafe { _mm256_cvtepu8_epi16(_mm256_castsi256_si128(diff_mod)) };
        let tw_lo = unsafe { _mm256_cvtepu8_epi16(_mm256_castsi256_si128(tw)) };
        let odd_lo =
            montgomery_mul_u8_radix::<Q>(diff_lo, tw_lo, q_vec_16, q_inv_vec, threshold_vec_16);
        let diff_hi =
            unsafe { _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(diff_mod) as __m128i) };
        let tw_hi = unsafe { _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(tw) as __m128i) };
        let odd_hi =
            montgomery_mul_u8_radix::<Q>(diff_hi, tw_hi, q_vec_16, q_inv_vec, threshold_vec_16);
        let odd_new = unsafe {
            _mm256_setr_m128i(
                _mm_packus_epi16(
                    _mm256_castsi256_si128(odd_lo),
                    _mm256_extracti128_si256::<1>(odd_lo) as __m128i,
                ),
                _mm_packus_epi16(
                    _mm256_castsi256_si128(odd_hi),
                    _mm256_extracti128_si256::<1>(odd_hi) as __m128i,
                ),
            )
        };

        unsafe { _mm256_storeu_si256(even.add(i).cast::<__m256i>(), even_new) };
        unsafe { _mm256_storeu_si256(odd.add(i).cast::<__m256i>(), odd_new) };
        i += 32;
    }
    // Scalar tail
    for j in i..len {
        let even_val = *even.add(j);
        let odd_val = *odd.add(j);
        let tw = *twiddles.add(j);
        let sum = PrimeField::<Q, u8>::add_raw_words(even_val, odd_val);
        let diff = PrimeField::<Q, u8>::sub_raw_words(even_val, odd_val);
        let odd_new = PrimeField::<Q, u8>::mul_raw_words(diff, tw);
        *even.add(j) = sum;
        *odd.add(j) = odd_new;
    }
}

// ---------------------------------------------------------------------------
// Public wrappers
// ---------------------------------------------------------------------------

#[inline]
pub(crate) unsafe fn add_assign_prime_u8(dst: &mut [u8], src: &[u8], modulus: u64) {
    debug_assert_eq!(dst.len(), src.len());
    add_assign_prime_u8_avx2(dst.as_mut_ptr(), src.as_ptr(), dst.len(), modulus);
}

#[inline]
pub(crate) unsafe fn sub_assign_prime_u8(dst: &mut [u8], src: &[u8], modulus: u64) {
    debug_assert_eq!(dst.len(), src.len());
    sub_assign_prime_u8_avx2(dst.as_mut_ptr(), src.as_ptr(), dst.len(), modulus);
}

#[inline]
pub(crate) unsafe fn mul_assign_prime_montgomery_u8<const Q: u64>(dst: &mut [u8], src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    mul_assign_prime_montgomery_u8_avx2::<Q>(dst.as_mut_ptr(), src.as_ptr(), dst.len());
}

#[inline]
pub(crate) unsafe fn scalar_mul_prime_montgomery_u8<const Q: u64>(dst: &mut [u8], scalar: u64) {
    scalar_mul_prime_montgomery_u8_avx2::<Q>(dst.as_mut_ptr(), dst.len(), scalar as u8);
}

#[inline]
pub(crate) unsafe fn butterfly_forward_prime_montgomery_u8<const Q: u64>(
    even: &mut [u8],
    odd: &mut [u8],
    twiddles: &[u8],
) {
    debug_assert_eq!(even.len(), odd.len());
    debug_assert_eq!(even.len(), twiddles.len());
    butterfly_forward_prime_montgomery_u8_avx2::<Q>(
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
    butterfly_inverse_prime_montgomery_u8_avx2::<Q>(
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

    type F127u8 = PrimeField<127, u8>;
    type F251u8 = PrimeField<251, u8>;

    const N: usize = 34; // one full SIMD iter (32) + 2 scalar tail

    fn skip_if_no_avx2() -> bool {
        !avx2_available()
    }

    #[test]
    fn test_u8_small_add_sub_matches_scalar() {
        if skip_if_no_avx2() {
            return;
        }
        let modulus = 127u64;
        let q = modulus as u8;
        let lhs: [u8; N] = core::array::from_fn(|i| (i * 17 + 1) as u8 % q);
        let rhs: [u8; N] = core::array::from_fn(|i| (i * 23 + 5) as u8 % q);
        let mut simd_add = lhs;
        let mut simd_sub = lhs;
        let scalar_add: [u8; N] = core::array::from_fn(|i| {
            let s = lhs[i].wrapping_add(rhs[i]);
            if s >= q { s - q } else { s }
        });
        let scalar_sub: [u8; N] = core::array::from_fn(|i| {
            if lhs[i] >= rhs[i] {
                lhs[i] - rhs[i]
            } else {
                q - rhs[i] + lhs[i]
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
        if skip_if_no_avx2() {
            return;
        }
        let lhs: [u8; N] = core::array::from_fn(|i| F127u8::from_u64((i * 17 + 1) as u64).raw());
        let rhs: [u8; N] = core::array::from_fn(|i| F127u8::from_u64((i * 23 + 5) as u64).raw());
        let mut simd = lhs;
        let scalar: [u8; N] = core::array::from_fn(|i| F127u8::mul_raw_words(lhs[i], rhs[i]));
        unsafe {
            mul_assign_prime_montgomery_u8::<127>(&mut simd, &rhs);
        }
        assert_eq!(simd, scalar);
    }

    #[test]
    fn test_u8_small_scalar_mul_matches_scalar() {
        if skip_if_no_avx2() {
            return;
        }
        let scalar = F127u8::from_u64(19);
        let vals: [u8; N] = core::array::from_fn(|i| F127u8::from_u64((i * 17 + 1) as u64).raw());
        let mut simd = vals;
        let expected: [u8; N] =
            core::array::from_fn(|i| F127u8::mul_raw_words(vals[i], scalar.raw()));
        unsafe {
            scalar_mul_prime_montgomery_u8::<127>(&mut simd, scalar.raw() as u64);
        }
        assert_eq!(simd, expected);
    }

    #[test]
    fn test_u8_small_butterfly_matches_scalar() {
        if skip_if_no_avx2() {
            return;
        }
        let mut fwd_even: [u8; N] =
            core::array::from_fn(|i| F127u8::from_u64((i * 17 + 1) as u64).raw());
        let mut fwd_odd: [u8; N] =
            core::array::from_fn(|i| F127u8::from_u64((i * 23 + 5) as u64).raw());
        let twiddles: [u8; N] =
            core::array::from_fn(|i| F127u8::from_u64((i * 7 + 3) as u64).raw());
        let mut s_e = fwd_even;
        let mut s_o = fwd_odd;
        for i in 0..N {
            let t = F127u8::mul_raw_words(s_o[i], twiddles[i]);
            let sum = F127u8::add_raw_words(s_e[i], t);
            let diff = F127u8::sub_raw_words(s_e[i], t);
            s_e[i] = sum;
            s_o[i] = diff;
        }
        unsafe {
            butterfly_forward_prime_montgomery_u8::<127>(&mut fwd_even, &mut fwd_odd, &twiddles);
        }
        assert_eq!(fwd_even, s_e);
        assert_eq!(fwd_odd, s_o);
    }

    #[test]
    fn test_u8_small_inverse_butterfly_matches_scalar() {
        if skip_if_no_avx2() {
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
        if skip_if_no_avx2() {
            return;
        }
        use crate::arith::ntt::NttPlan;
        use alloc::vec::Vec;
        // 193 is prime, 193-1 = 192 = 64*3 → supports n=64 NTT
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

    #[test]
    fn test_u8_large_butterfly_matches_scalar() {
        if skip_if_no_avx2() {
            return;
        }
        // 251 > 2^7, exercises carry detection in butterfly add path
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
        if skip_if_no_avx2() {
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

    #[test]
    fn test_u8_large_add_sub_matches_scalar() {
        if skip_if_no_avx2() {
            return;
        }
        // 251 > 2^7, uses carry detection
        let modulus = 251u64;
        let q = modulus as u8;
        let lhs: [u8; N] = core::array::from_fn(|i| (i as u64 * 50 + 1) as u8 % q);
        let rhs: [u8; N] = core::array::from_fn(|i| (i as u64 * 70 + 5) as u8 % q);
        let mut simd_add = lhs;
        let mut simd_sub = lhs;
        let scalar_add: [u8; N] = core::array::from_fn(|i| {
            let (s, c) = lhs[i].overflowing_add(rhs[i]);
            if c || s >= q { s.wrapping_sub(q) } else { s }
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
        if skip_if_no_avx2() {
            return;
        }
        let lhs: [u8; N] = core::array::from_fn(|i| F251u8::from_u64((i * 50 + 1) as u64).raw());
        let rhs: [u8; N] = core::array::from_fn(|i| F251u8::from_u64((i * 70 + 5) as u64).raw());
        let mut simd = lhs;
        let scalar: [u8; N] = core::array::from_fn(|i| F251u8::mul_raw_words(lhs[i], rhs[i]));
        unsafe {
            mul_assign_prime_montgomery_u8::<251>(&mut simd, &rhs);
        }
        assert_eq!(simd, scalar);
    }
}
