//! x86_64 SIMD backend hooks.
#![allow(unsafe_op_in_unsafe_fn)]

use crate::arith::prime::PrimeField;
use core::arch::x86_64::{
    __m256i, _mm256_add_epi64, _mm256_and_si256, _mm256_cmpgt_epi64, _mm256_loadu_si256,
    _mm256_mul_epu32, _mm256_or_si256, _mm256_set1_epi64x, _mm256_slli_epi64, _mm256_srli_epi64,
    _mm256_storeu_si256, _mm256_sub_epi64, _mm256_xor_si256,
};

#[cfg(all(feature = "std", target_arch = "x86_64"))]
pub(crate) fn avx2_available() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}

#[cfg(all(not(feature = "std"), target_arch = "x86_64", target_feature = "avx2"))]
pub(crate) fn avx2_available() -> bool {
    true
}

#[cfg(not(any(
    all(feature = "std", target_arch = "x86_64"),
    all(not(feature = "std"), target_arch = "x86_64", target_feature = "avx2")
)))]
pub(crate) fn avx2_available() -> bool {
    false
}

#[target_feature(enable = "avx2")]
unsafe fn add_assign_u64_masked_avx2(dst: *mut u64, src: *const u64, len: usize, mask: u64) {
    let mut i = 0usize;
    let mask_vec = _mm256_set1_epi64x(mask as i64);
    while i + 4 <= len {
        let lhs = unsafe { _mm256_loadu_si256(dst.add(i) as *const __m256i) };
        let rhs = unsafe { _mm256_loadu_si256(src.add(i) as *const __m256i) };
        let sum = _mm256_and_si256(_mm256_add_epi64(lhs, rhs), mask_vec);
        unsafe { _mm256_storeu_si256(dst.add(i) as *mut __m256i, sum) };
        i += 4;
    }
    while i < len {
        unsafe {
            *dst.add(i) = dst.add(i).read().wrapping_add(src.add(i).read()) & mask;
        }
        i += 1;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn sub_assign_u64_masked_avx2(dst: *mut u64, src: *const u64, len: usize, mask: u64) {
    let mut i = 0usize;
    let mask_vec = _mm256_set1_epi64x(mask as i64);
    while i + 4 <= len {
        let lhs = unsafe { _mm256_loadu_si256(dst.add(i) as *const __m256i) };
        let rhs = unsafe { _mm256_loadu_si256(src.add(i) as *const __m256i) };
        let diff = _mm256_and_si256(_mm256_sub_epi64(lhs, rhs), mask_vec);
        unsafe { _mm256_storeu_si256(dst.add(i) as *mut __m256i, diff) };
        i += 4;
    }
    while i < len {
        unsafe {
            *dst.add(i) = dst.add(i).read().wrapping_sub(src.add(i).read()) & mask;
        }
        i += 1;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn add_assign_prime_u64_avx2(dst: *mut u64, src: *const u64, len: usize, modulus: u64) {
    let modulus_vec = _mm256_set1_epi64x(modulus as i64);
    let threshold_vec = _mm256_set1_epi64x((modulus - 1) as i64);
    let mut i = 0usize;
    while i + 4 <= len {
        let lhs = unsafe { _mm256_loadu_si256(dst.add(i) as *const __m256i) };
        let rhs = unsafe { _mm256_loadu_si256(src.add(i) as *const __m256i) };
        let reduced = add_mod_u64_avx2(lhs, rhs, modulus_vec, threshold_vec);
        unsafe { _mm256_storeu_si256(dst.add(i) as *mut __m256i, reduced) };
        i += 4;
    }
    while i < len {
        unsafe {
            let sum = dst.add(i).read() + src.add(i).read();
            *dst.add(i) = if sum >= modulus { sum - modulus } else { sum };
        }
        i += 1;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn sub_assign_prime_u64_avx2(dst: *mut u64, src: *const u64, len: usize, modulus: u64) {
    let modulus_vec = _mm256_set1_epi64x(modulus as i64);
    let mut i = 0usize;
    while i + 4 <= len {
        let lhs = unsafe { _mm256_loadu_si256(dst.add(i) as *const __m256i) };
        let rhs = unsafe { _mm256_loadu_si256(src.add(i) as *const __m256i) };
        let diff = sub_mod_u64_avx2(lhs, rhs, modulus_vec);
        unsafe { _mm256_storeu_si256(dst.add(i) as *mut __m256i, diff) };
        i += 4;
    }
    while i < len {
        unsafe {
            let lhs = dst.add(i).read();
            let rhs = src.add(i).read();
            *dst.add(i) = if lhs >= rhs {
                lhs - rhs
            } else {
                lhs + modulus - rhs
            };
        }
        i += 1;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn mul_assign_u64_low32_masked_avx2(dst: *mut u64, src: *const u64, len: usize, mask: u64) {
    let mut i = 0usize;
    let mask_vec = _mm256_set1_epi64x(mask as i64);
    while i + 4 <= len {
        let lhs = unsafe { _mm256_loadu_si256(dst.add(i) as *const __m256i) };
        let rhs = unsafe { _mm256_loadu_si256(src.add(i) as *const __m256i) };
        let product = _mm256_and_si256(_mm256_mul_epu32(lhs, rhs), mask_vec);
        unsafe { _mm256_storeu_si256(dst.add(i) as *mut __m256i, product) };
        i += 4;
    }
    while i < len {
        unsafe {
            *dst.add(i) = dst.add(i).read().wrapping_mul(src.add(i).read()) & mask;
        }
        i += 1;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn scalar_mul_u64_low32_masked_avx2(dst: *mut u64, len: usize, scalar: u64, mask: u64) {
    let mut i = 0usize;
    let scalar_vec = _mm256_set1_epi64x(scalar as i64);
    let mask_vec = _mm256_set1_epi64x(mask as i64);
    while i + 4 <= len {
        let lhs = unsafe { _mm256_loadu_si256(dst.add(i) as *const __m256i) };
        let product = _mm256_and_si256(_mm256_mul_epu32(lhs, scalar_vec), mask_vec);
        unsafe { _mm256_storeu_si256(dst.add(i) as *mut __m256i, product) };
        i += 4;
    }
    while i < len {
        unsafe {
            *dst.add(i) = dst.add(i).read().wrapping_mul(scalar) & mask;
        }
        i += 1;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn cmpgt_epu64_avx2(lhs: __m256i, rhs: __m256i) -> __m256i {
    let sign_bit = _mm256_set1_epi64x((1u64 << 63) as i64);
    _mm256_cmpgt_epi64(
        _mm256_xor_si256(lhs, sign_bit),
        _mm256_xor_si256(rhs, sign_bit),
    )
}

#[target_feature(enable = "avx2")]
unsafe fn add_mod_u64_avx2(
    lhs: __m256i,
    rhs: __m256i,
    modulus_vec: __m256i,
    threshold_vec: __m256i,
) -> __m256i {
    let sum = _mm256_add_epi64(lhs, rhs);
    let carry = unsafe { cmpgt_epu64_avx2(lhs, sum) };
    let ge_modulus = unsafe { cmpgt_epu64_avx2(sum, threshold_vec) };
    let reduce = _mm256_or_si256(carry, ge_modulus);
    _mm256_sub_epi64(sum, _mm256_and_si256(reduce, modulus_vec))
}

#[target_feature(enable = "avx2")]
unsafe fn sub_mod_u64_avx2(lhs: __m256i, rhs: __m256i, modulus_vec: __m256i) -> __m256i {
    let diff = _mm256_sub_epi64(lhs, rhs);
    let borrow = unsafe { cmpgt_epu64_avx2(rhs, lhs) };
    _mm256_add_epi64(diff, _mm256_and_si256(borrow, modulus_vec))
}

#[target_feature(enable = "avx2")]
unsafe fn mul_lo_epu64_avx2(lhs: __m256i, rhs: __m256i) -> __m256i {
    let lhs_hi = _mm256_srli_epi64::<32>(lhs);
    let rhs_hi = _mm256_srli_epi64::<32>(rhs);
    let p0 = _mm256_mul_epu32(lhs, rhs);
    let p1 = _mm256_mul_epu32(lhs, rhs_hi);
    let p2 = _mm256_mul_epu32(lhs_hi, rhs);
    let cross = _mm256_add_epi64(p1, p2);
    _mm256_add_epi64(p0, _mm256_slli_epi64::<32>(cross))
}

#[target_feature(enable = "avx2")]
unsafe fn mul_epu64_wide_avx2(lhs: __m256i, rhs: __m256i) -> (__m256i, __m256i) {
    let lhs_hi = _mm256_srli_epi64::<32>(lhs);
    let rhs_hi = _mm256_srli_epi64::<32>(rhs);

    let p0 = _mm256_mul_epu32(lhs, rhs);
    let p1 = _mm256_mul_epu32(lhs, rhs_hi);
    let p2 = _mm256_mul_epu32(lhs_hi, rhs);
    let p3 = _mm256_mul_epu32(lhs_hi, rhs_hi);

    let cross = _mm256_add_epi64(p1, p2);
    let cross_carry = unsafe { cmpgt_epu64_avx2(p1, cross) };

    let lo = _mm256_add_epi64(p0, _mm256_slli_epi64::<32>(cross));
    let lo_carry = unsafe { cmpgt_epu64_avx2(p0, lo) };

    let hi = _mm256_add_epi64(
        _mm256_add_epi64(
            p3,
            _mm256_add_epi64(
                _mm256_srli_epi64::<32>(cross),
                _mm256_and_si256(cross_carry, _mm256_set1_epi64x((1u64 << 32) as i64)),
            ),
        ),
        _mm256_and_si256(lo_carry, _mm256_set1_epi64x(1)),
    );

    (lo, hi)
}

#[target_feature(enable = "avx2")]
unsafe fn montgomery_mul_u64_avx2<const Q: u64>(lhs: __m256i, rhs: __m256i) -> __m256i {
    let modulus_vec = _mm256_set1_epi64x(Q as i64);
    let threshold_vec = _mm256_set1_epi64x((Q - 1) as i64);
    let q_inv_vec = _mm256_set1_epi64x(PrimeField::<Q>::Q_INV_U64 as i64);
    let carry_bit_vec = _mm256_set1_epi64x(1);

    let (t_lo, t_hi) = unsafe { mul_epu64_wide_avx2(lhs, rhs) };
    let m = unsafe { mul_lo_epu64_avx2(t_lo, q_inv_vec) };
    let (mq_lo, mq_hi) = unsafe { mul_epu64_wide_avx2(m, modulus_vec) };

    let sum_lo = _mm256_add_epi64(t_lo, mq_lo);
    let carry_lo = unsafe { cmpgt_epu64_avx2(t_lo, sum_lo) };

    let sum_hi_a = _mm256_add_epi64(t_hi, mq_hi);
    let carry_hi_a = unsafe { cmpgt_epu64_avx2(t_hi, sum_hi_a) };
    let sum_hi = _mm256_add_epi64(sum_hi_a, _mm256_and_si256(carry_lo, carry_bit_vec));
    let carry_hi_b = unsafe { cmpgt_epu64_avx2(sum_hi_a, sum_hi) };

    let reduce = _mm256_or_si256(_mm256_or_si256(carry_hi_a, carry_hi_b), unsafe {
        cmpgt_epu64_avx2(sum_hi, threshold_vec)
    });
    _mm256_sub_epi64(sum_hi, _mm256_and_si256(reduce, modulus_vec))
}

#[target_feature(enable = "avx2")]
unsafe fn mul_assign_prime_wide_u64_avx2<const Q: u64>(dst: *mut u64, src: *const u64, len: usize) {
    let mut i = 0usize;
    while i + 4 <= len {
        let lhs = unsafe { _mm256_loadu_si256(dst.add(i) as *const __m256i) };
        let rhs = unsafe { _mm256_loadu_si256(src.add(i) as *const __m256i) };
        let reduced = unsafe { montgomery_mul_u64_avx2::<Q>(lhs, rhs) };
        unsafe { _mm256_storeu_si256(dst.add(i) as *mut __m256i, reduced) };
        i += 4;
    }
    while i < len {
        unsafe {
            let lhs = dst.add(i).read();
            let rhs = src.add(i).read();
            *dst.add(i) = PrimeField::<Q>::mul_raw_words(lhs, rhs);
        }
        i += 1;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn scalar_mul_prime_wide_u64_avx2<const Q: u64>(dst: *mut u64, len: usize, scalar: u64) {
    let mut i = 0usize;
    let scalar_vec = _mm256_set1_epi64x(scalar as i64);
    while i + 4 <= len {
        let lhs = unsafe { _mm256_loadu_si256(dst.add(i) as *const __m256i) };
        let reduced = unsafe { montgomery_mul_u64_avx2::<Q>(lhs, scalar_vec) };
        unsafe { _mm256_storeu_si256(dst.add(i) as *mut __m256i, reduced) };
        i += 4;
    }
    while i < len {
        unsafe {
            let lhs = dst.add(i).read();
            *dst.add(i) = PrimeField::<Q>::mul_raw_words(lhs, scalar);
        }
        i += 1;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn butterfly_forward_prime_wide_u64_avx2<const Q: u64>(
    even: *mut u64,
    odd: *mut u64,
    twiddles: *const u64,
    len: usize,
) {
    let mut i = 0usize;
    let modulus_vec = _mm256_set1_epi64x(Q as i64);
    let threshold_vec = _mm256_set1_epi64x((Q - 1) as i64);
    while i + 4 <= len {
        let u_vec = unsafe { _mm256_loadu_si256(even.add(i) as *const __m256i) };
        let v_raw_vec = unsafe { _mm256_loadu_si256(odd.add(i) as *const __m256i) };
        let w_vec = unsafe { _mm256_loadu_si256(twiddles.add(i) as *const __m256i) };
        let v = unsafe { montgomery_mul_u64_avx2::<Q>(v_raw_vec, w_vec) };
        let even_out = unsafe { add_mod_u64_avx2(u_vec, v, modulus_vec, threshold_vec) };
        let odd_out = unsafe { sub_mod_u64_avx2(u_vec, v, modulus_vec) };
        unsafe {
            _mm256_storeu_si256(even.add(i) as *mut __m256i, even_out);
            _mm256_storeu_si256(odd.add(i) as *mut __m256i, odd_out);
        }
        i += 4;
    }

    while i < len {
        unsafe {
            let u = even.add(i).read();
            let v = PrimeField::<Q>::mul_raw_words(odd.add(i).read(), twiddles.add(i).read());
            even.add(i).write(PrimeField::<Q>::add_raw_words(u, v));
            odd.add(i).write(PrimeField::<Q>::sub_raw_words(u, v));
        }
        i += 1;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn butterfly_inverse_prime_wide_u64_avx2<const Q: u64>(
    even: *mut u64,
    odd: *mut u64,
    twiddles: *const u64,
    len: usize,
) {
    let mut i = 0usize;
    let modulus_vec = _mm256_set1_epi64x(Q as i64);
    let threshold_vec = _mm256_set1_epi64x((Q - 1) as i64);
    while i + 4 <= len {
        let u_vec = unsafe { _mm256_loadu_si256(even.add(i) as *const __m256i) };
        let v_vec = unsafe { _mm256_loadu_si256(odd.add(i) as *const __m256i) };
        let sum = unsafe { add_mod_u64_avx2(u_vec, v_vec, modulus_vec, threshold_vec) };
        let diff = unsafe { sub_mod_u64_avx2(u_vec, v_vec, modulus_vec) };
        let w_vec = unsafe { _mm256_loadu_si256(twiddles.add(i) as *const __m256i) };
        let odd_out = unsafe { montgomery_mul_u64_avx2::<Q>(diff, w_vec) };
        unsafe {
            _mm256_storeu_si256(even.add(i) as *mut __m256i, sum);
            _mm256_storeu_si256(odd.add(i) as *mut __m256i, odd_out);
        }
        i += 4;
    }

    while i < len {
        unsafe {
            let u = even.add(i).read();
            let v = odd.add(i).read();
            let sum = PrimeField::<Q>::add_raw_words(u, v);
            let diff = PrimeField::<Q>::sub_raw_words(u, v);
            even.add(i).write(sum);
            odd.add(i)
                .write(PrimeField::<Q>::mul_raw_words(diff, twiddles.add(i).read()));
        }
        i += 1;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn mul_assign_prime_montgomery_u64_avx2<const Q: u64>(
    dst: *mut u64,
    src: *const u64,
    len: usize,
) {
    if Q >= (1u64 << 32) {
        unsafe { mul_assign_prime_wide_u64_avx2::<Q>(dst, src, len) };
        return;
    }

    let mut i = 0usize;
    while i + 4 <= len {
        let lhs = unsafe { _mm256_loadu_si256(dst.add(i) as *const __m256i) };
        let rhs = unsafe { _mm256_loadu_si256(src.add(i) as *const __m256i) };
        let products = _mm256_mul_epu32(lhs, rhs);
        let mut lanes = [0u64; 4];
        unsafe { _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, products) };
        for lane in &mut lanes {
            *lane = PrimeField::<Q>::mul_raw_words(*lane, 1);
        }
        let reduced = unsafe { _mm256_loadu_si256(lanes.as_ptr() as *const __m256i) };
        unsafe { _mm256_storeu_si256(dst.add(i) as *mut __m256i, reduced) };
        i += 4;
    }
    while i < len {
        unsafe {
            let product = dst.add(i).read().wrapping_mul(src.add(i).read());
            *dst.add(i) = PrimeField::<Q>::mul_raw_words(product, 1);
        }
        i += 1;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn scalar_mul_prime_montgomery_u64_avx2<const Q: u64>(
    dst: *mut u64,
    len: usize,
    scalar: u64,
) {
    if Q >= (1u64 << 32) {
        unsafe { scalar_mul_prime_wide_u64_avx2::<Q>(dst, len, scalar) };
        return;
    }

    let mut i = 0usize;
    let scalar_vec = _mm256_set1_epi64x(scalar as i64);
    while i + 4 <= len {
        let lhs = unsafe { _mm256_loadu_si256(dst.add(i) as *const __m256i) };
        let products = _mm256_mul_epu32(lhs, scalar_vec);
        let mut lanes = [0u64; 4];
        unsafe { _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, products) };
        for lane in &mut lanes {
            *lane = PrimeField::<Q>::mul_raw_words(*lane, 1);
        }
        let reduced = unsafe { _mm256_loadu_si256(lanes.as_ptr() as *const __m256i) };
        unsafe { _mm256_storeu_si256(dst.add(i) as *mut __m256i, reduced) };
        i += 4;
    }
    while i < len {
        unsafe {
            let product = dst.add(i).read().wrapping_mul(scalar);
            *dst.add(i) = PrimeField::<Q>::mul_raw_words(product, 1);
        }
        i += 1;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn butterfly_forward_prime_montgomery_u64_avx2<const Q: u64>(
    even: *mut u64,
    odd: *mut u64,
    twiddles: *const u64,
    len: usize,
) {
    if Q >= (1u64 << 32) {
        unsafe { butterfly_forward_prime_wide_u64_avx2::<Q>(even, odd, twiddles, len) };
        return;
    }

    let mut i = 0usize;
    let modulus_vec = _mm256_set1_epi64x(Q as i64);
    let threshold_vec = _mm256_set1_epi64x((Q - 1) as i64);
    while i + 4 <= len {
        let u = unsafe { _mm256_loadu_si256(even.add(i) as *const __m256i) };
        let v_raw = unsafe { _mm256_loadu_si256(odd.add(i) as *const __m256i) };
        let w = unsafe { _mm256_loadu_si256(twiddles.add(i) as *const __m256i) };
        let products = _mm256_mul_epu32(v_raw, w);
        let mut reduced_lanes = [0u64; 4];
        unsafe { _mm256_storeu_si256(reduced_lanes.as_mut_ptr() as *mut __m256i, products) };
        for lane in &mut reduced_lanes {
            *lane = PrimeField::<Q>::mul_raw_words(*lane, 1);
        }
        let v = unsafe { _mm256_loadu_si256(reduced_lanes.as_ptr() as *const __m256i) };

        let sum = _mm256_add_epi64(u, v);
        let reduce_mask = _mm256_cmpgt_epi64(sum, threshold_vec);
        let even_out = _mm256_sub_epi64(sum, _mm256_and_si256(reduce_mask, modulus_vec));

        let borrow_mask = _mm256_cmpgt_epi64(v, u);
        let odd_out = _mm256_add_epi64(
            _mm256_sub_epi64(u, v),
            _mm256_and_si256(borrow_mask, modulus_vec),
        );

        unsafe {
            _mm256_storeu_si256(even.add(i) as *mut __m256i, even_out);
            _mm256_storeu_si256(odd.add(i) as *mut __m256i, odd_out);
        }
        i += 4;
    }

    while i < len {
        unsafe {
            let u = even.add(i).read();
            let v = PrimeField::<Q>::mul_raw_words(odd.add(i).read(), twiddles.add(i).read());
            even.add(i)
                .write(if u + v >= Q { u + v - Q } else { u + v });
            odd.add(i).write(if u >= v { u - v } else { u + Q - v });
        }
        i += 1;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn butterfly_inverse_prime_montgomery_u64_avx2<const Q: u64>(
    even: *mut u64,
    odd: *mut u64,
    twiddles: *const u64,
    len: usize,
) {
    if Q >= (1u64 << 32) {
        unsafe { butterfly_inverse_prime_wide_u64_avx2::<Q>(even, odd, twiddles, len) };
        return;
    }

    let mut i = 0usize;
    let modulus_vec = _mm256_set1_epi64x(Q as i64);
    let threshold_vec = _mm256_set1_epi64x((Q - 1) as i64);
    while i + 4 <= len {
        let u = unsafe { _mm256_loadu_si256(even.add(i) as *const __m256i) };
        let v = unsafe { _mm256_loadu_si256(odd.add(i) as *const __m256i) };

        let sum = _mm256_add_epi64(u, v);
        let reduce_mask = _mm256_cmpgt_epi64(sum, threshold_vec);
        let even_out = _mm256_sub_epi64(sum, _mm256_and_si256(reduce_mask, modulus_vec));

        let borrow_mask = _mm256_cmpgt_epi64(v, u);
        let diff = _mm256_add_epi64(
            _mm256_sub_epi64(u, v),
            _mm256_and_si256(borrow_mask, modulus_vec),
        );
        let w = unsafe { _mm256_loadu_si256(twiddles.add(i) as *const __m256i) };
        let products = _mm256_mul_epu32(diff, w);
        let mut reduced_lanes = [0u64; 4];
        unsafe { _mm256_storeu_si256(reduced_lanes.as_mut_ptr() as *mut __m256i, products) };
        for lane in &mut reduced_lanes {
            *lane = PrimeField::<Q>::mul_raw_words(*lane, 1);
        }
        let odd_out = unsafe { _mm256_loadu_si256(reduced_lanes.as_ptr() as *const __m256i) };

        unsafe {
            _mm256_storeu_si256(even.add(i) as *mut __m256i, even_out);
            _mm256_storeu_si256(odd.add(i) as *mut __m256i, odd_out);
        }
        i += 4;
    }

    while i < len {
        unsafe {
            let u = even.add(i).read();
            let v = odd.add(i).read();
            let sum = if u + v >= Q { u + v - Q } else { u + v };
            let diff = if u >= v { u - v } else { u + Q - v };
            even.add(i).write(sum);
            odd.add(i)
                .write(PrimeField::<Q>::mul_raw_words(diff, twiddles.add(i).read()));
        }
        i += 1;
    }
}

/// AVX2-backed masked addition for contiguous `u64` slices.
pub(crate) unsafe fn add_assign_u64_masked(dst: &mut [u64], src: &[u64], mask: u64) {
    debug_assert_eq!(dst.len(), src.len());
    unsafe { add_assign_u64_masked_avx2(dst.as_mut_ptr(), src.as_ptr(), dst.len(), mask) };
}

/// AVX2-backed masked subtraction for contiguous `u64` slices.
pub(crate) unsafe fn sub_assign_u64_masked(dst: &mut [u64], src: &[u64], mask: u64) {
    debug_assert_eq!(dst.len(), src.len());
    unsafe { sub_assign_u64_masked_avx2(dst.as_mut_ptr(), src.as_ptr(), dst.len(), mask) };
}

/// AVX2-backed modular addition for contiguous `u64` slices with small prime moduli.
pub(crate) unsafe fn add_assign_prime_u64(dst: &mut [u64], src: &[u64], modulus: u64) {
    debug_assert_eq!(dst.len(), src.len());
    unsafe { add_assign_prime_u64_avx2(dst.as_mut_ptr(), src.as_ptr(), dst.len(), modulus) };
}

/// AVX2-backed modular subtraction for contiguous `u64` slices with small prime moduli.
pub(crate) unsafe fn sub_assign_prime_u64(dst: &mut [u64], src: &[u64], modulus: u64) {
    debug_assert_eq!(dst.len(), src.len());
    unsafe { sub_assign_prime_u64_avx2(dst.as_mut_ptr(), src.as_ptr(), dst.len(), modulus) };
}

/// AVX2-backed masked multiplication for contiguous `u64` slices whose values fit in 32 bits.
pub(crate) unsafe fn mul_assign_u64_low32_masked(dst: &mut [u64], src: &[u64], mask: u64) {
    debug_assert_eq!(dst.len(), src.len());
    unsafe { mul_assign_u64_low32_masked_avx2(dst.as_mut_ptr(), src.as_ptr(), dst.len(), mask) };
}

/// AVX2-backed masked scalar multiplication for contiguous `u64` slices whose values fit in 32 bits.
pub(crate) unsafe fn scalar_mul_u64_low32_masked(dst: &mut [u64], scalar: u64, mask: u64) {
    unsafe { scalar_mul_u64_low32_masked_avx2(dst.as_mut_ptr(), dst.len(), scalar, mask) };
}

/// AVX2-backed Montgomery pointwise multiplication for prime-field raw-value slices.
pub(crate) unsafe fn mul_assign_prime_montgomery_u64<const Q: u64>(dst: &mut [u64], src: &[u64]) {
    debug_assert_eq!(dst.len(), src.len());
    unsafe { mul_assign_prime_montgomery_u64_avx2::<Q>(dst.as_mut_ptr(), src.as_ptr(), dst.len()) };
}

/// AVX2-backed Montgomery scalar multiplication for prime-field raw-value slices.
pub(crate) unsafe fn scalar_mul_prime_montgomery_u64<const Q: u64>(dst: &mut [u64], scalar: u64) {
    unsafe { scalar_mul_prime_montgomery_u64_avx2::<Q>(dst.as_mut_ptr(), dst.len(), scalar) };
}

/// AVX2-backed in-place forward butterflies for prime-field raw-value slices.
pub(crate) unsafe fn butterfly_forward_prime_montgomery_u64<const Q: u64>(
    even: &mut [u64],
    odd: &mut [u64],
    twiddles: &[u64],
) {
    debug_assert_eq!(even.len(), odd.len());
    debug_assert_eq!(even.len(), twiddles.len());
    unsafe {
        butterfly_forward_prime_montgomery_u64_avx2::<Q>(
            even.as_mut_ptr(),
            odd.as_mut_ptr(),
            twiddles.as_ptr(),
            even.len(),
        )
    };
}

/// AVX2-backed in-place inverse butterflies for prime-field raw-value slices.
pub(crate) unsafe fn butterfly_inverse_prime_montgomery_u64<const Q: u64>(
    even: &mut [u64],
    odd: &mut [u64],
    twiddles: &[u64],
) {
    debug_assert_eq!(even.len(), odd.len());
    debug_assert_eq!(even.len(), twiddles.len());
    unsafe {
        butterfly_inverse_prime_montgomery_u64_avx2::<Q>(
            even.as_mut_ptr(),
            odd.as_mut_ptr(),
            twiddles.as_ptr(),
            even.len(),
        )
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arith::prime::GOLDILOCKS_MODULUS;
    use crate::arith::ring::{IntegerRing, Ring};

    #[test]
    fn test_masked_add_sub_matches_scalar_when_avx2_available() {
        if !avx2_available() {
            return;
        }

        let mask = (1u64 << 13) - 1;
        let lhs = [1u64, 2, 3, 4, 8191, 10, 17];
        let rhs = [5u64, 6, 7, 8, 1, 20, 40];

        let mut simd_add = lhs;
        let mut simd_sub = lhs;
        let scalar_add = core::array::from_fn::<u64, 7, _>(|i| lhs[i].wrapping_add(rhs[i]) & mask);
        let scalar_sub = core::array::from_fn::<u64, 7, _>(|i| lhs[i].wrapping_sub(rhs[i]) & mask);

        unsafe {
            add_assign_u64_masked(&mut simd_add, &rhs, mask);
            sub_assign_u64_masked(&mut simd_sub, &rhs, mask);
        }

        assert_eq!(simd_add, scalar_add);
        assert_eq!(simd_sub, scalar_sub);
    }

    #[test]
    fn test_masked_mul_matches_scalar_when_avx2_available() {
        if !avx2_available() {
            return;
        }

        let mask = (1u64 << 16) - 1;
        let lhs = [1u64, 2, 300, 400, 5000, 6000, 700];
        let rhs = [5u64, 6, 70, 80, 9, 10, 11];

        let mut simd_mul = lhs;
        let scalar_mul = core::array::from_fn::<u64, 7, _>(|i| lhs[i].wrapping_mul(rhs[i]) & mask);

        unsafe {
            mul_assign_u64_low32_masked(&mut simd_mul, &rhs, mask);
        }

        assert_eq!(simd_mul, scalar_mul);
    }

    #[test]
    fn test_prime_add_sub_matches_scalar_when_avx2_available() {
        if !avx2_available() {
            return;
        }

        let modulus = 12289u64;
        let lhs = [1u64, 2, 300, 400, 5000, 6000, 12288];
        let rhs = [5u64, 6, 70, 80, 9000, 7000, 1];

        let mut simd_add = lhs;
        let mut simd_sub = lhs;
        let scalar_add = core::array::from_fn::<u64, 7, _>(|i| {
            let sum = lhs[i] + rhs[i];
            if sum >= modulus { sum - modulus } else { sum }
        });
        let scalar_sub = core::array::from_fn::<u64, 7, _>(|i| {
            if lhs[i] >= rhs[i] {
                lhs[i] - rhs[i]
            } else {
                lhs[i] + modulus - rhs[i]
            }
        });

        unsafe {
            add_assign_prime_u64(&mut simd_add, &rhs, modulus);
            sub_assign_prime_u64(&mut simd_sub, &rhs, modulus);
        }

        assert_eq!(simd_add, scalar_add);
        assert_eq!(simd_sub, scalar_sub);
    }

    #[test]
    fn test_prime_montgomery_mul_matches_scalar_when_avx2_available() {
        if !avx2_available() {
            return;
        }

        let lhs = [
            PrimeField::<12289>::from_u64(1).raw(),
            PrimeField::<12289>::from_u64(2).raw(),
            PrimeField::<12289>::from_u64(300).raw(),
            PrimeField::<12289>::from_u64(400).raw(),
            PrimeField::<12289>::from_u64(5000).raw(),
            PrimeField::<12289>::from_u64(6000).raw(),
            PrimeField::<12289>::from_u64(12_288).raw(),
        ];
        let rhs = [
            PrimeField::<12289>::from_u64(5).raw(),
            PrimeField::<12289>::from_u64(6).raw(),
            PrimeField::<12289>::from_u64(70).raw(),
            PrimeField::<12289>::from_u64(80).raw(),
            PrimeField::<12289>::from_u64(9).raw(),
            PrimeField::<12289>::from_u64(10).raw(),
            PrimeField::<12289>::from_u64(11).raw(),
        ];

        let mut simd_mul = lhs;
        let scalar_mul = core::array::from_fn::<u64, 7, _>(|i| {
            PrimeField::<12289>::montgomery_reduce_word(lhs[i].wrapping_mul(rhs[i]))
        });

        unsafe {
            mul_assign_prime_montgomery_u64::<12289>(&mut simd_mul, &rhs);
        }

        assert_eq!(simd_mul, scalar_mul);
    }

    #[test]
    fn test_prime_forward_butterfly_matches_scalar_when_avx2_available() {
        if !avx2_available() {
            return;
        }

        let mut even = [
            PrimeField::<12289>::from_u64(1).raw(),
            PrimeField::<12289>::from_u64(2).raw(),
            PrimeField::<12289>::from_u64(300).raw(),
            PrimeField::<12289>::from_u64(400).raw(),
            PrimeField::<12289>::from_u64(5000).raw(),
        ];
        let mut odd = [
            PrimeField::<12289>::from_u64(5).raw(),
            PrimeField::<12289>::from_u64(6).raw(),
            PrimeField::<12289>::from_u64(70).raw(),
            PrimeField::<12289>::from_u64(80).raw(),
            PrimeField::<12289>::from_u64(9).raw(),
        ];
        let twiddles = [
            PrimeField::<12289>::one().raw(),
            PrimeField::<12289>::from_u64(7).raw(),
            PrimeField::<12289>::from_u64(11).raw(),
            PrimeField::<12289>::from_u64(17).raw(),
            PrimeField::<12289>::from_u64(19).raw(),
        ];

        let mut scalar_even = even;
        let mut scalar_odd = odd;
        for i in 0..even.len() {
            let u = scalar_even[i];
            let v = PrimeField::<12289>::montgomery_reduce_word(
                scalar_odd[i].wrapping_mul(twiddles[i]),
            );
            scalar_even[i] = if u + v >= 12289 { u + v - 12289 } else { u + v };
            scalar_odd[i] = if u >= v { u - v } else { u + 12289 - v };
        }

        unsafe {
            butterfly_forward_prime_montgomery_u64::<12289>(&mut even, &mut odd, &twiddles);
        }

        assert_eq!(even, scalar_even);
        assert_eq!(odd, scalar_odd);
    }

    #[test]
    fn test_prime_inverse_butterfly_matches_scalar_when_avx2_available() {
        if !avx2_available() {
            return;
        }

        let mut even = [
            PrimeField::<12289>::from_u64(1).raw(),
            PrimeField::<12289>::from_u64(2).raw(),
            PrimeField::<12289>::from_u64(300).raw(),
            PrimeField::<12289>::from_u64(400).raw(),
            PrimeField::<12289>::from_u64(5000).raw(),
        ];
        let mut odd = [
            PrimeField::<12289>::from_u64(5).raw(),
            PrimeField::<12289>::from_u64(6).raw(),
            PrimeField::<12289>::from_u64(70).raw(),
            PrimeField::<12289>::from_u64(80).raw(),
            PrimeField::<12289>::from_u64(9).raw(),
        ];
        let twiddles = [
            PrimeField::<12289>::one().raw(),
            PrimeField::<12289>::from_u64(7).raw(),
            PrimeField::<12289>::from_u64(11).raw(),
            PrimeField::<12289>::from_u64(17).raw(),
            PrimeField::<12289>::from_u64(19).raw(),
        ];

        let mut scalar_even = even;
        let mut scalar_odd = odd;
        for i in 0..even.len() {
            let u = scalar_even[i];
            let v = scalar_odd[i];
            let sum = if u + v >= 12289 { u + v - 12289 } else { u + v };
            let diff = if u >= v { u - v } else { u + 12289 - v };
            scalar_even[i] = sum;
            scalar_odd[i] =
                PrimeField::<12289>::montgomery_reduce_word(diff.wrapping_mul(twiddles[i]));
        }

        unsafe {
            butterfly_inverse_prime_montgomery_u64::<12289>(&mut even, &mut odd, &twiddles);
        }

        assert_eq!(even, scalar_even);
        assert_eq!(odd, scalar_odd);
    }

    #[test]
    fn test_wide_prime_montgomery_mul_matches_scalar_when_avx2_available() {
        if !avx2_available() {
            return;
        }

        let lhs = [
            PrimeField::<4294967311>::from_u64(1).raw(),
            PrimeField::<4294967311>::from_u64(2).raw(),
            PrimeField::<4294967311>::from_u64(300).raw(),
            PrimeField::<4294967311>::from_u64(400).raw(),
            PrimeField::<4294967311>::from_u64(5000).raw(),
            PrimeField::<4294967311>::from_u64(6000).raw(),
            PrimeField::<4294967311>::from_u64(4_294_967_310).raw(),
        ];
        let rhs = [
            PrimeField::<4294967311>::from_u64(5).raw(),
            PrimeField::<4294967311>::from_u64(6).raw(),
            PrimeField::<4294967311>::from_u64(70).raw(),
            PrimeField::<4294967311>::from_u64(80).raw(),
            PrimeField::<4294967311>::from_u64(9).raw(),
            PrimeField::<4294967311>::from_u64(10).raw(),
            PrimeField::<4294967311>::from_u64(11).raw(),
        ];

        let mut simd_mul = lhs;
        let scalar_mul = core::array::from_fn::<u64, 7, _>(|i| {
            PrimeField::<4294967311>::mul_raw_words(lhs[i], rhs[i])
        });

        unsafe {
            mul_assign_prime_montgomery_u64::<4294967311>(&mut simd_mul, &rhs);
        }

        assert_eq!(simd_mul, scalar_mul);
    }

    #[test]
    fn test_high_band_prime_montgomery_mul_matches_scalar_when_avx2_available() {
        if !avx2_available() {
            return;
        }

        let lhs = [
            PrimeField::<9223372036854775783>::from_u64(1).raw(),
            PrimeField::<9223372036854775783>::from_u64(2).raw(),
            PrimeField::<9223372036854775783>::from_u64(300).raw(),
            PrimeField::<9223372036854775783>::from_u64(400).raw(),
            PrimeField::<9223372036854775783>::from_u64(5_000).raw(),
            PrimeField::<9223372036854775783>::from_u64(6_000).raw(),
            PrimeField::<9223372036854775783>::from_u64(9_223_372_036_854_775_000).raw(),
        ];
        let rhs = [
            PrimeField::<9223372036854775783>::from_u64(5).raw(),
            PrimeField::<9223372036854775783>::from_u64(6).raw(),
            PrimeField::<9223372036854775783>::from_u64(70).raw(),
            PrimeField::<9223372036854775783>::from_u64(80).raw(),
            PrimeField::<9223372036854775783>::from_u64(9).raw(),
            PrimeField::<9223372036854775783>::from_u64(10).raw(),
            PrimeField::<9223372036854775783>::from_u64(11).raw(),
        ];

        let mut simd_mul = lhs;
        let scalar_mul = core::array::from_fn::<u64, 7, _>(|i| {
            PrimeField::<9223372036854775783>::mul_raw_words(lhs[i], rhs[i])
        });

        unsafe {
            mul_assign_prime_montgomery_u64::<9223372036854775783>(&mut simd_mul, &rhs);
        }

        assert_eq!(simd_mul, scalar_mul);
    }

    #[test]
    fn test_goldilocks_prime_montgomery_mul_matches_scalar_when_avx2_available() {
        if !avx2_available() {
            return;
        }

        let lhs = [
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64(1).raw(),
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64(2).raw(),
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64((1u64 << 32) - 1).raw(),
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64((1u64 << 32) + 7).raw(),
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64(123_456_789).raw(),
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64(9_876_543_210).raw(),
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64(GOLDILOCKS_MODULUS - 2).raw(),
        ];
        let rhs = [
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64(5).raw(),
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64(6).raw(),
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64((1u64 << 32) + 11).raw(),
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64(80).raw(),
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64(9).raw(),
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64(10).raw(),
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64(GOLDILOCKS_MODULUS - 3).raw(),
        ];

        let mut simd_mul = lhs;
        let scalar_mul = core::array::from_fn::<u64, 7, _>(|i| {
            PrimeField::<GOLDILOCKS_MODULUS>::mul_raw_words(lhs[i], rhs[i])
        });

        unsafe {
            mul_assign_prime_montgomery_u64::<GOLDILOCKS_MODULUS>(&mut simd_mul, &rhs);
        }

        assert_eq!(simd_mul, scalar_mul);
    }

    #[test]
    fn test_wide_prime_forward_butterfly_matches_scalar_when_avx2_available() {
        if !avx2_available() {
            return;
        }

        let mut even = [
            PrimeField::<4294967311>::from_u64(1).raw(),
            PrimeField::<4294967311>::from_u64(2).raw(),
            PrimeField::<4294967311>::from_u64(300).raw(),
            PrimeField::<4294967311>::from_u64(400).raw(),
            PrimeField::<4294967311>::from_u64(5000).raw(),
        ];
        let mut odd = [
            PrimeField::<4294967311>::from_u64(5).raw(),
            PrimeField::<4294967311>::from_u64(6).raw(),
            PrimeField::<4294967311>::from_u64(70).raw(),
            PrimeField::<4294967311>::from_u64(80).raw(),
            PrimeField::<4294967311>::from_u64(9).raw(),
        ];
        let twiddles = [
            PrimeField::<4294967311>::one().raw(),
            PrimeField::<4294967311>::from_u64(7).raw(),
            PrimeField::<4294967311>::from_u64(11).raw(),
            PrimeField::<4294967311>::from_u64(17).raw(),
            PrimeField::<4294967311>::from_u64(19).raw(),
        ];

        let mut scalar_even = even;
        let mut scalar_odd = odd;
        for i in 0..even.len() {
            let u = scalar_even[i];
            let v = PrimeField::<4294967311>::mul_raw_words(scalar_odd[i], twiddles[i]);
            scalar_even[i] = PrimeField::<4294967311>::add_raw_words(u, v);
            scalar_odd[i] = PrimeField::<4294967311>::sub_raw_words(u, v);
        }

        unsafe {
            butterfly_forward_prime_montgomery_u64::<4294967311>(&mut even, &mut odd, &twiddles);
        }

        assert_eq!(even, scalar_even);
        assert_eq!(odd, scalar_odd);
    }

    #[test]
    fn test_wide_prime_inverse_butterfly_matches_scalar_when_avx2_available() {
        if !avx2_available() {
            return;
        }

        let mut even = [
            PrimeField::<4294967311>::from_u64(1).raw(),
            PrimeField::<4294967311>::from_u64(2).raw(),
            PrimeField::<4294967311>::from_u64(300).raw(),
            PrimeField::<4294967311>::from_u64(400).raw(),
            PrimeField::<4294967311>::from_u64(5000).raw(),
        ];
        let mut odd = [
            PrimeField::<4294967311>::from_u64(5).raw(),
            PrimeField::<4294967311>::from_u64(6).raw(),
            PrimeField::<4294967311>::from_u64(70).raw(),
            PrimeField::<4294967311>::from_u64(80).raw(),
            PrimeField::<4294967311>::from_u64(9).raw(),
        ];
        let twiddles = [
            PrimeField::<4294967311>::one().raw(),
            PrimeField::<4294967311>::from_u64(7).raw(),
            PrimeField::<4294967311>::from_u64(11).raw(),
            PrimeField::<4294967311>::from_u64(17).raw(),
            PrimeField::<4294967311>::from_u64(19).raw(),
        ];

        let mut scalar_even = even;
        let mut scalar_odd = odd;
        for i in 0..even.len() {
            let u = scalar_even[i];
            let v = scalar_odd[i];
            let sum = PrimeField::<4294967311>::add_raw_words(u, v);
            let diff = PrimeField::<4294967311>::sub_raw_words(u, v);
            scalar_even[i] = sum;
            scalar_odd[i] = PrimeField::<4294967311>::mul_raw_words(diff, twiddles[i]);
        }

        unsafe {
            butterfly_inverse_prime_montgomery_u64::<4294967311>(&mut even, &mut odd, &twiddles);
        }

        assert_eq!(even, scalar_even);
        assert_eq!(odd, scalar_odd);
    }

    #[test]
    fn test_goldilocks_forward_butterfly_matches_scalar_when_avx2_available() {
        if !avx2_available() {
            return;
        }

        let mut even = [
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64(1).raw(),
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64(2).raw(),
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64((1u64 << 32) - 1).raw(),
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64((1u64 << 32) + 7).raw(),
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64(GOLDILOCKS_MODULUS - 2).raw(),
        ];
        let mut odd = [
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64(5).raw(),
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64(6).raw(),
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64(70).raw(),
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64(80).raw(),
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64(9).raw(),
        ];
        let twiddles = [
            PrimeField::<GOLDILOCKS_MODULUS>::one().raw(),
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64(7).raw(),
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64(11).raw(),
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64(17).raw(),
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64(19).raw(),
        ];

        let mut scalar_even = even;
        let mut scalar_odd = odd;
        for i in 0..even.len() {
            let u = scalar_even[i];
            let v = PrimeField::<GOLDILOCKS_MODULUS>::mul_raw_words(scalar_odd[i], twiddles[i]);
            scalar_even[i] = PrimeField::<GOLDILOCKS_MODULUS>::add_raw_words(u, v);
            scalar_odd[i] = PrimeField::<GOLDILOCKS_MODULUS>::sub_raw_words(u, v);
        }

        unsafe {
            butterfly_forward_prime_montgomery_u64::<GOLDILOCKS_MODULUS>(
                &mut even, &mut odd, &twiddles,
            );
        }

        assert_eq!(even, scalar_even);
        assert_eq!(odd, scalar_odd);
    }

    #[test]
    fn test_goldilocks_inverse_butterfly_matches_scalar_when_avx2_available() {
        if !avx2_available() {
            return;
        }

        let mut even = [
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64(1).raw(),
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64(2).raw(),
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64((1u64 << 32) - 1).raw(),
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64((1u64 << 32) + 7).raw(),
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64(GOLDILOCKS_MODULUS - 2).raw(),
        ];
        let mut odd = [
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64(5).raw(),
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64(6).raw(),
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64(70).raw(),
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64(80).raw(),
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64(9).raw(),
        ];
        let twiddles = [
            PrimeField::<GOLDILOCKS_MODULUS>::one().raw(),
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64(7).raw(),
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64(11).raw(),
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64(17).raw(),
            PrimeField::<GOLDILOCKS_MODULUS>::from_u64(19).raw(),
        ];

        let mut scalar_even = even;
        let mut scalar_odd = odd;
        for i in 0..even.len() {
            let u = scalar_even[i];
            let v = scalar_odd[i];
            let sum = PrimeField::<GOLDILOCKS_MODULUS>::add_raw_words(u, v);
            let diff = PrimeField::<GOLDILOCKS_MODULUS>::sub_raw_words(u, v);
            scalar_even[i] = sum;
            scalar_odd[i] = PrimeField::<GOLDILOCKS_MODULUS>::mul_raw_words(diff, twiddles[i]);
        }

        unsafe {
            butterfly_inverse_prime_montgomery_u64::<GOLDILOCKS_MODULUS>(
                &mut even, &mut odd, &twiddles,
            );
        }

        assert_eq!(even, scalar_even);
        assert_eq!(odd, scalar_odd);
    }
}
