//! aarch64 SIMD backend hooks.

use crate::arith::prime::PrimeField;

use core::arch::aarch64::{
    uint64x2_t, vaddq_u64, vandq_u64, vcgtq_u64, vdupq_n_u64, vld1q_u64, vmovn_u64, vmull_u32,
    vorrq_u64, vshlq_n_u64, vshrn_n_u64, vshrq_n_u64, vst1q_u64, vsubq_u64,
};

#[cfg(all(feature = "std", target_arch = "aarch64"))]
pub(crate) fn neon_available() -> bool {
    std::arch::is_aarch64_feature_detected!("neon")
}

#[cfg(all(not(feature = "std"), target_arch = "aarch64", target_feature = "neon"))]
pub(crate) fn neon_available() -> bool {
    true
}

#[cfg(not(any(
    all(feature = "std", target_arch = "aarch64"),
    all(not(feature = "std"), target_arch = "aarch64", target_feature = "neon")
)))]
pub(crate) fn neon_available() -> bool {
    false
}

#[target_feature(enable = "neon")]
unsafe fn add_assign_u64_masked_neon(dst: *mut u64, src: *const u64, len: usize, mask: u64) {
    let mut i = 0usize;
    let mask_vec = vdupq_n_u64(mask);
    while i + 2 <= len {
        let lhs = unsafe { vld1q_u64(dst.add(i)) };
        let rhs = unsafe { vld1q_u64(src.add(i)) };
        let sum = vandq_u64(vaddq_u64(lhs, rhs), mask_vec);
        unsafe { vst1q_u64(dst.add(i), sum) };
        i += 2;
    }
    while i < len {
        unsafe {
            *dst.add(i) = dst.add(i).read().wrapping_add(src.add(i).read()) & mask;
        }
        i += 1;
    }
}

#[target_feature(enable = "neon")]
unsafe fn sub_assign_u64_masked_neon(dst: *mut u64, src: *const u64, len: usize, mask: u64) {
    let mut i = 0usize;
    let mask_vec = vdupq_n_u64(mask);
    while i + 2 <= len {
        let lhs = unsafe { vld1q_u64(dst.add(i)) };
        let rhs = unsafe { vld1q_u64(src.add(i)) };
        let diff = vandq_u64(vsubq_u64(lhs, rhs), mask_vec);
        unsafe { vst1q_u64(dst.add(i), diff) };
        i += 2;
    }
    while i < len {
        unsafe {
            *dst.add(i) = dst.add(i).read().wrapping_sub(src.add(i).read()) & mask;
        }
        i += 1;
    }
}

#[target_feature(enable = "neon")]
unsafe fn lanes_from_u64x2(vec: uint64x2_t) -> [u64; 2] {
    let mut lanes = [0u64; 2];
    unsafe { vst1q_u64(lanes.as_mut_ptr(), vec) };
    lanes
}

#[target_feature(enable = "neon")]
unsafe fn u64x2_from_lanes(lanes: &[u64; 2]) -> uint64x2_t {
    unsafe { vld1q_u64(lanes.as_ptr()) }
}

#[target_feature(enable = "neon")]
unsafe fn mul_assign_u64_low32_masked_neon(dst: *mut u64, src: *const u64, len: usize, mask: u64) {
    let mut i = 0usize;
    while i + 2 <= len {
        let lhs = unsafe { vld1q_u64(dst.add(i)) };
        let rhs = unsafe { vld1q_u64(src.add(i)) };
        let mut products = unsafe { lanes_from_u64x2(lhs) };
        let rhs_lanes = unsafe { lanes_from_u64x2(rhs) };
        for (lane, rhs_lane) in products.iter_mut().zip(rhs_lanes.iter()) {
            *lane = lane.wrapping_mul(*rhs_lane) & mask;
        }
        let product_vec = unsafe { u64x2_from_lanes(&products) };
        unsafe { vst1q_u64(dst.add(i), product_vec) };
        i += 2;
    }
    while i < len {
        unsafe {
            *dst.add(i) = dst.add(i).read().wrapping_mul(src.add(i).read()) & mask;
        }
        i += 1;
    }
}

#[target_feature(enable = "neon")]
unsafe fn scalar_mul_u64_low32_masked_neon(dst: *mut u64, len: usize, scalar: u64, mask: u64) {
    let mut i = 0usize;
    while i + 2 <= len {
        let lhs = unsafe { vld1q_u64(dst.add(i)) };
        let mut products = unsafe { lanes_from_u64x2(lhs) };
        for lane in &mut products {
            *lane = lane.wrapping_mul(scalar) & mask;
        }
        let product_vec = unsafe { u64x2_from_lanes(&products) };
        unsafe { vst1q_u64(dst.add(i), product_vec) };
        i += 2;
    }
    while i < len {
        unsafe {
            *dst.add(i) = dst.add(i).read().wrapping_mul(scalar) & mask;
        }
        i += 1;
    }
}

#[target_feature(enable = "neon")]
unsafe fn add_assign_prime_u64_neon(dst: *mut u64, src: *const u64, len: usize, modulus: u64) {
    let mut i = 0usize;
    let modulus_vec = vdupq_n_u64(modulus);
    let threshold_vec = vdupq_n_u64(modulus - 1);
    while i + 2 <= len {
        let lhs = unsafe { vld1q_u64(dst.add(i)) };
        let rhs = unsafe { vld1q_u64(src.add(i)) };
        let sum = vaddq_u64(lhs, rhs);
        let reduce_mask = vcgtq_u64(sum, threshold_vec);
        let reduced = vsubq_u64(sum, vandq_u64(reduce_mask, modulus_vec));
        unsafe { vst1q_u64(dst.add(i), reduced) };
        i += 2;
    }
    while i < len {
        unsafe {
            let sum = dst.add(i).read() + src.add(i).read();
            *dst.add(i) = if sum >= modulus { sum - modulus } else { sum };
        }
        i += 1;
    }
}

#[target_feature(enable = "neon")]
unsafe fn sub_assign_prime_u64_neon(dst: *mut u64, src: *const u64, len: usize, modulus: u64) {
    let mut i = 0usize;
    let modulus_vec = vdupq_n_u64(modulus);
    while i + 2 <= len {
        let lhs = unsafe { vld1q_u64(dst.add(i)) };
        let rhs = unsafe { vld1q_u64(src.add(i)) };
        let borrow_mask = vcgtq_u64(rhs, lhs);
        let diff = vaddq_u64(vsubq_u64(lhs, rhs), vandq_u64(borrow_mask, modulus_vec));
        unsafe { vst1q_u64(dst.add(i), diff) };
        i += 2;
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

#[target_feature(enable = "neon")]
unsafe fn mul_epu64_wide_neon(lhs: uint64x2_t, rhs: uint64x2_t) -> (uint64x2_t, uint64x2_t) {
    let lo32_a = vmovn_u64(lhs);
    let lo32_b = vmovn_u64(rhs);
    let hi32_a = vshrn_n_u64::<32>(lhs);
    let hi32_b = vshrn_n_u64::<32>(rhs);

    let p0 = vmull_u32(lo32_a, lo32_b);
    let p1 = vmull_u32(lo32_a, hi32_b);
    let p2 = vmull_u32(hi32_a, lo32_b);
    let p3 = vmull_u32(hi32_a, hi32_b);

    let carry32 = vdupq_n_u64(1u64 << 32);
    let carry1 = vdupq_n_u64(1);

    let cross = vaddq_u64(p1, p2);
    let cross_carry = vcgtq_u64(p1, cross);

    let lo = vaddq_u64(p0, vshlq_n_u64::<32>(cross));
    let lo_carry = vcgtq_u64(p0, lo);

    let hi = vaddq_u64(
        vaddq_u64(
            p3,
            vaddq_u64(vshrq_n_u64::<32>(cross), vandq_u64(cross_carry, carry32)),
        ),
        vandq_u64(lo_carry, carry1),
    );

    (lo, hi)
}

#[target_feature(enable = "neon")]
unsafe fn montgomery_mul_u64_neon<const Q: u64>(lhs: uint64x2_t, rhs: uint64x2_t) -> uint64x2_t {
    let modulus_vec = vdupq_n_u64(Q);
    let threshold_vec = vdupq_n_u64(Q - 1);
    let q_inv_vec = vdupq_n_u64(PrimeField::<Q>::Q_INV_U64);
    let carry_bit_vec = vdupq_n_u64(1);

    let (t_lo, t_hi) = unsafe { mul_epu64_wide_neon(lhs, rhs) };
    let (m, _) = unsafe { mul_epu64_wide_neon(t_lo, q_inv_vec) };
    let (mq_lo, mq_hi) = unsafe { mul_epu64_wide_neon(m, modulus_vec) };

    let sum_lo = vaddq_u64(t_lo, mq_lo);
    let carry_lo = vcgtq_u64(t_lo, sum_lo);

    let sum_hi_a = vaddq_u64(t_hi, mq_hi);
    let carry_hi_a = vcgtq_u64(t_hi, sum_hi_a);
    let sum_hi = vaddq_u64(sum_hi_a, vandq_u64(carry_lo, carry_bit_vec));
    let carry_hi_b = vcgtq_u64(sum_hi_a, sum_hi);

    let reduce = vorrq_u64(
        vorrq_u64(carry_hi_a, carry_hi_b),
        vcgtq_u64(sum_hi, threshold_vec),
    );
    vsubq_u64(sum_hi, vandq_u64(reduce, modulus_vec))
}

#[target_feature(enable = "neon")]
unsafe fn mul_assign_prime_montgomery_u64_neon<const Q: u64>(
    dst: *mut u64,
    src: *const u64,
    len: usize,
) {
    let mut i = 0usize;
    while i + 2 <= len {
        let lhs = unsafe { vld1q_u64(dst.add(i)) };
        let rhs = unsafe { vld1q_u64(src.add(i)) };
        let reduced_vec = unsafe { montgomery_mul_u64_neon::<Q>(lhs, rhs) };
        unsafe { vst1q_u64(dst.add(i), reduced_vec) };
        i += 2;
    }
    while i < len {
        unsafe {
            let product = dst.add(i).read().wrapping_mul(src.add(i).read());
            *dst.add(i) = PrimeField::<Q>::montgomery_reduce_word(product);
        }
        i += 1;
    }
}

#[target_feature(enable = "neon")]
unsafe fn scalar_mul_prime_montgomery_u64_neon<const Q: u64>(
    dst: *mut u64,
    len: usize,
    scalar: u64,
) {
    let mut i = 0usize;
    let scalar_vec = vdupq_n_u64(scalar);
    while i + 2 <= len {
        let lhs = unsafe { vld1q_u64(dst.add(i)) };
        let reduced_vec = unsafe { montgomery_mul_u64_neon::<Q>(lhs, scalar_vec) };
        unsafe { vst1q_u64(dst.add(i), reduced_vec) };
        i += 2;
    }
    while i < len {
        unsafe {
            let product = dst.add(i).read().wrapping_mul(scalar);
            *dst.add(i) = PrimeField::<Q>::montgomery_reduce_word(product);
        }
        i += 1;
    }
}

#[target_feature(enable = "neon")]
unsafe fn butterfly_forward_prime_montgomery_u64_neon<const Q: u64>(
    even: *mut u64,
    odd: *mut u64,
    twiddles: *const u64,
    len: usize,
) {
    let mut i = 0usize;
    let modulus_vec = vdupq_n_u64(Q);
    let threshold_vec = vdupq_n_u64(Q - 1);
    while i + 2 <= len {
        let u = unsafe { vld1q_u64(even.add(i)) };
        let v_raw = unsafe { vld1q_u64(odd.add(i)) };
        let w = unsafe { vld1q_u64(twiddles.add(i)) };
        let v = unsafe { montgomery_mul_u64_neon::<Q>(v_raw, w) };

        let sum = vaddq_u64(u, v);
        let reduce_mask = vcgtq_u64(sum, threshold_vec);
        let even_out = vsubq_u64(sum, vandq_u64(reduce_mask, modulus_vec));

        let borrow_mask = vcgtq_u64(v, u);
        let odd_out = vaddq_u64(vsubq_u64(u, v), vandq_u64(borrow_mask, modulus_vec));

        unsafe {
            vst1q_u64(even.add(i), even_out);
            vst1q_u64(odd.add(i), odd_out);
        }
        i += 2;
    }

    while i < len {
        unsafe {
            let u = even.add(i).read();
            let v = PrimeField::<Q>::montgomery_reduce_word(
                odd.add(i).read().wrapping_mul(twiddles.add(i).read()),
            );
            even.add(i)
                .write(if u + v >= Q { u + v - Q } else { u + v });
            odd.add(i).write(if u >= v { u - v } else { u + Q - v });
        }
        i += 1;
    }
}

#[target_feature(enable = "neon")]
unsafe fn butterfly_inverse_prime_montgomery_u64_neon<const Q: u64>(
    even: *mut u64,
    odd: *mut u64,
    twiddles: *const u64,
    len: usize,
) {
    let mut i = 0usize;
    let modulus_vec = vdupq_n_u64(Q);
    let threshold_vec = vdupq_n_u64(Q - 1);
    while i + 2 <= len {
        let u = unsafe { vld1q_u64(even.add(i)) };
        let v = unsafe { vld1q_u64(odd.add(i)) };

        let sum = vaddq_u64(u, v);
        let reduce_mask = vcgtq_u64(sum, threshold_vec);
        let even_out = vsubq_u64(sum, vandq_u64(reduce_mask, modulus_vec));

        let borrow_mask = vcgtq_u64(v, u);
        let diff = vaddq_u64(vsubq_u64(u, v), vandq_u64(borrow_mask, modulus_vec));

        let w = unsafe { vld1q_u64(twiddles.add(i)) };
        let odd_out = unsafe { montgomery_mul_u64_neon::<Q>(diff, w) };

        unsafe {
            vst1q_u64(even.add(i), even_out);
            vst1q_u64(odd.add(i), odd_out);
        }
        i += 2;
    }

    while i < len {
        unsafe {
            let u = even.add(i).read();
            let v = odd.add(i).read();
            let sum = if u + v >= Q { u + v - Q } else { u + v };
            let diff = if u >= v { u - v } else { u + Q - v };
            even.add(i).write(sum);
            odd.add(i).write(PrimeField::<Q>::montgomery_reduce_word(
                diff.wrapping_mul(twiddles.add(i).read()),
            ));
        }
        i += 1;
    }
}

/// NEON-backed masked addition for contiguous `u64` slices.
pub(crate) unsafe fn add_assign_u64_masked(dst: &mut [u64], src: &[u64], mask: u64) {
    debug_assert_eq!(dst.len(), src.len());
    unsafe { add_assign_u64_masked_neon(dst.as_mut_ptr(), src.as_ptr(), dst.len(), mask) };
}

/// NEON-backed masked subtraction for contiguous `u64` slices.
pub(crate) unsafe fn sub_assign_u64_masked(dst: &mut [u64], src: &[u64], mask: u64) {
    debug_assert_eq!(dst.len(), src.len());
    unsafe { sub_assign_u64_masked_neon(dst.as_mut_ptr(), src.as_ptr(), dst.len(), mask) };
}

/// NEON-backed modular addition for contiguous `u64` slices with small prime moduli.
pub(crate) unsafe fn add_assign_prime_u64(dst: &mut [u64], src: &[u64], modulus: u64) {
    debug_assert_eq!(dst.len(), src.len());
    unsafe { add_assign_prime_u64_neon(dst.as_mut_ptr(), src.as_ptr(), dst.len(), modulus) };
}

/// NEON-backed modular subtraction for contiguous `u64` slices with small prime moduli.
pub(crate) unsafe fn sub_assign_prime_u64(dst: &mut [u64], src: &[u64], modulus: u64) {
    debug_assert_eq!(dst.len(), src.len());
    unsafe { sub_assign_prime_u64_neon(dst.as_mut_ptr(), src.as_ptr(), dst.len(), modulus) };
}

/// NEON-backed masked multiplication for contiguous `u64` slices whose values fit in 32 bits.
pub(crate) unsafe fn mul_assign_u64_low32_masked(dst: &mut [u64], src: &[u64], mask: u64) {
    debug_assert_eq!(dst.len(), src.len());
    unsafe { mul_assign_u64_low32_masked_neon(dst.as_mut_ptr(), src.as_ptr(), dst.len(), mask) };
}

/// NEON-backed masked scalar multiplication for contiguous `u64` slices whose values fit in 32 bits.
pub(crate) unsafe fn scalar_mul_u64_low32_masked(dst: &mut [u64], scalar: u64, mask: u64) {
    unsafe { scalar_mul_u64_low32_masked_neon(dst.as_mut_ptr(), dst.len(), scalar, mask) };
}

/// NEON-backed Montgomery pointwise multiplication for prime-field raw-value slices.
pub(crate) unsafe fn mul_assign_prime_montgomery_u64<const Q: u64>(dst: &mut [u64], src: &[u64]) {
    debug_assert_eq!(dst.len(), src.len());
    unsafe { mul_assign_prime_montgomery_u64_neon::<Q>(dst.as_mut_ptr(), src.as_ptr(), dst.len()) };
}

/// NEON-backed Montgomery scalar multiplication for prime-field raw-value slices.
pub(crate) unsafe fn scalar_mul_prime_montgomery_u64<const Q: u64>(dst: &mut [u64], scalar: u64) {
    unsafe { scalar_mul_prime_montgomery_u64_neon::<Q>(dst.as_mut_ptr(), dst.len(), scalar) };
}

/// NEON-backed in-place forward butterflies for prime-field raw-value slices.
pub(crate) unsafe fn butterfly_forward_prime_montgomery_u64<const Q: u64>(
    even: &mut [u64],
    odd: &mut [u64],
    twiddles: &[u64],
) {
    debug_assert_eq!(even.len(), odd.len());
    debug_assert_eq!(even.len(), twiddles.len());
    unsafe {
        butterfly_forward_prime_montgomery_u64_neon::<Q>(
            even.as_mut_ptr(),
            odd.as_mut_ptr(),
            twiddles.as_ptr(),
            even.len(),
        )
    };
}

/// NEON-backed in-place inverse butterflies for prime-field raw-value slices.
pub(crate) unsafe fn butterfly_inverse_prime_montgomery_u64<const Q: u64>(
    even: &mut [u64],
    odd: &mut [u64],
    twiddles: &[u64],
) {
    debug_assert_eq!(even.len(), odd.len());
    debug_assert_eq!(even.len(), twiddles.len());
    unsafe {
        butterfly_inverse_prime_montgomery_u64_neon::<Q>(
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
    use crate::arith::ring::{IntegerRing, Ring};

    #[test]
    fn test_masked_add_sub_matches_scalar_when_neon_available() {
        if !neon_available() {
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
    fn test_prime_add_sub_matches_scalar_when_neon_available() {
        if !neon_available() {
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
    fn test_masked_mul_matches_scalar_when_neon_available() {
        if !neon_available() {
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
    fn test_prime_montgomery_mul_matches_scalar_when_neon_available() {
        if !neon_available() {
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
    fn test_prime_forward_butterfly_matches_scalar_when_neon_available() {
        if !neon_available() {
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
    fn test_prime_inverse_butterfly_matches_scalar_when_neon_available() {
        if !neon_available() {
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
}
