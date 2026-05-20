//! aarch64 SIMD backend modules.

pub(crate) mod gf2;
pub(crate) mod u16_arith;
pub(crate) mod u32_arith;
pub(crate) mod u64_arith;
pub(crate) mod u8_arith;

pub(crate) use u64_arith::neon_available;
// Re-export u64 API for backward compatibility with existing callers.
pub(crate) use u64_arith::{
    add_assign_prime_u64, add_assign_u64_masked, butterfly_forward_prime_montgomery_u64,
    butterfly_inverse_prime_montgomery_u64, mul_assign_prime_montgomery_u64,
    mul_assign_u64_low32_masked, scalar_mul_prime_montgomery_u64, scalar_mul_u64_low32_masked,
    sub_assign_prime_u64, sub_assign_u64_masked,
};
