//! Profiles for [`LargeZm`](super::large_zm::LargeZm).

use super::large_zm::LargeZmProfile;

/// Profile for the Fermat ring `Z/(2^64 + 1)Z`.
pub enum Fermat64Profile {}

impl LargeZmProfile<2> for Fermat64Profile {
    const MODULUS: [u64; 2] = [1, 1]; // 2^64 + 1
    const MONTGOMERY: bool = true;
    // R = 2^128 ≡ 1 mod (2^64+1).
    const MONT_ONE: [u64; 2] = [1, 0];
    // R = 2^128 ≡ 1 mod (2^64+1), so R^2 ≡ 1 = [1, 0].
    const MONT_R2: [u64; 2] = [1, 0];
    // -MODULUS[0]^{-1} mod 2^64 = -1 = u64::MAX.
    const MONT_NEG_INV: u64 = u64::MAX;
    // floor(2^128 / (2^64+1)) = floor(2^64 - 1 + eps) = 2^64 - 1.
    const BARRETT_MU: [u64; 2] = [u64::MAX, 0];
}
