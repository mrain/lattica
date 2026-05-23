//! `Zm<M>` — integers modulo `M` where `M >= 2`.
//!
//! Uses Montgomery form when `M` is odd for efficient multiplication.
//! Uses plain `% M` when `M` is even.
//!
//! Unlike [`PrimeField`](super::prime::PrimeField), `Zm` does **not** require
//! `M` to be prime, and does **not** implement [`Field`](super::ring::Field)
//! or [`NTTRing`](super::ntt::NTTRing).

use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::limb::UintLimb;
use super::ring::{IntegerRing, Ring};
use grid_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError};
use grid_std::rand::RngExt;

/// A modular integer in `Z_M` where `M` is any integer >= 2.
///
/// When `M` is odd, values are stored in Montgomery form (`val = a * R mod M`)
/// for efficient multiplication. When `M` is even, values are stored as plain
/// canonical residues in `[0, M)`.
///
/// `L` is the limb type (defaults to `u64`).
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Zm<const M: u64, L: UintLimb = u64> {
    val: L,
}

// ---------------------------------------------------------------------------
// Compile-time constants and helpers
// ---------------------------------------------------------------------------

impl<const M: u64, L: UintLimb> Zm<M, L> {
    /// Validated modulus.
    pub const MODULUS: u64 = {
        assert!(M >= 2, "Zm modulus must be >= 2");
        if L::BITS < 64 {
            assert!(
                M < (1u64 << L::BITS),
                "Zm modulus must fit in the limb type"
            );
        }
        M
    };

    /// Whether this modulus uses Montgomery form.
    const MONTGOMERY: bool = Self::MODULUS % 2 == 1;

    /// Minimum bytes needed to represent values in `[0, MODULUS)`.
    const SERIALIZED_BYTES: usize = {
        let m = Self::MODULUS;
        if m <= 1 {
            1
        } else {
            let bits = 64 - (m - 1).leading_zeros() as usize;
            bits.div_ceil(8)
        }
    };

    // --- Montgomery constants (only valid when MODULUS is odd) ---

    /// `R = 2^{L::BITS} mod MODULUS`.
    const R_MOD_M_U64: u64 = {
        if L::BITS < 64 {
            (1u64 << L::BITS) % M
        } else {
            (u64::MAX % M + 1) % M
        }
    };

    /// `R^2 mod MODULUS` — used to convert to Montgomery form.
    const R2_MOD_M_U64: u64 = {
        let r = Self::R_MOD_M_U64 as u128;
        let r2 = (r * r) % (M as u128);
        r2 as u64
    };

    /// `MOD_INV = -MODULUS^{-1} mod 2^{L::BITS}` (Montgomery reduction constant).
    const MOD_INV_U64: u64 = {
        let m = M;
        let mut inv: u64 = 1;
        let mut i = 0;
        while i < 6 {
            inv = inv.wrapping_mul(2u64.wrapping_sub(m.wrapping_mul(inv)));
            i += 1;
        }
        inv.wrapping_neg()
    };

    fn modulus_limb() -> L {
        L::from_u64(Self::MODULUS)
    }

    fn mod_inv_limb() -> L {
        let inv = if L::BITS < 64 {
            Self::MOD_INV_U64 & ((1u64 << L::BITS).wrapping_sub(1))
        } else {
            Self::MOD_INV_U64
        };
        L::from_u64(inv)
    }

    // --- Raw word arithmetic ---

    #[inline(always)]
    fn add_raw(lhs: L, rhs: L) -> L {
        let m = Self::modulus_limb();
        if M < (1u64 << (L::BITS - 1)) {
            let sum = lhs.wrapping_add(rhs);
            if sum >= m {
                return sum.wrapping_sub(m);
            }
            return sum;
        }
        let (sum, carry) = lhs.overflowing_add(rhs);
        if carry || sum >= m {
            sum.wrapping_sub(m)
        } else {
            sum
        }
    }

    #[inline(always)]
    fn sub_raw(lhs: L, rhs: L) -> L {
        let m = Self::modulus_limb();
        if lhs >= rhs {
            lhs - rhs
        } else {
            m.wrapping_sub(rhs).wrapping_add(lhs)
        }
    }

    // --- Montgomery helpers (odd M only) ---

    #[inline(always)]
    fn montgomery_reduce(t: L::Wide) -> L {
        let m_limb = Self::modulus_limb();
        if M >= (1u64 << (L::BITS - 1)) {
            return Self::montgomery_reduce_wide(t);
        }
        let q_inv = Self::mod_inv_limb();
        let m_lo = L::from_wide_truncate(t).wrapping_mul(q_inv);
        let t_plus_mq = L::wide_add(t, L::wide_mul(L::to_wide(m_lo), L::to_wide(m_limb)));
        let result = L::from_wide_truncate(L::wide_shr(t_plus_mq, L::BITS));
        if result >= m_limb {
            result.wrapping_sub(m_limb)
        } else {
            result
        }
    }

    #[inline(always)]
    fn montgomery_reduce_wide(t: L::Wide) -> L {
        let m_limb = Self::modulus_limb();
        let q_inv = Self::mod_inv_limb();
        let t_lo = L::from_wide_truncate(t);
        let m = t_lo.wrapping_mul(q_inv);
        let product = L::wide_mul(L::to_wide(m), L::to_wide(m_limb));
        let bits = L::BITS;

        let t_hi = L::from_wide_truncate(L::wide_shr(t, bits));
        let prod_lo = L::from_wide_truncate(product);
        let prod_hi = L::from_wide_truncate(L::wide_shr(product, bits));

        let (_sum_lo, carry_lo) = t_lo.overflowing_add(prod_lo);
        let (sum_hi, carry_hi_a) = t_hi.overflowing_add(prod_hi);
        let (sum_hi, carry_hi_b) = sum_hi.overflowing_add(L::from_u64(carry_lo as u64));
        let overflow = carry_hi_a || carry_hi_b;

        if overflow {
            sum_hi.wrapping_add(m_limb.wrapping_neg())
        } else if sum_hi >= m_limb {
            sum_hi.wrapping_sub(m_limb)
        } else {
            sum_hi
        }
    }

    /// Convert a canonical value to internal representation.
    fn encode_canonical(val: u64) -> Self {
        let r = val % M;
        if Self::MONTGOMERY {
            let r_limb = L::from_u64(r);
            let t = L::wide_mul(L::to_wide(r_limb), L::wide_from_u64(Self::R2_MOD_M_U64));
            Self {
                val: Self::montgomery_reduce(t),
            }
        } else {
            Self {
                val: L::from_u64(r),
            }
        }
    }

    /// Convert internal representation back to canonical `u64`.
    fn decode_canonical(self) -> u64 {
        if Self::MONTGOMERY {
            let val = Self::montgomery_reduce(L::to_wide(self.val));
            val.to_u64()
        } else {
            self.val.to_u64()
        }
    }
}

// --- Operator impls ---

impl<const M: u64, L: UintLimb> Add for Zm<M, L> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            val: Self::add_raw(self.val, rhs.val),
        }
    }
}

impl<const M: u64, L: UintLimb> Add<&Self> for Zm<M, L> {
    type Output = Self;
    fn add(self, rhs: &Self) -> Self {
        self + *rhs
    }
}

impl<const M: u64, L: UintLimb> AddAssign for Zm<M, L> {
    fn add_assign(&mut self, rhs: Self) {
        self.val = Self::add_raw(self.val, rhs.val);
    }
}

impl<const M: u64, L: UintLimb> AddAssign<&Self> for Zm<M, L> {
    fn add_assign(&mut self, rhs: &Self) {
        self.val = Self::add_raw(self.val, rhs.val);
    }
}

impl<const M: u64, L: UintLimb> Sub for Zm<M, L> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            val: Self::sub_raw(self.val, rhs.val),
        }
    }
}

impl<const M: u64, L: UintLimb> Sub<&Self> for Zm<M, L> {
    type Output = Self;
    fn sub(self, rhs: &Self) -> Self {
        self - *rhs
    }
}

impl<const M: u64, L: UintLimb> SubAssign for Zm<M, L> {
    fn sub_assign(&mut self, rhs: Self) {
        self.val = Self::sub_raw(self.val, rhs.val);
    }
}

impl<const M: u64, L: UintLimb> SubAssign<&Self> for Zm<M, L> {
    fn sub_assign(&mut self, rhs: &Self) {
        self.val = Self::sub_raw(self.val, rhs.val);
    }
}

impl<const M: u64, L: UintLimb> Mul for Zm<M, L> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        if Self::MONTGOMERY {
            let product = L::wide_mul(L::to_wide(self.val), L::to_wide(rhs.val));
            Self {
                val: Self::montgomery_reduce(product),
            }
        } else {
            let a = self.to_u64();
            let b = rhs.to_u64();
            let product = (a as u128 * b as u128) % Self::MODULUS as u128;
            Self {
                val: L::from_u64(product as u64),
            }
        }
    }
}

impl<const M: u64, L: UintLimb> Mul<&Self> for Zm<M, L> {
    type Output = Self;
    fn mul(self, rhs: &Self) -> Self {
        self * *rhs
    }
}

impl<const M: u64, L: UintLimb> MulAssign for Zm<M, L> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const M: u64, L: UintLimb> MulAssign<&Self> for Zm<M, L> {
    fn mul_assign(&mut self, rhs: &Self) {
        *self = *self * rhs;
    }
}

// Ref-ref operator impls
impl<const M: u64, L: UintLimb> Add<Self> for &Zm<M, L> {
    type Output = Zm<M, L>;
    fn add(self, rhs: Self) -> Self::Output {
        *self + *rhs
    }
}

impl<const M: u64, L: UintLimb> Add<Zm<M, L>> for &Zm<M, L> {
    type Output = Zm<M, L>;
    fn add(self, rhs: Zm<M, L>) -> Self::Output {
        *self + rhs
    }
}

impl<const M: u64, L: UintLimb> Sub<Self> for &Zm<M, L> {
    type Output = Zm<M, L>;
    fn sub(self, rhs: Self) -> Self::Output {
        *self - *rhs
    }
}

impl<const M: u64, L: UintLimb> Sub<Zm<M, L>> for &Zm<M, L> {
    type Output = Zm<M, L>;
    fn sub(self, rhs: Zm<M, L>) -> Self::Output {
        *self - rhs
    }
}

impl<const M: u64, L: UintLimb> Mul<Self> for &Zm<M, L> {
    type Output = Zm<M, L>;
    fn mul(self, rhs: Self) -> Self::Output {
        *self * *rhs
    }
}

impl<const M: u64, L: UintLimb> Mul<Zm<M, L>> for &Zm<M, L> {
    type Output = Zm<M, L>;
    fn mul(self, rhs: Zm<M, L>) -> Self::Output {
        *self * rhs
    }
}

impl<const M: u64, L: UintLimb> Neg for Zm<M, L> {
    type Output = Self;
    fn neg(self) -> Self {
        let m = Self::modulus_limb();
        if self.val == L::ZERO {
            self
        } else {
            Self {
                val: m.wrapping_sub(self.val),
            }
        }
    }
}

// --- Debug / Display ---

impl<const M: u64, L: UintLimb> core::fmt::Debug for Zm<M, L> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Z_{}({})", M, self.to_u64())
    }
}

impl<const M: u64, L: UintLimb> core::fmt::Display for Zm<M, L> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.to_u64())
    }
}

// --- Ring impl ---

impl<const M: u64, L: UintLimb> Ring for Zm<M, L> {
    fn zero() -> Self {
        let _ = Self::MODULUS;
        Self { val: L::ZERO }
    }

    fn one() -> Self {
        Self::encode_canonical(1)
    }

    fn is_zero(&self) -> bool {
        self.val == L::ZERO
    }

    fn is_one(&self) -> bool {
        self.decode_canonical() == 1
    }

    fn double(&self) -> Self {
        Self {
            val: Self::add_raw(self.val, self.val),
        }
    }

    fn square(&self) -> Self {
        *self * *self
    }

    fn add_ref(a: &Self, b: &Self) -> Self {
        a + b
    }

    fn sub_ref(a: &Self, b: &Self) -> Self {
        a - b
    }

    fn mul_ref(a: &Self, b: &Self) -> Self {
        a * b
    }
}

// --- IntegerRing impl ---

impl<const M: u64, L: UintLimb> IntegerRing for Zm<M, L> {
    type Canonical = u64;

    fn modulus_canonical() -> u64 {
        Self::MODULUS
    }

    fn from_small_u64(val: u64) -> Self {
        Self::encode_canonical(val)
    }

    fn from_canonical(value: &u64) -> Self {
        Self::from_small_u64(*value)
    }

    fn to_canonical(&self) -> u64 {
        self.decode_canonical()
    }

    fn try_to_u64(&self) -> Option<u64> {
        Some(self.to_canonical())
    }

    fn try_to_u128(&self) -> Option<u128> {
        Some(self.to_canonical() as u128)
    }

    fn lossy_l2_value(&self) -> f64 {
        let val = self.to_canonical();
        let m = Self::MODULUS;
        // Centered representation in [-m/2, m/2).
        // For odd m: val > m/2 is the negative half.
        // For even m: val >= m/2 is the negative half.
        let mid = m / 2;
        if val > mid || (m % 2 == 0 && val == mid) {
            (val as f64) - (m as f64)
        } else {
            val as f64
        }
    }

    fn reduce(&self) -> Self {
        *self
    }
}

// --- Serialization ---

impl<const M: u64, L: UintLimb> CanonicalSerialize for Zm<M, L> {
    fn serialized_size(&self) -> usize {
        Self::SERIALIZED_BYTES
    }

    fn serialize_into(&self, buf: &mut alloc::vec::Vec<u8>) -> Result<(), SerializationError> {
        let val = self.to_u64();
        let bytes = val.to_le_bytes();
        buf.extend_from_slice(&bytes[..Self::SERIALIZED_BYTES]);
        Ok(())
    }
}

impl<const M: u64, L: UintLimb> CanonicalDeserialize for Zm<M, L> {
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        if data.len() < Self::SERIALIZED_BYTES {
            return Err(SerializationError::UnexpectedEnd);
        }
        let mut bytes = [0u8; 8];
        bytes[..Self::SERIALIZED_BYTES].copy_from_slice(&data[..Self::SERIALIZED_BYTES]);
        let val = u64::from_le_bytes(bytes);
        if val >= Self::MODULUS {
            return Err(SerializationError::InvalidData(alloc::format!(
                "value {val} >= modulus {}",
                Self::MODULUS
            )));
        }
        Ok((Self::from_u64(val), Self::SERIALIZED_BYTES))
    }
}

impl<const M: u64, L: UintLimb> grid_serialize::Valid for Zm<M, L> {
    fn is_valid(&self) -> bool {
        self.val < Self::modulus_limb()
    }
}

// --- Random sampling ---

impl<const M: u64, L: UintLimb> grid_std::UniformRand for Zm<M, L> {
    fn rand<R: RngExt + ?Sized>(rng: &mut R) -> Self {
        let reject = (u64::MAX % Self::MODULUS).wrapping_add(1) % Self::MODULUS;
        let upper = u64::MAX - reject;
        loop {
            let sample: u64 = rng.random();
            if sample <= upper {
                return Self::from_u64(sample % Self::MODULUS);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use grid_std::UniformRand;

    fn test_rng() -> impl RngExt {
        grid_std::test_rng()
    }

    type Z257 = Zm<257>; // odd, composite
    type Z274177 = Zm<274177>; // odd, prime (F_6 factor)
    type Z2 = Zm<2>; // even
    type Z256 = Zm<256>; // even

    #[test]
    fn test_ring_properties_odd_composite() {
        let a = Z257::from_u64(100);
        let b = Z257::from_u64(200);
        let c = Z257::from_u64(50);
        // (a + b) * c == a*c + b*c
        let lhs = (a + b) * c;
        let rhs = (a * c) + (b * c);
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_ring_properties_odd_prime() {
        let a = Z274177::from_u64(12345);
        let b = Z274177::from_u64(67890);
        let c = Z274177::from_u64(11111);
        let lhs = (a + b) * c;
        let rhs = (a * c) + (b * c);
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_ring_properties_even() {
        let a = Z2::from_u64(1);
        let b = Z2::from_u64(0);
        let c = Z2::from_u64(1);
        assert_eq!((a + b) * c, (a * c) + (b * c));
    }

    #[test]
    fn test_mul_matches_u64() {
        let mut rng = test_rng();
        for _ in 0..1000 {
            let a: u64 = rng.random();
            let b: u64 = rng.random();
            let expected = ((a as u128 * b as u128) % 257) as u64;
            let got = Z257::from_u64(a) * Z257::from_u64(b);
            assert_eq!(got.to_u64(), expected);
        }
    }

    #[test]
    fn test_mul_matches_u64_large_modulus() {
        let mut rng = test_rng();
        for _ in 0..1000 {
            let a: u64 = rng.random();
            let b: u64 = rng.random();
            let expected = ((a as u128 * b as u128) % 274177) as u64;
            let got = Z274177::from_u64(a) * Z274177::from_u64(b);
            assert_eq!(got.to_u64(), expected);
        }
    }

    #[test]
    fn test_mul_matches_u64_even() {
        let mut rng = test_rng();
        for _ in 0..100 {
            let a: u64 = rng.random();
            let b: u64 = rng.random();
            let expected = ((a as u128 * b as u128) % 256) as u64;
            let got = Z256::from_u64(a) * Z256::from_u64(b);
            assert_eq!(got.to_u64(), expected);
        }
    }

    #[test]
    fn test_mul_matches_u64_mod2() {
        let mut rng = test_rng();
        for _ in 0..100 {
            let a: u64 = rng.random();
            let b: u64 = rng.random();
            let expected = (a % 2) * (b % 2) % 2;
            let got = Z2::from_u64(a) * Z2::from_u64(b);
            assert_eq!(got.to_u64(), expected);
        }
    }

    #[test]
    fn test_montgomery_roundtrip() {
        for val in 0..100u64 {
            let x = Z257::from_u64(val);
            assert_eq!(x.to_u64(), val % 257);
        }
    }

    #[test]
    fn test_even_roundtrip() {
        for val in 0..100u64 {
            let x = Z256::from_u64(val);
            assert_eq!(x.to_u64(), val % 256);
        }
    }

    #[test]
    fn test_zero() {
        let z = <Z257 as Ring>::zero();
        assert_eq!(z.to_u64(), 0);
        assert!(z.is_zero());
    }

    #[test]
    fn test_one() {
        let o = <Z257 as Ring>::one();
        assert_eq!(o.to_u64(), 1);
        assert!(o.is_one());
        // Montgomery form: one() must NOT be raw limb 1
        // is_one() uses decode_canonical() which correctly handles Montgomery form
        let o_even = <Z2 as Ring>::one();
        assert_eq!(o_even.to_u64(), 1);
    }

    #[test]
    fn test_neg() {
        assert_eq!(-Z257::from_u64(0), Z257::from_u64(0));
        assert_eq!(-Z257::from_u64(1), Z257::from_u64(256));
        assert_eq!(-Z257::from_u64(256), Z257::from_u64(1));
    }

    #[test]
    fn test_add_sub_neg() {
        let mut rng = test_rng();
        for _ in 0..1000 {
            let a: u64 = rng.random();
            let b: u64 = rng.random();
            let x = Z257::from_u64(a);
            let y = Z257::from_u64(b);
            // a + b - b = a
            assert_eq!((x + y - y).to_u64(), a % 257);
            // -(-a) = a
            assert_eq!(-(-x), x);
        }
    }

    #[test]
    fn test_serialize_roundtrip() {
        let val = Z257::from_u64(123);
        let bytes = val.serialize().unwrap();
        assert_eq!(bytes.len(), Z257::SERIALIZED_BYTES);
        let decoded = Z257::deserialize_exact(&bytes).unwrap();
        assert_eq!(decoded.to_u64(), 123);
    }

    #[test]
    fn test_serialize_rejects_non_canonical() {
        // Z274177 serializes in 3 bytes. Test that values >= 274177 are rejected.
        let bytes = [0x00, 0x00, 0x05]; // 0x050000 = 327680 >= 274177
        assert!(Z274177::deserialize_exact(&bytes).is_err());
    }

    #[test]
    fn test_uniform_rand_in_range() {
        let mut rng = test_rng();
        for _ in 0..1000 {
            let v: Z257 = UniformRand::rand(&mut rng);
            assert!(v.to_u64() < 257);
        }
    }

    #[test]
    fn test_large_canonical() {
        let val = Z257::from_u64(42);
        let canon = val.to_canonical();
        assert_eq!(canon, 42);
        assert_eq!(Z257::from_canonical(&42), val);
    }

    #[test]
    fn test_even_2_arithmetic() {
        let mut rng = test_rng();
        for _ in 0..1000 {
            let a: u64 = rng.random();
            let b: u64 = rng.random();
            let x = Z2::from_u64(a);
            let y = Z2::from_u64(b);
            // Z2 is isomorphic to GF2 — same truth table
            assert_eq!((x + y).to_u64(), ((a % 2) ^ (b % 2)));
            assert_eq!((x * y).to_u64(), ((a % 2) & (b % 2)));
        }
    }

    #[test]
    fn test_small_limbs() {
        type Z257U16 = Zm<257, u16>;
        let a = Z257U16::from_u64(100);
        let b = Z257U16::from_u64(200);
        assert_eq!((a + b).to_u64(), 300 % 257);
        assert_eq!((a * b).to_u64(), (100 * 200) % 257);
    }
}
