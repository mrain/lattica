//! Multi-limb unsigned integer type for modular arithmetic.
//!
//! Provides a fixed-size big integer represented as an array of `u64` limbs
//! in little-endian order (least significant limb first).

use core::fmt;

use grid_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError};

/// A fixed-size unsigned integer with `N` 64-bit limbs (little-endian).
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct BigUint<const N: usize> {
    /// Limbs in little-endian order: `limbs[0]` is the least significant.
    pub limbs: [u64; N],
}

impl<const N: usize> BigUint<N> {
    /// Zero value.
    pub const ZERO: Self = Self { limbs: [0u64; N] };

    /// Maximum representable value.
    pub const MAX: Self = Self {
        limbs: [u64::MAX; N],
    };

    /// One value.
    pub const fn one() -> Self {
        let mut limbs = [0u64; N];
        limbs[0] = 1;
        Self { limbs }
    }

    /// Create from a single `u64`.
    pub const fn from_u64(val: u64) -> Self {
        let mut limbs = [0u64; N];
        limbs[0] = val;
        Self { limbs }
    }

    /// Create from a single `u128`.
    pub const fn from_u128(val: u128) -> Self {
        assert!(
            N > 1 || (val >> 64) == 0,
            "BigUint::from_u128 requires at least two limbs for values above 2^64 - 1"
        );
        let mut limbs = [0u64; N];
        if N > 0 {
            limbs[0] = val as u64;
        }
        if N > 1 {
            limbs[1] = (val >> 64) as u64;
        }
        Self { limbs }
    }

    /// Returns `true` if the value is zero.
    pub fn is_zero(&self) -> bool {
        self.limbs.iter().all(|&l| l == 0)
    }

    /// L2-canonical value as `f64` via Horner's method (MSB-first).
    ///
    /// Precision degrades for values above 2^53. Sufficient for norm-bound
    /// comparisons where relative accuracy suffices.
    pub fn lossy_l2_value(&self) -> f64 {
        const SHIFT: f64 = 18446744073709551616.0; // 2^64
        let mut acc = 0.0;
        for &limb in self.limbs.iter().rev() {
            acc = acc * SHIFT + limb as f64;
        }
        acc
    }

    /// Add two big integers, returning (result, carry).
    pub fn add_with_carry(&self, other: &Self) -> (Self, bool) {
        let mut result = Self::ZERO;
        let mut carry = 0u64;
        for i in 0..N {
            let (sum1, c1) = self.limbs[i].overflowing_add(other.limbs[i]);
            let (sum2, c2) = sum1.overflowing_add(carry);
            result.limbs[i] = sum2;
            carry = (c1 as u64) + (c2 as u64);
        }
        (result, carry > 0)
    }

    /// Add a single `u64`, returning (result, carry).
    pub fn add_small(&self, small: u64) -> (Self, bool) {
        let mut result = *self;
        let (sum, carry0) = result.limbs[0].overflowing_add(small);
        result.limbs[0] = sum;

        let mut carry = carry0 as u64;
        for limb in result.limbs.iter_mut().skip(1) {
            if carry == 0 {
                break;
            }
            let (sum, next_carry) = limb.overflowing_add(carry);
            *limb = sum;
            carry = next_carry as u64;
        }

        (result, carry != 0)
    }

    /// Subtract two big integers, returning (result, borrow).
    pub fn sub_with_borrow(&self, other: &Self) -> (Self, bool) {
        let mut result = Self::ZERO;
        let mut borrow = 0u64;
        for i in 0..N {
            let (diff1, b1) = self.limbs[i].overflowing_sub(other.limbs[i]);
            let (diff2, b2) = diff1.overflowing_sub(borrow);
            result.limbs[i] = diff2;
            borrow = (b1 as u64) + (b2 as u64);
        }
        (result, borrow > 0)
    }

    /// Subtract a single `u64`, returning (result, borrow).
    pub fn sub_small(&self, small: u64) -> (Self, bool) {
        let mut result = *self;
        let (diff, borrow0) = result.limbs[0].overflowing_sub(small);
        result.limbs[0] = diff;

        let mut borrow = borrow0 as u64;
        for limb in result.limbs.iter_mut().skip(1) {
            if borrow == 0 {
                break;
            }
            let (diff, next_borrow) = limb.overflowing_sub(borrow);
            *limb = diff;
            borrow = next_borrow as u64;
        }

        (result, borrow != 0)
    }

    /// Multiply by a single `u64` limb, returning (result, carry_limb).
    pub fn mul_by_limb(&self, limb: u64) -> (Self, u64) {
        let mut result = Self::ZERO;
        let mut carry = 0u128;
        for i in 0..N {
            let prod = (self.limbs[i] as u128) * (limb as u128) + carry;
            result.limbs[i] = prod as u64;
            carry = prod >> 64;
        }
        (result, carry as u64)
    }

    /// Divide by a single `u64` limb, returning (quotient, remainder).
    pub fn div_rem_small(&self, divisor: u64) -> (Self, u64) {
        assert!(divisor != 0, "division by zero");

        let mut quotient = Self::ZERO;
        let mut remainder = 0u128;
        for i in (0..N).rev() {
            let wide = (remainder << 64) | self.limbs[i] as u128;
            quotient.limbs[i] = (wide / divisor as u128) as u64;
            remainder = wide % divisor as u128;
        }
        (quotient, remainder as u64)
    }

    /// Full multiplication of two N-limb integers.
    /// Returns the lower N limbs and upper N limbs separately.
    pub fn widening_mul(&self, other: &Self) -> (Self, Self) {
        let mut lo = Self::ZERO;
        let mut hi = Self::ZERO;

        for i in 0..N {
            let mut carry = 0u128;
            for j in 0..N {
                let pos = i + j;
                let prod = (self.limbs[i] as u128) * (other.limbs[j] as u128) + carry;
                if pos < N {
                    let sum = (lo.limbs[pos] as u128) + (prod & 0xFFFF_FFFF_FFFF_FFFF);
                    lo.limbs[pos] = sum as u64;
                    carry = (prod >> 64) + (sum >> 64); // carry from product high bits + sum overflow
                } else if pos < 2 * N {
                    let hi_pos = pos - N;
                    let sum = (hi.limbs[hi_pos] as u128) + (prod & 0xFFFF_FFFF_FFFF_FFFF);
                    hi.limbs[hi_pos] = sum as u64;
                    carry = (prod >> 64) + (sum >> 64); // carry from product high bits + sum overflow
                } else {
                    break;
                }
            }
            // Propagate final carry
            let mut pos = i + N;
            while carry > 0 && pos < 2 * N {
                let hi_pos = pos - N;
                let sum = (hi.limbs[hi_pos] as u128) + carry;
                hi.limbs[hi_pos] = sum as u64;
                carry = sum >> 64;
                pos += 1;
            }
        }

        (lo, hi)
    }

    /// Compare two big integers.
    pub fn compare(&self, other: &Self) -> core::cmp::Ordering {
        for i in (0..N).rev() {
            match self.limbs[i].cmp(&other.limbs[i]) {
                core::cmp::Ordering::Equal => continue,
                ord => return ord,
            }
        }
        core::cmp::Ordering::Equal
    }

    /// Returns the number of significant bits.
    pub fn bits(&self) -> u32 {
        for i in (0..N).rev() {
            if self.limbs[i] != 0 {
                return (i as u32) * 64 + (64 - self.limbs[i].leading_zeros());
            }
        }
        0
    }

    /// Shift left by a bit count smaller than 64, returning the shifted value and carry limb.
    pub fn shl_bits(&self, bits: u32) -> (Self, u64) {
        assert!(bits < 64, "bit shift must be less than 64");
        if bits == 0 {
            return (*self, 0);
        }

        let mut result = Self::ZERO;
        let mut carry = 0u64;
        for i in 0..N {
            let limb = self.limbs[i];
            result.limbs[i] = (limb << bits) | carry;
            carry = limb >> (64 - bits);
        }
        (result, carry)
    }

    /// Shift right by a bit count smaller than 64.
    pub fn shr_bits(&self, bits: u32) -> Self {
        assert!(bits < 64, "bit shift must be less than 64");
        if bits == 0 {
            return *self;
        }

        let mut result = Self::ZERO;
        let mut carry = 0u64;
        for i in (0..N).rev() {
            let limb = self.limbs[i];
            result.limbs[i] = (limb >> bits) | carry;
            carry = limb << (64 - bits);
        }
        result
    }

    /// Subtract `other` if `self >= other`, otherwise return `self` unchanged.
    pub fn sub_if_ge(&self, other: &Self) -> Self {
        let (diff, borrow) = self.sub_with_borrow(other);
        if borrow { *self } else { diff }
    }

    /// Return the value as `u64` if it fits exactly.
    pub fn try_to_u64(&self) -> Option<u64> {
        if self.limbs.iter().skip(1).any(|&limb| limb != 0) {
            None
        } else {
            Some(self.limbs[0])
        }
    }

    /// Return the value as `u128` if it fits exactly.
    pub fn try_to_u128(&self) -> Option<u128> {
        if self.limbs.iter().skip(2).any(|&limb| limb != 0) {
            return None;
        }

        let lo = self.limbs.first().copied().unwrap_or(0) as u128;
        let hi = self.limbs.get(1).copied().unwrap_or(0) as u128;
        Some(lo | (hi << 64))
    }

    /// Zero-extend into a larger fixed-width integer.
    pub fn zero_extend<const M: usize>(&self) -> BigUint<M> {
        let mut limbs = [0u64; M];
        let copy_len = core::cmp::min(N, M);
        limbs[..copy_len].copy_from_slice(&self.limbs[..copy_len]);
        BigUint { limbs }
    }
}

impl<const N: usize> PartialOrd for BigUint<N> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(Ord::cmp(self, other))
    }
}

impl<const N: usize> Ord for BigUint<N> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.compare(other)
    }
}

impl<const N: usize> fmt::Debug for BigUint<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BigUint(")?;
        for i in (0..N).rev() {
            if i < N - 1 {
                write!(f, "_{:016x}", self.limbs[i])?;
            } else {
                write!(f, "{:x}", self.limbs[i])?;
            }
        }
        write!(f, ")")
    }
}

impl<const N: usize> fmt::Display for BigUint<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl<const N: usize> CanonicalSerialize for BigUint<N> {
    fn serialized_size(&self) -> usize {
        N * 8
    }

    fn serialize_into(&self, buf: &mut alloc::vec::Vec<u8>) -> Result<(), SerializationError> {
        for &limb in &self.limbs {
            buf.extend_from_slice(&limb.to_le_bytes());
        }
        Ok(())
    }
}

impl<const N: usize> CanonicalDeserialize for BigUint<N> {
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        let size = N * 8;
        if data.len() < size {
            return Err(SerializationError::UnexpectedEnd);
        }
        let mut limbs = [0u64; N];
        for (i, limb) in limbs.iter_mut().enumerate() {
            let start = i * 8;
            *limb = u64::from_le_bytes(data[start..start + 8].try_into().unwrap());
        }
        Ok((Self { limbs }, size))
    }
}

impl<const N: usize> grid_serialize::Valid for BigUint<N> {
    fn is_valid(&self) -> bool {
        true // Any bit pattern is a valid unsigned integer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type U128 = BigUint<2>; // 128-bit integer

    #[test]
    fn test_add() {
        let a = U128::from_u64(u64::MAX);
        let b = U128::from_u64(1);
        let (sum, carry) = a.add_with_carry(&b);
        assert!(!carry);
        assert_eq!(sum.limbs[0], 0);
        assert_eq!(sum.limbs[1], 1);
    }

    #[test]
    fn test_sub() {
        let a = U128 {
            limbs: [0, 1], // 2^64
        };
        let b = U128::from_u64(1);
        let (diff, borrow) = a.sub_with_borrow(&b);
        assert!(!borrow);
        assert_eq!(diff.limbs[0], u64::MAX);
        assert_eq!(diff.limbs[1], 0);
    }

    #[test]
    fn test_mul_by_limb() {
        let a = U128::from_u64(u64::MAX);
        let (prod, carry) = a.mul_by_limb(2);
        assert_eq!(prod.limbs[0], u64::MAX - 1);
        assert_eq!(prod.limbs[1], 1);
        assert_eq!(carry, 0);
    }

    #[test]
    fn test_widening_mul() {
        let a = U128::from_u64(1_000_000);
        let b = U128::from_u64(1_000_000);
        let (lo, hi) = a.widening_mul(&b);
        assert_eq!(lo.limbs[0], 1_000_000_000_000);
        assert!(hi.is_zero());
    }

    #[test]
    fn test_cmp() {
        let a = U128::from_u64(100);
        let b = U128::from_u64(200);
        assert!(a < b);
        assert!(b > a);
        assert_eq!(a.cmp(&a), core::cmp::Ordering::Equal);
    }

    #[test]
    fn test_bits() {
        assert_eq!(U128::ZERO.bits(), 0);
        assert_eq!(U128::from_u64(1).bits(), 1);
        assert_eq!(U128::from_u64(255).bits(), 8);
        assert_eq!(U128::from_u64(256).bits(), 9);
    }

    #[test]
    fn test_from_u128_and_try_to_u128() {
        let value = (1u128 << 96) | 0xDEAD_BEEF_CAFE_BABEu128;
        let big = BigUint::<4>::from_u128(value);
        assert_eq!(big.try_to_u128(), Some(value));
    }

    #[test]
    #[should_panic(expected = "BigUint::from_u128 requires at least two limbs")]
    fn test_from_u128_rejects_truncation_for_single_limb() {
        let _ = BigUint::<1>::from_u128(1u128 << 64);
    }

    #[test]
    fn test_try_to_u64_rejects_high_limb() {
        let value = U128 { limbs: [5, 1] };
        assert_eq!(value.try_to_u64(), None);
        assert_eq!(value.try_to_u128(), Some((1u128 << 64) | 5));
    }

    #[test]
    fn test_shift_helpers() {
        let value = U128 {
            limbs: [0x0123_4567_89AB_CDEF, 0x0FED_CBA9_8765_4321],
        };
        let (shifted, carry) = value.shl_bits(4);
        assert_eq!(shifted.limbs[0], 0x1234_5678_9ABC_DEF0);
        assert_eq!(shifted.limbs[1], 0xFEDC_BA98_7654_3210);
        assert_eq!(carry, 0);

        let shrunk = shifted.shr_bits(4);
        assert_eq!(shrunk, value);
    }

    #[test]
    fn test_sub_if_ge() {
        let a = U128 { limbs: [10, 1] };
        let b = U128 { limbs: [3, 1] };
        let c = U128 { limbs: [0, 2] };

        assert_eq!(a.sub_if_ge(&b), U128 { limbs: [7, 0] });
        assert_eq!(a.sub_if_ge(&c), a);
    }

    #[test]
    fn test_add_small() {
        let value = U128 {
            limbs: [u64::MAX, 7],
        };
        let (sum, carry) = value.add_small(5);
        assert!(!carry);
        assert_eq!(sum.limbs, [4, 8]);
    }

    #[test]
    fn test_sub_small() {
        let value = U128 { limbs: [0, 7] };
        let (diff, borrow) = value.sub_small(5);
        assert!(!borrow);
        assert_eq!(diff.limbs, [u64::MAX - 4, 6]);
    }

    #[test]
    fn test_div_rem_small() {
        let value = BigUint::<3> { limbs: [0, 5, 1] };
        let (quotient, remainder) = value.div_rem_small(7);
        assert_eq!(remainder, 0);

        let (reconstructed, carry) = quotient.mul_by_limb(7);
        assert_eq!(carry, 0);
        let (reconstructed, overflow) = reconstructed.add_small(remainder);
        assert!(!overflow);
        assert_eq!(reconstructed, value);
    }

    #[test]
    fn test_zero_extend() {
        let value = U128 {
            limbs: [0xDEAD_BEEF, 0xCAFE_BABE],
        };
        let widened = value.zero_extend::<4>();
        assert_eq!(widened.limbs, [0xDEAD_BEEF, 0xCAFE_BABE, 0, 0]);
    }

    #[test]
    fn test_serialize_round_trip() {
        use grid_serialize::{CanonicalDeserialize, CanonicalSerialize};
        let a = U128 {
            limbs: [0xDEAD_BEEF, 0xCAFE_BABE],
        };
        let bytes = a.serialize().unwrap();
        assert_eq!(bytes.len(), 16);
        let (b, consumed) = U128::deserialize(&bytes).unwrap();
        assert_eq!(consumed, 16);
        assert_eq!(a, b);
    }
}
