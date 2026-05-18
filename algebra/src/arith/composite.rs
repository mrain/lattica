//! `CompositeModulus` — `Z_q` where `q` is a product of pairwise-coprime factors.
//!
//! Internally stores values in RNS (Residue Number System) representation
//! for efficient arithmetic. Each operation is performed component-wise
//! across the pairwise-coprime limbs.

use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use super::bigint::BigUint;
use super::rns::RnsBasis;

/// A modular integer in RNS representation for a composite modulus.
///
/// The modulus `q = p_0 * p_1 * ... * p_{k-1}` is stored as an [`RnsBasis`],
/// and the value is stored as residues `[x mod p_0, x mod p_1, ...]`.
#[derive(Clone, Debug)]
pub struct CompositeRing {
    /// The RNS basis (shared across all elements with the same modulus).
    basis: Arc<RnsBasis>,
    /// Residues: `residues[i] = value mod basis.primes[i]`.
    residues: Vec<u64>,
}

impl CompositeRing {
    /// Create a new element from a `u64` value and an RNS basis.
    pub fn from_u64_with_basis(val: u64, basis: Arc<RnsBasis>) -> Self {
        let residues = basis.decompose(val);
        Self { basis, residues }
    }

    /// Create a new element from canonical residues and an RNS basis.
    pub fn from_residues_with_basis(
        residues: Vec<u64>,
        basis: Arc<RnsBasis>,
    ) -> Result<Self, grid_serialize::SerializationError> {
        if residues.len() != basis.num_limbs() {
            return Err(grid_serialize::SerializationError::InvalidData(
                alloc::format!(
                    "expected {} limbs, got {}",
                    basis.num_limbs(),
                    residues.len()
                ),
            ));
        }

        for (residue, prime) in residues.iter().zip(basis.primes.iter()) {
            if residue >= prime {
                return Err(grid_serialize::SerializationError::InvalidData(
                    alloc::format!("residue {residue} >= prime {prime}"),
                ));
            }
        }

        Ok(Self { basis, residues })
    }

    /// Create a new element from a fixed-limb canonical integer and an RNS basis.
    pub fn from_canonical_with_basis<const N: usize>(
        value: &BigUint<N>,
        basis: Arc<RnsBasis>,
    ) -> Self {
        let residues = basis.decompose_biguint(value);
        Self { basis, residues }
    }

    /// Create the zero element for the given basis.
    pub fn zero_with_basis(basis: Arc<RnsBasis>) -> Self {
        let residues = vec![0u64; basis.num_limbs()];
        Self { basis, residues }
    }

    /// Create the one element for the given basis.
    pub fn one_with_basis(basis: Arc<RnsBasis>) -> Self {
        let residues = basis.decompose(1);
        Self { basis, residues }
    }

    /// Get the RNS basis.
    pub fn basis(&self) -> &Arc<RnsBasis> {
        &self.basis
    }

    /// Get the residues.
    pub fn residues(&self) -> &[u64] {
        &self.residues
    }

    /// Reconstruct the value using CRT.
    ///
    /// Returns the value as a `u128` (may be large for many limbs).
    pub fn to_u128(&self) -> u128 {
        self.try_to_u128()
            .expect("CompositeRing value does not fit in u128")
    }

    /// Reconstruct the value into a fixed-limb canonical integer.
    pub fn to_canonical<const N: usize>(&self) -> BigUint<N> {
        self.basis.reconstruct_biguint(&self.residues)
    }

    /// Return the value as `u128` if it fits exactly.
    pub fn try_to_u128(&self) -> Option<u128> {
        self.to_canonical::<2>().try_to_u128()
    }

    #[inline]
    fn assert_same_basis(&self, other: &Self) {
        assert_eq!(
            self.basis.primes, other.basis.primes,
            "CompositeRing basis mismatch"
        );
    }
}

impl PartialEq for CompositeRing {
    fn eq(&self, other: &Self) -> bool {
        self.basis.primes == other.basis.primes && self.residues == other.residues
    }
}

impl Eq for CompositeRing {}

impl core::fmt::Display for CompositeRing {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.to_u128())
    }
}

// --- Operator impls ---

impl Add for CompositeRing {
    type Output = Self;
    #[inline]
    fn add(mut self, rhs: Self) -> Self {
        self.assert_same_basis(&rhs);
        self.basis
            .add_assign_into(&mut self.residues, &rhs.residues);
        self
    }
}

impl Add<&Self> for CompositeRing {
    type Output = Self;
    #[inline]
    fn add(mut self, rhs: &Self) -> Self {
        self.assert_same_basis(rhs);
        self.basis
            .add_assign_into(&mut self.residues, &rhs.residues);
        self
    }
}

impl AddAssign for CompositeRing {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.assert_same_basis(&rhs);
        self.basis
            .add_assign_into(&mut self.residues, &rhs.residues);
    }
}

impl AddAssign<&Self> for CompositeRing {
    #[inline]
    fn add_assign(&mut self, rhs: &Self) {
        self.assert_same_basis(rhs);
        self.basis
            .add_assign_into(&mut self.residues, &rhs.residues);
    }
}

impl Sub for CompositeRing {
    type Output = Self;
    #[inline]
    fn sub(mut self, rhs: Self) -> Self {
        self.assert_same_basis(&rhs);
        self.basis
            .sub_assign_into(&mut self.residues, &rhs.residues);
        self
    }
}

impl Sub<&Self> for CompositeRing {
    type Output = Self;
    #[inline]
    fn sub(mut self, rhs: &Self) -> Self {
        self.assert_same_basis(rhs);
        self.basis
            .sub_assign_into(&mut self.residues, &rhs.residues);
        self
    }
}

impl SubAssign for CompositeRing {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.assert_same_basis(&rhs);
        self.basis
            .sub_assign_into(&mut self.residues, &rhs.residues);
    }
}

impl SubAssign<&Self> for CompositeRing {
    #[inline]
    fn sub_assign(&mut self, rhs: &Self) {
        self.assert_same_basis(rhs);
        self.basis
            .sub_assign_into(&mut self.residues, &rhs.residues);
    }
}

impl Mul for CompositeRing {
    type Output = Self;
    #[inline]
    fn mul(mut self, rhs: Self) -> Self {
        self.assert_same_basis(&rhs);
        self.basis
            .mul_assign_into(&mut self.residues, &rhs.residues);
        self
    }
}

impl Mul<&Self> for CompositeRing {
    type Output = Self;
    #[inline]
    fn mul(mut self, rhs: &Self) -> Self {
        self.assert_same_basis(rhs);
        self.basis
            .mul_assign_into(&mut self.residues, &rhs.residues);
        self
    }
}

impl MulAssign for CompositeRing {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.assert_same_basis(&rhs);
        self.basis
            .mul_assign_into(&mut self.residues, &rhs.residues);
    }
}

impl MulAssign<&Self> for CompositeRing {
    #[inline]
    fn mul_assign(&mut self, rhs: &Self) {
        self.assert_same_basis(rhs);
        self.basis
            .mul_assign_into(&mut self.residues, &rhs.residues);
    }
}

impl Add<Self> for &CompositeRing {
    type Output = CompositeRing;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        self.assert_same_basis(rhs);
        let mut residues = self.residues.clone();
        self.basis.add_assign_into(&mut residues, &rhs.residues);
        CompositeRing {
            residues,
            basis: self.basis.clone(),
        }
    }
}

impl Add<CompositeRing> for &CompositeRing {
    type Output = CompositeRing;
    #[inline]
    fn add(self, rhs: CompositeRing) -> Self::Output {
        self.assert_same_basis(&rhs);
        let mut residues = self.residues.clone();
        self.basis.add_assign_into(&mut residues, &rhs.residues);
        CompositeRing {
            residues,
            basis: self.basis.clone(),
        }
    }
}

impl Sub<Self> for &CompositeRing {
    type Output = CompositeRing;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        self.assert_same_basis(rhs);
        let mut residues = self.residues.clone();
        self.basis.sub_assign_into(&mut residues, &rhs.residues);
        CompositeRing {
            residues,
            basis: self.basis.clone(),
        }
    }
}

impl Sub<CompositeRing> for &CompositeRing {
    type Output = CompositeRing;
    #[inline]
    fn sub(self, rhs: CompositeRing) -> Self::Output {
        self.assert_same_basis(&rhs);
        let mut residues = self.residues.clone();
        self.basis.sub_assign_into(&mut residues, &rhs.residues);
        CompositeRing {
            residues,
            basis: self.basis.clone(),
        }
    }
}

impl Mul<Self> for &CompositeRing {
    type Output = CompositeRing;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.assert_same_basis(rhs);
        let mut residues = self.residues.clone();
        self.basis.mul_assign_into(&mut residues, &rhs.residues);
        CompositeRing {
            residues,
            basis: self.basis.clone(),
        }
    }
}

impl Mul<CompositeRing> for &CompositeRing {
    type Output = CompositeRing;
    #[inline]
    fn mul(self, rhs: CompositeRing) -> Self::Output {
        self.assert_same_basis(&rhs);
        let mut residues = self.residues.clone();
        self.basis.mul_assign_into(&mut residues, &rhs.residues);
        CompositeRing {
            residues,
            basis: self.basis.clone(),
        }
    }
}

impl Neg for CompositeRing {
    type Output = Self;
    #[inline]
    fn neg(mut self) -> Self {
        for (residue, prime) in self.residues.iter_mut().zip(self.basis.primes.iter()) {
            if *residue != 0 {
                *residue = *prime - *residue;
            }
        }
        self
    }
}

// --- Serialization ---

impl grid_serialize::CanonicalSerialize for CompositeRing {
    fn serialized_size(&self) -> usize {
        // 8 bytes per residue + 8 bytes for the limb count
        8 + self.residues.len() * 8
    }

    fn serialize_into(
        &self,
        buf: &mut alloc::vec::Vec<u8>,
    ) -> Result<(), grid_serialize::SerializationError> {
        buf.extend_from_slice(&(self.residues.len() as u64).to_le_bytes());
        for &r in &self.residues {
            buf.extend_from_slice(&r.to_le_bytes());
        }
        Ok(())
    }
}

// Note: CompositeRing deserialization requires an external basis, so we provide
// a method rather than implementing the trait (which has no basis parameter).
impl CompositeRing {
    /// Deserialize from bytes with a known basis.
    pub fn deserialize_with_basis(
        data: &[u8],
        basis: Arc<RnsBasis>,
    ) -> Result<(Self, usize), grid_serialize::SerializationError> {
        if data.len() < 8 {
            return Err(grid_serialize::SerializationError::UnexpectedEnd);
        }
        let num_limbs = usize::try_from(u64::from_le_bytes(data[..8].try_into().unwrap()))
            .map_err(|_| {
                grid_serialize::SerializationError::InvalidData("limb count too large".into())
            })?;
        if num_limbs != basis.num_limbs() {
            return Err(grid_serialize::SerializationError::InvalidData(
                alloc::format!("expected {} limbs, got {num_limbs}", basis.num_limbs()),
            ));
        }
        let total = 8usize
            .checked_add(num_limbs.checked_mul(8).ok_or_else(|| {
                grid_serialize::SerializationError::InvalidData("limb count too large".into())
            })?)
            .ok_or_else(|| {
                grid_serialize::SerializationError::InvalidData("limb count too large".into())
            })?;
        if data.len() < total {
            return Err(grid_serialize::SerializationError::UnexpectedEnd);
        }
        let mut residues = Vec::new();
        residues.try_reserve_exact(num_limbs).map_err(|_| {
            grid_serialize::SerializationError::InvalidData("limb count too large".into())
        })?;
        for i in 0..num_limbs {
            let start = 8 + i * 8;
            let r = u64::from_le_bytes(data[start..start + 8].try_into().unwrap());
            if r >= basis.primes[i] {
                return Err(grid_serialize::SerializationError::InvalidData(
                    alloc::format!("residue {r} >= prime {}", basis.primes[i]),
                ));
            }
            residues.push(r);
        }
        Ok((Self::from_residues_with_basis(residues, basis)?, total))
    }

    /// Deserialize from bytes with a known basis and require exact input consumption.
    pub fn deserialize_exact_with_basis(
        data: &[u8],
        basis: Arc<RnsBasis>,
    ) -> Result<Self, grid_serialize::SerializationError> {
        let (value, consumed) = Self::deserialize_with_basis(data, basis)?;
        if consumed == data.len() {
            Ok(value)
        } else {
            Err(grid_serialize::SerializationError::InvalidData(
                "trailing bytes after deserialized value".into(),
            ))
        }
    }
}

// --- Ring trait impl ---
// Note: CompositeRing doesn't implement Ring directly because Ring requires
// zero()/one() as associated functions (no &self), but CompositeRing needs
// a basis to construct elements. Instead, use from_u64_with_basis() etc.

#[cfg(test)]
mod tests {
    use super::*;

    fn test_basis() -> Arc<RnsBasis> {
        Arc::new(RnsBasis::new(vec![17, 19, 23])) // M = 7429
    }

    #[test]
    fn test_composite_from_u64() {
        let basis = test_basis();
        let a = CompositeRing::from_u64_with_basis(1234, basis);
        assert_eq!(a.to_u128(), 1234);
    }

    #[test]
    fn test_composite_add() {
        let basis = test_basis();
        let a = CompositeRing::from_u64_with_basis(1234, basis.clone());
        let b = CompositeRing::from_u64_with_basis(5678, basis);
        let sum = a + b;
        assert_eq!(sum.to_u128(), 6912); // (1234 + 5678) mod 7429
    }

    #[test]
    fn test_composite_mul() {
        let basis = test_basis();
        let a = CompositeRing::from_u64_with_basis(123, basis.clone());
        let b = CompositeRing::from_u64_with_basis(456, basis);
        let prod = a * b;
        assert_eq!(prod.to_u128(), (123u128 * 456) % 7429);
    }

    #[test]
    fn test_composite_sub() {
        let basis = test_basis();
        let a = CompositeRing::from_u64_with_basis(100, basis.clone());
        let b = CompositeRing::from_u64_with_basis(200, basis);
        let diff = a - b;
        // 100 - 200 mod 7429 = 7329
        assert_eq!(diff.to_u128(), 7329);
    }

    #[test]
    fn test_composite_neg() {
        let basis = test_basis();
        let a = CompositeRing::from_u64_with_basis(100, basis);
        let neg_a = -a;
        assert_eq!(neg_a.to_u128(), 7429 - 100);
    }

    #[test]
    fn test_composite_zero_one() {
        let basis = test_basis();
        let zero = CompositeRing::zero_with_basis(basis.clone());
        let one = CompositeRing::one_with_basis(basis.clone());
        let a = CompositeRing::from_u64_with_basis(42, basis);

        assert_eq!((a.clone() + &zero).to_u128(), 42);
        assert_eq!((a.clone() * &one).to_u128(), 42);
        assert_eq!((a.clone() * &zero).to_u128(), 0);
    }

    #[test]
    fn test_composite_round_trip_all() {
        let basis = Arc::new(RnsBasis::new(vec![7, 11, 13])); // M = 1001
        for x in 0..1001u64 {
            let elem = CompositeRing::from_u64_with_basis(x, basis.clone());
            assert_eq!(elem.to_u128(), x as u128, "round-trip failed for {x}");
        }
    }
    #[test]
    fn test_serialize_round_trip() {
        let basis = test_basis();
        for x in [0u64, 1, 42, 1234, 7428] {
            let a = CompositeRing::from_u64_with_basis(x, basis.clone());
            let bytes = grid_serialize::CanonicalSerialize::serialize(&a).unwrap();
            let b = CompositeRing::deserialize_exact_with_basis(&bytes, basis.clone()).unwrap();
            assert_eq!(a, b, "serialize round-trip failed for {x}");
        }
    }

    #[test]
    fn test_composite_equality_checks_basis() {
        let lhs = CompositeRing::from_u64_with_basis(1, Arc::new(RnsBasis::new(vec![7, 11])));
        let rhs = CompositeRing::from_u64_with_basis(1, Arc::new(RnsBasis::new(vec![13, 17])));
        assert_ne!(lhs, rhs);
    }

    #[test]
    #[should_panic(expected = "basis mismatch")]
    fn test_composite_add_basis_mismatch_panics() {
        let lhs = CompositeRing::from_u64_with_basis(1, Arc::new(RnsBasis::new(vec![7, 11])));
        let rhs = CompositeRing::from_u64_with_basis(1, Arc::new(RnsBasis::new(vec![13, 17])));
        let _ = lhs + rhs;
    }

    #[test]
    fn test_composite_one_round_trips_through_serialization() {
        let basis = Arc::new(RnsBasis::new(vec![7, 11]));
        let one = CompositeRing::one_with_basis(basis.clone());
        let bytes = grid_serialize::CanonicalSerialize::serialize(&one).unwrap();
        let decoded = CompositeRing::deserialize_exact_with_basis(&bytes, basis).unwrap();
        assert_eq!(decoded, one);
        assert_eq!(decoded.to_u128(), 1);
    }

    #[test]
    fn test_deserialize_exact_with_basis_rejects_trailing_bytes() {
        let basis = test_basis();
        let value = CompositeRing::from_u64_with_basis(42, basis.clone());
        let mut bytes = grid_serialize::CanonicalSerialize::serialize(&value).unwrap();
        bytes.push(0xAA);
        let err = CompositeRing::deserialize_exact_with_basis(&bytes, basis).unwrap_err();
        assert_eq!(
            err,
            grid_serialize::SerializationError::InvalidData(
                "trailing bytes after deserialized value".into()
            )
        );
    }

    #[test]
    fn test_deserialize_rejects_wrong_limb_count() {
        let basis = test_basis();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&2u64.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        let err = CompositeRing::deserialize_with_basis(&bytes, basis).unwrap_err();
        assert!(matches!(
            err,
            grid_serialize::SerializationError::InvalidData(_)
        ));
    }

    #[test]
    fn test_deserialize_rejects_out_of_range_residue() {
        let basis = Arc::new(RnsBasis::new(vec![7, 11]));
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&2u64.to_le_bytes());
        bytes.extend_from_slice(&7u64.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        let err = CompositeRing::deserialize_with_basis(&bytes, basis).unwrap_err();
        assert!(matches!(
            err,
            grid_serialize::SerializationError::InvalidData(_)
        ));
    }
}
