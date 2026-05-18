//! RNS (Residue Number System) layer.
//!
//! Provides operations for working with integers in RNS representation
//! across multiple pairwise-coprime moduli. Used by [`CompositeRing`](super::composite::CompositeRing).

use alloc::vec;
use alloc::vec::Vec;

use super::bigint::BigUint;

/// An RNS basis: a set of pairwise-coprime moduli for RNS representation.
///
/// Each element is represented as a vector of residues, one per basis prime.
/// This enables component-wise arithmetic without carries.
#[derive(Clone, Debug)]
pub struct RnsBasis {
    /// The component moduli forming the RNS basis.
    pub primes: Vec<u64>,
    /// Cached `2^64 mod p_i` values for Horner-style decomposition.
    radix_mods: Vec<u64>,
}

impl RnsBasis {
    /// Create a new RNS basis from a list of co-prime moduli.
    ///
    /// # Panics
    /// Panics if any pair of moduli shares a common factor.
    pub fn new(primes: Vec<u64>) -> Self {
        assert!(
            !primes.is_empty(),
            "RNS basis must contain at least one modulus"
        );
        for &prime in &primes {
            assert!(prime > 1, "RNS basis moduli must be greater than 1");
        }
        // Verify pairwise coprimality
        for i in 0..primes.len() {
            for j in (i + 1)..primes.len() {
                assert!(
                    gcd(primes[i], primes[j]) == 1,
                    "RNS basis moduli must be pairwise coprime: gcd({}, {}) != 1",
                    primes[i],
                    primes[j]
                );
            }
        }
        let radix_mods = primes
            .iter()
            .map(|&prime| ((1u128 << 64) % prime as u128) as u64)
            .collect();
        Self { primes, radix_mods }
    }

    /// Number of limbs (primes) in this basis.
    pub fn num_limbs(&self) -> usize {
        self.primes.len()
    }

    /// Decompose a value into RNS representation.
    ///
    /// Given `x`, returns `[x mod p_0, x mod p_1, ...]`.
    pub fn decompose(&self, x: u64) -> Vec<u64> {
        self.primes.iter().map(|&p| x % p).collect()
    }

    /// Decompose a fixed-limb canonical integer into RNS representation.
    pub fn decompose_biguint<const N: usize>(&self, x: &BigUint<N>) -> Vec<u64> {
        self.primes
            .iter()
            .zip(self.radix_mods.iter())
            .map(|(&prime, &radix_mod)| {
                let mut acc = 0u64;
                for &limb in x.limbs.iter().rev() {
                    acc = (((acc as u128) * (radix_mod as u128) + (limb as u128)) % (prime as u128))
                        as u64;
                }
                acc
            })
            .collect()
    }

    /// Reconstruct a value from RNS representation using CRT.
    ///
    /// Returns the unique value in `[0, M)` where `M = product of all primes`.
    /// Uses the Garner algorithm for numerical stability.
    pub fn reconstruct(&self, residues: &[u64]) -> u128 {
        self.reconstruct_biguint::<2>(residues)
            .try_to_u128()
            .expect("RNS reconstruction does not fit in u128")
    }

    /// Return the composite modulus product as a fixed-limb integer.
    pub fn modulus_biguint<const N: usize>(&self) -> BigUint<N> {
        let mut product = BigUint::<N>::one();
        for &prime in &self.primes {
            let (next, carry) = product.mul_by_limb(prime);
            assert!(
                carry == 0,
                "RNS modulus product does not fit in the requested canonical width"
            );
            product = next;
        }
        product
    }

    /// Reconstruct a value from RNS representation into a fixed-limb canonical integer.
    pub fn reconstruct_biguint<const N: usize>(&self, residues: &[u64]) -> BigUint<N> {
        assert_eq!(residues.len(), self.primes.len());

        if self.primes.is_empty() {
            return BigUint::ZERO;
        }

        let coeffs = self.garner_coefficients(residues);
        let mut result = BigUint::<N>::ZERO;
        let mut product = BigUint::<N>::one();

        for (coeff, &prime) in coeffs.iter().zip(self.primes.iter()) {
            let (term, carry_term) = product.mul_by_limb(*coeff);
            assert!(
                carry_term == 0,
                "RNS coefficient reconstruction does not fit in the requested canonical width"
            );
            let (sum, carry_sum) = result.add_with_carry(&term);
            assert!(
                !carry_sum,
                "RNS coefficient reconstruction does not fit in the requested canonical width"
            );
            result = sum;

            let (next_product, carry_product) = product.mul_by_limb(prime);
            assert!(
                carry_product == 0,
                "RNS modulus product does not fit in the requested canonical width"
            );
            product = next_product;
        }

        result
    }

    /// Component-wise addition in RNS into an existing residue buffer.
    pub fn add_assign_into(&self, dst: &mut [u64], src: &[u64]) {
        assert_eq!(dst.len(), self.primes.len());
        assert_eq!(src.len(), self.primes.len());
        for ((dst_limb, src_limb), &prime) in dst.iter_mut().zip(src.iter()).zip(self.primes.iter())
        {
            let sum = *dst_limb as u128 + *src_limb as u128;
            let prime = prime as u128;
            *dst_limb = if sum >= prime {
                (sum - prime) as u64
            } else {
                sum as u64
            };
        }
    }

    /// Component-wise addition in RNS.
    pub fn add(&self, a: &[u64], b: &[u64]) -> Vec<u64> {
        let mut out = a.to_vec();
        self.add_assign_into(&mut out, b);
        out
    }

    /// Component-wise subtraction in RNS into an existing residue buffer.
    pub fn sub_assign_into(&self, dst: &mut [u64], src: &[u64]) {
        assert_eq!(dst.len(), self.primes.len());
        assert_eq!(src.len(), self.primes.len());
        for ((dst_limb, src_limb), &prime) in dst.iter_mut().zip(src.iter()).zip(self.primes.iter())
        {
            *dst_limb = if *dst_limb >= *src_limb {
                *dst_limb - *src_limb
            } else {
                prime - *src_limb + *dst_limb
            };
        }
    }

    /// Component-wise subtraction in RNS.
    pub fn sub(&self, a: &[u64], b: &[u64]) -> Vec<u64> {
        let mut out = a.to_vec();
        self.sub_assign_into(&mut out, b);
        out
    }

    /// Component-wise multiplication in RNS into an existing residue buffer.
    pub fn mul_assign_into(&self, dst: &mut [u64], src: &[u64]) {
        assert_eq!(dst.len(), self.primes.len());
        assert_eq!(src.len(), self.primes.len());
        for ((dst_limb, src_limb), &prime) in dst.iter_mut().zip(src.iter()).zip(self.primes.iter())
        {
            *dst_limb = ((*dst_limb as u128 * *src_limb as u128) % prime as u128) as u64;
        }
    }

    /// Component-wise multiplication in RNS.
    pub fn mul(&self, a: &[u64], b: &[u64]) -> Vec<u64> {
        let mut out = a.to_vec();
        self.mul_assign_into(&mut out, b);
        out
    }

    fn garner_coefficients(&self, residues: &[u64]) -> Vec<u64> {
        assert_eq!(residues.len(), self.primes.len());

        if self.primes.is_empty() {
            return Vec::new();
        }

        let k = self.primes.len();
        let mut coeffs = vec![0u64; k];
        coeffs[0] = residues[0];

        for i in 1..k {
            let mut u = residues[i] as u128;
            let p_i = self.primes[i] as u128;
            for (j, coeff) in coeffs.iter().enumerate().take(i) {
                let coeff_mod = (*coeff as u128) % p_i;
                if u >= coeff_mod {
                    u = (u - coeff_mod) % p_i;
                } else {
                    u = p_i - ((coeff_mod - u) % p_i);
                }

                let inv = mod_inv_u64(self.primes[j], self.primes[i]) as u128;
                u = (u * inv) % p_i;
            }
            coeffs[i] = u as u64;
        }

        coeffs
    }
}

/// Compute `gcd(a, b)` using Euclid's algorithm.
#[inline]
pub fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Compute modular inverse `a^{-1} mod m` using extended Euclidean algorithm.
///
/// # Panics
/// Panics if `gcd(a, m) != 1`.
#[inline]
pub fn mod_inv_u64(a: u64, m: u64) -> u64 {
    let mut old_r = a as i128;
    let mut r = m as i128;
    let mut old_s: i128 = 1;
    let mut s: i128 = 0;

    while r != 0 {
        let q = old_r / r;
        let temp_r = r;
        r = old_r - q * r;
        old_r = temp_r;
        let temp_s = s;
        s = old_s - q * s;
        old_s = temp_s;
    }

    assert!(old_r == 1, "gcd({a}, {m}) != 1, no inverse exists");

    ((old_s % m as i128 + m as i128) % m as i128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(17, 13), 1);
        assert_eq!(gcd(100, 75), 25);
    }

    #[test]
    fn test_mod_inv() {
        // 3^{-1} mod 7 = 5 (since 3*5 = 15 = 2*7 + 1)
        assert_eq!(mod_inv_u64(3, 7), 5);
        // 5^{-1} mod 17 = 7
        assert_eq!(mod_inv_u64(5, 17), 7);
    }

    #[test]
    fn test_rns_decompose_reconstruct() {
        let basis = RnsBasis::new(vec![7, 11, 13]); // M = 1001
        let x = 500u64;
        let residues = basis.decompose(x);
        assert_eq!(residues, vec![500 % 7, 500 % 11, 500 % 13]);

        let reconstructed = basis.reconstruct(&residues);
        assert_eq!(reconstructed, 500u128);
    }

    #[test]
    fn test_rns_decompose_reconstruct_many() {
        let basis = RnsBasis::new(vec![7, 11, 13]); // M = 1001
        for x in 0..1001u64 {
            let residues = basis.decompose(x);
            let reconstructed = basis.reconstruct(&residues);
            assert_eq!(reconstructed, x as u128, "failed for x = {x}");
        }
    }

    #[test]
    fn test_rns_arithmetic() {
        let basis = RnsBasis::new(vec![7, 11, 13]); // M = 1001
        let a = 123u64;
        let b = 456u64;

        let ra = basis.decompose(a);
        let rb = basis.decompose(b);

        // Addition
        let sum = basis.add(&ra, &rb);
        let expected_sum = basis.decompose((a + b) % 1001);
        assert_eq!(sum, expected_sum, "RNS add failed");

        // Multiplication
        let prod = basis.mul(&ra, &rb);
        let expected_prod = basis.decompose(((a as u128 * b as u128) % 1001) as u64);
        assert_eq!(prod, expected_prod, "RNS mul failed");

        // Subtraction
        let diff = basis.sub(&ra, &rb);
        let expected_diff = basis.decompose((a + 1001 - b) % 1001);
        assert_eq!(diff, expected_diff, "RNS sub failed");
    }

    #[test]
    fn test_rns_larger_basis() {
        let basis = RnsBasis::new(vec![17, 19, 23, 29]);
        // M = 17*19*23*29 = 215441
        let x = 100000u64;
        let residues = basis.decompose(x);
        let reconstructed = basis.reconstruct(&residues);
        assert_eq!(reconstructed, x as u128);
    }

    #[test]
    fn test_rns_biguint_round_trip_above_u128() {
        let basis = RnsBasis::new(vec![
            0x0000_1000_01d0_0001,
            0x0000_1000_03b0_0001,
            0x0000_1000_0450_0001,
        ]);
        let value = BigUint::<3> {
            limbs: [0x0123_4567_89ab_cdef, 0x0fed_cba9_8765_4321, 0x0f],
        };

        let residues = basis.decompose_biguint(&value);
        let reconstructed = basis.reconstruct_biguint::<3>(&residues);
        assert_eq!(reconstructed, value);
        assert_eq!(
            basis.modulus_biguint::<3>(),
            BigUint::<3> {
                limbs: [0xb01e_9700_09d0_0001, 0x0009_d001_e970_1e0c, 0x10],
            }
        );
    }

    #[test]
    #[should_panic(expected = "pairwise coprime")]
    fn test_rns_non_coprime_panics() {
        RnsBasis::new(vec![6, 10]); // gcd(6,10) = 2
    }

    #[test]
    #[should_panic(expected = "at least one modulus")]
    fn test_rns_empty_basis_panics() {
        let _ = RnsBasis::new(vec![]);
    }

    #[test]
    #[should_panic(expected = "greater than 1")]
    fn test_rns_zero_limb_panics() {
        let _ = RnsBasis::new(vec![0, 7]);
    }

    #[test]
    #[should_panic(expected = "greater than 1")]
    fn test_rns_one_limb_panics() {
        let _ = RnsBasis::new(vec![1, 7]);
    }
}
