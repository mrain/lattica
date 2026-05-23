//! Application-ring abstraction for arithmetic R1CS reductions.
//!
//! The [`AppModRing`] trait extends [`IntegerRing`] with non-adjacent-form (NAF)
//! signed-digit encoding that bridges application-ring values into proof-ring
//! polynomials, plus uniform transcript challenge sampling.

use alloc::vec::Vec;

use grid_algebra::arith::UintLimb;
use grid_algebra::arith::bigint::BigUint;
use grid_algebra::arith::gf2::GF2;
use grid_algebra::arith::large_zm::{LargeZm, LargeZmProfile};
use grid_algebra::arith::ring::{IntegerRing, Ring};
use grid_algebra::arith::zm::Zm;
use grid_serialize::{CanonicalDeserialize, CanonicalSerialize};
use grid_std::UniformRand;
use grid_transcript::Transcript;

use crate::error::LabradorError;

/// Application ring for arithmetic R1CS reductions.
///
/// Extends [`IntegerRing`] with NAF signed-digit encoding that bridges
/// application-ring values into proof-ring polynomials.
///
/// `GF2` intentionally does **not** implement this trait — binary reduction is
/// concrete over `GF2`.
pub trait AppModRing:
    IntegerRing
    + CanonicalSerialize
    + CanonicalDeserialize
    + grid_serialize::Valid
    + UniformRand
    + Clone
    + Eq
    + core::fmt::Debug
{
    /// Convert this element into `N` NAF digits in \{-1, 0, 1\}.
    fn to_naf_digits<const N: usize>(&self) -> Result<[i8; N], LabradorError>;

    /// Evaluate `∑ digits[i] · 2^i` in this ring (modulo the app-ring modulus).
    fn eval_naf_digits(digits: &[i8]) -> Self;

    /// Sample a uniformly random element below the app modulus from a
    /// Fiat-Shamir transcript. Used for verifier challenge generation.
    fn sample_from_transcript<T: Transcript>(
        transcript: &mut T,
        label: &'static [u8],
    ) -> Result<Self, grid_transcript::TranscriptError>;

    /// Create the ring element for a single signed digit in \{-1, 0, 1\}.
    fn from_signed_digit(x: i8) -> Self {
        match x {
            0 => Self::zero(),
            1 => Self::one(),
            -1 => -Self::one(),
            _ => unreachable!("NAF digit out of range: {x}"),
        }
    }

    /// Whether this ring's modulus equals `2^N + 1` (the paper's Fermat ring).
    ///
    /// Figure 5/Theorem 6.3 are for R1CS over `Z_{2^d+1}` with proof-ring
    /// degree `d = N`. Returns `true` only when the modulus matches exactly.
    fn is_fermat_modulus_for_degree<const N: usize>() -> bool;

    /// Number of bits for the paper-style restricted binary encoding.
    ///
    /// Returns `Some(N)` when `is_fermat_modulus_for_degree::<N>()` holds.
    /// Values are restricted to `[0, 2^N)`, not the full canonical range.
    fn binary_encoding_width<const N: usize>() -> Option<usize>;

    /// Decode from exactly `N` LSB-first GF2 bits. Returns `None` if the bit
    /// string is not valid for the ring's restricted paper encoding or if
    /// the decoded value is outside `[0, 2^N)`.
    fn try_decode_from_gf2_bits<const N: usize>(bits: &[GF2]) -> Option<Self>;

    /// Encode to LSB-first GF2 bits of length `binary_encoding_width::<N>()`.
    /// Returns `None` if this value is outside the restricted encoding range.
    fn try_encode_to_gf2_bits<const N: usize>(&self) -> Option<Vec<GF2>>;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build `2^N + 1` as `BigUint<LIMBS>`. Returns `None` if N exceeds the
/// representable bit-width (`LIMBS * 64`), or if the addition overflows
/// (which shouldn't happen given the bit-width guard).
fn fermat_modulus_candidate<const N: usize, const LIMBS: usize>() -> Option<BigUint<LIMBS>> {
    if N >= LIMBS * 64 {
        return None;
    }
    // Build 2^N directly via limb indexing (shl_bits panics for N >= 64).
    let limb_idx = N / 64;
    let bit_pos = N % 64;
    let mut limbs = [0u64; LIMBS];
    limbs[limb_idx] = 1u64 << bit_pos;
    let pow2 = BigUint::<LIMBS> { limbs };
    // 2^N + 1
    let (sum, carry) = pow2.add_with_carry(&BigUint::one());
    if carry {
        return None;
    }
    Some(sum)
}

/// Build `2^N` as `BigUint<LIMBS>`. Returns `None` if N exceeds the
/// representable bit-width (`LIMBS * 64`).
fn restricted_encoding_limit<const N: usize, const LIMBS: usize>() -> Option<BigUint<LIMBS>> {
    if N >= LIMBS * 64 {
        return None;
    }
    let limb_idx = N / 64;
    let bit_pos = N % 64;
    let mut limbs = [0u64; LIMBS];
    limbs[limb_idx] = 1u64 << bit_pos;
    Some(BigUint { limbs })
}

// ---------------------------------------------------------------------------
// Zm<M, L> — small application moduli
// ---------------------------------------------------------------------------

impl<const M: u64, L: UintLimb> AppModRing for Zm<M, L> {
    fn sample_from_transcript<T: Transcript>(
        transcript: &mut T,
        label: &'static [u8],
    ) -> Result<Self, grid_transcript::TranscriptError> {
        let m = Self::MODULUS; // touches validated modulus constant
        let threshold = u64::MAX - (u64::MAX % m);
        loop {
            let bytes = transcript.challenge_bytes(label, 8)?;
            let val = u64::from_le_bytes(bytes.try_into().unwrap());
            if val < threshold {
                return Ok(Self::from_u64(val % m));
            }
        }
    }

    fn to_naf_digits<const N: usize>(&self) -> Result<[i8; N], LabradorError> {
        let v = self.to_canonical(); // u64 in [0, M-1]
        let half_m = (M - 1) / 2;

        let centered: i128 = if v > half_m {
            (v as i128) - (M as i128)
        } else {
            v as i128
        };

        let mut digits = [0i8; N];
        let mut val = centered;
        let mut i = 0;
        while val != 0 && i < N {
            if val & 1 != 0 {
                let rem = val.rem_euclid(4);
                if rem == 1 {
                    digits[i] = 1;
                    val -= 1;
                } else {
                    // rem == 3 (val is odd, so rem cannot be 0 or 2)
                    digits[i] = -1;
                    val += 1;
                }
            }
            val /= 2;
            i += 1;
        }
        if val != 0 {
            return Err(LabradorError::InvalidInput(alloc::format!(
                "value {v} does not fit in {N} NAF digits (mod {M})"
            )));
        }
        Ok(digits)
    }

    fn eval_naf_digits(digits: &[i8]) -> Self {
        let mut acc = Self::zero();
        let mut weight = Self::one();
        for &d in digits {
            if d == 1 {
                acc += &weight;
            } else if d == -1 {
                acc -= &weight;
            }
            weight += &weight.clone();
        }
        acc
    }

    fn is_fermat_modulus_for_degree<const N: usize>() -> bool {
        // Zm<M> stores modulus as u64; 2^N+1 fits iff N < 64.
        if N >= 64 {
            return false;
        }
        M == (1u64 << N) + 1
    }

    fn binary_encoding_width<const N: usize>() -> Option<usize> {
        if Self::is_fermat_modulus_for_degree::<N>() {
            Some(N)
        } else {
            None
        }
    }

    fn try_decode_from_gf2_bits<const N: usize>(bits: &[GF2]) -> Option<Self> {
        if bits.len() != N || !Self::is_fermat_modulus_for_degree::<N>() {
            return None;
        }
        let mut val: u64 = 0;
        for (i, bit) in bits.iter().enumerate() {
            if !bit.is_zero() {
                val |= 1u64 << i;
            }
        }
        // Restricted encoding: value must be in [0, 2^N).
        if val >= (1u64 << N) {
            return None;
        }
        Some(Self::from_u64(val))
    }

    fn try_encode_to_gf2_bits<const N: usize>(&self) -> Option<Vec<GF2>> {
        if !Self::is_fermat_modulus_for_degree::<N>() {
            return None;
        }
        let v = self.to_u64();
        // Restricted encoding: the residue 2^N is excluded.
        if v >= (1u64 << N) {
            return None;
        }
        let mut out = Vec::with_capacity(N);
        for i in 0..N {
            out.push(GF2::new(((v >> i) & 1) as u8));
        }
        Some(out)
    }
}

// ---------------------------------------------------------------------------
// LargeZm<P, LIMBS> — multi-limb application moduli (e.g. FermatRing64)
// ---------------------------------------------------------------------------

impl<P, const LIMBS: usize> AppModRing for LargeZm<P, LIMBS>
where
    P: LargeZmProfile<LIMBS>,
{
    fn sample_from_transcript<T: Transcript>(
        transcript: &mut T,
        label: &'static [u8],
    ) -> Result<Self, grid_transcript::TranscriptError> {
        // Rejection sampling over the modulus bit-width using transcript bytes.
        let modulus = Self::modulus_canonical();
        let modulus_bits = modulus.bits() as usize;
        let byte_count = modulus_bits.div_ceil(8);
        loop {
            let bytes = transcript.challenge_bytes(label, byte_count)?;
            let mut limbs = [0u64; LIMBS];
            for (i, chunk) in bytes.chunks(8).enumerate() {
                let mut arr = [0u8; 8];
                for (j, &b) in chunk.iter().enumerate() {
                    arr[j] = b;
                }
                limbs[i] = u64::from_le_bytes(arr);
            }
            let extra_bits = modulus_bits % 64;
            if extra_bits > 0 {
                let last_idx = modulus_bits / 64;
                if last_idx < LIMBS {
                    limbs[last_idx] &= (1u64 << extra_bits) - 1;
                }
            }
            let sample = BigUint { limbs };
            if sample < modulus {
                return Ok(Self::from_canonical(&sample));
            }
        }
    }

    fn to_naf_digits<const N: usize>(&self) -> Result<[i8; N], LabradorError> {
        let canon = self.to_canonical(); // BigUint<LIMBS>
        let modulus = Self::modulus_canonical(); // BigUint<LIMBS>

        // half_modulus = (modulus - 1) / 2
        let one = BigUint::<LIMBS>::one();
        let (mod_minus_1, _) = modulus.sub_with_borrow(&one);
        let (half_modulus, _) = mod_minus_1.div_rem_small(2u64);

        // Center into [-(M-1)/2, (M-1)/2].
        let (is_negative, mut mag) = if canon.compare(&half_modulus) == core::cmp::Ordering::Greater
        {
            let (mag, _) = modulus.sub_with_borrow(&canon);
            (true, mag)
        } else {
            (false, canon)
        };

        let mut digits = [0i8; N];
        let mut i = 0;
        while !mag.is_zero() && i < N {
            if mag.limbs[0] & 1 != 0 {
                let limb_mod_4 = mag.limbs[0] % 4; // 1 or 3 (value is odd)

                if is_negative {
                    // Signed value = -(mag).
                    // -(mag).rem_euclid(4):  mag%4==1 → 3,  mag%4==3 → 1
                    if limb_mod_4 == 1 {
                        // rem == 3 → digit = -1, val += 1
                        // val + 1 = -(mag) + 1 = -(mag - 1).  mag ← mag - 1.
                        digits[i] = -1;
                        let (new_mag, _) = mag.sub_with_borrow(&one);
                        mag = new_mag;
                    } else {
                        // limb_mod_4 == 3 → rem == 1 → digit = 1, val -= 1
                        // val - 1 = -(mag) - 1 = -(mag + 1).  mag ← mag + 1.
                        digits[i] = 1;
                        let (new_mag, _) = mag.add_with_carry(&one);
                        mag = new_mag;
                    }
                } else {
                    // Signed value = +(mag).
                    // mag.rem_euclid(4): mag%4==1 → 1,  mag%4==3 → 3
                    if limb_mod_4 == 1 {
                        digits[i] = 1;
                        let (new_mag, _) = mag.sub_with_borrow(&one);
                        mag = new_mag;
                    } else {
                        digits[i] = -1;
                        let (new_mag, _) = mag.add_with_carry(&one);
                        mag = new_mag;
                    }
                }
            }
            // mag /= 2
            mag = mag.shr_bits(1);
            i += 1;
        }
        if !mag.is_zero() {
            return Err(LabradorError::InvalidInput(alloc::format!(
                "value does not fit in {N} NAF digits"
            )));
        }
        Ok(digits)
    }

    fn eval_naf_digits(digits: &[i8]) -> Self {
        let mut acc = Self::zero();
        let mut weight = Self::one();
        for &d in digits {
            if d == 1 {
                acc += &weight;
            } else if d == -1 {
                acc -= &weight;
            }
            weight += &weight.clone();
        }
        acc
    }

    fn is_fermat_modulus_for_degree<const N: usize>() -> bool {
        let candidate = match fermat_modulus_candidate::<N, LIMBS>() {
            Some(c) => c,
            None => return false,
        };
        candidate == Self::modulus_canonical()
    }

    fn binary_encoding_width<const N: usize>() -> Option<usize> {
        if Self::is_fermat_modulus_for_degree::<N>() {
            Some(N)
        } else {
            None
        }
    }

    fn try_decode_from_gf2_bits<const N: usize>(bits: &[GF2]) -> Option<Self> {
        if bits.len() != N || !Self::is_fermat_modulus_for_degree::<N>() {
            return None;
        }
        // Build the integer from LSB-first bits.
        let mut limbs = [0u64; LIMBS];
        for (i, bit) in bits.iter().enumerate() {
            if bit.is_zero() {
                continue;
            }
            let limb_idx = i / 64;
            if limb_idx >= LIMBS {
                return None;
            }
            limbs[limb_idx] |= 1u64 << (i % 64);
        }
        let val = BigUint::<LIMBS> { limbs };
        // Restricted encoding: value must be in [0, 2^N).
        let max_val = restricted_encoding_limit::<N, LIMBS>()?;
        if val.compare(&max_val) != core::cmp::Ordering::Less {
            return None;
        }
        Some(Self::from_canonical(&val))
    }

    fn try_encode_to_gf2_bits<const N: usize>(&self) -> Option<Vec<GF2>> {
        if !Self::is_fermat_modulus_for_degree::<N>() {
            return None;
        }
        let canon = self.to_canonical();
        // Restricted encoding: the residue 2^N is excluded.
        let max_val = restricted_encoding_limit::<N, LIMBS>()?;
        if canon.compare(&max_val) != core::cmp::Ordering::Less {
            return None;
        }
        let mut out = Vec::with_capacity(N);
        for i in 0..N {
            let limb_idx = i / 64;
            let bit = if limb_idx < LIMBS {
                (canon.limbs[limb_idx] >> (i % 64)) & 1
            } else {
                0
            };
            out.push(GF2::new(bit as u8));
        }
        Some(out)
    }
}

// ---------------------------------------------------------------------------
// Bridge helpers
// ---------------------------------------------------------------------------

/// Convert a `GF2` value to a proof-ring constant.
#[inline]
pub fn gf2_to_proof_const<P: IntegerRing>(x: grid_algebra::arith::GF2) -> P {
    if x.is_zero() { P::zero() } else { P::one() }
}

/// Convert an app-ring value to a proof-ring constant if it fits in `u64`.
/// Returns `None` for multi-limb values (e.g. `FermatRing64` residues).
#[inline]
pub fn small_app_to_proof_const<A: AppModRing, P: IntegerRing>(x: &A) -> Option<P> {
    Some(P::from_u64(x.try_to_u64()?))
}
