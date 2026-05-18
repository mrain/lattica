//! Gadget decomposition utilities.

use alloc::vec;
use alloc::vec::Vec;

use crate::arith::bigint::BigUint;
use crate::arith::large_modulus::LargeCanonicalRing;
use crate::arith::ring::IntegerRing;

fn num_digits(mut modulus: u64, base: u64) -> usize {
    assert!(base >= 2, "base must be at least 2");
    let mut digits = 0usize;
    while modulus > 0 {
        digits += 1;
        modulus /= base;
    }
    digits.max(1)
}

fn max_canonical_value<R: IntegerRing<Uint = u64>>() -> u64 {
    let modulus = R::modulus();
    if modulus == 0 { u64::MAX } else { modulus - 1 }
}

fn num_digits_big<const N: usize>(mut modulus: BigUint<N>, base: u64) -> usize {
    assert!(base >= 2, "base must be at least 2");
    let mut digits = 0usize;
    while !modulus.is_zero() {
        digits += 1;
        modulus = modulus.div_rem_small(base).0;
    }
    digits.max(1)
}

fn max_canonical_value_big<R, const N: usize>() -> BigUint<N>
where
    R: LargeCanonicalRing<Canonical = BigUint<N>>,
{
    let (max, borrow) = R::modulus_canonical().sub_small(1);
    assert!(
        !borrow,
        "large-modulus gadget decomposition requires modulus greater than one"
    );
    max
}

/// Decompose coefficients into base-`B` digits.
pub fn gadget_decompose<R>(x: &[R], base: u64) -> Vec<Vec<R>>
where
    R: IntegerRing<Uint = u64>,
{
    let k = num_digits(max_canonical_value::<R>(), base);
    let mut out = vec![vec![R::zero(); x.len()]; k];
    let mut values = x.iter().map(IntegerRing::to_u64).collect::<Vec<_>>();
    for digit_vec in out.iter_mut().take(k) {
        for (digit, value) in digit_vec.iter_mut().zip(values.iter_mut()) {
            *digit = R::from_u64(*value % base);
            *value /= base;
        }
    }
    out
}

/// Recompose base-`B` digits into coefficients.
pub fn gadget_recompose<R>(digits: &[Vec<R>], base: u64) -> Vec<R>
where
    R: IntegerRing<Uint = u64>,
{
    assert!(base >= 2, "base must be at least 2");
    if digits.is_empty() {
        return Vec::new();
    }
    let n = digits[0].len();
    let mut out = vec![R::zero(); n];
    let k = digits.len();
    for j in 0..n {
        let mut acc = 0u64;
        let mut place = 1u64;
        for (idx, digit_vec) in digits.iter().enumerate() {
            assert_eq!(
                digit_vec.len(),
                n,
                "all digit vectors must have equal length"
            );
            let digit = digit_vec[j].to_u64();
            let (prod, overflow_prod) = digit.overflowing_mul(place);
            let (new_acc, overflow_acc) = acc.overflowing_add(prod);
            debug_assert!(
                !overflow_prod && !overflow_acc,
                "gadget recomposition overflowed u64 (too many digits or base too large)"
            );
            acc = new_acc;
            // Place value overflow on the last iteration is harmless (never used).
            if idx + 1 < k {
                let (new_place, overflow_place) = place.overflowing_mul(base);
                debug_assert!(
                    !overflow_place,
                    "gadget recomposition place value overflowed u64"
                );
                place = new_place;
            }
        }
        out[j] = R::from_u64(acc);
    }
    out
}

/// Return the gadget vector `[1, B, B^2, ...]`.
pub fn gadget_vector<R>(base: u64) -> Vec<R>
where
    R: IntegerRing<Uint = u64>,
{
    let k = num_digits(max_canonical_value::<R>(), base);
    let mut out = Vec::with_capacity(k);
    let mut value = 1u64;
    for _ in 0..k {
        out.push(R::from_u64(value));
        value = value.wrapping_mul(base);
    }
    out
}

/// Decompose coefficients into base-`B` digits for `BigUint`-backed large-modulus rings.
pub fn gadget_decompose_large<R, const N: usize>(x: &[R], base: u64) -> Vec<Vec<R>>
where
    R: LargeCanonicalRing<Canonical = BigUint<N>>,
{
    let k = num_digits_big(max_canonical_value_big::<R, N>(), base);
    let mut out = vec![vec![R::zero(); x.len()]; k];
    let mut values = x
        .iter()
        .map(LargeCanonicalRing::to_canonical)
        .collect::<Vec<_>>();
    for digit_vec in out.iter_mut().take(k) {
        for (digit, value) in digit_vec.iter_mut().zip(values.iter_mut()) {
            let (quotient, remainder) = value.div_rem_small(base);
            *digit = R::from_small_u64(remainder);
            *value = quotient;
        }
    }
    out
}

/// Recompose base-`B` digits into coefficients for `BigUint`-backed large-modulus rings.
pub fn gadget_recompose_large<R, const N: usize>(digits: &[Vec<R>], base: u64) -> Vec<R>
where
    R: LargeCanonicalRing<Canonical = BigUint<N>>,
{
    assert!(base >= 2, "base must be at least 2");
    if digits.is_empty() {
        return Vec::new();
    }
    let n = digits[0].len();
    let mut out = vec![R::zero(); n];
    for j in 0..n {
        let mut acc = BigUint::<N>::ZERO;
        for digit_vec in digits.iter().rev() {
            assert_eq!(
                digit_vec.len(),
                n,
                "all digit vectors must have equal length"
            );
            let (scaled, carry_scaled) = acc.mul_by_limb(base);
            assert_eq!(
                carry_scaled, 0,
                "large-modulus gadget recomposition overflowed the canonical width"
            );
            let digit = digit_vec[j]
                .try_to_u64()
                .expect("large-modulus gadget digits must fit in u64");
            assert!(digit < base, "digit must be less than the gadget base");
            let (next, carry_next) = scaled.add_small(digit);
            assert!(
                !carry_next,
                "large-modulus gadget recomposition overflowed the canonical width"
            );
            acc = next;
        }
        out[j] = R::from_canonical(&acc);
    }
    out
}

/// Return the gadget vector `[1, B, B^2, ...]` for `BigUint`-backed large-modulus rings.
pub fn gadget_vector_large<R, const N: usize>(base: u64) -> Vec<R>
where
    R: LargeCanonicalRing<Canonical = BigUint<N>>,
{
    let k = num_digits_big(max_canonical_value_big::<R, N>(), base);
    let mut out = Vec::with_capacity(k);
    let mut value = BigUint::<N>::one();
    for idx in 0..k {
        out.push(R::from_canonical(&value));
        if idx + 1 < k {
            let (next, carry) = value.mul_by_limb(base);
            assert_eq!(
                carry, 0,
                "large-modulus gadget vector overflowed the canonical width"
            );
            value = next;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arith::LargeCanonicalRing;
    use crate::arith::large_prime::Bn254Fr;
    use crate::arith::large_rns::Rns3V0;
    use crate::arith::prime::PrimeField;
    use crate::arith::ring::IntegerRing;
    use crate::arith::z2k::Z2K;
    use grid_std::UniformRand;

    type F17 = PrimeField<17>;
    type F12289 = PrimeField<12289>;
    type Z2_64 = Z2K<64>;

    #[test]
    fn test_gadget_round_trip_bases() {
        let coeffs = vec![
            F17::from_u64(0),
            F17::from_u64(1),
            F17::from_u64(5),
            F17::from_u64(16),
        ];
        for &base in &[2u64, 4, 16, 256] {
            let digits = gadget_decompose(&coeffs, base);
            for digit_vec in &digits {
                for digit in digit_vec {
                    assert!(digit.to_u64() < base);
                }
            }
            assert_eq!(gadget_recompose(&digits, base), coeffs);
        }
    }

    #[test]
    fn test_gadget_vector() {
        let vec = gadget_vector::<F12289>(4);
        assert_eq!(vec[0].to_u64(), 1);
        assert_eq!(vec[1].to_u64(), 4);
        assert_eq!(vec[2].to_u64(), 16);
    }

    #[test]
    fn test_gadget_round_trip_power_of_two_64() {
        let coeffs = vec![Z2_64::from_u64(0x0123_4567_89AB_CDEF)];
        let digits = gadget_decompose(&coeffs, 256);
        assert_eq!(digits.len(), 8);
        for digit_vec in &digits {
            for digit in digit_vec {
                assert!(digit.to_u64() < 256);
            }
        }
        assert_eq!(gadget_recompose(&digits, 256), coeffs);

        let gadget = gadget_vector::<Z2_64>(256);
        assert_eq!(gadget.len(), 8);
        assert_eq!(gadget[0].to_u64(), 1);
        assert_eq!(gadget[1].to_u64(), 256);
    }

    #[test]
    fn test_gadget_round_trip_multidigit_prime_field() {
        let mut rng = grid_std::test_rng();
        let coeffs = (0..64).map(|_| F12289::rand(&mut rng)).collect::<Vec<_>>();
        let digits = gadget_decompose(&coeffs, 16);
        assert!(digits.len() > 1);
        for digit_vec in &digits {
            for digit in digit_vec {
                assert!(digit.to_u64() < 16);
            }
        }
        assert_eq!(gadget_recompose(&digits, 16), coeffs);
    }

    fn exercise_large_gadget_round_trip<R, const N: usize>()
    where
        R: LargeCanonicalRing<Canonical = BigUint<N>> + UniformRand + PartialEq + core::fmt::Debug,
    {
        let mut rng = grid_std::test_rng();
        let coeffs = (0..16).map(|_| R::rand(&mut rng)).collect::<Vec<_>>();
        let digits = gadget_decompose_large(&coeffs, 16);
        assert!(digits.len() > 1);
        for digit_vec in &digits {
            for digit in digit_vec {
                assert!(digit.try_to_u64().is_some_and(|digit| digit < 16));
            }
        }
        assert_eq!(gadget_recompose_large(&digits, 16), coeffs);

        let gadget = gadget_vector_large::<R, N>(16);
        assert_eq!(gadget[0].try_to_u64(), Some(1));
        assert_eq!(gadget[1].try_to_u64(), Some(16));
        assert_eq!(gadget[2].try_to_u64(), Some(256));
    }

    #[test]
    fn test_gadget_round_trip_large_prime_backend() {
        exercise_large_gadget_round_trip::<Bn254Fr, 4>();
    }

    #[test]
    fn test_gadget_round_trip_large_rns_backend() {
        exercise_large_gadget_round_trip::<Rns3V0, 3>();
    }
}
