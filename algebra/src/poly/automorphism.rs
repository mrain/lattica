//! Ring automorphisms for negacyclic polynomial rings.

use alloc::vec;

use crate::arith::ring::Ring;

/// Apply the automorphism `X -> X^k` in `Z_q[X] / (X^n + 1)`.
pub fn apply_automorphism<R: Ring>(coeffs: &mut [R], k: usize, n: usize) {
    assert_eq!(coeffs.len(), n, "coefficient length must equal n");
    assert!(k % 2 == 1, "k must be odd for an automorphism on X^n + 1");
    let modulus = 2 * n;
    assert_eq!(gcd(k, modulus), 1, "k must be coprime to 2n");

    let input = coeffs.to_vec();
    let mut out = vec![R::zero(); n];
    for (i, coeff) in input.into_iter().enumerate() {
        let mapped = (i * k) % modulus;
        if mapped < n {
            out[mapped] += coeff;
        } else {
            out[mapped - n] -= coeff;
        }
    }
    coeffs.clone_from_slice(&out);
}

/// Apply the Frobenius-like automorphism `sigma_p`.
pub fn frobenius<R: Ring>(coeffs: &mut [R], p: usize, n: usize) {
    apply_automorphism(coeffs, p, n);
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arith::prime::PrimeField;
    use crate::arith::ring::IntegerRing;
    use crate::poly::ring::{CyclotomicPolyRing, PolyRing};

    type F17 = PrimeField<17>;
    type Poly8 = CyclotomicPolyRing<F17, 8>;

    fn coeffs(poly: &Poly8) -> Vec<F17> {
        poly.coeffs().to_vec()
    }

    #[test]
    fn test_identity_automorphism() {
        let poly = Poly8::from_array([
            F17::from_u64(1),
            F17::from_u64(2),
            F17::from_u64(3),
            F17::from_u64(4),
            F17::from_u64(5),
            F17::from_u64(6),
            F17::from_u64(7),
            F17::from_u64(8),
        ]);
        let mut transformed = coeffs(&poly);
        apply_automorphism(&mut transformed, 1, 8);
        assert_eq!(transformed, coeffs(&poly));
    }

    #[test]
    fn test_inverse_automorphism() {
        let poly = Poly8::from_array([
            F17::from_u64(1),
            F17::from_u64(2),
            F17::from_u64(3),
            F17::from_u64(4),
            F17::from_u64(5),
            F17::from_u64(6),
            F17::from_u64(7),
            F17::from_u64(8),
        ]);
        let mut transformed = coeffs(&poly);
        apply_automorphism(&mut transformed, 3, 8);
        apply_automorphism(&mut transformed, 11, 8);
        assert_eq!(transformed, coeffs(&poly));
    }

    #[test]
    fn test_automorphism_homomorphism() {
        let f = Poly8::from_array([
            F17::from_u64(1),
            F17::from_u64(2),
            F17::from_u64(0),
            F17::from_u64(1),
            F17::from_u64(0),
            F17::from_u64(0),
            F17::from_u64(0),
            F17::from_u64(0),
        ]);
        let g = Poly8::from_array([
            F17::from_u64(3),
            F17::from_u64(1),
            F17::from_u64(4),
            F17::from_u64(1),
            F17::from_u64(0),
            F17::from_u64(0),
            F17::from_u64(0),
            F17::from_u64(0),
        ]);
        let fg = Poly8::neg_cyclic_mul(&f, &g);

        let mut lhs = coeffs(&fg);
        apply_automorphism(&mut lhs, 3, 8);

        let mut f_auto = coeffs(&f);
        let mut g_auto = coeffs(&g);
        apply_automorphism(&mut f_auto, 3, 8);
        apply_automorphism(&mut g_auto, 3, 8);
        let rhs = Poly8::neg_cyclic_mul(
            &Poly8::try_from_coeffs(&f_auto).unwrap(),
            &Poly8::try_from_coeffs(&g_auto).unwrap(),
        );

        assert_eq!(lhs, coeffs(&rhs));
    }
}
