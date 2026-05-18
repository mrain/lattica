//! Binary-norm decomposition of z (§5.3).
//!
//! Decomposes the amortized witness z = z⁰ + b·z¹ so that each coefficient
//! of z⁰ lies in [-b/2, b/2]. Also bundles the commitment openings
//! v = t ‖ g ‖ h for the next recursion level.

use alloc::vec::Vec;

use grid_algebra::arith::ring::IntegerRing;
use grid_algebra::arith::ring::Ring;
use grid_algebra::poly::ring::{CyclotomicPolyRing, NegacyclicMulRing, PolyRing};

/// Binary-norm decomposition of z: z = z⁰ + b·z¹.
///
/// Each coefficient c of z is split into a "low" part z⁰ ∈ [-b/2, b/2]
/// and a "high" part z¹ such that c ≡ z⁰ + b·z¹ (mod q).
///
/// Returns `[z⁰[0], ..., z⁰[n-1], z¹[0], ..., z¹[n-1]]` — 2n polynomials total.
///
/// # Panics
///
/// Panics if `z` is empty.
pub fn decompose_z<R, const N: usize>(
    z: &[CyclotomicPolyRing<R, N>],
    b: u64,
) -> Vec<CyclotomicPolyRing<R, N>>
where
    R: IntegerRing<Uint = u64> + NegacyclicMulRing<N>,
{
    let n = z.len();
    assert!(n > 0, "z must be non-empty");

    let q = R::modulus();
    let b_i128 = b as i128;
    let q_i128 = q as i128;
    let half_q = q / 2;

    let mut z0 = Vec::with_capacity(n);
    let mut z1 = Vec::with_capacity(n);

    for z_k in z.iter().take(n) {
        let mut p0 = CyclotomicPolyRing::<R, N>::zero();
        let mut p1 = CyclotomicPolyRing::<R, N>::zero();

        for i in 0..N {
            let c = z_k.coeff(i).to_u64();
            let c_signed = if c > half_q {
                (c as i128) - q_i128
            } else {
                c as i128
            };

            let low = centered_mod_i128(c_signed, b_i128);
            let low_u64 = if low >= 0 {
                low as u64
            } else {
                q.wrapping_sub((-low) as u64)
            };
            p0.set_coeff(i, R::from_u64(low_u64));

            let high = (c_signed - low) / b_i128;
            let high_wrapped = ((high % q_i128) + q_i128) % q_i128;
            p1.set_coeff(i, R::from_u64(high_wrapped as u64));
        }

        z0.push(p0);
        z1.push(p1);
    }

    [z0, z1].concat()
}

/// Centered modulo: returns `v mod base` in range `[-base/2, base/2]`.
fn centered_mod_i128(v: i128, base: i128) -> i128 {
    let half = base / 2;
    let r = v % base;
    if r > half {
        r - base
    } else if r < -half {
        r + base
    } else {
        r
    }
}

/// Bundle commitment openings into v = t ‖ g ‖ h.
///
/// The limbs are already decomposed from the main protocol (t at base b₁,
/// g at base b₂, h at base b₁). This function simply concatenates them.
///
/// Returns a flat vector of length `r·κ·t₁ + garbage_count(r)·(t₂ + t₁)`.
pub fn bundle_v<R, const N: usize>(
    t_limbs: &[CyclotomicPolyRing<R, N>],
    g_limbs: &[CyclotomicPolyRing<R, N>],
    h_limbs: &[CyclotomicPolyRing<R, N>],
) -> Vec<CyclotomicPolyRing<R, N>>
where
    R: IntegerRing<Uint = u64> + NegacyclicMulRing<N>,
{
    let total = t_limbs.len() + g_limbs.len() + h_limbs.len();
    let mut v = Vec::with_capacity(total);
    v.extend_from_slice(t_limbs);
    v.extend_from_slice(g_limbs);
    v.extend_from_slice(h_limbs);
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    use grid_algebra::arith::prime::PrimeField;
    use grid_algebra::arith::ring::Ring;
    use grid_algebra::poly::ring::PolyRing;

    type F = PrimeField<12289>;

    fn make_poly<const D: usize>(coeffs: &[u64]) -> CyclotomicPolyRing<F, D> {
        let mut p = CyclotomicPolyRing::<F, D>::zero();
        for (i, &c) in coeffs.iter().enumerate().take(D) {
            p.set_coeff(i, F::from_u64(c));
        }
        p
    }

    #[test]
    fn test_decompose_z_roundtrip() {
        const N: usize = 64;
        let b = 16;
        let q = F::modulus();

        let cases = vec![
            // Small positive coefficients
            vec![make_poly::<N>(&[3, 5, 0]), make_poly::<N>(&[7, 11, 0])],
            // Coefficients exceeding decomposition base
            vec![make_poly::<N>(&[17, 100, 0]), make_poly::<N>(&[200, 1, 0])],
            // Negative coefficients (Montgomery representation)
            vec![make_poly::<N>(&[q - 3, q - 7, 0])],
        ];

        for z in cases {
            let result = decompose_z(&z, b);
            assert_eq!(result.len(), z.len() * 2, "2n polynomials");

            let n = z.len();
            let z0 = &result[0..n];
            let z1 = &result[n..];

            for k in 0..n {
                for i in 0..N {
                    let orig = z[k].coeff(i).to_u64();
                    let low = z0[k].coeff(i).to_u64();
                    let high = z1[k].coeff(i).to_u64();
                    let reconstructed = (low + b.wrapping_mul(high)) % q;
                    assert_eq!(
                        orig, reconstructed,
                        "roundtrip failed at z[{}].coeff({}): orig={}, low={}, high={}, recon={}",
                        k, i, orig, low, high, reconstructed
                    );
                }
            }
        }
    }

    #[test]
    fn test_decompose_z_all_zeros() {
        const N: usize = 64;
        let z = vec![CyclotomicPolyRing::<F, N>::zero(); 3];
        let result = decompose_z(&z, 16);
        assert_eq!(result.len(), 6);
        for p in &result {
            assert!(p.is_zero(), "decomposition of zero should be zero");
        }
    }

    #[test]
    fn test_bundle_v() {
        const N: usize = 64;
        let t = vec![make_poly::<N>(&[1]), make_poly::<N>(&[2])];
        let g = vec![make_poly::<N>(&[3])];
        let h = vec![
            make_poly::<N>(&[4]),
            make_poly::<N>(&[5]),
            make_poly::<N>(&[6]),
        ];

        let v = bundle_v(&t, &g, &h);
        assert_eq!(v.len(), 6, "2 + 1 + 3 = 6");
        assert_eq!(v[0].coeff(0).to_u64(), 1);
        assert_eq!(v[1].coeff(0).to_u64(), 2);
        assert_eq!(v[2].coeff(0).to_u64(), 3);
        assert_eq!(v[3].coeff(0).to_u64(), 4);
        assert_eq!(v[4].coeff(0).to_u64(), 5);
        assert_eq!(v[5].coeff(0).to_u64(), 6);
    }

    #[test]
    #[should_panic(expected = "z must be non-empty")]
    fn test_decompose_z_empty_panics() {
        let z: Vec<CyclotomicPolyRing<F, 64>> = vec![];
        let _ = decompose_z(&z, 16);
    }
}
