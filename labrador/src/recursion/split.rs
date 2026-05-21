//! Split decomposed witness into r' vectors (§5.3).
//!
//! The decomposed polynomials (z⁰, z¹ from decomposition, v = t‖g‖h from bundling)
//! are split into r' = 2ν + μ vectors of rank n', zero-padded for uniformity.

use alloc::vec::Vec;

use grid_algebra::arith::ring::IntegerRing;
use grid_algebra::arith::ring::Ring;
use grid_algebra::poly::ring::{CyclotomicPolyRing, NegacyclicMulRing};

use crate::relation::LabradorWitness;

/// Compute the next-level rank and multiplicity.
///
/// Per §5.3: z⁽⁰⁾, z⁽¹⁾ ∈ R_q^n are each split into ν parts of rank ⌈n/ν⌉,
/// and v ∈ R_q^m is split into μ parts of rank ⌈m/μ⌉. All parts are zero-padded
/// to n' = max(⌈n/ν⌉, ⌈m/μ⌉).
///
/// Returns `(r', n')` where `r' = 2ν + μ`.
pub fn compute_next_level_shape(n: usize, m: usize, nu: usize, mu: usize) -> (usize, usize) {
    let r_prime = 2 * nu + mu;
    let z_part_rank = n.div_ceil(nu);
    let v_part_rank = m.div_ceil(mu);
    let n_prime = z_part_rank.max(v_part_rank);
    (r_prime, n_prime)
}

/// Split decomposed witness polynomials into r' vectors of rank n'.
///
/// Layout:
/// - `z_parts` (length 2n): first n = z⁰, next n = z¹ → split into 2ν parts
/// - `v` (length m): bundled t‖g‖h limbs → split into μ parts
///
/// Each part is zero-padded to rank `n_prime`. Returns a [`LabradorWitness`]
/// with `r' = 2ν + μ` parts, each of length `n'`.
pub fn split_witness<R, const N: usize>(
    z_parts: &[CyclotomicPolyRing<R, N>],
    v: &[CyclotomicPolyRing<R, N>],
    n: usize,
    nu: usize,
    mu: usize,
    n_prime: usize,
) -> LabradorWitness<CyclotomicPolyRing<R, N>>
where
    R: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    assert_eq!(
        z_parts.len(),
        2 * n,
        "z_parts must have length 2n, got {} for n={}",
        z_parts.len(),
        n
    );

    let r_prime = 2 * nu + mu;
    let zero = CyclotomicPolyRing::<R, N>::zero();
    let mut parts = Vec::with_capacity(r_prime);

    // z⁰ → parts 0..ν
    let z0 = &z_parts[0..n];
    let z0_per_part = n.div_ceil(nu);
    for i in 0..nu {
        let start = i * z0_per_part;
        let end = (start + z0_per_part).min(n);
        let mut part = z0[start..end].to_vec();
        while part.len() < n_prime {
            part.push(zero.clone());
        }
        parts.push(part);
    }

    // z¹ → parts ν..2ν
    let z1 = &z_parts[n..2 * n];
    for i in 0..nu {
        let start = i * z0_per_part;
        let end = (start + z0_per_part).min(n);
        let mut part = z1[start..end].to_vec();
        while part.len() < n_prime {
            part.push(zero.clone());
        }
        parts.push(part);
    }

    // v → parts 2ν..(2ν+μ)
    let m = v.len();
    let v_per_part = m.div_ceil(mu);
    for i in 0..mu {
        let start = i * v_per_part;
        let end = (start + v_per_part).min(m);
        let mut part = v[start..end].to_vec();
        while part.len() < n_prime {
            part.push(zero.clone());
        }
        parts.push(part);
    }

    LabradorWitness { parts }
}

/// Split proof z and bundled v into a last-level witness (no z decomposition).
///
/// Unlike [`split_witness`] which decomposes z into z⁰/z¹ for r' = 2ν+μ parts,
/// the last level has a single z split into ν parts, plus v split into μ parts,
/// for r_last = ν+μ total parts.
pub fn split_last_level_witness<R, const N: usize>(
    z: &[CyclotomicPolyRing<R, N>],
    t_decomposed: &[CyclotomicPolyRing<R, N>],
    g_decomposed: &[CyclotomicPolyRing<R, N>],
    h_decomposed: &[CyclotomicPolyRing<R, N>],
    nu: usize,
    mu: usize,
    n_prime: usize,
) -> LabradorWitness<CyclotomicPolyRing<R, N>>
where
    R: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    let zero = CyclotomicPolyRing::<R, N>::zero();
    let n = z.len();
    let mut parts = Vec::with_capacity(nu + mu);

    // z → parts 0..nu
    let z_per_part = n.div_ceil(nu);
    for i in 0..nu {
        let start = i * z_per_part;
        let end = (start + z_per_part).min(n);
        let mut part = z[start..end].to_vec();
        while part.len() < n_prime {
            part.push(zero.clone());
        }
        parts.push(part);
    }

    // v = t || g || h → parts nu..(nu+mu)
    let mut v = Vec::with_capacity(t_decomposed.len() + g_decomposed.len() + h_decomposed.len());
    v.extend_from_slice(t_decomposed);
    v.extend_from_slice(g_decomposed);
    v.extend_from_slice(h_decomposed);

    let m = v.len();
    let v_per_part = m.div_ceil(mu);
    for i in 0..mu {
        let start = i * v_per_part;
        let end = (start + v_per_part).min(m);
        let mut part = v[start..end].to_vec();
        while part.len() < n_prime {
            part.push(zero.clone());
        }
        parts.push(part);
    }

    LabradorWitness { parts }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    use grid_algebra::arith::prime::PrimeField;
    use grid_algebra::arith::ring::Ring;
    use grid_algebra::poly::ring::PolyRing;

    type F = PrimeField<12289>;

    fn make_poly<const D: usize>(v: u64) -> CyclotomicPolyRing<F, D> {
        let mut p = CyclotomicPolyRing::<F, D>::zero();
        p.set_coeff(0, F::from_u64(v));
        p
    }

    #[test]
    fn test_compute_next_level_shape_equal() {
        // n=8, m=24, ν=2, μ=2 → z_part=4, v_part=12, n'=12, r'=6
        let (r_prime, n_prime) = compute_next_level_shape(8, 24, 2, 2);
        assert_eq!(r_prime, 6, "2*2 + 2 = 6");
        assert_eq!(n_prime, 12, "max(⌈8/2⌉, ⌈24/2⌉) = 12");
    }

    #[test]
    fn test_compute_next_level_shape_z_dominates() {
        // n=100, m=10, ν=1, μ=1 → z_part=100, v_part=10, n'=100, r'=3
        let (r_prime, n_prime) = compute_next_level_shape(100, 10, 1, 1);
        assert_eq!(r_prime, 3);
        assert_eq!(n_prime, 100);
    }

    #[test]
    fn test_compute_next_level_shape_v_dominates() {
        // n=5, m=100, ν=1, μ=1 → z_part=5, v_part=100, n'=100, r'=3
        let (r_prime, n_prime) = compute_next_level_shape(5, 100, 1, 1);
        assert_eq!(r_prime, 3);
        assert_eq!(n_prime, 100);
    }

    #[test]
    fn test_split_witness_basic() {
        const N: usize = 64;
        // z_parts: [1,2,3,4] (z⁰=[1,2], z¹=[3,4]), n=2
        let z_parts = vec![
            make_poly::<N>(1),
            make_poly::<N>(2),
            make_poly::<N>(3),
            make_poly::<N>(4),
        ];
        // v: [10, 11, 12], m=3
        let v = vec![make_poly::<N>(10), make_poly::<N>(11), make_poly::<N>(12)];

        let witness = split_witness(&z_parts, &v, 2, 1, 1, 3);

        assert_eq!(witness.num_parts(), 3, "r' = 2*1 + 1 = 3");
        assert_eq!(witness.rank(), 3, "n' = max(⌈2/1⌉, ⌈3/1⌉) = 3");

        // Part 0 (z⁰): [1, 2, 0]
        assert_eq!(witness.parts[0][0].coeff(0).to_u64(), 1);
        assert_eq!(witness.parts[0][1].coeff(0).to_u64(), 2);
        assert!(witness.parts[0][2].is_zero());

        // Part 1 (z¹): [3, 4, 0]
        assert_eq!(witness.parts[1][0].coeff(0).to_u64(), 3);
        assert_eq!(witness.parts[1][1].coeff(0).to_u64(), 4);
        assert!(witness.parts[1][2].is_zero());

        // Part 2 (v): [10, 11, 12]
        assert_eq!(witness.parts[2][0].coeff(0).to_u64(), 10);
        assert_eq!(witness.parts[2][1].coeff(0).to_u64(), 11);
        assert_eq!(witness.parts[2][2].coeff(0).to_u64(), 12);
    }

    #[test]
    fn test_split_witness_multi_split() {
        const N: usize = 64;
        // z_parts: 6 polys (z⁰=[1,2,3], z¹=[4,5,6]), n=3, ν=3
        let z_parts = vec![
            make_poly::<N>(1),
            make_poly::<N>(2),
            make_poly::<N>(3),
            make_poly::<N>(4),
            make_poly::<N>(5),
            make_poly::<N>(6),
        ];
        // v: [100], m=1, μ=1
        let v = vec![make_poly::<N>(100)];

        let witness = split_witness(&z_parts, &v, 3, 3, 1, 1);

        assert_eq!(witness.num_parts(), 7, "r' = 2*3 + 1 = 7");
        assert_eq!(witness.rank(), 1, "n' = max(⌈3/3⌉, ⌈1/1⌉) = 1");

        // z⁰ split into 3 parts: [1], [2], [3]
        assert_eq!(witness.parts[0][0].coeff(0).to_u64(), 1);
        assert_eq!(witness.parts[1][0].coeff(0).to_u64(), 2);
        assert_eq!(witness.parts[2][0].coeff(0).to_u64(), 3);

        // z¹ split into 3 parts: [4], [5], [6]
        assert_eq!(witness.parts[3][0].coeff(0).to_u64(), 4);
        assert_eq!(witness.parts[4][0].coeff(0).to_u64(), 5);
        assert_eq!(witness.parts[5][0].coeff(0).to_u64(), 6);

        // v: [100]
        assert_eq!(witness.parts[6][0].coeff(0).to_u64(), 100);
    }

    #[test]
    fn test_split_witness_padding() {
        const N: usize = 64;
        // z⁰=[1,2,3,4], z¹=[5,6,7,8], n=4, ν=3
        // z0_per_part = ⌈4/3⌉ = 2, so parts: [1,2], [3,4], [5,6] for z⁰... wait
        // z⁰ = [1,2,3,4] split into 3 parts of ⌈4/3⌉=2: [1,2], [3,4], []
        // z¹ = [5,6,7,8] split into 3 parts of 2: [5,6], [7,8], []
        // v = [100, 200], m=2, μ=1, v_per_part=2: [100, 200]
        // n' = max(2, 2) = 2
        let z_parts = vec![
            make_poly::<N>(1),
            make_poly::<N>(2),
            make_poly::<N>(3),
            make_poly::<N>(4),
            make_poly::<N>(5),
            make_poly::<N>(6),
            make_poly::<N>(7),
            make_poly::<N>(8),
        ];
        let v = vec![make_poly::<N>(100), make_poly::<N>(200)];

        let witness = split_witness(&z_parts, &v, 4, 3, 1, 2);

        assert_eq!(witness.num_parts(), 7, "2*3 + 1 = 7");
        assert_eq!(witness.rank(), 2);

        // z⁰ split into 3 parts of 2: [1,2], [3,4], []
        assert_eq!(witness.parts[0][0].coeff(0).to_u64(), 1);
        assert_eq!(witness.parts[0][1].coeff(0).to_u64(), 2);
        assert_eq!(witness.parts[1][0].coeff(0).to_u64(), 3);
        assert_eq!(witness.parts[1][1].coeff(0).to_u64(), 4);
        // Third part of z⁰ is empty → padded to [0, 0]
        assert!(witness.parts[2][0].is_zero());
        assert!(witness.parts[2][1].is_zero());
    }

    #[test]
    #[should_panic(expected = "z_parts must have length 2n")]
    fn test_split_witness_bad_length() {
        const N: usize = 64;
        let z_parts = vec![make_poly::<N>(1)];
        let v: Vec<CyclotomicPolyRing<F, N>> = vec![];
        let _ = split_witness(&z_parts, &v, 3, 1, 1, 1);
    }
}
