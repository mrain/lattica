//! Garbage polynomial computation (§5.2 steps 3, 13).
//!
//! Computes quadratic garbage `g_ij = ⟨s_i, s_j⟩` and linear garbage
//! `h_ij = (⟨φ_i, s_j⟩ + ⟨φ_j, s_i⟩) / 2`, then balanced-decomposes.

use alloc::vec::Vec;

use grid_algebra::arith::ring::IntegerRing;
use grid_algebra::arith::ring::Ring;
use grid_algebra::poly::ring::{CyclotomicPolyRing, NegacyclicMulRing, PolyRing};

use crate::main_protocol::{DecomposedPolys, decompose_polys, garbage_count};
use crate::params::LabradorParams;
use crate::relation::LabradorWitness;

/// Result of garbage computation.
///
/// `g_polys` and `g_decomposed` are available immediately (g_ij independent of challenges).
/// `h_polys` and `h_decomposed` require aggregated φ_i from the aggregation step.
#[derive(Debug, Clone)]
pub struct GarbageData<R, const N: usize>
where
    R: IntegerRing<Uint = u64> + NegacyclicMulRing<N>,
{
    /// g_ij polynomials: upper-triangular, (r²+r)/2 entries.
    /// g_ij = ring dot product of s_i and s_j.
    pub g_polys: Vec<CyclotomicPolyRing<R, N>>,

    /// Decomposed g: balanced base b2 into t2 centered limbs.
    /// Flat length: garbage_count(r) * t2.
    pub g_decomposed: DecomposedPolys<CyclotomicPolyRing<R, N>>,

    /// h_ij polynomials: upper-triangular, (r²+r)/2 entries.
    /// h_ij = (⟨φ_i, s_j⟩ + ⟨φ_j, s_i⟩) · 2⁻¹ mod q
    /// Only computable AFTER aggregation determines φ_i.
    pub h_polys: Option<Vec<CyclotomicPolyRing<R, N>>>,

    /// Decomposed h: balanced base b1 into t1 centered limbs.
    /// Flat length: garbage_count(r) * t1.
    pub h_decomposed: Option<DecomposedPolys<CyclotomicPolyRing<R, N>>>,
}

impl<R, const N: usize> GarbageData<R, N>
where
    R: IntegerRing<Uint = u64> + NegacyclicMulRing<N>,
{
    /// Create from g_polys only (h not yet computed).
    pub fn from_g(g_polys: Vec<CyclotomicPolyRing<R, N>>, b2: u64, t2: usize) -> Self {
        let g_decomposed = decompose_polys(&g_polys, b2, t2);
        Self {
            g_polys,
            g_decomposed,
            h_polys: None,
            h_decomposed: None,
        }
    }

    /// Fill in h from aggregated φ_i.
    pub fn fill_h(
        &mut self,
        witness: &LabradorWitness<CyclotomicPolyRing<R, N>>,
        aggregated_phis: &[Vec<CyclotomicPolyRing<R, N>>],
        params: &LabradorParams,
    ) {
        let h_polys = compute_garbage_h(witness, aggregated_phis);
        let h_decomposed = decompose_polys(&h_polys, params.b1, params.t1);
        self.h_polys = Some(h_polys);
        self.h_decomposed = Some(h_decomposed);
    }
}

/// Compute g_ij = ⟨s_i, s_j⟩ for all upper-triangular pairs.
///
/// Ring dot product: `Σ_k s_i[k] · s_j[k]` where · is polynomial multiplication.
/// Result is a polynomial in R_q (NOT a scalar).
pub fn compute_garbage_g<R, const N: usize>(
    witness: &LabradorWitness<CyclotomicPolyRing<R, N>>,
) -> Vec<CyclotomicPolyRing<R, N>>
where
    R: IntegerRing<Uint = u64> + NegacyclicMulRing<N>,
{
    let r = witness.num_parts();
    let garbage_len = garbage_count(r);
    let mut g_polys = Vec::with_capacity(garbage_len);

    for i in 0..r {
        for j in i..r {
            let g_ij = Ring::dot_product(&witness.parts[i], &witness.parts[j]);
            g_polys.push(g_ij);
        }
    }
    g_polys
}

/// Compute h_ij = (⟨φ_i, s_j⟩ + ⟨φ_j, s_i⟩) · 2⁻¹ mod q.
///
/// φ_i ∈ R_q^n is the aggregated linear term for witness part i.
/// 2⁻¹ mod q exists because q is always odd.
///
/// REQUIRES: `2` is invertible in the coefficient ring (i.e., `q` is odd).
pub fn compute_garbage_h<R, const N: usize>(
    witness: &LabradorWitness<CyclotomicPolyRing<R, N>>,
    aggregated_phis: &[Vec<CyclotomicPolyRing<R, N>>],
) -> Vec<CyclotomicPolyRing<R, N>>
where
    R: IntegerRing<Uint = u64> + NegacyclicMulRing<N>,
{
    let r = witness.num_parts();
    let garbage_len = garbage_count(r);
    let q = R::modulus();
    debug_assert!(q % 2 == 1, "modulus must be odd for division by 2");
    // q is always odd, so (q+1)/2 = 2⁻¹ mod q.
    // Equivalent to q.div_ceil(2), spelled out for clarity.
    let two_inv = R::from_u64(q / 2 + 1);

    let mut h_polys = Vec::with_capacity(garbage_len);

    for i in 0..r {
        for j in i..r {
            let term1 = Ring::dot_product(&aggregated_phis[i], &witness.parts[j]);
            let term2 = Ring::dot_product(&aggregated_phis[j], &witness.parts[i]);
            let sum = term1 + term2;
            // Multiply each coefficient by 2⁻¹ mod q
            let mut h_ij = CyclotomicPolyRing::<R, N>::zero();
            for k in 0..N {
                let c = sum.coeff(k) * two_inv.clone();
                h_ij.set_coeff(k, c);
            }
            h_polys.push(h_ij);
        }
    }
    h_polys
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    use grid_algebra::arith::prime::PrimeField;
    use grid_algebra::arith::ring::Ring;

    type F = PrimeField<12289>;

    fn fake_params() -> LabradorParams {
        use crate::params::{ChallengeProfile, JLProfile};
        LabradorParams {
            jl: JLProfile::default(),
            challenge: ChallengeProfile::paper_default(),
            security_bits: 8,
            soundness_error: 0.0,
            l: 1,
            arith_p: 274177,
            n: 4,
            r: 3,
            beta: 100.0,
            d: 64,
            q: 12289.0,
            sigma: 1.0,
            b: 2,
            b1: 16,
            b2: 16,
            t1: 4,
            t2: 4,
            kappa: 2,
            kappa1: 2,
            kappa2: 2,
            gamma: 100.0,
            gamma1_sq: 10_000,
            gamma2_sq: 10_000,
            beta_prime: 100.0,
            nu: 1,
            mu: 1,
            num_levels: 1,
        }
    }

    fn make_witness(r: usize, n: usize) -> LabradorWitness<CyclotomicPolyRing<F, 64>> {
        LabradorWitness {
            parts: (0..r)
                .map(|i| {
                    (0..n)
                        .map(|j| {
                            let mut p = CyclotomicPolyRing::<F, 64>::zero();
                            p.set_coeff(0, F::from_u64((i * n + j + 1) as u64));
                            p
                        })
                        .collect()
                })
                .collect(),
        }
    }

    #[test]
    fn test_garbage_g_count() {
        let params = fake_params();
        let witness = make_witness(params.r, params.n);
        let g_polys = compute_garbage_g(&witness);
        assert_eq!(g_polys.len(), garbage_count(params.r));
        assert_eq!(garbage_count(3), 6);
    }

    #[test]
    fn test_garbage_g_zero_witness() {
        let params = fake_params();
        let zero_witness = LabradorWitness {
            parts: (0..params.r)
                .map(|_| vec![CyclotomicPolyRing::<F, 64>::zero(); params.n])
                .collect(),
        };
        let g_polys = compute_garbage_g(&zero_witness);
        for g in &g_polys {
            assert!(g.is_zero(), "g_ij should be zero for zero witness");
        }
    }

    #[test]
    fn test_garbage_h_zero_phi() {
        let params = fake_params();
        let witness = make_witness(params.r, params.n);
        let zero_phis: Vec<Vec<CyclotomicPolyRing<F, 64>>> = (0..params.r)
            .map(|_| vec![CyclotomicPolyRing::<F, 64>::zero(); params.n])
            .collect();

        let h_polys = compute_garbage_h(&witness, &zero_phis);
        assert_eq!(h_polys.len(), garbage_count(params.r));
        for h in &h_polys {
            assert!(h.is_zero(), "h_ij should be zero for zero phi");
        }
    }

    #[test]
    fn test_garbage_division_by_2() {
        // Verify that x * 2⁻¹ * 2 ≡ x mod q for odd q
        let q = F::modulus();
        assert!(q % 2 == 1, "q must be odd");
        let two_inv = F::from_u64(q.div_ceil(2));
        let two = F::from_u64(2);

        for v in 0..100u64 {
            let val = F::from_u64(v);
            let halved = val * two_inv;
            let doubled = halved * two;
            assert_eq!(
                doubled.to_u64(),
                v,
                "division by 2 roundtrip failed for {}",
                v
            );
        }
    }

    #[test]
    fn test_garbage_data_from_g_and_fill_h() {
        let params = fake_params();
        let witness = make_witness(params.r, params.n);
        let g_polys = compute_garbage_g(&witness);

        let mut garbage = GarbageData::from_g(g_polys, params.b2, params.t2);
        assert_eq!(garbage.g_polys.len(), garbage_count(params.r));
        assert_eq!(garbage.g_decomposed.num_polys, garbage_count(params.r));
        assert_eq!(garbage.g_decomposed.num_limbs, params.t2);
        assert_eq!(garbage.g_decomposed.base, params.b2);
        assert!(garbage.h_polys.is_none());
        assert!(garbage.h_decomposed.is_none());

        let aggregated_phis: Vec<Vec<CyclotomicPolyRing<F, 64>>> = (0..params.r)
            .map(|i| {
                (0..params.n)
                    .map(|j| {
                        let mut p = CyclotomicPolyRing::<F, 64>::zero();
                        p.set_coeff(0, F::from_u64((i + j + 1) as u64));
                        p
                    })
                    .collect()
            })
            .collect();

        garbage.fill_h(&witness, &aggregated_phis, &params);

        assert!(garbage.h_polys.is_some());
        assert!(garbage.h_decomposed.is_some());
        let h_polys = garbage.h_polys.as_ref().unwrap();
        let h_decomposed = garbage.h_decomposed.as_ref().unwrap();
        assert_eq!(h_polys.len(), garbage_count(params.r));
        assert_eq!(h_decomposed.num_polys, garbage_count(params.r));
        assert_eq!(h_decomposed.num_limbs, params.t1);
        assert_eq!(h_decomposed.base, params.b1);
    }

    #[test]
    fn test_garbage_decompose_roundtrip_and_limb_bounds() {
        let params = fake_params();
        let witness = make_witness(params.r, params.n);
        let g_polys = compute_garbage_g(&witness);

        let garbage = GarbageData::from_g(g_polys.clone(), params.b2, params.t2);

        let reconstructed = garbage.g_decomposed.reconstruct();
        assert_eq!(reconstructed.len(), g_polys.len());
        for (orig, recon) in g_polys.iter().zip(reconstructed.iter()) {
            assert_eq!(orig, recon, "g roundtrip mismatch");
        }

        let half_base = garbage.g_decomposed.base as i128 / 2;
        for limb in &garbage.g_decomposed.flat {
            for i in 0..64 {
                let v = limb.coeff(i).to_u64();
                let q = F::modulus();
                let centered = if v as i128 > half_base {
                    (v as i128) - q as i128
                } else {
                    v as i128
                };
                assert!(
                    centered.abs() <= half_base,
                    "g limb coeff {} out of centered range",
                    v
                );
            }
        }
    }
}
