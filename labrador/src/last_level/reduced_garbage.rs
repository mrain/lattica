//! Reduced garbage computation for the last recursion level (§5.6).
//!
//! In the last level, the witness is split into r = ν + μ parts (z not decomposed),
//! and garbage is computed per-challenge-round rather than for all r²+r pairs.
//! This produces 2r-1 h-polynomials and 2ν+1 g-polynomials (vs (r²+r)/2 each).

use alloc::vec::Vec;

use grid_algebra::arith::ring::IntegerRing;
use grid_algebra::arith::ring::Ring;
use grid_algebra::poly::ring::{CyclotomicPolyRing, NegacyclicMulRing};

use crate::relation::LabradorWitness;

/// Compute reduced h-garbage for challenge round i (1-indexed, i = 1..r).
/// Returns (h_{2i-1}, h_{2i}).
///
/// For i=1: returns (zero, h₂). The prover skips serializing the zero h₁ term
/// into the transcript (h₁ is an empty sum, always zero).
///
/// witness: full witness, parts accessed via 0-indexed permutation (see translation table)
/// aggregated_phi: `phi[part_idx]` is the aggregated φ for witness part `part_idx` (0-indexed)
/// challenges_so_far: c₁..c_{i-1} as `&[CyclotomicPolyRing<R, N>]` (0-indexed: `challenges[0..i-1]`)
/// round_i: i (1-indexed). Maps to 0-indexed witness part via pi = if i <= mu: nu+i-1 else: i-mu-1
pub fn compute_h_round<R, const N: usize>(
    witness: &LabradorWitness<CyclotomicPolyRing<R, N>>,
    aggregated_phi: &[Vec<CyclotomicPolyRing<R, N>>],
    challenges_so_far: &[CyclotomicPolyRing<R, N>],
    round_i: usize,
    nu: usize,
    mu: usize,
) -> (CyclotomicPolyRing<R, N>, CyclotomicPolyRing<R, N>)
where
    R: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    let pi = super::witness_part_index(round_i, nu, mu);
    let s_pi = &witness.parts[pi];
    let phi_pi = &aggregated_phi[pi];

    // h_{2i-1} = Σ_{j=1}^{i-1} (⟨φ_{π(i)}, s_{π(j)}⟩ + ⟨φ_{π(j)}, s_{π(i)}⟩) · c_j
    let mut h_odd = CyclotomicPolyRing::<R, N>::zero();

    for (k, c_j) in challenges_so_far.iter().enumerate() {
        let pj = super::witness_part_index(k + 1, nu, mu);
        let s_pj = &witness.parts[pj];
        let phi_pj = &aggregated_phi[pj];

        let term1 = Ring::dot_product(phi_pi, s_pj);
        let term2 = Ring::dot_product(phi_pj, s_pi);
        let sum = term1 + term2;

        // Multiply by c_j
        let scaled = sum * c_j;
        h_odd += scaled;
    }

    // h_{2i} = ⟨φ_{π(i)}, s_{π(i)}⟩
    let h_even = Ring::dot_product(phi_pi, s_pi);

    (h_odd, h_even)
}

/// Compute reduced g₀: v×v block only.
/// Called once, before the first z-challenge (round 2μ+1).
///
/// g₀ = Σᵢ₌₁..ᵐ Σⱼ₌₁..ᵐ ⟨s_{ν+i}, s_{ν+j}⟩ · cᵢ · cⱼ
/// The v×z and z×v cross-terms belong in g_{2j-1} (computed by compute_g_pair), NOT in g₀.
///
/// challenges: all μ v-challenges c₁..c_μ (0-indexed: challenges[0..mu]).
/// Returns g₀.
pub fn compute_g0<R, const N: usize>(
    witness: &LabradorWitness<CyclotomicPolyRing<R, N>>,
    v_challenges: &[CyclotomicPolyRing<R, N>],
    nu: usize,
    mu: usize,
) -> CyclotomicPolyRing<R, N>
where
    R: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    let mut g0 = CyclotomicPolyRing::<R, N>::zero();

    for i in 0..mu {
        let vi = nu + i;
        for j in 0..mu {
            let vj = nu + j;
            let dot = Ring::dot_product(&witness.parts[vi], &witness.parts[vj]);
            let scaled = dot * &v_challenges[i] * &v_challenges[j];
            g0 += scaled;
        }
    }

    g0
}

/// Compute reduced g pair (g_{2j-1}, g_{2j}) for z-index j (1-indexed, j = 1..ν).
/// Called during challenge rounds μ+1..μ+ν (the z-challenge phase).
///
/// witness: full witness, z-parts at witness.parts[0..nu], v-parts at witness.parts[nu..r]
/// challenges_so_far: v-challenges c₁..c_μ followed by known z-challenges c_{μ+1}..c_{μ+j-1}
/// z_index_j: j (1-indexed). z-part j maps to witness.parts[j-1].
/// nu: number of z-parts, mu: number of v-parts
pub fn compute_g_pair<R, const N: usize>(
    witness: &LabradorWitness<CyclotomicPolyRing<R, N>>,
    challenges_so_far: &[CyclotomicPolyRing<R, N>],
    z_index_j: usize,
    nu: usize,
    mu: usize,
) -> (CyclotomicPolyRing<R, N>, CyclotomicPolyRing<R, N>)
where
    R: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    let z_j = z_index_j - 1;

    // g_{2j} = ⟨sⱼ, sⱼ⟩
    let g_even = Ring::dot_product(&witness.parts[z_j], &witness.parts[z_j]);

    // g_{2j-1} = Σᵢ₌₁..ᵐ (⟨s_{ν+i}, sⱼ⟩ + ⟨sⱼ, s_{ν+i}⟩) · cᵢ
    //          + Σₖ₌₁..ⱼ₋₁ (⟨sₖ, sⱼ⟩ + ⟨sⱼ, sₖ⟩) · c_{μ+k}
    let mut g_odd = CyclotomicPolyRing::<R, N>::zero();

    // v-challenge terms: c₁..c_μ
    for (i, c_i) in challenges_so_far.iter().enumerate().take(mu) {
        let vi = nu + i;
        let cross = Ring::dot_product(&witness.parts[vi], &witness.parts[z_j])
            + Ring::dot_product(&witness.parts[z_j], &witness.parts[vi]);
        g_odd += cross * c_i;
    }

    // z-challenge terms: c_{μ+1}..c_{μ+j-1} (already known from previous rounds)
    let num_known_z = challenges_so_far.len() - mu;
    for k in 0..num_known_z {
        let z_k = k;
        let cross = Ring::dot_product(&witness.parts[z_k], &witness.parts[z_j])
            + Ring::dot_product(&witness.parts[z_j], &witness.parts[z_k]);
        let c_idx = mu + k;
        g_odd += cross * &challenges_so_far[c_idx];
    }

    (g_odd, g_even)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::last_level::witness_part_index;
    use alloc::vec;

    use grid_algebra::arith::prime::PrimeField;
    use grid_algebra::arith::ring::Ring;
    use grid_algebra::poly::ring::PolyRing;

    type F = PrimeField<12289>;

    fn make_poly<const N: usize>(v: u64) -> CyclotomicPolyRing<F, N> {
        let mut p = CyclotomicPolyRing::<F, N>::zero();
        p.set_coeff(0, F::from_u64(v));
        p
    }

    fn make_witness<const N: usize>(
        r: usize,
        n: usize,
    ) -> LabradorWitness<CyclotomicPolyRing<F, N>> {
        LabradorWitness {
            parts: (0..r)
                .map(|i| {
                    (0..n)
                        .map(|j| make_poly::<N>((i * n + j + 1) as u64))
                        .collect()
                })
                .collect(),
        }
    }

    #[test]
    fn test_compute_h_round_first_round() {
        const N: usize = 64;
        let nu = 1;
        let mu = 1;
        let n = 4;

        let witness = make_witness::<N>(nu + mu, n);
        let aggregated_phi: Vec<Vec<CyclotomicPolyRing<F, N>>> = (0..nu + mu)
            .map(|i| {
                (0..n)
                    .map(|j| make_poly::<N>((10 * (i + 1) + j) as u64))
                    .collect()
            })
            .collect();

        // Round 1: no prior challenges, h₁ should be zero
        let (h_odd, h_even) = compute_h_round::<F, N>(&witness, &aggregated_phi, &[], 1, nu, mu);

        assert!(h_odd.is_zero(), "h₁ should be zero (empty sum)");
        assert!(!h_even.is_zero(), "h₂ should be nonzero");
    }

    #[test]
    fn test_compute_h_round_second_round() {
        const N: usize = 64;
        let nu = 1;
        let mu = 1;
        let n = 4;

        let witness = make_witness::<N>(nu + mu, n);
        let aggregated_phi: Vec<Vec<CyclotomicPolyRing<F, N>>> = (0..nu + mu)
            .map(|i| {
                (0..n)
                    .map(|j| make_poly::<N>((10 * (i + 1) + j) as u64))
                    .collect()
            })
            .collect();

        let c1 = make_poly::<N>(5);

        // Round 2: one prior challenge c₁
        let (h_odd, h_even) = compute_h_round::<F, N>(&witness, &aggregated_phi, &[c1], 2, nu, mu);

        assert!(
            !h_odd.is_zero(),
            "h₃ should be nonzero with prior challenge"
        );
        assert!(!h_even.is_zero(), "h₄ should be nonzero");
    }

    #[test]
    fn test_compute_g0_basic() {
        const N: usize = 64;
        let nu = 1;
        let mu = 1;
        let n = 4;

        let witness = make_witness::<N>(nu + mu, n);
        let c1 = make_poly::<N>(3);

        let g0 = compute_g0::<F, N>(&witness, &[c1], nu, mu);

        // g₀ = ⟨s_{ν+1}, s_{ν+1}⟩ · c₁ · c₁ = ⟨s₁, s₁⟩ · 9
        let expected =
            Ring::dot_product(&witness.parts[nu], &witness.parts[nu]) * make_poly::<N>(9);
        assert_eq!(g0, expected);
    }

    #[test]
    fn test_compute_g_pair_first_z() {
        const N: usize = 64;
        let nu = 2;
        let mu = 1;
        let n = 4;

        let witness = make_witness::<N>(nu + mu, n);
        let c1 = make_poly::<N>(2);

        // z-index 1: only v-challenge terms
        let (g_odd, g_even) =
            compute_g_pair::<F, N>(&witness, core::slice::from_ref(&c1), 1, nu, mu);

        // g₂ = ⟨s₀, s₀⟩
        let expected_even = Ring::dot_product(&witness.parts[0], &witness.parts[0]);
        assert_eq!(g_even, expected_even);

        // g₁ = (⟨s_{ν}, s₀⟩ + ⟨s₀, s_{ν}⟩) · c₁
        let vi = nu;
        let cross = Ring::dot_product(&witness.parts[vi], &witness.parts[0])
            + Ring::dot_product(&witness.parts[0], &witness.parts[vi]);
        let expected_odd = cross * c1;
        assert_eq!(g_odd, expected_odd);
    }

    #[test]
    fn test_witness_part_index() {
        let nu = 2;
        let mu = 1;

        // Round 1 (v-challenge): maps to v-part 0 → nu + 0 = 2
        assert_eq!(witness_part_index(1, nu, mu), 2);

        // Round 2 (z-challenge): maps to z-part 0 → 0
        assert_eq!(witness_part_index(2, nu, mu), 0);

        // Round 3 (z-challenge): maps to z-part 1 → 1
        assert_eq!(witness_part_index(3, nu, mu), 1);
    }

    #[test]
    fn test_compute_g0_zero_witness() {
        const N: usize = 64;
        let nu = 1;
        let mu = 1;

        let zero_witness = LabradorWitness {
            parts: vec![
                vec![CyclotomicPolyRing::<F, N>::zero(); 4],
                vec![CyclotomicPolyRing::<F, N>::zero(); 4],
            ],
        };
        let c1 = make_poly::<N>(5);

        let g0 = compute_g0::<F, N>(&zero_witness, &[c1], nu, mu);
        assert!(g0.is_zero(), "g₀ should be zero for zero witness");
    }

    #[test]
    fn test_prove_verify_last_level_roundtrip() {
        use grid_algebra::lattice::types::RingMat;
        use grid_transcript::hash::ShakeTranscript;

        use crate::crs::CommitKey;
        use crate::last_level::{prove_last_level, verify_last_level};
        use crate::params::{ChallengeProfile, JLProfile, LabradorParams};
        use crate::relation::{LabradorStatement, LabradorWitness, QuadraticFunction};

        const N: usize = 64;

        let nu = 1;
        let mu = 1;
        let r_last = nu + mu;
        let n_last = 4;
        let kappa = 2;
        let beta_current = 100.0;

        let params = LabradorParams {
            jl: JLProfile::default(),
            challenge: ChallengeProfile::paper_default(),
            security_bits: 8,
            soundness_error: 0.0,
            l: 1,
            arith_p: 274177,
            n: 4,
            r: r_last,
            beta: beta_current,
            d: 64,
            q: 12289.0,
            sigma: 1.0,
            b: 2,
            b1: 16,
            b2: 16,
            t1: 4,
            t2: 4,
            kappa,
            kappa1: 2,
            kappa2: 2,
            gamma: 100.0,
            gamma1_sq: 40_000,
            gamma2_sq: 40_000,
            beta_prime: 300.0,
            nu,
            mu,
            num_levels: 1,
        };

        // Build CRS
        let mut rng = grid_std::test_rng();
        let key = CommitKey::<F, N>::generate_from_params(&mut rng, &params);

        // Build A matrix for inner commitments
        let a = RingMat::new(
            key.a.rows(),
            n_last,
            (0..key.a.rows() * n_last)
                .map(|i| {
                    let mut p = CyclotomicPolyRing::<F, N>::zero();
                    p.set_coeff(0, F::from_u64((i % 10 + 1) as u64));
                    p
                })
                .collect(),
        );

        // Build zero statement (all phi zero, b zero, no f_prime)
        let statement = LabradorStatement {
            f: vec![QuadraticFunction::from_parts(
                vec![],
                (0..r_last)
                    .map(|_| vec![CyclotomicPolyRing::<F, N>::zero(); n_last])
                    .collect(),
                CyclotomicPolyRing::<F, N>::zero(),
            )],
            f_prime: vec![],
        };

        // Build zero witness (passes JL norm check trivially)
        let witness = LabradorWitness {
            parts: (0..r_last)
                .map(|_| vec![CyclotomicPolyRing::<F, N>::zero(); n_last])
                .collect(),
        };

        // Prove and verify
        let mut prover_transcript = ShakeTranscript::default();
        let proof = prove_last_level::<F, N, _>(
            &a,
            &statement,
            &witness,
            &params,
            r_last,
            n_last,
            beta_current,
            &mut prover_transcript,
        )
        .expect("prove_last_level_coeff should succeed");

        let mut verifier_transcript = ShakeTranscript::default();
        let result = verify_last_level::<F, N, _>(
            &a,
            &statement,
            &proof,
            &params,
            r_last,
            n_last,
            beta_current,
            &mut verifier_transcript,
        );

        assert!(
            result.is_ok(),
            "verify_last_level should accept honest proof"
        );
    }

    #[test]
    fn test_verify_last_level_rejects_off_diagonal_a_ij() {
        use alloc::format;

        use grid_algebra::lattice::types::RingMat;
        use grid_transcript::hash::ShakeTranscript;

        use crate::crs::CommitKey;
        use crate::last_level::{prove_last_level, verify_last_level};
        use crate::params::{ChallengeProfile, JLProfile, LabradorParams};
        use crate::relation::{LabradorStatement, LabradorWitness, QuadraticFunction};

        const N: usize = 64;

        let nu = 1;
        let mu = 1;
        let r_last = nu + mu;
        let n_last = 4;
        let beta_current = 100.0;

        let params = LabradorParams {
            jl: JLProfile::default(),
            challenge: ChallengeProfile::paper_default(),
            security_bits: 8,
            soundness_error: 0.0,
            l: 1,
            arith_p: 274177,
            n: 4,
            r: r_last,
            beta: beta_current,
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
            gamma1_sq: 40_000,
            gamma2_sq: 40_000,
            beta_prime: 300.0,
            nu,
            mu,
            num_levels: 1,
        };

        let mut rng = grid_std::test_rng();
        let key = CommitKey::<F, N>::generate_from_params(&mut rng, &params);

        let a = RingMat::new(
            key.a.rows(),
            n_last,
            (0..key.a.rows() * n_last)
                .map(|_| CyclotomicPolyRing::<F, N>::zero())
                .collect(),
        );

        // Statement with off-diagonal a_ij (i != j) — should be rejected by last-level verifier
        let mut one = CyclotomicPolyRing::<F, N>::zero();
        one.set_coeff(0, F::from_u64(1));

        let statement = LabradorStatement {
            f: vec![QuadraticFunction::from_parts(
                vec![(0, 1, one.clone())], // Off-diagonal term: i=0, j=1
                (0..r_last)
                    .map(|_| vec![CyclotomicPolyRing::<F, N>::zero(); n_last])
                    .collect(),
                CyclotomicPolyRing::<F, N>::zero(),
            )],
            f_prime: vec![],
        };

        let witness = LabradorWitness {
            parts: (0..r_last)
                .map(|_| vec![CyclotomicPolyRing::<F, N>::zero(); n_last])
                .collect(),
        };

        let mut prover_transcript = ShakeTranscript::default();
        let proof = prove_last_level::<F, N, _>(
            &a,
            &statement,
            &witness,
            &params,
            r_last,
            n_last,
            beta_current,
            &mut prover_transcript,
        )
        .expect("prove_last_level_coeff should succeed");

        let mut verifier_transcript = ShakeTranscript::default();
        let result = verify_last_level::<F, N, _>(
            &a,
            &statement,
            &proof,
            &params,
            r_last,
            n_last,
            beta_current,
            &mut verifier_transcript,
        );

        assert!(
            result.is_err(),
            "verify_last_level should reject off-diagonal a_ij"
        );
        let err_str = format!("{}", result.err().unwrap());
        assert!(
            err_str.contains("diagonal-only"),
            "error should mention diagonal-only constraint, got: {}",
            err_str
        );
    }
}
