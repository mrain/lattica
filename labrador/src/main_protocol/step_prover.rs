//! Prover orchestrator for one LaBRADOR recursion step (§5.2).
//!
//! Follows the exact transcript binding order:
//! ```text
//! u1 → absorb → JL seed → p → absorb → ψ,ω → b'' → absorb → α,β → u2 → absorb → c_i → z
//! ```

use alloc::format;
use alloc::vec::Vec;

use grid_algebra::arith::ring::{IntegerRing, Ring};
use grid_algebra::lattice::types::RingVec;
use grid_algebra::poly::ring::PolyRing;
use grid_algebra::poly::ring::{CyclotomicPolyRing, NegacyclicMulRing};
use grid_serialize::{CanonicalDeserialize, CanonicalSerialize};
use grid_std::UniformRand;
use grid_transcript::Transcript;

use crate::crs::CommitKey;
use crate::error::LabradorError;
use crate::jl::JLMatrix;
use crate::main_protocol::aggregation::{
    aggregate_step2_flat, compute_b_double_prime, sample_step1_challenges,
};
use crate::main_protocol::amortization::compute_amortized_witness;
use crate::main_protocol::garbage::{compute_garbage_g, compute_garbage_h};

use crate::main_protocol::{
    DecomposedPolys, challenge_poly, decompose_polys, jl_rows_flat_to_conjugated_polys,
};
use crate::params::LabradorParams;

use crate::relation::{LabradorStatement, LabradorWitness};

/// Public proof data for one recursion level.
///
/// Only the verifier-visible messages are included here. The witness data
/// (`z`, decomposed limbs) lives in [`LevelPrivateWitness`] and is not
/// serialized. The verifier derives the target relation shape from params.
#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct LevelProof<R, const N: usize>
where
    R: IntegerRing<Uint = u64> + NegacyclicMulRing<N> + UniformRand,
{
    pub u1: RingVec<CyclotomicPolyRing<R, N>>,
    pub jl_seed: [u8; 32],
    pub jl_retry: u32,
    pub p: Vec<R>,
    pub b_double_prime: Vec<CyclotomicPolyRing<R, N>>,
    pub u2: RingVec<CyclotomicPolyRing<R, N>>,
}

/// Private witness for one recursion level.
///
/// Not serialized into the proof bytes. Used by the prover to build the next
/// level's target relation. The verifier does not have access to this data;
/// it derives target statements from public proof messages and verifies
/// the proof chain.
#[derive(Debug, Clone)]
pub struct LevelPrivateWitness<R, const N: usize>
where
    R: IntegerRing<Uint = u64> + NegacyclicMulRing<N> + UniformRand,
{
    pub z: Vec<CyclotomicPolyRing<R, N>>,
    pub t_decomposed: DecomposedPolys<CyclotomicPolyRing<R, N>>,
    pub g_decomposed: DecomposedPolys<CyclotomicPolyRing<R, N>>,
    pub h_decomposed: DecomposedPolys<CyclotomicPolyRing<R, N>>,
}

/// Complete prover output for one recursion step.
///
/// Contains the public [`LevelProof`] sent to the verifier, the private
/// [`LevelPrivateWitness`] used for target derivation across levels, and
/// aggregation data (`AggregatedFunction` and amortization challenges)
/// needed to build the target relation for the next level via
/// [`crate::recursion::build_target_relation`].
#[derive(Debug, Clone)]
pub struct StepProverOutput<R, const N: usize>
where
    R: IntegerRing<Uint = u64> + NegacyclicMulRing<N> + UniformRand,
{
    /// Public proof data sent to the verifier.
    pub level_proof: LevelProof<R, N>,
    /// Private witness — not serialized, used for target derivation across levels.
    pub private_witness: LevelPrivateWitness<R, N>,
    /// Aggregated quadratic function from step 2 aggregation.
    /// Contains the `phi` terms used in garbage h computation.
    pub aggregated: crate::main_protocol::aggregation::AggregatedFunction<R, N>,
    /// Amortization challenges c_1..c_r used to build the target relation.
    pub challenges: Vec<CyclotomicPolyRing<R, N>>,
}

/// Prove one recursion step.
///
/// Returns [`StepProverOutput`] containing the [`LevelProof`] sent to the
/// verifier, the [`LevelPrivateWitness`] for target derivation, and the
/// aggregation data (`AggregatedFunction` and amortization challenges)
/// needed to build the target relation for the next level via
/// [`crate::recursion::build_target_relation`].
pub fn prove_step<R, const N: usize, T>(
    key: &CommitKey<R, N>,
    statement: &LabradorStatement<CyclotomicPolyRing<R, N>>,
    witness: &LabradorWitness<CyclotomicPolyRing<R, N>>,
    params: &LabradorParams,
    transcript: &mut T,
    level: usize,
) -> Result<StepProverOutput<R, N>, LabradorError>
where
    R: IntegerRing<Uint = u64>
        + NegacyclicMulRing<N>
        + UniformRand
        + CanonicalSerialize
        + CanonicalDeserialize,
    T: Transcript,
{
    // ── Commit phase ──

    // 1. Inner commitments + balanced decomposition
    let t_decomposed = key.inner_commit_decomposed(witness, params);

    // 2. Garbage g + balanced decomposition (bound early, before challenges)
    let g_polys = compute_garbage_g(witness);
    let g_decomposed = decompose_polys(&g_polys, params.b2, params.t2);

    // 3. Outer commitment u1 = B·t + C·g
    let u1 = key.outer_commit_u1_slice(&t_decomposed.flat, &g_decomposed.flat);
    transcript.append_serializable(b"labrador_u1", &u1)?;

    // ── Project phase ──

    // 4. JL projection with rejection sampling retry
    // Per paper §5.2: project only first r parts (Σ_i Π_i s_i for i ∈ [r])
    let p_coeffs: Vec<Vec<R>> = witness
        .parts
        .iter()
        .take(params.r)
        .map(|part| {
            part.iter()
                .flat_map(|poly| poly.coeffs().iter().cloned())
                .collect()
        })
        .collect();

    const JL_MAX_RETRY: u32 = 100;
    let mut retry: u32 = 0;
    let (jl_matrix, p, jl_rows_flat) = loop {
        if retry >= JL_MAX_RETRY {
            return Err(LabradorError::Prover(format!(
                "JL projection retry exceeded {} attempts (parameters may be too tight)",
                JL_MAX_RETRY
            )));
        }
        let retry_bytes = retry.to_le_bytes();
        transcript.append_bytes(b"labrador_jl_retry", &retry_bytes)?;
        let jl_seed_bytes = transcript.challenge_bytes(b"labrador_jl_seed_squeeze", 32)?;
        let jl_seed_candidate: [u8; 32] = jl_seed_bytes
            .try_into()
            .expect("challenge_bytes returns exact length");
        let jl_matrix = JLMatrix::from_seed(&params.jl, params.n * params.d, jl_seed_candidate);

        // Combined projection + extraction: single pass over keystream.
        let p_slices: Vec<&[R]> = p_coeffs.iter().map(|v| v.as_slice()).collect();
        let (candidate_p, jl_flat) =
            jl_matrix.project_and_extract_jl_rows(&p_slices, params.r, params.d);

        if crate::jl::verify_norm(&params.jl, &candidate_p, params.beta) {
            break (jl_matrix, candidate_p, jl_flat);
        }
        retry += 1;
    };
    let jl_retry = retry;
    // Commit seed and projection to transcript
    let jl_seed = *jl_matrix.seed();
    transcript.append_bytes(b"labrador_jl_seed_bind", &jl_seed)?;
    transcript.append_serializable(b"labrador_p", &p)?;

    // 5. Aggregation step 1 — verifier samples ψ, ω
    let (psi, omega) = sample_step1_challenges(
        statement.f_prime.len(),
        params.jl.rows,
        params.q as u64,
        transcript,
    )?;

    // Convert flat JL entries to conjugated polynomials (single allocation)
    let q = R::modulus();
    let jl_rows_poly = jl_rows_flat_to_conjugated_polys::<R, N>(
        &jl_rows_flat,
        params.jl.rows,
        params.r,
        params.n,
        q,
    );

    // Compute JL polynomial evaluations (only first r parts per paper §5.2)
    let jl_evals: Vec<CyclotomicPolyRing<R, N>> = (0..params.jl.rows)
        .map(|m| {
            let mut jl_eval = CyclotomicPolyRing::<R, N>::zero();
            for (i, s_i) in witness.parts.iter().enumerate().take(params.r) {
                for (j, s_ij) in s_i.iter().enumerate() {
                    jl_eval += jl_rows_poly.get(m, i, j) * s_ij;
                }
            }
            jl_eval
        })
        .collect();

    // 6. Compute b'' and send
    let f_prime_evals: Vec<_> = statement
        .f_prime
        .iter()
        .map(|f| f.evaluate(witness))
        .collect();
    let f_prime_b: Vec<_> = statement.f_prime.iter().map(|f| f.b().clone()).collect();
    let b_double_prime =
        compute_b_double_prime(&f_prime_evals, &f_prime_b, &jl_evals, &psi, &omega);
    transcript.append_serializable(b"labrador_bdp", &b_double_prime)?;

    // 7. Aggregation step 2 — verifier samples α, β (polynomials)
    let level_bytes = (level as u32).to_le_bytes();
    transcript.append_bytes(b"labrador_level", &level_bytes)?;
    let alpha: Vec<CyclotomicPolyRing<R, N>> = (0..statement.f.len())
        .map(|_| challenge_poly::<CyclotomicPolyRing<R, N>, _>(transcript, b"labrador_alpha"))
        .collect::<Result<_, _>>()?;
    let beta: Vec<CyclotomicPolyRing<R, N>> = (0..psi.len())
        .map(|_| challenge_poly::<CyclotomicPolyRing<R, N>, _>(transcript, b"labrador_beta"))
        .collect::<Result<_, _>>()?;

    let aggregated = aggregate_step2_flat(
        &statement.f,
        &statement.f_prime,
        &alpha,
        &beta,
        &b_double_prime,
        &psi,
        &omega,
        &jl_rows_poly,
    );

    // ── Post-aggregation phase ──

    // 8. Garbage h (φ_i known from aggregation)
    let h_polys = compute_garbage_h(witness, &aggregated.phi);
    let h_decomposed = decompose_polys(&h_polys, params.b1, params.t1);

    // 9. Outer commitment u2 = D·h
    let u2 = key.outer_commit_u2_slice(&h_decomposed.flat);
    transcript.append_serializable(b"labrador_u2", &u2)?;

    // ── Amortize phase ──

    // 10. Amortization — challenges from transcript, compute z
    let amort = compute_amortized_witness(witness, params, transcript, level)?;

    // Build outputs
    let level_proof = LevelProof {
        u1,
        jl_seed,
        jl_retry,
        p,
        b_double_prime,
        u2,
    };
    let private_witness = LevelPrivateWitness {
        z: amort.z.clone(),
        t_decomposed,
        g_decomposed,
        h_decomposed,
    };

    Ok(StepProverOutput {
        level_proof,
        private_witness,
        aggregated,
        challenges: amort.challenges,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    use crate::params::{ChallengeProfile, JLProfile};
    use grid_algebra::arith::prime::PrimeField;

    type F = PrimeField<12289>;

    fn fake_params() -> LabradorParams {
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

    fn fake_witness(params: &LabradorParams) -> LabradorWitness<CyclotomicPolyRing<F, 64>> {
        LabradorWitness {
            parts: (0..params.r)
                .map(|_| {
                    (0..params.n)
                        .map(|i| {
                            let mut p = CyclotomicPolyRing::<F, 64>::zero();
                            p.set_coeff(0, F::from_u64((i + 1) as u64));
                            p
                        })
                        .collect()
                })
                .collect(),
        }
    }

    fn fake_statement(params: &LabradorParams) -> LabradorStatement<CyclotomicPolyRing<F, 64>> {
        LabradorStatement {
            f: vec![QuadraticFunction::from_parts(
                vec![],
                (0..params.r)
                    .map(|_| vec![CyclotomicPolyRing::<F, 64>::zero(); params.n])
                    .collect(),
                CyclotomicPolyRing::<F, 64>::zero(),
            )],
            f_prime: vec![],
        }
    }

    use crate::relation::QuadraticFunction;
    use grid_transcript::hash::ShakeTranscript;

    #[test]
    fn test_prove_step_shapes() {
        let mut rng = grid_std::test_rng();
        let params = fake_params();
        let key = CommitKey::<F, 64>::generate_from_params(&mut rng, &params);
        let witness = fake_witness(&params);
        let statement = fake_statement(&params);

        let mut transcript = ShakeTranscript::default();
        let output =
            prove_step::<F, 64, _>(&key, &statement, &witness, &params, &mut transcript, 0)
                .expect("prove_step should succeed");
        let proof = &output.level_proof;

        // LevelProof shapes (public fields only)
        assert_eq!(proof.u1.len(), params.kappa1);
        assert_eq!(proof.p.len(), params.jl.rows);
        assert_eq!(
            proof.b_double_prime.len(),
            crate::main_protocol::aggregation::num_agg_batches(params.q as u64)
        );
        assert_eq!(proof.u2.len(), params.kappa2);
        // Private witness shapes
        let pw = &output.private_witness;
        assert_eq!(pw.z.len(), params.n);
        assert_eq!(
            pw.t_decomposed.flat.len(),
            params.r * params.kappa * params.t1
        );
        let garbage_len = crate::main_protocol::garbage_count(params.r);
        assert_eq!(pw.g_decomposed.flat.len(), garbage_len * params.t2);
        assert_eq!(pw.h_decomposed.flat.len(), garbage_len * params.t1);
    }
}
