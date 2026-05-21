//! Top-level LaBRADOR prover (Phase 5).
//!
//! Orchestrates the full proof: R1CS reduction → recursive main protocol steps
//! → last-level proof. Produces a [`LabradorProof`] containing public messages
//! for each main level (`u1`, `jl_seed`, `jl_retry`, `p`, `b_double_prime`, `u2`)
//! plus the last-level proof (which includes `z` and reduced garbage). Private
//! openings (`z`, decomposed `t`/`g`/`h` limbs) are retained by the prover for
//! target derivation across recursion levels; the verifier derives target
//! statements from public messages alone.

use alloc::format;
use alloc::string::ToString;
use alloc::vec::Vec;

use grid_algebra::arith::ring::IntegerRing;
use grid_algebra::poly::ring::{CyclotomicPolyRing, NegacyclicMulRing};
use grid_serialize::{CanonicalDeserialize, CanonicalSerialize};
use grid_std::UniformRand;
use grid_transcript::Transcript;

use crate::crs::{CRS, CommitKey};
use crate::error::LabradorError;
use crate::last_level::prove_last_level;
use crate::main_protocol::aggregation::AggregatedFunction;
use crate::main_protocol::step_prover::{
    LevelPrivateWitness, LevelProof, StepProverOutput, prove_step,
};
use crate::params::LabradorParams;
use crate::proof::LabradorProof;
use crate::recursion::split::split_last_level_witness;
use crate::recursion::target_relation::build_last_level_target;
use crate::recursion::{
    RecursiveTarget, build_target_relation, compute_next_level_shape, split_witness,
};
use crate::relation::{LabradorStatement, LabradorWitness};
use crate::traits::LabradorProofRing;

/// Complete prover output: public proof (serializable) + private witnesses.
///
/// The [`LabradorProof`] only carries the public messages per level
/// (`u1`, `jl_seed`, `jl_retry`, `p`, `b_double_prime`, `u2`).
/// Private witnesses (`z`, decomposed limbs) are retained by the prover
/// for target derivation across recursion levels and debug inspection.
/// The verifier derives statements from public messages and verifies the
/// proof chain without access to private data.
#[derive(Debug, Clone)]
pub struct ProverOutput<R, const N: usize>
where
    R: IntegerRing<Canonical = u64> + NegacyclicMulRing<N> + UniformRand,
{
    pub proof: LabradorProof<R, N>,
    pub private_witnesses: Vec<LevelPrivateWitness<R, N>>,
}

/// Prove a LaBRADOR statement with multi-level recursion.
#[allow(clippy::too_many_arguments)]
pub fn prove<R, const N: usize, T>(
    crs: &CRS,
    params: &LabradorParams,
    statement: &LabradorStatement<CyclotomicPolyRing<R, N>>,
    witness: &LabradorWitness<CyclotomicPolyRing<R, N>>,
    num_main_levels: usize,
    transcript: &mut T,
) -> Result<LabradorProof<R, N>, LabradorError>
where
    R: LabradorProofRing<N>,
    T: Transcript,
{
    // Enforce proof depth matches public parameter profile
    if num_main_levels + 1 != params.num_levels {
        return Err(LabradorError::InvalidInput(format!(
            "num_main_levels ({}) + 1 != params.num_levels ({})",
            num_main_levels, params.num_levels
        )));
    }

    // Absorb public inputs before any protocol message (Fiat-Shamir soundness)
    crate::main_protocol::absorb_public_input(transcript, crs, statement, params)
        .map_err(|e| LabradorError::Internal(format!("Failed to absorb public input: {}", e)))?;

    let mut current_key = crs.expand(params);
    let mut levels = Vec::with_capacity(num_main_levels);
    let mut current_statement = statement.clone();
    let mut current_witness = witness.clone();

    let mut r_current = params.r;
    let mut n_current = params.n;
    let mut beta_current = params.beta;

    let mut a_last_opt: Option<_> = if num_main_levels == 0 {
        let last_crs = CRS::derive_last(transcript)
            .map_err(|e| LabradorError::Internal(format!("Last CRS derivation failed: {}", e)))?;
        Some(last_crs.expand_a::<R, N>(params.kappa, params.n))
    } else {
        None
    };

    for level in 0..num_main_levels {
        let mut level_params = params.clone();
        level_params.r = r_current;
        level_params.n = n_current;
        level_params.beta = beta_current;

        if level > 0 {
            let seed = transcript
                .challenge_bytes(b"labrador_crs_seed", 32)
                .map_err(|e| {
                    LabradorError::Internal(format!("CRS seed derivation failed: {:?}", e))
                })?;
            let seed_array: [u8; 32] = seed
                .try_into()
                .map_err(|_| LabradorError::Internal("CRS seed length mismatch".to_string()))?;
            current_key = CommitKey::from_seed(seed_array, &level_params);
        }

        let is_final_step = level + 1 == num_main_levels;

        let output = prove_step(
            &current_key,
            &current_statement,
            &current_witness,
            &level_params,
            transcript,
            level,
        )?;

        let target = if is_final_step {
            let r_last = params.nu + params.mu;
            let n_last = compute_n_last(&output, params, n_current);

            let last_crs = CRS::derive_last(transcript).map_err(|e| {
                LabradorError::Internal(format!("Last CRS derivation failed: {}", e))
            })?;
            a_last_opt = Some(last_crs.expand_a::<R, N>(params.kappa, n_last));

            let last_witness = build_last_level_witness(&output, params, n_current);

            build_last_level_target(
                &current_key,
                &output.level_proof.u1,
                &output.level_proof.u2,
                &output.challenges,
                &output.aggregated,
                &level_params,
                last_witness,
                r_last,
                n_last,
                r_current,
            )?
        } else {
            derive_target_for_prover(
                &current_key,
                &output.level_proof,
                &output.private_witness,
                &level_params,
                &output.aggregated,
                &output.challenges,
            )
        };

        levels.push(output.level_proof);
        current_statement = target.statement;
        current_witness = target.witness;
        r_current = target.r_prime;
        n_current = target.n_prime;
        beta_current = target.beta_prime;
    }

    let r_last = params.nu + params.mu;
    let a_for_last = a_last_opt
        .as_ref()
        .expect("a_last should be set after final main step");

    let last = prove_last_level(
        a_for_last,
        &current_statement,
        &current_witness,
        params,
        r_last,
        n_current,
        beta_current,
        transcript,
    )?;

    Ok(LabradorProof { levels, last })
}

/// Build last-level witness from prover output.
fn build_last_level_witness<R, const N: usize>(
    output: &StepProverOutput<R, N>,
    params: &LabradorParams,
    n_current: usize,
) -> LabradorWitness<CyclotomicPolyRing<R, N>>
where
    R: LabradorProofRing<N>,
{
    let n_last = compute_n_last(output, params, n_current);
    let pw = &output.private_witness;
    split_last_level_witness(
        &pw.z,
        &pw.t_decomposed.flat,
        &pw.g_decomposed.flat,
        &pw.h_decomposed.flat,
        params.nu,
        params.mu,
        n_last,
    )
}

/// Compute n_last from current witness rank and bundled v size.
fn compute_n_last<R, const N: usize>(
    output: &StepProverOutput<R, N>,
    params: &LabradorParams,
    n_current: usize,
) -> usize
where
    R: LabradorProofRing<N>,
{
    let pw = &output.private_witness;
    let v = bundle_v(
        &pw.t_decomposed.flat,
        &pw.g_decomposed.flat,
        &pw.h_decomposed.flat,
    );
    let m = v.len();
    n_current.div_ceil(params.nu).max(m.div_ceil(params.mu))
}

/// Derive the target relation for the next recursion level from prover output.
///
/// Decomposes z into binary parts, bundles t/g/h limbs, splits into r' = 2nu+mu
/// parts, then builds the target relation.
fn derive_target_for_prover<R, const N: usize>(
    key: &CommitKey<R, N>,
    proof: &LevelProof<R, N>,
    private_witness: &LevelPrivateWitness<R, N>,
    params: &LabradorParams,
    aggregated: &AggregatedFunction<R, N>,
    challenges: &[CyclotomicPolyRing<R, N>],
) -> RecursiveTarget<R, N>
where
    R: IntegerRing<Canonical = u64> + NegacyclicMulRing<N> + UniformRand,
{
    let z_parts = crate::recursion::decompose::decompose_z(&private_witness.z, params.b);
    let v = bundle_v(
        &private_witness.t_decomposed.flat,
        &private_witness.g_decomposed.flat,
        &private_witness.h_decomposed.flat,
    );
    let m = v.len();
    let (r_prime, n_prime) = compute_next_level_shape(params.n, m, params.nu, params.mu);
    let witness = split_witness(&z_parts, &v, params.n, params.nu, params.mu, n_prime);
    build_target_relation(
        key, &proof.u1, &proof.u2, challenges, aggregated, params, witness, r_prime, n_prime,
    )
}

/// Bundle t, g, h decomposed limbs into a flat v vector.
fn bundle_v<R, const N: usize>(
    t_decomposed: &[CyclotomicPolyRing<R, N>],
    g_decomposed: &[CyclotomicPolyRing<R, N>],
    h_decomposed: &[CyclotomicPolyRing<R, N>],
) -> Vec<CyclotomicPolyRing<R, N>>
where
    R: IntegerRing<Canonical = u64> + CanonicalSerialize + CanonicalDeserialize,
{
    let mut v = Vec::with_capacity(t_decomposed.len() + g_decomposed.len() + h_decomposed.len());
    v.extend_from_slice(t_decomposed);
    v.extend_from_slice(g_decomposed);
    v.extend_from_slice(h_decomposed);
    v
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::verify;
    use alloc::vec;

    use crate::params::{ChallengeProfile, JLProfile};
    use crate::relation::QuadraticFunction;
    use grid_algebra::arith::prime::PrimeField;
    use grid_algebra::arith::ring::{IntegerRing, Ring};
    use grid_algebra::poly::ring::{CyclotomicPolyRing, PolyRing};
    use grid_transcript::hash::ShakeTranscript;

    type F = PrimeField<12289>;
    const N: usize = 64;
    const Q: u64 = 12289;

    fn cp(v: u64) -> CyclotomicPolyRing<F, N> {
        let mut p = CyclotomicPolyRing::<F, N>::zero();
        p.set_coeff(0, F::from_u64(v));
        p
    }

    /// Build Fibonacci: x0=0, x1=1, x(i+1)=x(i)+x(i-1), r parts total.
    fn build_fib(
        r: usize,
        n: usize,
    ) -> (
        LabradorStatement<CyclotomicPolyRing<F, N>>,
        LabradorWitness<CyclotomicPolyRing<F, N>>,
    ) {
        // Compute Fibonacci values mod Q
        let mut fib = vec![0u64, 1];
        for i in 2..r {
            fib.push((fib[i - 1] + fib[i - 2]) % Q);
        }

        let zero = cp(0);
        let blank_phi = vec![vec![zero.clone(); n]; r];
        let mut f_funcs = Vec::new();

        // x0 = 0
        let mut phi = blank_phi.clone();
        phi[0][0] = cp(1);
        f_funcs.push(QuadraticFunction::from_parts(
            vec![],
            phi,
            CyclotomicPolyRing::<F, N>::zero(),
        ));

        // x1 = 1
        let mut phi = blank_phi.clone();
        phi[1][0] = cp(1);
        f_funcs.push(QuadraticFunction::from_parts(vec![], phi, cp(1)));

        // xi + x(i-1) - x(i+1) = 0 for i=1..r-2
        for i in 1..r - 1 {
            let mut phi = blank_phi.clone();
            phi[i][0] = cp(1);
            phi[i - 1][0] = cp(1);
            phi[i + 1][0] = cp(Q - 1);
            f_funcs.push(QuadraticFunction::from_parts(
                vec![],
                phi,
                CyclotomicPolyRing::<F, N>::zero(),
            ));
        }

        let witness = LabradorWitness::new(
            (0..r)
                .map(|i| {
                    let mut part = vec![zero.clone(); n];
                    part[0] = cp(fib[i]);
                    part
                })
                .collect(),
        );

        (
            LabradorStatement {
                f: f_funcs,
                f_prime: vec![],
            },
            witness,
        )
    }

    /// 2-level smoke test: r=5, nu=1, mu=2 -> r'=4 (main), r_last=3 (last).
    /// This exercises: main step -> build_last_level_target -> prove_last_level,
    /// and the verifier mirror: verify_step -> build_last_level_target -> verify_last_level.
    #[test]
    fn test_two_level_linear() {
        let r = 5;
        let nu = 1;
        let mu = 2;
        assert!(
            r > 2 * nu + mu,
            "r={} must exceed r'={} for shrinking",
            r,
            2 * nu + mu
        );

        let params = LabradorParams {
            jl: JLProfile::default(),
            challenge: ChallengeProfile::paper_default(),
            security_bits: 8,
            soundness_error: 0.0,
            l: 1,
            arith_p: 274177,
            n: 4,
            r,
            beta: 1000.0,
            d: N,
            q: Q as f64,
            sigma: 1.0,
            b: 2,
            b1: 16,
            b2: 16,
            t1: 4,
            t2: 4,
            kappa: 2,
            kappa1: 2,
            kappa2: 2,
            gamma: 150.0,
            gamma1_sq: 62_500,
            gamma2_sq: 62_500,
            beta_prime: 384.0,
            nu,
            mu,
            num_levels: 2,
        };

        let (statement, witness) = build_fib(r, params.n);

        let mut rng = grid_std::test_rng();
        let crs = CRS::random(&mut rng);

        let num_main = params.num_levels - 1;
        assert_eq!(num_main, 1, "expecting exactly 1 main level");

        // Prove
        let mut pt = ShakeTranscript::default();
        let proof = prove(&crs, &params, &statement, &witness, num_main, &mut pt).expect("prove");
        assert_eq!(proof.num_levels(), 2, "2 total: 1 main + 1 last");
        assert_eq!(proof.levels.len(), 1, "1 main level proof");

        // Verify using fresh transcript (verifier re-absorbs proof data)
        let mut vt = ShakeTranscript::default();
        verify(&crs, &statement, &proof, &params, &mut vt).expect("verify");
    }

    /// Verify that derive_target_for_prover uses the real private witness (not template zeros).
    /// This is the key fix for 3+ level recursion: the old code called
    /// derive_target_relation which built a zero witness, breaking multi-level recursion.
    #[test]
    fn test_derive_target_for_prover_uses_real_witness() {
        use crate::main_protocol::step_prover::prove_step;
        use crate::recursion::decompose::{bundle_v, decompose_z};
        use crate::recursion::split::split_witness;

        let mut rng = grid_std::test_rng();
        let params = LabradorParams {
            jl: JLProfile::default(),
            challenge: ChallengeProfile::paper_default(),
            security_bits: 8,
            soundness_error: 0.0,
            l: 1,
            arith_p: 274177,
            n: 4,
            r: 5,
            beta: 1000.0,
            d: N,
            q: Q as f64,
            sigma: 1.0,
            b: 2,
            b1: 16,
            b2: 16,
            t1: 4,
            t2: 4,
            kappa: 2,
            kappa1: 2,
            kappa2: 2,
            gamma: 150.0,
            gamma1_sq: 62_500,
            gamma2_sq: 62_500,
            beta_prime: 384.0,
            nu: 1,
            mu: 2,
            num_levels: 2,
        };

        let (statement, witness) = build_fib(params.r, params.n);
        let key = CommitKey::<F, N>::generate_from_params(&mut rng, &params);

        let mut pt = ShakeTranscript::default();
        let output =
            prove_step(&key, &statement, &witness, &params, &mut pt, 0).expect("prove_step");

        // Derive target using the prover's function
        let target = derive_target_for_prover(
            &key,
            &output.level_proof,
            &output.private_witness,
            &params,
            &output.aggregated,
            &output.challenges,
        );

        // Verify the target witness has the right shape
        let r_prime = 2 * params.nu + params.mu; // 4
        assert_eq!(target.witness.num_parts(), r_prime, "witness has r' parts");
        assert_eq!(target.r_prime, r_prime, "r' matches");

        // Verify the target witness is NOT all zeros (proves we used real private witness)
        let any_nonzero = target
            .witness
            .parts
            .iter()
            .any(|part| part.iter().any(|p| p.coeffs().iter().any(|c| !c.is_zero())));
        assert!(
            any_nonzero,
            "target witness should have nonzero values (real private witness, not template)"
        );

        // Verify the target witness matches what we expect from the private witness
        let expected_z_parts = decompose_z(&output.private_witness.z, params.b);
        let expected_v = bundle_v(
            &output.private_witness.t_decomposed.flat,
            &output.private_witness.g_decomposed.flat,
            &output.private_witness.h_decomposed.flat,
        );
        let m = expected_v.len();
        let (_, n_prime) =
            crate::recursion::split::compute_next_level_shape(params.n, m, params.nu, params.mu);
        let expected_witness = split_witness(
            &expected_z_parts,
            &expected_v,
            params.n,
            params.nu,
            params.mu,
            n_prime,
        );

        // The target witness should match the expected witness from private data
        assert_eq!(
            target.witness.num_parts(),
            expected_witness.num_parts(),
            "witness part count matches"
        );
        for (i, (actual, expected)) in target
            .witness
            .parts
            .iter()
            .zip(expected_witness.parts.iter())
            .enumerate()
        {
            assert_eq!(actual.len(), expected.len(), "part {} length matches", i);
            for (j, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
                assert_eq!(a, e, "part[{}][{}] matches", i, j);
            }
        }
    }

    /// 3-level recursion: main -> main -> last.
    /// Exercises derive_target_for_prover at intermediate level, then
    /// build_last_level_target on the final main step.
    #[test]
    fn test_three_level_linear() {
        let r = 5;
        let nu = 1;
        let mu = 2;
        assert!(
            r > 2 * nu + mu,
            "r={} must exceed r'={} for shrinking",
            r,
            2 * nu + mu
        );

        // r' = 2*nu+mu = 4, r_last = nu+mu = 3
        // Level 1 witness has 160 decomposition limb polys (t/g/h), each with 64 coeffs
        // in [-8,8]. JL projection of 20480 coeffs into 256 rows needs large beta.
        let beta = 50_000.0;
        let beta_prime = 500_000.0;

        let params = LabradorParams {
            jl: JLProfile::default(),
            challenge: ChallengeProfile::paper_default(),
            security_bits: 8,
            soundness_error: 0.0,
            l: 1,
            arith_p: 274177,
            n: 4,
            r,
            beta,
            d: N,
            q: Q as f64,
            sigma: 1.0,
            b: 2,
            b1: 16,
            b2: 16,
            t1: 4,
            t2: 4,
            kappa: 2,
            kappa1: 2,
            kappa2: 2,
            gamma: beta * 2.5,
            gamma1_sq: (beta * 3.0) as u128 * (beta * 3.0) as u128,
            gamma2_sq: (beta * 3.0) as u128 * (beta * 3.0) as u128,
            beta_prime,
            nu,
            mu,
            num_levels: 3,
        };

        let (statement, witness) = build_fib(r, params.n);

        let mut rng = grid_std::test_rng();
        let crs = CRS::random(&mut rng);

        let num_main = params.num_levels - 1;
        assert_eq!(num_main, 2, "expecting exactly 2 main levels");

        // Prove
        let mut pt = ShakeTranscript::default();
        let proof = prove(&crs, &params, &statement, &witness, num_main, &mut pt).expect("prove");
        assert_eq!(proof.num_levels(), 3, "3 total: 2 main + 1 last");
        assert_eq!(proof.levels.len(), 2, "2 main level proofs");

        // Verify using fresh transcript
        let mut vt = ShakeTranscript::default();
        verify(&crs, &statement, &proof, &params, &mut vt).expect("verify");
    }
}
