//! Top-level LaBRADOR verifier (Phase 5).
//!
//! [`verify`] checks the full multi-level proof.

use alloc::format;
use alloc::string::ToString;
use alloc::vec;

use crate::error::LabradorError;
use crate::main_protocol::garbage_count;
use crate::recursion::target_relation::build_last_level_target;
use crate::relation::LabradorWitness;
use grid_algebra::poly::ring::CyclotomicPolyRing;
use grid_transcript::Transcript;

use crate::crs::CRS;
use crate::crs::CommitKey;
use crate::last_level::verify_last_level;
use crate::main_protocol::step_verifier::verify_step;
use crate::params::LabradorParams;
use crate::proof::LabradorProof;
use crate::relation::LabradorStatement;
use crate::traits::LabradorProofRing;

/// Verify a multi-level LaBRADOR proof.
#[allow(clippy::too_many_arguments)]
pub fn verify<R, const N: usize, T>(
    crs: &CRS,
    statement: &LabradorStatement<CyclotomicPolyRing<R, N>>,
    proof: &LabradorProof<R, N>,
    params: &LabradorParams,
    transcript: &mut T,
) -> Result<(), LabradorError>
where
    R: LabradorProofRing<N>,
    T: Transcript,
{
    // Enforce proof depth matches public parameter profile
    if proof.num_levels() != params.num_levels {
        return Err(LabradorError::Verification(format!(
            "proof.num_levels ({}) != params.num_levels ({})",
            proof.num_levels(),
            params.num_levels
        )));
    }

    // Absorb public inputs before any protocol message (Fiat-Shamir soundness)
    crate::main_protocol::absorb_public_input(transcript, crs, statement, params)?;

    let num_main_levels = proof.levels.len();
    let mut current_statement = statement.clone();
    let mut current_key = crs.expand(params);
    let mut r_current = params.r;
    let mut n_current = params.n;
    let mut beta_current = params.beta;

    for (level, level_proof) in proof.levels.iter().enumerate() {
        let mut level_params = params.clone();
        level_params.r = r_current;
        level_params.n = n_current;
        level_params.beta = beta_current;

        if level > 0 {
            let seed = transcript
                .challenge_bytes(b"labrador_crs_seed", 32)
                .map_err(|e| format!("CRS seed derivation failed: {:?}", e))?;
            let seed_array: [u8; 32] = seed
                .try_into()
                .map_err(|_| "CRS seed length mismatch".to_string())?;
            current_key = CommitKey::from_seed(seed_array, &level_params);
        }

        let is_final_step = level + 1 == num_main_levels;
        let step_output = verify_step(
            &current_statement,
            level_proof,
            &level_params,
            transcript,
            level,
        )?;

        // Build target: last-level target on final main step (r_last = nu + mu),
        // regular recursive target on intermediate steps (r' = 2*nu + mu)
        let v_len = r_current * params.kappa * params.t1
            + garbage_count(r_current) * params.t2
            + garbage_count(r_current) * params.t1;
        let target = if is_final_step {
            let r_last = params.nu + params.mu;
            let n_last = n_current.div_ceil(params.nu).max(v_len.div_ceil(params.mu));
            let witness = LabradorWitness::new(vec![]);
            build_last_level_target(
                &current_key,
                &level_proof.u1,
                &level_proof.u2,
                &step_output.challenges,
                &step_output.aggregated,
                &level_params,
                witness,
                r_last,
                n_last,
                r_current,
            )
            .map_err(|e| LabradorError::Internal(format!("last-level target: {}", e)))?
        } else {
            crate::main_protocol::step_verifier::derive_target_relation(
                &current_key,
                level_proof,
                &level_params,
                &step_output.aggregated,
                &step_output.challenges,
            )
        };

        current_statement = target.statement;
        r_current = target.r_prime;
        n_current = target.n_prime;
        beta_current = target.beta_prime;
    }

    // Derive last-level CRS and expand a_last after all main levels
    let a_last = {
        let last_crs = CRS::derive_last(transcript)
            .map_err(|e| format!("Last CRS derivation failed: {}", e))?;
        last_crs.expand_a::<R, N>(params.kappa, n_current)
    };

    verify_last_level(
        &a_last,
        &current_statement,
        &proof.last,
        params,
        params.nu + params.mu,
        n_current,
        beta_current,
        transcript,
    )
}
