//! Last-level LaBRADOR protocol (§5.6).
//!
//! The last recursion level skips producing new outer commitments but still
//! proves all constraints from the previous level's target. Uses reduced
//! garbage: 2r-1 h-polynomials and 2ν+1 g-polynomials.

use alloc::vec::Vec;

use grid_algebra::arith::IntegerRing;
use grid_transcript::Transcript;

mod prover;
mod reduced_garbage;
mod verifier;

pub use prover::{LastLevelProof, prove_last_level};
pub use reduced_garbage::{compute_g_pair, compute_g0, compute_h_round};
pub use verifier::verify_last_level;

/// Map 1-indexed challenge round to 0-indexed witness part index.
/// - For round_i ≤ μ: v-part, witness index = nu + (round_i - 1)
/// - For round_i > μ: z-part, witness index = (round_i - μ - 1)
///
/// # Panics
///
/// Panics in debug mode if `round_i == 0`, `round_i > nu + mu`, or `mu == 0`.
#[inline]
pub(super) fn witness_part_index(round_i: usize, nu: usize, mu: usize) -> usize {
    debug_assert!(
        round_i >= 1 && round_i <= nu + mu,
        "round_i must be in [1, nu+mu] = [1, {}], got {}",
        nu + mu,
        round_i
    );
    debug_assert!(mu > 0, "mu must be > 0 (at least one v-challenge required)");
    if round_i <= mu {
        nu + (round_i - 1)
    } else {
        round_i - mu - 1
    }
}

/// Sample step 1 aggregation challenges with last-level labels.
#[allow(clippy::type_complexity)]
pub(super) fn sample_last_level_step1_challenges<R, T>(
    num_f_prime: usize,
    num_jl_entries: usize,
    q: f64,
    transcript: &mut T,
) -> Result<(Vec<Vec<R>>, Vec<Vec<R>>), super::LabradorError>
where
    R: IntegerRing<Canonical = u64>,
    T: Transcript,
{
    let ell = grid_std::ceil(128.0 / grid_std::log2(q)) as usize;
    let mut psi = Vec::with_capacity(ell);
    let mut omega = Vec::with_capacity(ell);

    for k in 0..ell {
        let k_bytes = (k as u32).to_le_bytes();
        transcript.append_bytes(b"labrador_last_agg_k", &k_bytes)?;

        let psi_k: Vec<_> = (0..num_f_prime)
            .map(|_| transcript.challenge_scalar(b"labrador_last_psi"))
            .collect::<Result<_, _>>()?;

        let omega_k: Vec<_> = (0..num_jl_entries)
            .map(|_| transcript.challenge_scalar(b"labrador_last_omega"))
            .collect::<Result<_, _>>()?;

        psi.push(psi_k);
        omega.push(omega_k);
    }

    Ok((psi, omega))
}
