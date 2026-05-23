//! Last-level prover (§5.6).
//!
//! Proves the last recursion level without producing new outer commitments.
//! Uses reduced garbage (2r-1 h-polynomials, 2ν+1 g-polynomials) and
//! challenge reordering (v-challenges first, z-challenges second).
//!
//! The witness multiplicity at the last level is `r_last = ν + μ` (not
//! `r' = 2ν + μ` from regular recursion), so the garbage rounds iterate
//! `r_last` times rather than `r` times.

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;

use grid_algebra::arith::ring::IntegerRing;
use grid_algebra::arith::ring::Ring;
use grid_algebra::lattice::types::{RingMat, RingVec};
use grid_algebra::poly::ring::PolyRing;
use grid_algebra::poly::ring::{CyclotomicPolyRing, NegacyclicMulRing};
use grid_serialize::{CanonicalDeserialize, CanonicalSerialize};
use grid_std::UniformRand;
use grid_transcript::Transcript;

use crate::challenges::FoldingChallengeSampler;
use crate::error::LabradorError;
use crate::jl::JLMatrix;
use crate::last_level::reduced_garbage::{compute_g_pair, compute_g0, compute_h_round};
use crate::main_protocol::aggregation::{aggregate_step2_flat, compute_b_double_prime};
use crate::main_protocol::{challenge_poly, jl_rows_flat_to_conjugated_polys};
use crate::params::LabradorParams;
use crate::relation::{LabradorStatement, LabradorWitness};

/// Last-level proof data.
#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct LastLevelProof<R, const N: usize>
where
    R: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    /// Inner commitments: r vectors, each of length κ.
    /// Flat RingVec of length `r*κ`, row-major: `t_vecs[i*κ + d] = t_{π(i)}[d]`.
    pub t_vecs: RingVec<CyclotomicPolyRing<R, N>>,
    /// JL seed for deterministic matrix reconstruction.
    pub jl_seed: [u8; 32],
    /// Number of retries before finding a valid JL projection.
    pub jl_retry: u32,
    /// JL projection coefficients.
    pub p: Vec<R>,
    /// b'' polynomials from aggregation step 1.
    pub b_double_prime: Vec<CyclotomicPolyRing<R, N>>,
    /// Reduced h-garbage: 2r-1 polynomials. Array layout (0-indexed):
    /// `h_garbage = [h₂, h₃, h₄, h₅, ..., h_{2r}]` (length 2r-1)
    /// Indexing: `h_garbage[k] = h_{k+2}` for k=0..2r-2
    /// Verification loop (1-indexed i=1..r):
    ///   h_{2i-1} (odd term): 0 when i=1; else `h_garbage[2*i-3]`
    ///   h_{2i}   (even term): `h_garbage[2*i-2]`
    pub h_garbage: Vec<CyclotomicPolyRing<R, N>>,
    /// Reduced g-garbage: 2ν+1 polynomials. Array layout (0-indexed):
    /// `g_garbage[0] = g₀`
    /// `g_garbage[2j-1] = g_{2j-1}` for j=1..ν
    /// `g_garbage[2j] = g_{2j}` for j=1..ν
    /// So: `g_garbage = [g₀, g₁, g₂, ..., g_{2ν}]` (length 2ν+1)
    pub g_garbage: Vec<CyclotomicPolyRing<R, N>>,
    /// Amortized witness z = Σ cᵢ · s_{π(i)} (n polynomials).
    pub z: Vec<CyclotomicPolyRing<R, N>>,
}

/// Last-level prover.
#[allow(
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::needless_range_loop
)]
pub fn prove_last_level<R, const N: usize, T>(
    a: &RingMat<CyclotomicPolyRing<R, N>>,
    statement: &LabradorStatement<CyclotomicPolyRing<R, N>>,
    witness: &LabradorWitness<CyclotomicPolyRing<R, N>>,
    params: &LabradorParams,
    r_last: usize,
    n_last: usize,
    beta_current: f64,
    transcript: &mut T,
) -> Result<LastLevelProof<R, N>, LabradorError>
where
    R: IntegerRing<Canonical = u64>
        + NegacyclicMulRing<N>
        + UniformRand
        + CanonicalSerialize
        + CanonicalDeserialize,
    T: Transcript,
{
    let nu = params.nu;
    let mu = params.mu;
    let kappa = params.kappa;

    // --- Invariant checks ---
    if r_last != nu + mu {
        return Err(LabradorError::Prover(format!(
            "r_last ({}) != nu + mu ({})",
            r_last,
            nu + mu
        )));
    }
    if n_last == 0 {
        return Err(LabradorError::Prover("n_last must be > 0".into()));
    }

    // --- 1. Inner commitments (in π permutation order for challenge matching) ---
    let mut t_vecs_data = Vec::with_capacity(r_last * kappa);
    for challenge_round in 0..r_last {
        let pi = super::witness_part_index(challenge_round + 1, nu, mu);
        let t_pi = a.mul_slice(&witness.parts[pi]);
        for d in 0..kappa {
            t_vecs_data.push(t_pi.entries()[d].clone());
        }
    }
    let t_vecs_ring = RingVec::new(t_vecs_data.clone());
    transcript.append_serializable(b"labrador_last_t", &t_vecs_ring)?;
    let t_vecs = t_vecs_ring;

    // --- 2. JL projection with rejection sampling retry ---
    let p_coeffs: Vec<Vec<R>> = witness
        .parts
        .iter()
        .take(r_last)
        .map(|part| {
            part.iter()
                .flat_map(|poly| poly.coeffs().iter().cloned())
                .collect()
        })
        .collect();

    /// Maximum JL retry attempts before giving up.
    const JL_MAX_RETRY: u32 = 100;

    let mut jl_retry: u32 = 0;
    let (jl_matrix, p) = loop {
        if jl_retry >= JL_MAX_RETRY {
            return Err(LabradorError::Prover(format!(
                "JL projection retry exceeded {} attempts (parameters may be too tight)",
                JL_MAX_RETRY
            )));
        }
        let retry_bytes = jl_retry.to_le_bytes();
        transcript.append_bytes(b"labrador_last_jl_retry", &retry_bytes)?;
        let jl_seed_bytes = transcript.challenge_bytes(b"labrador_last_jl_seed", 32)?;
        let jl_seed_candidate: [u8; 32] = jl_seed_bytes
            .try_into()
            .expect("challenge_bytes returns exact length");
        let jl_matrix = JLMatrix::from_seed(&params.jl, n_last * params.d, jl_seed_candidate);

        let p_slices: Vec<&[R]> = p_coeffs.iter().map(|v| v.as_slice()).collect();
        let candidate_p = jl_matrix.project_multi(&p_slices);

        if crate::jl::verify_norm(&params.jl, &candidate_p, beta_current) {
            break (jl_matrix, candidate_p);
        }
        jl_retry += 1;
    };
    let jl_seed = *jl_matrix.seed();
    let num_jl_entries = params.jl.rows;
    // Commit seed and projection to transcript
    transcript.append_serializable(b"labrador_last_jl_seed", &jl_seed)?;
    transcript.append_serializable(b"labrador_last_p", &p)?;

    // --- 3. Aggregation step 1 ---
    let num_f_prime = statement.num_f_prime();
    let (psi, omega) = super::sample_last_level_step1_challenges(
        num_f_prime,
        num_jl_entries,
        params.q,
        transcript,
    )?;

    // Compute b''
    let f_prime_evals: Vec<_> = statement
        .f_prime
        .iter()
        .map(|f| f.evaluate(witness))
        .collect();
    let f_prime_b: Vec<_> = statement.f_prime.iter().map(|f| f.b().clone()).collect();

    let q = R::modulus();
    let jl_rows_flat = jl_matrix.extract_jl_rows_flat(r_last, params.d);
    let jl_rows_poly =
        jl_rows_flat_to_conjugated_polys::<R, N>(&jl_rows_flat, params.jl.rows, r_last, n_last, q);

    let jl_evals: Vec<CyclotomicPolyRing<R, N>> = (0..num_jl_entries)
        .map(|m| {
            let mut jl_eval = CyclotomicPolyRing::<R, N>::zero();
            for (i, s_i) in witness.parts.iter().enumerate().take(r_last) {
                for (j, s_ij) in s_i.iter().enumerate() {
                    jl_eval += jl_rows_poly.get(m, i, j) * s_ij;
                }
            }
            jl_eval
        })
        .collect();

    let b_double_prime =
        compute_b_double_prime(&f_prime_evals, &f_prime_b, &jl_evals, &psi, &omega);

    transcript.append_serializable(b"labrador_last_bdp", &b_double_prime)?;

    // --- 4. Aggregation step 2 ---
    let num_alpha = statement.num_f();
    let num_beta = b_double_prime.len();
    let alpha: Vec<CyclotomicPolyRing<R, N>> = (0..num_alpha)
        .map(|_| challenge_poly(transcript, b"labrador_last_alpha"))
        .collect::<Result<_, _>>()?;
    let beta: Vec<CyclotomicPolyRing<R, N>> = (0..num_beta)
        .map(|_| challenge_poly(transcript, b"labrador_last_beta"))
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

    // --- 5. Reduced garbage rounds ---
    // h_garbage has length 2*r_last - 1 (h₁ omitted, always zero since
    // first round has no prior cross-term). Prover sends h₂, h₃, ..., h_{2r_last}.
    // Verifier skips i=1's odd-indexed commitment to stay in sync.
    let mut h_garbage = Vec::with_capacity(2 * r_last - 1);
    let mut g_garbage = Vec::with_capacity(2 * nu + 1);
    let mut challenges = Vec::with_capacity(r_last);
    let mut g0_computed = false;
    let sampler = FoldingChallengeSampler::new(params.challenge.clone());

    for i in 1..=r_last {
        let (h_odd, h_even) = compute_h_round(witness, &aggregated.phi, &challenges, i, nu, mu);

        let round_byte = (i as u32).to_le_bytes();
        transcript.append_bytes(b"labrador_last_garbage_domain", &round_byte)?;

        // Skip h₁ (always zero for first round — no prior cross-term)
        if i > 1 {
            transcript.append_serializable(b"labrador_last_h", &h_odd)?;
            h_garbage.push(h_odd);
        }
        transcript.append_serializable(b"labrador_last_h", &h_even)?;
        h_garbage.push(h_even);

        // g polynomials during z-challenge phase
        if i > mu {
            let z_index_j = i - mu;
            if !g0_computed {
                let g0_val = compute_g0(witness, &challenges, nu, mu);
                g0_computed = true;
                g_garbage.push(g0_val);
                transcript.append_serializable(b"labrador_last_g", &g_garbage[0])?;
            }

            let (g_odd, g_even) = compute_g_pair(witness, &challenges, z_index_j, nu, mu);
            transcript.append_serializable(b"labrador_last_g", &g_odd)?;
            g_garbage.push(g_odd);
            transcript.append_serializable(b"labrador_last_g", &g_even)?;
            g_garbage.push(g_even);
        }

        // Sample challenge
        let amortize_round_byte = (i as u32).to_le_bytes();
        transcript.append_bytes(b"labrador_last_amortize_domain", &amortize_round_byte)?;

        let c_i = sampler.sample_transcript(transcript, b"labrador_last_amortize")?;
        challenges.push(c_i);
    }

    // --- 8. Amortization ---
    let mut z = vec![CyclotomicPolyRing::<R, N>::zero(); n_last];
    for (k_idx, c_i) in challenges.iter().enumerate() {
        let pi = super::witness_part_index(k_idx + 1, nu, mu);
        let part_len = witness.parts[pi].len();
        for k in 0..n_last.min(part_len) {
            z[k] += c_i * &witness.parts[pi][k];
        }
    }

    transcript.append_serializable(b"labrador_last_z", &z)?;

    // Self-check: z norm
    let gamma_last = beta_current * grid_std::sqrt(params.challenge.tau());
    let z_coeffs: Vec<_> = z.iter().flat_map(|poly| poly.coeffs()).cloned().collect();
    let z_norm_sq = crate::main_protocol::squared_l2_norm(z_coeffs).map_err(|_| {
        LabradorError::Prover("z norm overflow (squared sum exceeds u128::MAX)".into())
    })?;
    if z_norm_sq > gamma_last * gamma_last {
        return Err(LabradorError::Prover(format!(
            "z norm squared ({}) exceeds gamma_last² ({})",
            z_norm_sq,
            gamma_last * gamma_last
        )));
    }

    Ok(LastLevelProof {
        t_vecs,
        jl_seed,
        jl_retry,
        p,
        b_double_prime,
        h_garbage,
        g_garbage,
        z,
    })
}
