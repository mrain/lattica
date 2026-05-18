//! Last-level verifier (§5.6).
//!
//! Verifies the last recursion level proof: inner commitment check,
//! reduced garbage equations (h, g), and aggregated garbage equation.

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use grid_algebra::arith::ring::IntegerRing;
use grid_algebra::arith::ring::Ring;
use grid_algebra::lattice::types::RingMat;
use grid_algebra::poly::ring::PolyRing;
use grid_algebra::poly::ring::{CyclotomicPolyRing, NegacyclicMulRing};
use grid_serialize::{CanonicalDeserialize, CanonicalSerialize};
use grid_transcript::Transcript;

use crate::challenges::FoldingChallengeSampler;
use crate::error::LabradorError;
use crate::jl::JLMatrix;
use crate::last_level::prover::LastLevelProof;
use crate::main_protocol::aggregation::{aggregate_step2_flat, verify_b_double_prime};
use crate::main_protocol::{challenge_poly, jl_rows_flat_to_conjugated_polys};
use crate::params::LabradorParams;
use crate::relation::LabradorStatement;

/// Verify the last recursion level proof.
#[allow(
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::needless_range_loop
)]
pub fn verify_last_level<R, const N: usize, T>(
    a: &RingMat<CyclotomicPolyRing<R, N>>,
    statement: &LabradorStatement<CyclotomicPolyRing<R, N>>,
    proof: &LastLevelProof<R, N>,
    params: &LabradorParams,
    r_last: usize,
    n_last: usize,
    beta_current: f64,
    transcript: &mut T,
) -> Result<(), LabradorError>
where
    R: IntegerRing<Uint = u64> + NegacyclicMulRing<N> + CanonicalSerialize + CanonicalDeserialize,
    T: Transcript,
{
    let nu = params.nu;
    let mu = params.mu;
    let kappa = params.kappa;

    // --- Invariant checks ---
    if r_last != nu + mu {
        return Err(LabradorError::Verification(format!(
            "r_last ({}) != nu + mu ({})",
            r_last,
            nu + mu
        )));
    }
    if n_last == 0 {
        return Err(LabradorError::Verification("n_last must be > 0".into()));
    }

    // --- Shape checks ---
    if proof.t_vecs.len() != r_last * kappa {
        return Err(LabradorError::Verification(format!(
            "t_vecs length {} != r_last * kappa ({})",
            proof.t_vecs.len(),
            r_last * kappa
        )));
    }
    if proof.z.len() != n_last {
        return Err(LabradorError::Verification(format!(
            "z length {} != n_last ({})",
            proof.z.len(),
            n_last
        )));
    }
    if proof.h_garbage.len() != 2 * r_last - 1 {
        return Err(LabradorError::Verification(format!(
            "h_garbage length {} != 2*r_last - 1 ({})",
            proof.h_garbage.len(),
            2 * r_last - 1
        )));
    }
    if proof.g_garbage.len() != 2 * nu + 1 {
        return Err(LabradorError::Verification(format!(
            "g_garbage length {} != 2*nu + 1 ({})",
            proof.g_garbage.len(),
            2 * nu + 1
        )));
    }

    // b'' length must match expected number of aggregation batches
    let expected_ell = crate::main_protocol::aggregation::num_agg_batches(params.q as u64);
    if proof.b_double_prime.len() != expected_ell {
        return Err(LabradorError::Verification(format!(
            "b_double_prime length {} != expected aggregation batches ({})",
            proof.b_double_prime.len(),
            expected_ell
        )));
    }

    // JL projection length must match profile rows
    if proof.p.len() != params.jl.rows {
        return Err(LabradorError::Verification(format!(
            "p length {} != JL profile rows ({})",
            proof.p.len(),
            params.jl.rows
        )));
    }

    // --- Statement shape validation ---
    validate_last_level_statement(statement, r_last, n_last, nu)
        .map_err(LabradorError::Verification)?;

    // --- Replay transcript (verifier re-sends proof data) ---

    // 1. Inner commitments
    transcript.append_serializable(b"labrador_last_t", &proof.t_vecs)?;

    // 2. JL retry loop replay + seed binding + projection
    // Prover stops at retry >= JL_MAX_RETRY, so valid indices are 0..99.
    // Reject >= to close the off-by-one that lets forged proofs gain one extra seed attempt.
    if proof.jl_retry >= crate::jl::JL_MAX_RETRY {
        return Err(LabradorError::Verification(format!(
            "jl_retry ({}) exceeds maximum ({})",
            proof.jl_retry,
            crate::jl::JL_MAX_RETRY
        )));
    }
    let num_jl_entries = params.jl.rows;
    let mut last_challenged_seed: Option<[u8; 32]> = None;
    for retry in 0..=proof.jl_retry {
        let retry_bytes = retry.to_le_bytes();
        transcript.append_bytes(b"labrador_last_jl_retry", &retry_bytes)?;
        let jl_seed_challenge = transcript.challenge_bytes(b"labrador_last_jl_seed", 32)?;
        last_challenged_seed = Some(
            jl_seed_challenge
                .try_into()
                .map_err(|_| LabradorError::Verification("JL seed length mismatch".into()))?,
        );
    }
    let expected_seed = last_challenged_seed
        .ok_or_else(|| LabradorError::Verification("JL retry loop produced no seed".into()))?;
    if proof.jl_seed != expected_seed {
        return Err(LabradorError::Verification(
            "JL seed not bound to transcript — proof.jl_seed != challenged seed".into(),
        ));
    }
    let jl_matrix = JLMatrix::from_seed(&params.jl, n_last * params.d, proof.jl_seed);
    let q = R::modulus();
    let jl_rows_flat = jl_matrix.extract_jl_rows_flat(r_last, params.d);
    let jl_rows_poly =
        jl_rows_flat_to_conjugated_polys::<R, N>(&jl_rows_flat, params.jl.rows, r_last, n_last, q);
    transcript.append_serializable(b"labrador_last_jl_seed", &proof.jl_seed)?;
    transcript.append_serializable(b"labrador_last_p", &proof.p)?;

    // 3. Aggregation step 1
    let num_f_prime = statement.num_f_prime();
    let (psi, omega) = super::sample_last_level_step1_challenges(
        num_f_prime,
        num_jl_entries,
        params.q,
        transcript,
    )?;

    // b''
    transcript.append_serializable(b"labrador_last_bdp", &proof.b_double_prime)?;

    // Verify ct(b'')
    let b0_primes: Vec<R> = statement
        .f_prime
        .iter()
        .map(|f| f.b().coeff(0).clone())
        .collect();
    verify_b_double_prime(&proof.b_double_prime, &psi, &omega, &proof.p, &b0_primes)
        .map_err(LabradorError::Verification)?;

    // 4. Aggregation step 2
    let num_alpha = statement.num_f();
    let num_beta = proof.b_double_prime.len();
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
        &proof.b_double_prime,
        &psi,
        &omega,
        &jl_rows_poly,
    );

    // 5. Replay garbage rounds
    let mut challenges = Vec::with_capacity(r_last);
    let sampler = FoldingChallengeSampler::new(params.challenge.clone());

    for i in 1..=r_last {
        let round_byte = (i as u32).to_le_bytes();
        transcript.append_bytes(b"labrador_last_garbage_domain", &round_byte)?;

        // h_odd (skip for i=1, h1 is always 0)
        // h_garbage layout: [h2, h3, h4, ..., h_{2r}]
        // h_{2i-1} at index 2*i-3, h_{2i} at index 2*i-2
        if i > 1 {
            transcript.append_serializable(b"labrador_last_h", &proof.h_garbage[2 * i - 3])?;
        }
        // h_even
        transcript.append_serializable(b"labrador_last_h", &proof.h_garbage[2 * i - 2])?;

        // g polynomials during z-challenge phase
        if i > mu {
            let z_index_j = i - mu;
            if z_index_j == 1 {
                // g₀
                transcript.append_serializable(b"labrador_last_g", &proof.g_garbage[0])?;
            }
            // g_{2j-1} and g_{2j}
            let g_odd_idx = 2 * z_index_j - 1;
            let g_even_idx = g_odd_idx + 1;
            transcript.append_serializable(b"labrador_last_g", &proof.g_garbage[g_odd_idx])?;
            transcript.append_serializable(b"labrador_last_g", &proof.g_garbage[g_even_idx])?;
        }

        let amortize_round_byte = (i as u32).to_le_bytes();
        transcript.append_bytes(b"labrador_last_amortize_domain", &amortize_round_byte)?;

        let c_i = sampler.sample_transcript(transcript, b"labrador_last_amortize")?;
        challenges.push(c_i);
    }

    // 6. Amortized witness
    transcript.append_serializable(b"labrador_last_z", &proof.z)?;

    // --- Verification checks ---

    // Check (1): A · z = Σᵢ cᵢ · t_{π(i)}
    let lhs = a.mul_slice(&proof.z);
    let mut rhs = vec![CyclotomicPolyRing::<R, N>::zero(); kappa];
    for (k_idx, c_i) in challenges.iter().enumerate() {
        for d in 0..kappa {
            let t_idx = k_idx * kappa + d;
            let scaled = &proof.t_vecs.entries()[t_idx] * c_i;
            rhs[d] += scaled;
        }
    }
    for d in 0..kappa {
        if lhs.entries()[d] != rhs[d] {
            return Err(LabradorError::Verification(format!(
                "Check (1) inner commitment mismatch at dimension {}",
                d
            )));
        }
    }

    // Check (2): h-garbage equation
    let mut lhs_2 = CyclotomicPolyRing::<R, N>::zero();
    let mut rhs_2 = CyclotomicPolyRing::<R, N>::zero();

    for (k_idx, c_i) in challenges.iter().enumerate() {
        let i = k_idx + 1;
        let pi = super::witness_part_index(i, nu, mu);
        let phi_pi = &aggregated.phi[pi];

        let dot = Ring::dot_product(phi_pi, &proof.z);
        lhs_2 += dot * c_i;

        let h_odd = if i == 1 {
            CyclotomicPolyRing::<R, N>::zero()
        } else {
            proof.h_garbage[2 * i - 3].clone()
        };
        let h_even = &proof.h_garbage[2 * i - 2];

        rhs_2 += h_odd * c_i;
        rhs_2 += h_even * (c_i * c_i);
    }

    if lhs_2 != rhs_2 {
        return Err(LabradorError::Verification(
            "Check (2) h-garbage equation mismatch".into(),
        ));
    }

    // Check (3): g-garbage equation
    let z_dot_z = Ring::dot_product(&proof.z, &proof.z);
    let mut rhs_3 = proof.g_garbage[0].clone();

    for j in 1..=nu {
        let c_idx = mu + j - 1;
        let c_j = &challenges[c_idx];
        let g_odd = &proof.g_garbage[2 * j - 1];
        let g_even = &proof.g_garbage[2 * j];

        rhs_3 += g_odd * c_j;
        rhs_3 += g_even * (c_j * c_j);
    }

    if z_dot_z != rhs_3 {
        return Err(LabradorError::Verification(
            "Check (3) g-garbage equation mismatch".into(),
        ));
    }

    // Check (4): Aggregated relation value — Σ a_ii·g_even_i + Σ h_even_i − aggregated.b = 0.
    // Structural validation of the statement is handled globally by
    // validate_last_level_statement. The verifier does not assume a reserved
    // final (3c) function; that invariant only holds for generated recursive
    // targets, not for direct last-level proofs (num_main_levels == 0).
    //
    // g_even_i = g_garbage[2*(i+1)] = ⟨s_i, s_i⟩ for z-part i (0-indexed)
    // h_even_k = h_garbage[2*k] = ⟨φ_pi, s_pi⟩ for witness part k (0-indexed)
    {
        let mut lhs_4 = CyclotomicPolyRing::<R, N>::zero();

        // Quadratic part: Σ a_ii · g_even_i  (diagonal-only in last level)
        for &(i, j, ref a_coeff) in &aggregated.a_ij {
            if i != j || i >= nu {
                return Err(LabradorError::Verification(format!(
                    "Check (4) aggregated has off-diagonal/out-of-range term a_{{{}, {}}} — last level requires diagonal-only a_{{i,i}} for i < nu",
                    i, j
                )));
            }
            let g_even_idx = 2 * (i + 1);
            lhs_4 += a_coeff * &proof.g_garbage[g_even_idx];
        }

        // Linear part: Σ h_even_k  over all witness parts
        for k in 0..r_last {
            let h_even_idx = 2 * k;
            lhs_4 += &proof.h_garbage[h_even_idx];
        }

        if lhs_4 != aggregated.b {
            return Err(LabradorError::Verification(
                "Check (4) aggregated relation value mismatch".into(),
            ));
        }
    }

    // Check (5): JL norm — ||p|| ≤ sqrt(128) · beta_current
    if !crate::jl::verify_norm(&params.jl, &proof.p, beta_current) {
        return Err(LabradorError::Verification(
            "Check (5) JL projection norm exceeds sqrt(128) · beta_current".into(),
        ));
    }

    // Check (6): z norm — ||z||² ≤ gamma_last² = (beta_current · sqrt(τ))²
    let gamma_last = beta_current * grid_std::sqrt(params.challenge.tau());
    let z_coeffs: Vec<_> = proof
        .z
        .iter()
        .flat_map(|poly| poly.coeffs())
        .cloned()
        .collect();
    let z_norm_sq = crate::main_protocol::squared_l2_norm(z_coeffs).map_err(|_| {
        LabradorError::Verification(
            "Check (6) z norm overflow (squared sum exceeds u128::MAX)".into(),
        )
    })?;
    if z_norm_sq > gamma_last * gamma_last {
        return Err(LabradorError::Verification(format!(
            "Check (6) z norm squared ({}) exceeds gamma_last² ({})",
            z_norm_sq,
            gamma_last * gamma_last
        )));
    }

    Ok(())
}

/// Validate last-level statement shape against (r_last, n_last, nu).
///
/// Ensures:
/// 1. Every f/f_prime has phi.len() == r_last
/// 2. Every phi[i].len() == n_last
/// 3. a.len() == ij.len()
/// 4. Last-level diagonal-only invariant: only i == j && i < nu allowed
fn validate_last_level_statement<R, const N: usize>(
    statement: &LabradorStatement<CyclotomicPolyRing<R, N>>,
    r_last: usize,
    n_last: usize,
    nu: usize,
) -> Result<(), String>
where
    R: IntegerRing<Uint = u64> + NegacyclicMulRing<N>,
{
    for (fi, f) in statement.f.iter().enumerate() {
        validate_last_level_function(f, fi, "F", r_last, n_last, nu, false)?;
    }
    for (fi, f) in statement.f_prime.iter().enumerate() {
        validate_last_level_function(f, fi, "F'", r_last, n_last, nu, true)?;
    }
    Ok(())
}

fn validate_last_level_function<R, const N: usize>(
    f: &crate::relation::QuadraticFunction<CyclotomicPolyRing<R, N>>,
    fi: usize,
    kind: &str,
    r_last: usize,
    n_last: usize,
    nu: usize,
    fprime: bool,
) -> Result<(), String>
where
    R: IntegerRing<Uint = u64> + NegacyclicMulRing<N>,
{
    match f {
        crate::relation::QuadraticFunction::Dense(d) => {
            if d.a.len() != d.ij.len() {
                return Err(format!(
                    "statement {}: function {} has {} coefficients but {} index pairs",
                    kind,
                    fi,
                    d.a.len(),
                    d.ij.len()
                ));
            }
            for (idx, &(i, j)) in d.ij.iter().enumerate() {
                if i != j || i >= nu {
                    return Err(format!(
                        "statement {}: function {} ij[{}]=({},{}) — last level requires diagonal-only a_{{i,i}} for i < nu",
                        kind, fi, idx, i, j
                    ));
                }
            }
            if d.phi.len() != r_last {
                return Err(format!(
                    "statement {}: function {} has {} phi vectors, expected r_last={}",
                    kind,
                    fi,
                    d.phi.len(),
                    r_last
                ));
            }
            for (pi, phi_i) in d.phi.iter().enumerate() {
                if phi_i.len() != n_last {
                    return Err(format!(
                        "statement {}: function {} phi[{}] has length {}, expected n_last={}",
                        kind,
                        fi,
                        pi,
                        phi_i.len(),
                        n_last
                    ));
                }
            }
        }
        crate::relation::QuadraticFunction::Sparse(s) => {
            for (idx, &(i, j, _)) in s.ij_a.iter().enumerate() {
                if i != j || i >= nu {
                    return Err(format!(
                        "statement {}: function {} ij_a[{}]=({},{}) — last level requires diagonal-only a_{{i,i}} for i < nu",
                        kind, fi, idx, i, j
                    ));
                }
            }
            for (idx, &(part, entry, _)) in s.phi.iter().enumerate() {
                if part >= r_last {
                    return Err(format!(
                        "statement {}: function {} phi[{}] part index {} out of bounds for r_last={}",
                        kind, fi, idx, part, r_last
                    ));
                }
                if entry >= n_last {
                    return Err(format!(
                        "statement {}: function {} phi[{}] entry index {} out of bounds for n_last={}",
                        kind, fi, idx, entry, n_last
                    ));
                }
            }
        }
    }

    // F' functions must have constant b (only coeff 0 nonzero).
    if fprime && !f.b().coeffs().iter().skip(1).all(|c| c.is_zero()) {
        return Err(format!(
            "statement {}: function {} has non-constant b — F' b must be a constant polynomial (only coefficient 0 may be nonzero)",
            kind, fi
        ));
    }

    Ok(())
}
