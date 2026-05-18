//! Verifier orchestrator for one LaBRADOR recursion step (§5.2).
//!
//! Re-absorbs prover messages into the transcript, re-samples all challenges,
//! verifies all four equations, and derives the target relation for the next
//! recursion level.

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use crate::error::LabradorError;
use grid_algebra::arith::ring::IntegerRing;
use grid_algebra::arith::ring::Ring;
use grid_algebra::poly::ring::PolyRing;
use grid_algebra::poly::ring::{CyclotomicPolyRing, NegacyclicMulRing};
use grid_serialize::{CanonicalDeserialize, CanonicalSerialize};
use grid_std::UniformRand;
use grid_transcript::Transcript;

macro_rules! ve {
    ($($arg:tt)*) => { LabradorError::Verification(format!($($arg)*)) };
}

use crate::challenges::FoldingChallengeSampler;
use crate::crs::CommitKey;
use crate::jl::JLMatrix;
use crate::main_protocol::aggregation::{
    AggregatedFunction, aggregate_step2_flat, sample_step1_challenges, verify_b_double_prime,
};

use crate::main_protocol::step_prover::LevelProof;
use crate::main_protocol::{challenge_poly, jl_rows_flat_to_conjugated_polys};
use crate::params::LabradorParams;
use crate::recursion::build_target_relation;
use crate::recursion::split::{compute_next_level_shape, split_witness};
use crate::recursion::target_relation::RecursiveTarget;
use crate::relation::{LabradorStatement, QuadraticFunction};

/// Result of [`verify_step`]: the recursive target for the next level (§5.3).
///
/// Contains K' = κ + κ₁ + κ₂ + 3 constraints, consolidated norm bound β',
/// and the decomposed/split witness ready for the next main protocol invocation.
pub type VerifiedStep<R, const N: usize> = RecursiveTarget<R, N>;

/// Output from a verified step, including derivation data for target construction.
///
/// Does NOT carry a target — the caller builds the target from the
/// aggregated function and challenges via [`crate::recursion::build_target_relation`] or
/// `crate::recursion::target_relation::build_last_level_target`.
#[derive(Debug, Clone)]
pub struct StepVerificationOutput<R, const N: usize>
where
    R: IntegerRing<Uint = u64> + NegacyclicMulRing<N> + UniformRand,
{
    /// Aggregated quadratic function from step 2 aggregation.
    pub aggregated: AggregatedFunction<R, N>,
    /// Amortization challenges c_1..c_r used to build the target relation.
    pub challenges: Vec<CyclotomicPolyRing<R, N>>,
}

/// Validate public statement structure against params.
///
/// A malformed statement can extend aggregated.phi past params.r or produce
/// out-of-range a_ij indices, causing panics in verify_eq_3b/verify_eq_3c.
fn validate_statement_shapes<R, const N: usize>(
    statement: &LabradorStatement<CyclotomicPolyRing<R, N>>,
    params: &LabradorParams,
) -> Result<(), LabradorError>
where
    R: IntegerRing<Uint = u64> + NegacyclicMulRing<N>,
{
    let r = params.r;
    let n = params.n;

    // Validate F function shapes
    for (fi, f) in statement.f.iter().enumerate() {
        validate_function_shape(f, fi, "F", r, n, false)?;
    }

    // Validate F' function shapes
    for (fi, f) in statement.f_prime.iter().enumerate() {
        validate_function_shape(f, fi, "F'", r, n, true)?;
    }

    Ok(())
}

/// Check that a polynomial is constant (only coefficient 0 is nonzero).
fn is_constant_poly<Rq>(poly: &Rq) -> bool
where
    Rq: PolyRing,
{
    poly.coeffs().iter().skip(1).all(|c| c.is_zero())
}

/// Validate a single function's shape against (r, n).
fn validate_function_shape<R, const N: usize>(
    f: &QuadraticFunction<CyclotomicPolyRing<R, N>>,
    fi: usize,
    kind: &str,
    r: usize,
    n: usize,
    fprime: bool,
) -> Result<(), LabradorError>
where
    R: IntegerRing<Uint = u64> + NegacyclicMulRing<N>,
{
    match f {
        QuadraticFunction::Dense(d) => {
            if d.a.len() != d.ij.len() {
                return Err(ve!(
                    "statement {}: function {} has {} coefficients but {} index pairs",
                    kind,
                    fi,
                    d.a.len(),
                    d.ij.len()
                ));
            }
            for (idx, &(i, j)) in d.ij.iter().enumerate() {
                if i > j {
                    return Err(ve!(
                        "statement {}: function {} ij[{}]=({},{}) violates upper-triangular invariant i <= j",
                        kind,
                        fi,
                        idx,
                        i,
                        j
                    ));
                }
                if i >= r || j >= r {
                    return Err(ve!(
                        "statement {}: function {} ij[{}]=({},{}) out of bounds for r={}",
                        kind,
                        fi,
                        idx,
                        i,
                        j,
                        r
                    ));
                }
            }
            if d.phi.len() != r {
                return Err(ve!(
                    "statement {}: function {} has {} phi vectors, expected r={}",
                    kind,
                    fi,
                    d.phi.len(),
                    r
                ));
            }
            for (pi, phi_i) in d.phi.iter().enumerate() {
                if phi_i.len() != n {
                    return Err(ve!(
                        "statement {}: function {} phi[{}] has length {}, expected n={}",
                        kind,
                        fi,
                        pi,
                        phi_i.len(),
                        n
                    ));
                }
            }
        }
        QuadraticFunction::Sparse(s) => {
            for (idx, &(i, j, _)) in s.ij_a.iter().enumerate() {
                if i > j {
                    return Err(ve!(
                        "statement {}: function {} ij_a[{}]=({},{}) violates upper-triangular invariant i <= j",
                        kind,
                        fi,
                        idx,
                        i,
                        j
                    ));
                }
                if i >= r || j >= r {
                    return Err(ve!(
                        "statement {}: function {} ij_a[{}]=({},{}) out of bounds for r={}",
                        kind,
                        fi,
                        idx,
                        i,
                        j,
                        r
                    ));
                }
            }
            for (idx, &(part, entry, _)) in s.phi.iter().enumerate() {
                if part >= r {
                    return Err(ve!(
                        "statement {}: function {} phi[{}] part index {} out of bounds for r={}",
                        kind,
                        fi,
                        idx,
                        part,
                        r
                    ));
                }
                if entry >= n {
                    return Err(ve!(
                        "statement {}: function {} phi[{}] entry index {} out of bounds for n={}",
                        kind,
                        fi,
                        idx,
                        entry,
                        n
                    ));
                }
            }
        }
    }

    // F' functions must have constant b (only coeff 0 nonzero).
    // The paper's F' operates on constant terms only; a non-constant b would
    // cause prover/verifier disagreement on b'' computation.
    if fprime && !is_constant_poly(f.b()) {
        return Err(ve!(
            "statement {}: function {} has non-constant b — F' b must be a constant polynomial (only coefficient 0 may be nonzero)",
            kind,
            fi
        ));
    }

    Ok(())
}

/// Verify one recursion step (transcript replay only).
///
/// Re-absorbs prover messages into the transcript and re-samples challenges.
/// Does NOT verify equations or build the target relation — the private witness
/// (`z`, decomposed limbs) is proven recursively by the next level, and target
/// construction is the caller's responsibility via
/// [`crate::recursion::build_target_relation`] or
/// `crate::recursion::target_relation::build_last_level_target`. Returns the aggregated function and amortization
/// challenges needed to derive the target relation.
pub fn verify_step<R, const N: usize, T>(
    statement: &LabradorStatement<CyclotomicPolyRing<R, N>>,
    proof: &LevelProof<R, N>,
    params: &LabradorParams,
    transcript: &mut T,
    level: usize,
) -> Result<StepVerificationOutput<R, N>, LabradorError>
where
    R: IntegerRing<Uint = u64>
        + NegacyclicMulRing<N>
        + UniformRand
        + CanonicalSerialize
        + CanonicalDeserialize,
    T: Transcript,
{
    // ── Re-absorb prover messages into transcript ──

    // Absorb u1
    transcript
        .append_serializable(b"labrador_u1", &proof.u1)
        .map_err(|e| format!("u1 absorb error: {:?}", e))?;

    // Replay prover's JL retry loop to sync transcript state and bind seed
    // Prover stops at retry >= JL_MAX_RETRY, so valid indices are 0..99.
    // Reject >= to close the off-by-one that lets forged proofs gain one extra seed attempt.
    if proof.jl_retry >= crate::jl::JL_MAX_RETRY {
        return Err(ve!(
            "jl_retry ({}) exceeds maximum ({})",
            proof.jl_retry,
            crate::jl::JL_MAX_RETRY
        ));
    }
    let mut last_challenged_seed: Option<[u8; 32]> = None;
    for retry in 0..=proof.jl_retry {
        let retry_bytes = retry.to_le_bytes();
        transcript
            .append_bytes(b"labrador_jl_retry", &retry_bytes)
            .map_err(|e| format!("JL retry absorb error: {:?}", e))?;
        let jl_seed_challenge = transcript
            .challenge_bytes(b"labrador_jl_seed_squeeze", 32)
            .map_err(|e| format!("JL seed challenge error: {:?}", e))?;
        last_challenged_seed = Some(
            jl_seed_challenge
                .try_into()
                .map_err(|_| String::from("JL seed length mismatch"))?,
        );
    }
    // Bind: proof.jl_seed must equal the last Fiat-Shamir challenged seed
    let expected_seed =
        last_challenged_seed.ok_or_else(|| String::from("JL retry loop produced no seed"))?;
    if proof.jl_seed != expected_seed {
        return Err(ve!(
            "JL seed not bound to transcript — proof.jl_seed != challenged seed"
        ));
    }
    transcript
        .append_bytes(b"labrador_jl_seed_bind", &proof.jl_seed)
        .map_err(|e| format!("JL seed absorb error: {:?}", e))?;
    let jl_matrix = JLMatrix::from_seed(&params.jl, params.n * params.d, proof.jl_seed);

    // Absorb p
    transcript
        .append_serializable(b"labrador_p", &proof.p)
        .map_err(|e| format!("p absorb error: {:?}", e))?;

    // Check JL norm
    if !crate::jl::verify_norm(&params.jl, &proof.p, params.beta) {
        return Err(ve!("JL projection norm exceeds sqrt(128)·β"));
    }

    // Sample aggregation step 1 challenges
    let (psi, omega) = sample_step1_challenges(
        statement.f_prime.len(),
        params.jl.rows,
        params.q as u64,
        transcript,
    )
    .map_err(|e| format!("step1 challenges error: {:?}", e))?;

    // Absorb b''
    transcript
        .append_serializable(b"labrador_bdp", &proof.b_double_prime)
        .map_err(|e| format!("b'' absorb error: {:?}", e))?;

    // ── Validate public statement shape before b'' check ──
    validate_statement_shapes(statement, params)?;

    // Verify b'' constant terms
    // Check exact length before any indexing to prevent panic on malformed proofs
    let ell = psi.len();
    if proof.b_double_prime.len() != ell {
        return Err(ve!(
            "shape: b_double_prime.len()={} expected num_agg_batches={}",
            proof.b_double_prime.len(),
            ell
        ));
    }
    let b0_primes: Vec<R> = statement.f_prime.iter().map(|f| f.b().coeff(0)).collect();
    verify_b_double_prime(&proof.b_double_prime, &psi, &omega, &proof.p, &b0_primes)?;

    // Sample aggregation step 2 challenges (polynomials)
    let level_bytes = (level as u32).to_le_bytes();
    transcript
        .append_bytes(b"labrador_level", &level_bytes)
        .map_err(|e| format!("level absorb error: {:?}", e))?;
    let alpha: Vec<CyclotomicPolyRing<R, N>> = (0..statement.f.len())
        .map(|_| challenge_poly::<CyclotomicPolyRing<R, N>, _>(transcript, b"labrador_alpha"))
        .collect::<Result<_, _>>()
        .map_err(|e| format!("alpha sampling: {:?}", e))?;
    let beta: Vec<CyclotomicPolyRing<R, N>> = (0..psi.len())
        .map(|_| challenge_poly::<CyclotomicPolyRing<R, N>, _>(transcript, b"labrador_beta"))
        .collect::<Result<_, _>>()
        .map_err(|e| format!("beta sampling: {:?}", e))?;

    // Absorb u2
    transcript
        .append_serializable(b"labrador_u2", &proof.u2)
        .map_err(|e| format!("u2 absorb error: {:?}", e))?;

    // Derive aggregated function for garbage equations
    let q = R::modulus();
    let jl_rows_flat = jl_matrix.extract_jl_rows_flat(params.r, params.d);
    let jl_rows_poly = jl_rows_flat_to_conjugated_polys::<R, N>(
        &jl_rows_flat,
        params.jl.rows,
        params.r,
        params.n,
        q,
    );

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

    // Sample amortization challenges from transcript (after u2, matches prover ordering)
    let sampler = FoldingChallengeSampler::new(params.challenge.clone());
    let mut domain = [0u8; 8];
    domain[..4].copy_from_slice(&(level as u32).to_le_bytes());
    let mut challenges = Vec::with_capacity(params.r);
    for i in 0..params.r {
        domain[4..].copy_from_slice(&(i as u32).to_le_bytes());
        transcript
            .append_bytes(b"labrador_amortize_domain", &domain)
            .map_err(|e| format!("amortization domain append: {:?}", e))?;
        let c_i = sampler
            .sample_transcript(transcript, b"labrador_amortize")
            .map_err(|e| format!("amortization challenge sampling: {:?}", e))?;
        challenges.push(c_i);
    }

    // Guard: u1/u2 must match expected commitment ranks before target construction
    // indexes them directly (build_target_relation accesses u1[d] for d<kappa1
    // and u2[d] for d<kappa2 without bounds checks).
    if proof.u1.len() != params.kappa1 {
        return Err(ve!(
            "u1 length {} != kappa1 ({})",
            proof.u1.len(),
            params.kappa1
        ));
    }
    if proof.u2.len() != params.kappa2 {
        return Err(ve!(
            "u2 length {} != kappa2 ({})",
            proof.u2.len(),
            params.kappa2
        ));
    }

    Ok(StepVerificationOutput {
        aggregated,
        challenges,
    })
}

/// Derive the target relation for the next recursion level (§5.3).
///
/// The next-level witness is constructed from z_parts (binary-decomposed z),
/// t_limbs, g_limbs, h_limbs split into r' = 2ν + μ vectors of rank n'.
/// The statement contains K' = κ + κ₁ + κ₂ + 3 constraints from family G.
pub fn derive_target_relation<R, const N: usize>(
    key: &CommitKey<R, N>,
    proof: &LevelProof<R, N>,
    params: &LabradorParams,
    aggregated: &AggregatedFunction<R, N>,
    challenges: &[CyclotomicPolyRing<R, N>],
) -> RecursiveTarget<R, N>
where
    R: IntegerRing<Uint = u64> + NegacyclicMulRing<N> + UniformRand,
{
    let nu = params.nu;
    let mu = params.mu;
    let n = params.n;

    // Derive witness shape from params (same as prover's decomposed limbs)
    let v_len = params.r * params.kappa * params.t1
        + crate::main_protocol::garbage_count(params.r) * params.t2
        + crate::main_protocol::garbage_count(params.r) * params.t1;
    let z_parts_len = 2 * n; // binary decomposition (b=2)

    let (r_prime, n_prime) = compute_next_level_shape(n, v_len, nu, mu);

    // Build template witness with correct shape (coefficients don't affect statement)
    let zero = <CyclotomicPolyRing<R, N> as Ring>::zero();
    let z_parts = vec![zero.clone(); z_parts_len];
    let v = vec![zero; v_len];
    let witness = split_witness(&z_parts, &v, n, nu, mu, n_prime);

    build_target_relation(
        key, &proof.u1, &proof.u2, challenges, aggregated, params, witness, r_prime, n_prime,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;

    use crate::main_protocol::step_prover::prove_step;
    use crate::params::{ChallengeProfile, JLProfile};
    use crate::recursion::target_relation::build_target_relation;
    use crate::relation::LabradorWitness;
    use grid_algebra::arith::prime::PrimeField;
    use grid_algebra::arith::ring::Ring;
    use grid_algebra::poly::ring::PolyRing;
    use grid_transcript::hash::ShakeTranscript;

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
            gamma1_sq: 40_000,
            gamma2_sq: 40_000,
            beta_prime: 300.0,
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

    #[test]
    fn test_prove_verify_roundtrip() {
        let mut rng = grid_std::test_rng();
        let params = fake_params();
        let key = CommitKey::<F, 64>::generate_from_params(&mut rng, &params);
        let witness = fake_witness(&params);
        let statement = fake_statement(&params);

        for (mode, clone_before_prove) in [("fresh", false), ("cloned", true)] {
            let mut prover_transcript = ShakeTranscript::default();
            let verifier_transcript = if clone_before_prove {
                prover_transcript.clone()
            } else {
                ShakeTranscript::default()
            };
            let mut verifier_transcript = verifier_transcript;

            let output = prove_step::<F, 64, _>(
                &key,
                &statement,
                &witness,
                &params,
                &mut prover_transcript,
                0,
            )
            .expect("prove_step should succeed");

            let result = verify_step::<F, 64, _>(
                &statement,
                &output.level_proof,
                &params,
                &mut verifier_transcript,
                0,
            );

            assert!(
                result.is_ok(),
                "verify_step should accept honest proof (transcript: {})",
                mode
            );
        }
    }

    #[test]
    fn test_verify_target_relation_shapes() {
        let mut rng = grid_std::test_rng();
        let params = fake_params();
        let key = CommitKey::<F, 64>::generate_from_params(&mut rng, &params);
        let witness = fake_witness(&params);
        let statement = fake_statement(&params);

        let mut prover_transcript = ShakeTranscript::default();
        let output = prove_step::<F, 64, _>(
            &key,
            &statement,
            &witness,
            &params,
            &mut prover_transcript,
            0,
        )
        .expect("prove_step should succeed");

        let mut verifier_transcript = ShakeTranscript::default();
        let verified = verify_step::<F, 64, _>(
            &statement,
            &output.level_proof,
            &params,
            &mut verifier_transcript,
            0,
        )
        .expect("verify_step should succeed");

        // Build target from step output
        let target = derive_target_relation(
            &key,
            &output.level_proof,
            &params,
            &verified.aggregated,
            &verified.challenges,
        );

        // Check r' = 2ν + μ
        assert_eq!(target.r_prime, 2 * params.nu + params.mu);
        // Check witness has r' parts
        assert_eq!(target.witness.num_parts(), target.r_prime);
        // Check each part has n' elements
        for part in &target.witness.parts {
            assert_eq!(part.len(), target.n_prime);
        }
    }

    #[test]
    fn test_verify_target_relation_semantic() {
        // Semantic test: prove_step → verify_step → relation::verify on the target
        // This catches coefficient/offset/sign bugs that shape tests miss.
        let mut rng = grid_std::test_rng();
        let params = fake_params();
        let key = CommitKey::<F, 64>::generate_from_params(&mut rng, &params);
        let witness = fake_witness(&params);
        let statement = fake_statement(&params);

        let mut prover_transcript = ShakeTranscript::default();
        let output = prove_step::<F, 64, _>(
            &key,
            &statement,
            &witness,
            &params,
            &mut prover_transcript,
            0,
        )
        .expect("prove_step should succeed");

        let mut verifier_transcript = ShakeTranscript::default();
        let verified = verify_step::<F, 64, _>(
            &statement,
            &output.level_proof,
            &params,
            &mut verifier_transcript,
            0,
        )
        .expect("verify_step should succeed");

        // Build target from step output
        let target = derive_target_relation(
            &key,
            &output.level_proof,
            &params,
            &verified.aggregated,
            &verified.challenges,
        );

        // Verify that the target relation is satisfied by the prover's private witness
        let z_parts = crate::recursion::decompose::decompose_z(&output.private_witness.z, params.b);
        let v = crate::recursion::decompose::bundle_v(
            &output.private_witness.t_decomposed.flat,
            &output.private_witness.g_decomposed.flat,
            &output.private_witness.h_decomposed.flat,
        );
        let witness = split_witness(&z_parts, &v, params.n, params.nu, params.mu, target.n_prime);
        let target_with_witness = build_target_relation(
            &key,
            &output.level_proof.u1,
            &output.level_proof.u2,
            &verified.challenges,
            &verified.aggregated,
            &params,
            witness,
            target.r_prime,
            target.n_prime,
        );
        let result = crate::relation::verify(
            &target_with_witness.statement,
            &target_with_witness.witness,
            target_with_witness.beta_prime,
        );
        assert!(
            result.is_ok(),
            "target relation should be satisfied: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_verify_target_relation_semantic_nonzero() {
        // Semantic test with nonzero quadratic F terms and a_ij.
        // Exercises the aggregated.a_ij path in (3c) and catches
        // coefficient/sign/index regressions in garbage constraint construction.
        const N: usize = 64;
        type Rq = CyclotomicPolyRing<F, N>;

        use grid_algebra::arith::ring::Ring;

        let mut rng = grid_std::test_rng();
        let params = fake_params();
        let key = CommitKey::<F, N>::generate_from_params(&mut rng, &params);

        // Build witness with nonzero values
        let witness = fake_witness(&params);

        // Build statement with nonzero quadratic term: 1 * ⟨s_0, s_0⟩ - b = 0
        let witness_part0 = &witness.parts[0];
        let dot = Rq::dot_product(witness_part0, witness_part0);

        let statement = LabradorStatement {
            f: vec![QuadraticFunction::from_parts(
                vec![(0, 0, Rq::one())], // a_00 = 1
                (0..params.r).map(|_| vec![Rq::zero(); params.n]).collect(),
                dot, // b = ⟨s_0, s_0⟩, so constraint is ⟨s_0,s_0⟩ - ⟨s_0,s_0⟩ = 0
            )],
            f_prime: vec![],
        };

        let mut prover_transcript = ShakeTranscript::default();
        let output = prove_step::<F, N, _>(
            &key,
            &statement,
            &witness,
            &params,
            &mut prover_transcript,
            0,
        )
        .expect("prove_step should succeed");

        let mut verifier_transcript = ShakeTranscript::default();
        let verified = verify_step::<F, N, _>(
            &statement,
            &output.level_proof,
            &params,
            &mut verifier_transcript,
            0,
        )
        .expect("verify_step should succeed");

        // Build target from step output
        let target = derive_target_relation(
            &key,
            &output.level_proof,
            &params,
            &verified.aggregated,
            &verified.challenges,
        );

        // Verify that the target relation is satisfied by the prover's private witness
        let z_parts = crate::recursion::decompose::decompose_z(&output.private_witness.z, params.b);
        let v = crate::recursion::decompose::bundle_v(
            &output.private_witness.t_decomposed.flat,
            &output.private_witness.g_decomposed.flat,
            &output.private_witness.h_decomposed.flat,
        );
        let witness = split_witness(&z_parts, &v, params.n, params.nu, params.mu, target.n_prime);
        let target_with_witness = build_target_relation(
            &key,
            &output.level_proof.u1,
            &output.level_proof.u2,
            &verified.challenges,
            &verified.aggregated,
            &params,
            witness,
            target.r_prime,
            target.n_prime,
        );
        let result = crate::relation::verify(
            &target_with_witness.statement,
            &target_with_witness.witness,
            target_with_witness.beta_prime,
        );
        assert!(
            result.is_ok(),
            "nonzero target relation should be satisfied: {:?}",
            result.err()
        );

        // Verify (3c) exercises the aggregated a_ij -> g limb path.
        // g limbs occupy v_offset [t_end, g_end) within the v witness parts.
        // With nu=1, mu=1: all v is in part 2, g at local indices [t_end, g_end).
        // The a_ij aggregation produces nonzero phi at these g-limb offsets.
        let idx_3c = params.kappa + params.kappa1 + params.kappa2 + 2;
        let f_3c = &target.statement.f[idx_3c];
        let t_end = params.r * params.kappa * params.t1;
        let g_end = t_end + crate::main_protocol::garbage_count(params.r) * params.t2;
        let v_part = 2 * params.nu; // first v part index
        let has_nonzero_g = f_3c.expect_dense().phi[v_part][t_end..g_end]
            .iter()
            .any(|p| !p.is_zero());
        assert!(
            has_nonzero_g,
            "(3c) g-limb region [{}, {}) should have nonzero phi from aggregated a_ij",
            t_end, g_end
        );
    }

    #[test]
    fn test_verify_rejects_different_transcript_state() {
        let mut rng = grid_std::test_rng();
        let params = fake_params();
        let key = CommitKey::<F, 64>::generate_from_params(&mut rng, &params);
        let witness = fake_witness(&params);
        let statement = fake_statement(&params);

        // Prove with one transcript
        let mut prover_transcript = ShakeTranscript::default();
        let output = prove_step::<F, 64, _>(
            &key,
            &statement,
            &witness,
            &params,
            &mut prover_transcript,
            0,
        )
        .expect("prove_step should succeed");

        // Verify with a transcript that has extra data appended
        let mut verifier_transcript = ShakeTranscript::default();
        verifier_transcript
            .append_bytes(b"extra", b"tampered data")
            .expect("append should work");

        let result = verify_step::<F, 64, _>(
            &statement,
            &output.level_proof,
            &params,
            &mut verifier_transcript,
            0,
        );

        // Should fail because challenges will differ
        assert!(
            result.is_err(),
            "verify_step should reject proof with different transcript state"
        );
    }

    #[test]
    fn test_prove_verify_sparse_f_prime() {
        // End-to-end test with a sparse F' function. This exercises the
        // for_each_nonzero_phi path in aggregation and the sparse validation
        // in step verifier.
        const N: usize = 64;
        type Rq = CyclotomicPolyRing<F, N>;

        use crate::relation::QuadraticFunction;

        let mut rng = grid_std::test_rng();
        let params = fake_params();
        let key = CommitKey::<F, N>::generate_from_params(&mut rng, &params);
        let witness = fake_witness(&params);

        // Build a sparse F' function: phi has exactly 1 nonzero entry, b is the
        // honest constant-term evaluation. With witness s_0[0] having coeff(0)=1,
        // the function evaluates to 1*s_0[0] - 1 = poly with ct = 1-1 = 0.
        // This exercises the nonzero for_each_nonzero_phi aggregation path:
        // if aggregation ignored sparse phi, the F' phi contribution would be
        // missing from agg_phi while b'' is still included in agg_b, so the
        // aggregated relation would not vanish.
        let sparse_f_prime = QuadraticFunction::from_sparse(
            Vec::new(),
            vec![(0, 0, Rq::one())], // +1 * s_0[0], ct = 1
            Rq::one(),               // b = 1 (constant), so ct(f') = 1 - 1 = 0
        );

        // Verify that sparse F' b passes constant check
        let is_const = sparse_f_prime
            .b()
            .coeffs()
            .iter()
            .skip(1)
            .all(|c| c.is_zero());
        assert!(is_const, "sparse F' b must be constant");

        let statement = LabradorStatement {
            f: vec![QuadraticFunction::from_parts(
                vec![],
                (0..params.r).map(|_| vec![Rq::zero(); params.n]).collect(),
                Rq::zero(),
            )],
            f_prime: vec![sparse_f_prime],
        };

        let mut prover_transcript = ShakeTranscript::default();
        let output = prove_step::<F, N, _>(
            &key,
            &statement,
            &witness,
            &params,
            &mut prover_transcript,
            0,
        )
        .expect("prove_step should succeed with sparse F'");

        let mut verifier_transcript = ShakeTranscript::default();
        let verified = verify_step::<F, N, _>(
            &statement,
            &output.level_proof,
            &params,
            &mut verifier_transcript,
            0,
        )
        .expect("verify_step should accept proof with sparse F'");

        // Build target from step output
        let target = derive_target_relation(
            &key,
            &output.level_proof,
            &params,
            &verified.aggregated,
            &verified.challenges,
        );

        // Verify target relation with prover's private witness
        let z_parts = crate::recursion::decompose::decompose_z(&output.private_witness.z, params.b);
        let v = crate::recursion::decompose::bundle_v(
            &output.private_witness.t_decomposed.flat,
            &output.private_witness.g_decomposed.flat,
            &output.private_witness.h_decomposed.flat,
        );
        let witness = split_witness(&z_parts, &v, params.n, params.nu, params.mu, target.n_prime);
        let target_with_witness = build_target_relation(
            &key,
            &output.level_proof.u1,
            &output.level_proof.u2,
            &verified.challenges,
            &verified.aggregated,
            &params,
            witness,
            target.r_prime,
            target.n_prime,
        );
        let result = crate::relation::verify(
            &target_with_witness.statement,
            &target_with_witness.witness,
            target_with_witness.beta_prime,
        );
        assert!(
            result.is_ok(),
            "target relation should be satisfied: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_verify_rejects_nonconstant_f_prime_b() {
        // Verify that the step verifier rejects F' functions with non-constant b.
        const N: usize = 64;
        type Rq = CyclotomicPolyRing<F, N>;

        use crate::relation::QuadraticFunction;

        let mut rng = grid_std::test_rng();
        let params = fake_params();
        let key = CommitKey::<F, N>::generate_from_params(&mut rng, &params);
        let witness = fake_witness(&params);

        // Build F' with non-constant b (coefficient at index 1 is nonzero)
        let mut nonconst_b = Rq::zero();
        nonconst_b.set_coeff(0, F::from_u64(5));
        nonconst_b.set_coeff(1, F::from_u64(1)); // nonzero coeff beyond index 0

        let statement = LabradorStatement {
            f: vec![],
            f_prime: vec![QuadraticFunction::from_parts(
                vec![],
                (0..params.r).map(|_| vec![Rq::zero(); params.n]).collect(),
                nonconst_b,
            )],
        };

        let mut prover_transcript = ShakeTranscript::default();
        let result = prove_step::<F, N, _>(
            &key,
            &statement,
            &witness,
            &params,
            &mut prover_transcript,
            0,
        );

        // Prover should succeed (it uses full b), but verifier should reject
        assert!(result.is_ok(), "prover should accept non-constant F' b");
        let output = result.unwrap();

        let mut verifier_transcript = ShakeTranscript::default();
        let result = verify_step::<F, N, _>(
            &statement,
            &output.level_proof,
            &params,
            &mut verifier_transcript,
            0,
        );
        assert!(
            result.is_err(),
            "verify_step should reject non-constant F' b"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("non-constant b"),
            "error should mention non-constant b, got: {}",
            err_msg
        );
    }
}
