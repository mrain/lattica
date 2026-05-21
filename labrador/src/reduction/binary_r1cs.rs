//! Binary R1CS → R reduction (§6, Figure 4, Theorem 6.2).
//!
//! Reduces a binary R1CS instance (matrices mod 2) to a LaBRADOR principal
//! relation R instance. Soundness error: `2^(-l)` where `l` is the number of
//! F2-linear combination challenges.
//!
//! # Witness layout
//!
//! All parts share the same rank `max_rank = max(k, n)`. Parts shorter than
//! `max_rank` are zero-padded:
//! - Parts 0-2 (a, b, c) and 4-6 (ã, b̃, c̃): length k, zero-padded if k < n
//! - Parts 3 (w) and 7 (w̃): length n, zero-padded if n < k

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use grid_algebra::arith::ring::{IntegerRing, Ring};
use grid_algebra::lattice::types::{RingMat, RingVec};
use grid_algebra::poly::ring::{CyclotomicPolyRing, NegacyclicMulRing, PolyRing};
use grid_std::rand::RngExt;
use grid_transcript::TranscriptError;

use crate::error::LabradorError;
use crate::relation::{LabradorStatement, LabradorWitness, QuadraticFunction, conjugation, verify};

macro_rules! bail {
    ($cond:expr, $msg:expr) => {
        if $cond {
            return Err(LabradorError::InvalidInput(alloc::string::String::from($msg)));
        }
    };
    ($cond:expr, $($arg:tt)*) => {
        if $cond {
            return Err(LabradorError::InvalidInput(alloc::format!($($arg)*)));
        }
    };
}

macro_rules! bail_t {
    ($cond:expr, $msg:expr) => {
        if $cond {
            return Err(TranscriptError::InvalidInput(alloc::string::String::from($msg)));
        }
    };
    ($cond:expr, $($arg:tt)*) => {
        if $cond {
            return Err(TranscriptError::InvalidInput(alloc::format!($($arg)*)));
        }
    };
}

/// Triple of binary challenge vectors (alpha, beta, gamma).
pub type BinaryChallenges<R> = (Vec<R>, Vec<R>, Vec<R>);

/// Binary R1CS instance.
#[derive(Debug, Clone)]
pub struct BinaryR1CSInstance<R: IntegerRing<Canonical = u64>> {
    pub a_r1cs: RingMat<R>,
    pub b_r1cs: RingMat<R>,
    pub c_r1cs: RingMat<R>,
}

/// Result of the binary R1CS → R reduction.
#[derive(Debug, Clone)]
pub struct BinaryR1CSReduction<R, const N: usize>
where
    R: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    /// LaBRADOR statement (F: Ajtai opening; F': binary + verifier checks)
    pub statement: LabradorStatement<CyclotomicPolyRing<R, N>>,
    /// LaBRADOR witness: (a, b, c, w, ã, b̃, c̃, w̃), all same rank
    pub witness: LabradorWitness<CyclotomicPolyRing<R, N>>,
    /// Ajtai commitment `t = A * (a||b||c||w)`
    pub commitment: RingVec<CyclotomicPolyRing<R, N>>,
    /// Number of F2-linear combination challenges (soundness: 2^(-l))
    pub l: usize,
    /// Integer values g_i = <α_i, a> + <β_i, b> + <γ_i, c> - <δ_i, w>.
    /// Verifier must check g_i ≡ 0 (mod 2) per paper §6 Figure 4.
    pub g_values: Vec<i64>,
    /// Number of R1CS constraints (rows of A, B, C matrices)
    pub k: usize,
    /// Number of R1CS variables (columns of A, B, C matrices)
    pub n: usize,
}

/// Create a constant polynomial from a u64 value.
fn constant_poly<R, const N: usize>(val: u64) -> CyclotomicPolyRing<R, N>
where
    R: IntegerRing + NegacyclicMulRing<N>,
{
    let mut poly = CyclotomicPolyRing::<R, N>::zero();
    poly.set_coeff(0, R::from_u64(val));
    poly
}

/// Create a monomial polynomial X^exp in Z[X]/(X^N+1).
fn monomial_poly<R, const N: usize>(exp: usize) -> CyclotomicPolyRing<R, N>
where
    R: IntegerRing + NegacyclicMulRing<N>,
{
    let mut poly = CyclotomicPolyRing::<R, N>::zero();
    poly.set_coeff(exp, R::one());
    poly
}

/// Pack integer coefficients into constant polynomials.
fn pack_to_poly<R, const N: usize>(coeffs: &[u64]) -> Vec<CyclotomicPolyRing<R, N>>
where
    R: IntegerRing + NegacyclicMulRing<N>,
{
    coeffs.iter().map(|v| constant_poly::<R, N>(*v)).collect()
}

/// Zero-pad a polynomial vector to the target rank.
fn pad_to_rank<R, const N: usize>(
    polys: &[CyclotomicPolyRing<R, N>],
    target: usize,
) -> Vec<CyclotomicPolyRing<R, N>>
where
    R: IntegerRing + NegacyclicMulRing<N>,
{
    let zero = CyclotomicPolyRing::<R, N>::zero();
    let mut result = polys.to_vec();
    result.resize(target, zero.clone());
    result
}

/// Compute `a = A_r1cs * w mod 2` for integer matrices.
fn mat_vec_mod2<R: IntegerRing<Canonical = u64>>(mat: &RingMat<R>, w: &[R]) -> Vec<u64> {
    let k = mat.rows();
    let n = mat.cols();
    let mut result = Vec::with_capacity(k);
    for i in 0..k {
        let mut sum: u64 = 0;
        for (j, wi) in w.iter().take(n).enumerate() {
            sum = (sum + mat.get(i, j).to_u64() * wi.to_u64()) % 2;
        }
        result.push(sum);
    }
    result
}

/// Build the binary R1CS → R reduction.
pub fn build_binary_r1cs_reduction<R, Rng, const N: usize>(
    instance: &BinaryR1CSInstance<R>,
    witness: &[R],
    crs_a: &RingMat<CyclotomicPolyRing<R, N>>,
    rng: &mut Rng,
    l: usize,
) -> Result<BinaryR1CSReduction<R, N>, LabradorError>
where
    R: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
    Rng: RngExt,
{
    let k = instance.a_r1cs.rows();
    let n = instance.a_r1cs.cols();

    bail!(
        instance.b_r1cs.rows() != k,
        "binary R1CS: b_r1cs rows ({}) != a_r1cs rows ({})",
        instance.b_r1cs.rows(),
        k
    );
    bail!(
        instance.b_r1cs.cols() != n,
        "binary R1CS: b_r1cs cols ({}) != a_r1cs cols ({})",
        instance.b_r1cs.cols(),
        n
    );
    bail!(
        instance.c_r1cs.rows() != k,
        "binary R1CS: c_r1cs rows ({}) != a_r1cs rows ({})",
        instance.c_r1cs.rows(),
        k
    );
    bail!(
        instance.c_r1cs.cols() != n,
        "binary R1CS: c_r1cs cols ({}) != a_r1cs cols ({})",
        instance.c_r1cs.cols(),
        n
    );
    bail!(
        witness.len() != n,
        "binary R1CS: witness len ({}) != expected ({})",
        witness.len(),
        n
    );

    // All witness parts must share the same rank
    let max_rank = k.max(n);

    // Step 1: Compute opened values a = A*w mod 2, etc.
    let a_vals = mat_vec_mod2(&instance.a_r1cs, witness);
    let b_vals = mat_vec_mod2(&instance.b_r1cs, witness);
    let c_vals = mat_vec_mod2(&instance.c_r1cs, witness);

    // Step 2: Pack into polynomials, then pad to max_rank
    let a_polys = pad_to_rank::<R, N>(&pack_to_poly::<R, N>(&a_vals), max_rank);
    let b_polys = pad_to_rank::<R, N>(&pack_to_poly::<R, N>(&b_vals), max_rank);
    let c_polys = pad_to_rank::<R, N>(&pack_to_poly::<R, N>(&c_vals), max_rank);
    let w_polys = pad_to_rank::<R, N>(
        &pack_to_poly::<R, N>(&witness.iter().map(|v| v.to_u64()).collect::<Vec<u64>>()),
        max_rank,
    );

    // Step 3: Conjugates
    let a_tilde: Vec<_> = a_polys.iter().map(conjugation).collect();
    let b_tilde: Vec<_> = b_polys.iter().map(conjugation).collect();
    let c_tilde: Vec<_> = c_polys.iter().map(conjugation).collect();
    let w_tilde: Vec<_> = w_polys.iter().map(conjugation).collect();

    // Step 4: Ajtai commitment t = A * (a[k]||b[k]||c[k]||w[n])
    let total_rank = 3 * k + n;
    let mut concat = Vec::with_capacity(total_rank);
    concat.extend_from_slice(&a_polys[..k]);
    concat.extend_from_slice(&b_polys[..k]);
    concat.extend_from_slice(&c_polys[..k]);
    concat.extend_from_slice(&w_polys[..n]);
    bail!(
        crs_a.cols() != total_rank,
        "binary R1CS: crs_a cols ({}) != expected total rank ({})",
        crs_a.cols(),
        total_rank
    );
    let commitment = crs_a.mul_slice(&concat);

    // Step 5: Witness parts: (a, b, c, w, ã, b̃, c̃, w̃), all max_rank
    let num_parts = 8;
    let labrador_witness = LabradorWitness::new(vec![
        a_polys.clone(),
        b_polys.clone(),
        c_polys.clone(),
        w_polys.clone(),
        a_tilde.clone(),
        b_tilde.clone(),
        c_tilde.clone(),
        w_tilde.clone(),
    ]);

    // Part indices
    const A: usize = 0;
    const B: usize = 1;
    const C: usize = 2;
    const W: usize = 3;
    const A_T: usize = 4;
    const B_T: usize = 5;
    const C_T: usize = 6;
    const W_T: usize = 7;

    // Step 6: F family — Ajtai commitment opening (fully vanishing)
    // Each row j of A: <row_j, concat> = t_j
    // phi segments: first k entries for a/b/c, next n entries for w, rest zero-padded
    let kappa = crs_a.rows();
    let mut f_functions: Vec<QuadraticFunction<CyclotomicPolyRing<R, N>>> =
        Vec::with_capacity(kappa);
    let zero = CyclotomicPolyRing::<R, N>::zero();

    for j in 0..kappa {
        let row = crs_a.row(j);
        let row_entries = row.entries();
        let mut offset = 0usize;

        // Build phi: each segment sliced from row, then zero-padded to max_rank
        let slice_pad = |seg: &[CyclotomicPolyRing<R, N>]| {
            let mut v: Vec<CyclotomicPolyRing<R, N>> = seg.to_vec();
            while v.len() < max_rank {
                v.push(zero.clone());
            }
            v
        };

        let phi: Vec<Vec<CyclotomicPolyRing<R, N>>> = vec![
            slice_pad(&row_entries[offset..offset + k]), // A
            {
                offset += k;
                slice_pad(&row_entries[offset..offset + k])
            }, // B
            {
                offset += k;
                slice_pad(&row_entries[offset..offset + k])
            }, // C
            {
                offset += k;
                slice_pad(&row_entries[offset..offset + n])
            }, // W
            vec![zero.clone(); max_rank],                // A_T (not in commitment)
            vec![zero.clone(); max_rank],                // B_T
            vec![zero.clone(); max_rank],                // C_T
            vec![zero.clone(); max_rank],                // W_T
        ];
        let b_val = commitment.get(j).clone();
        f_functions.push(QuadraticFunction::from_parts(Vec::new(), phi, b_val));
    }

    // Step 7: F' family (constant-term vanishing)
    let mut f_prime_functions: Vec<QuadraticFunction<CyclotomicPolyRing<R, N>>> = Vec::new();

    // Conjugacy constraints: x̃ = σ₋₁(x) enforced coefficient-wise in F' (paper §6 Figure 4, F₂).
    // σ₋₁(X^i) = X^(-i): in Z[X]/(X^N+1), σ₋₁(x)[0]=x[0], σ₋₁(x)[k]=-x[N-k] for k>0.
    // Each F' checks ct(monomial * x̃[j] ± monomial * x[j]) = 0 for ONE packed polynomial j.
    // Coefficient k=0: ct(x̃[j] - x[j]) = 0 → x̃[j][0] = x[j][0]
    // Coefficient k>0: ct(X^(N-k)*x̃[j] + X^k*x[j]) = -x̃[j][k] - x[j][N-k] = 0 → x̃[j][k] = -x[j][N-k]
    // Per-entry isolation: phi only has non-zero at index j (not aggregated across all entries).
    // Paper-aligned count: conjugacy for a(k), b(k), c(k), w(n) = N*(3k+n) total F' constraints.
    // Uses Sparse variant: each constraint has exactly 2 non-zero phi entries.
    {
        let zero_b = CyclotomicPolyRing::<R, N>::zero();
        for (part, part_tilde, num_entries) in [(A, A_T, k), (B, B_T, k), (C, C_T, k), (W, W_T, n)]
        {
            for j in 0..num_entries {
                for coeff in 0..N {
                    let phi_entries = if coeff == 0 {
                        // ct(x̃[j] - x[j]) = 0
                        vec![
                            (part_tilde, j, constant_poly::<R, N>(1)),
                            (part, j, -constant_poly::<R, N>(1)),
                        ]
                    } else {
                        // ct(X^(N-coeff)*x̃[j] + X^coeff*x[j]) = 0
                        vec![
                            (part_tilde, j, monomial_poly::<R, N>(N - coeff)),
                            (part, j, monomial_poly::<R, N>(coeff)),
                        ]
                    };
                    f_prime_functions.push(QuadraticFunction::from_sparse(
                        Vec::new(),
                        phi_entries,
                        zero_b.clone(),
                    ));
                }
            }
        }
    }

    // F' ordering: N*(3k+n) conjugacy + 4 binary + 1 Hadamard + l F2 challenges
    // 7a-d: Binary checks — ct(<ã, a> - <1, ã>) = 0
    // QuadraticFunction requires i <= j (upper-triangular).
    // Since dot_product is commutative, <parts[i], parts[j]> = <parts[j], parts[i]>.
    // We store (min, max) to satisfy the invariant.
    let parts_info = [(A, A_T, k), (B, B_T, k), (C, C_T, k), (W, W_T, n)];

    for (part_orig, part_tilde, len) in &parts_info {
        let one_poly = constant_poly::<R, N>(1);
        // Ensure i <= j: store (min(orig, tilde), max(orig, tilde))
        let (i, j) = if part_orig < part_tilde {
            (*part_orig, *part_tilde)
        } else {
            (*part_tilde, *part_orig)
        };
        let quad = vec![(i, j, one_poly.clone())];

        let mut phi: Vec<Vec<CyclotomicPolyRing<R, N>>> =
            vec![vec![zero.clone(); max_rank]; num_parts];
        // phi[part_tilde] = -1 for first `len` entries
        for p in phi[*part_tilde].iter_mut().take(*len) {
            *p = -one_poly.clone();
        }

        let zero_b = CyclotomicPolyRing::<R, N>::zero();
        f_prime_functions.push(QuadraticFunction::from_parts(quad, phi, zero_b));
    }

    // 7e: Hadamard product check — ct(<a+b-2c, ã+b̃-2c̃> - <a+b-2c, 1>) = 0
    {
        let one_poly = constant_poly::<R, N>(1);
        let two_poly = constant_poly::<R, N>(2);
        let four_poly = constant_poly::<R, N>(4);

        // Expand <a+b-2c, ã+b̃-2c̃> with i <= j for each term:
        let quad = vec![
            // <a, ã> = (A, A_T) → (0, 4) ✓
            (A, A_T, one_poly.clone()),
            // <a, b̃> = (A, B_T) → (0, 5) ✓
            (A, B_T, one_poly.clone()),
            // <a, c̃> = (A, C_T) → (0, 6) ✓
            (A, C_T, -two_poly.clone()),
            // <b, ã> = (B, A_T) → (1, 4) ✓
            (B, A_T, one_poly.clone()),
            // <b, b̃> = (B, B_T) → (1, 5) ✓
            (B, B_T, one_poly.clone()),
            // <b, c̃> = (B, C_T) → (1, 6) ✓
            (B, C_T, -two_poly.clone()),
            // <c, ã> = (C, A_T) → (2, 4) ✓
            (C, A_T, -two_poly.clone()),
            // <c, b̃> = (C, B_T) → (2, 5) ✓
            (C, B_T, -two_poly.clone()),
            // <c, c̃> = (C, C_T) → (2, 6) ✓
            (C, C_T, four_poly),
        ];

        let mut phi: Vec<Vec<CyclotomicPolyRing<R, N>>> =
            vec![vec![zero.clone(); max_rank]; num_parts];
        // Linear: -<a+b-2c, 1> = <-a,1> + <-b,1> + <2c,1>
        #[allow(clippy::needless_range_loop)]
        for idx in 0..k {
            phi[A][idx] = -one_poly.clone();
            phi[B][idx] = -one_poly.clone();
            phi[C][idx] = two_poly.clone();
        }

        let zero_b = CyclotomicPolyRing::<R, N>::zero();
        f_prime_functions.push(QuadraticFunction::from_parts(quad, phi, zero_b));
    }

    // 7f: Verifier challenges (F2-linear combinations)
    let mut g_values: Vec<i64> = Vec::with_capacity(l);
    for _ in 0..l {
        let alpha: Vec<u64> = (0..k).map(|_| rng.random_range(0..2)).collect();
        let beta: Vec<u64> = (0..k).map(|_| rng.random_range(0..2)).collect();
        let gamma: Vec<u64> = (0..k).map(|_| rng.random_range(0..2)).collect();

        // δ = (α * A^T + β * B^T + γ * C^T) mod 2
        // A^T is n×k, so α * A^T: entry j = Σ_i α_i * A[i][j]
        // A.get(i, j) = A[i][j] (row i, col j)
        let delta: Vec<u64> = (0..n)
            .map(|j| {
                let mut sum: u64 = 0;
                for i in 0..k {
                    sum = (sum + alpha[i] * instance.a_r1cs.get(i, j).to_u64()) % 2;
                    sum = (sum + beta[i] * instance.b_r1cs.get(i, j).to_u64()) % 2;
                    sum = (sum + gamma[i] * instance.c_r1cs.get(i, j).to_u64()) % 2;
                }
                sum
            })
            .collect();

        // Compute g_i as integer
        let g_i: i128 = alpha
            .iter()
            .zip(a_vals.iter())
            .map(|(a, v)| a * v)
            .sum::<u64>() as i128
            + beta
                .iter()
                .zip(b_vals.iter())
                .map(|(a, v)| a * v)
                .sum::<u64>() as i128
            + gamma
                .iter()
                .zip(c_vals.iter())
                .map(|(a, v)| a * v)
                .sum::<u64>() as i128
            - delta
                .iter()
                .zip(witness.iter())
                .map(|(d, w)| d * w.to_u64())
                .sum::<u64>() as i128;
        debug_assert!(g_i % 2 == 0, "g_i should be even for honest prover");
        g_values.push(g_i as i64);

        // Build F' constraint:
        // ct(<σ₋₁(α), a> + <σ₋₁(β), b> + <σ₋₁(γ), c> - <σ₋₁(δ), w>) = g_i
        let alpha_polys = pad_to_rank::<R, N>(&pack_to_poly::<R, N>(&alpha), max_rank);
        let beta_polys = pad_to_rank::<R, N>(&pack_to_poly::<R, N>(&beta), max_rank);
        let gamma_polys = pad_to_rank::<R, N>(&pack_to_poly::<R, N>(&gamma), max_rank);
        let delta_polys = pad_to_rank::<R, N>(&pack_to_poly::<R, N>(&delta), max_rank);

        let alpha_tilde: Vec<_> = alpha_polys.iter().map(conjugation).collect();
        let beta_tilde: Vec<_> = beta_polys.iter().map(conjugation).collect();
        let gamma_tilde: Vec<_> = gamma_polys.iter().map(conjugation).collect();
        let delta_tilde: Vec<_> = delta_polys.iter().map(conjugation).collect();

        let mut phi: Vec<Vec<CyclotomicPolyRing<R, N>>> =
            vec![vec![zero.clone(); max_rank]; num_parts];
        phi[A] = alpha_tilde;
        phi[B] = beta_tilde;
        phi[C] = gamma_tilde;
        for (p, d) in phi[W].iter_mut().zip(delta_tilde.iter()) {
            *p = -d.clone();
        }

        // b = constant polynomial with value g_i mod q
        let q: u64 = R::modulus();
        let g_abs: u64 = if g_i < 0 { (-g_i) as u64 } else { g_i as u64 };
        let g_mod = g_abs % q;
        let g_sign: u64 = if g_i < 0 {
            if g_mod == 0 { 0 } else { q - g_mod }
        } else {
            g_mod
        };
        let b_val = constant_poly::<R, N>(g_sign);

        f_prime_functions.push(QuadraticFunction::from_parts(Vec::new(), phi, b_val));
    }

    let statement = LabradorStatement {
        f: f_functions,
        f_prime: f_prime_functions,
    };

    Ok(BinaryR1CSReduction {
        statement,
        witness: labrador_witness,
        commitment,
        l,
        g_values,
        k,
        n,
    })
}

/// Verifier-side: generate binary R1CS challenges from an RNG.
pub fn sample_binary_challenges<R, Rng>(
    k: usize,
    l: usize,
    rng: &mut Rng,
) -> Vec<BinaryChallenges<R>>
where
    R: IntegerRing,
    Rng: RngExt,
{
    (0..l)
        .map(|_| {
            let alpha: Vec<R> = (0..k)
                .map(|_| R::from_u64(rng.random_range(0..2)))
                .collect();
            let beta: Vec<R> = (0..k)
                .map(|_| R::from_u64(rng.random_range(0..2)))
                .collect();
            let gamma: Vec<R> = (0..k)
                .map(|_| R::from_u64(rng.random_range(0..2)))
                .collect();
            (alpha, beta, gamma)
        })
        .collect()
}

/// Sample binary F2 challenges from a Fiat-Shamir transcript.
///
/// Each challenge vector is domain-separated by round index. The per-round
/// domain tag includes the round number to prevent cross-round collision.
pub fn sample_binary_challenges_transcript<R, T: grid_transcript::Transcript>(
    k: usize,
    l: usize,
    transcript: &mut T,
) -> Result<Vec<BinaryChallenges<R>>, TranscriptError>
where
    R: IntegerRing,
{
    let mut challenges = Vec::with_capacity(l);
    for round in 0..l {
        let round_bytes: Vec<u8> = round.to_le_bytes().to_vec();
        transcript.append_bytes(b"labrador_bin_round", &round_bytes)?;

        let alpha_bytes = transcript.challenge_bytes(b"labrador_bin_alpha", k)?;
        let beta_bytes = transcript.challenge_bytes(b"labrador_bin_beta", k)?;
        let gamma_bytes = transcript.challenge_bytes(b"labrador_bin_gamma", k)?;

        let alpha: Vec<R> = alpha_bytes
            .iter()
            .map(|b| R::from_u64((b & 1) as u64))
            .collect();
        let beta: Vec<R> = beta_bytes
            .iter()
            .map(|b| R::from_u64((b & 1) as u64))
            .collect();
        let gamma: Vec<R> = gamma_bytes
            .iter()
            .map(|b| R::from_u64((b & 1) as u64))
            .collect();

        challenges.push((alpha, beta, gamma));
    }
    Ok(challenges)
}

/// Build the binary R1CS → R reduction using transcript for verifier challenges.
///
/// Same as [`build_binary_r1cs_reduction`] but derives F2 challenges from the
/// Fiat-Shamir transcript instead of the prover's RNG. Commits `t` to the
/// transcript before sampling challenges.
pub fn build_binary_r1cs_reduction_transcript<R, T, const N: usize>(
    instance: &BinaryR1CSInstance<R>,
    witness: &[R],
    crs_a: &RingMat<CyclotomicPolyRing<R, N>>,
    transcript: &mut T,
    l: usize,
) -> Result<BinaryR1CSReduction<R, N>, TranscriptError>
where
    R: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
    T: grid_transcript::Transcript,
{
    let k = instance.a_r1cs.rows();
    let n = instance.a_r1cs.cols();

    bail_t!(
        instance.b_r1cs.rows() != k,
        "b_r1cs rows ({}) != a_r1cs rows ({})",
        instance.b_r1cs.rows(),
        k
    );
    bail_t!(
        instance.b_r1cs.cols() != n,
        "b_r1cs cols ({}) != a_r1cs cols ({})",
        instance.b_r1cs.cols(),
        n
    );
    bail_t!(
        instance.c_r1cs.rows() != k,
        "c_r1cs rows ({}) != a_r1cs rows ({})",
        instance.c_r1cs.rows(),
        k
    );
    bail_t!(
        instance.c_r1cs.cols() != n,
        "c_r1cs cols ({}) != a_r1cs cols ({})",
        instance.c_r1cs.cols(),
        n
    );
    bail_t!(
        witness.len() != n,
        "witness len ({}) != expected ({})",
        witness.len(),
        n
    );

    let max_rank = k.max(n);

    // Step 1: Compute opened values
    let a_vals = mat_vec_mod2(&instance.a_r1cs, witness);
    let b_vals = mat_vec_mod2(&instance.b_r1cs, witness);
    let c_vals = mat_vec_mod2(&instance.c_r1cs, witness);

    // Step 2: Pack into polynomials, then pad to max_rank
    let a_polys = pad_to_rank::<R, N>(&pack_to_poly::<R, N>(&a_vals), max_rank);
    let b_polys = pad_to_rank::<R, N>(&pack_to_poly::<R, N>(&b_vals), max_rank);
    let c_polys = pad_to_rank::<R, N>(&pack_to_poly::<R, N>(&c_vals), max_rank);
    let w_polys = pad_to_rank::<R, N>(
        &pack_to_poly::<R, N>(&witness.iter().map(|v| v.to_u64()).collect::<Vec<u64>>()),
        max_rank,
    );

    // Step 3: Conjugates
    let a_tilde: Vec<_> = a_polys.iter().map(conjugation).collect();
    let b_tilde: Vec<_> = b_polys.iter().map(conjugation).collect();
    let c_tilde: Vec<_> = c_polys.iter().map(conjugation).collect();
    let w_tilde: Vec<_> = w_polys.iter().map(conjugation).collect();

    // Step 4: Ajtai commitment t = A * (a||b||c||w)
    let total_rank = 3 * k + n;
    let mut concat = Vec::with_capacity(total_rank);
    concat.extend_from_slice(&a_polys[..k]);
    concat.extend_from_slice(&b_polys[..k]);
    concat.extend_from_slice(&c_polys[..k]);
    concat.extend_from_slice(&w_polys[..n]);
    bail_t!(
        crs_a.cols() != total_rank,
        "crs_a cols ({}) != expected total rank ({})",
        crs_a.cols(),
        total_rank
    );
    let commitment = crs_a.mul_slice(&concat);

    // Commit t to transcript before sampling challenges
    transcript.append_serializable(b"labrador_bin_t", &commitment)?;

    // Step 5: Witness parts
    let labrador_witness = LabradorWitness::new(vec![
        a_polys.clone(),
        b_polys.clone(),
        c_polys.clone(),
        w_polys.clone(),
        a_tilde.clone(),
        b_tilde.clone(),
        c_tilde.clone(),
        w_tilde.clone(),
    ]);

    const A: usize = 0;
    const B: usize = 1;
    const C: usize = 2;
    const W: usize = 3;
    const A_T: usize = 4;
    const B_T: usize = 5;
    const C_T: usize = 6;
    const W_T: usize = 7;
    let num_parts = 8;

    // Step 6: F family — Ajtai commitment opening
    let kappa = crs_a.rows();
    let mut f_functions: Vec<QuadraticFunction<CyclotomicPolyRing<R, N>>> =
        Vec::with_capacity(kappa);
    let zero = CyclotomicPolyRing::<R, N>::zero();

    for j in 0..kappa {
        let row = crs_a.row(j);
        let row_entries = row.entries();
        let mut offset = 0usize;

        let slice_pad = |seg: &[CyclotomicPolyRing<R, N>]| {
            let mut v: Vec<CyclotomicPolyRing<R, N>> = seg.to_vec();
            while v.len() < max_rank {
                v.push(zero.clone());
            }
            v
        };

        let phi: Vec<Vec<CyclotomicPolyRing<R, N>>> = vec![
            slice_pad(&row_entries[offset..offset + k]),
            {
                offset += k;
                slice_pad(&row_entries[offset..offset + k])
            },
            {
                offset += k;
                slice_pad(&row_entries[offset..offset + k])
            },
            {
                offset += k;
                slice_pad(&row_entries[offset..offset + n])
            },
            vec![zero.clone(); max_rank],
            vec![zero.clone(); max_rank],
            vec![zero.clone(); max_rank],
            vec![zero.clone(); max_rank],
        ];
        let b_val = commitment.get(j).clone();
        f_functions.push(QuadraticFunction::from_parts(Vec::new(), phi, b_val));
    }

    // Step 7: F' family (constant-term vanishing)
    let mut f_prime_functions: Vec<QuadraticFunction<CyclotomicPolyRing<R, N>>> = Vec::new();

    // Conjugacy constraints: x̃ = σ₋₁(x) enforced coefficient-wise in F' (paper §6 Figure 4, F₂).
    // σ₋₁(X^i) = X^(-i): in Z[X]/(X^N+1), σ₋₁(x)[0]=x[0], σ₋₁(x)[k]=-x[N-k] for k>0.
    // Each F' checks ct(monomial * x̃[j] ± monomial * x[j]) = 0 for ONE packed polynomial j.
    // Per-entry isolation: phi only has non-zero at index j (not aggregated across all entries).
    // Paper-aligned count: conjugacy for a(k), b(k), c(k), w(n) = N*(3k+n) total F' constraints.
    // Uses Sparse variant: each constraint has exactly 2 non-zero phi entries.
    {
        let zero_b = CyclotomicPolyRing::<R, N>::zero();
        for (part, part_tilde, num_entries) in [(A, A_T, k), (B, B_T, k), (C, C_T, k), (W, W_T, n)]
        {
            for j in 0..num_entries {
                for coeff in 0..N {
                    let phi_entries = if coeff == 0 {
                        vec![
                            (part_tilde, j, constant_poly::<R, N>(1)),
                            (part, j, -constant_poly::<R, N>(1)),
                        ]
                    } else {
                        vec![
                            (part_tilde, j, monomial_poly::<R, N>(N - coeff)),
                            (part, j, monomial_poly::<R, N>(coeff)),
                        ]
                    };
                    f_prime_functions.push(QuadraticFunction::from_sparse(
                        Vec::new(),
                        phi_entries,
                        zero_b.clone(),
                    ));
                }
            }
        }
    }

    // F' ordering: N*(3k+n) conjugacy + 4 binary + 1 Hadamard + l F2 challenges
    // 7a-d: Binary checks
    let parts_info = [(A, A_T, k), (B, B_T, k), (C, C_T, k), (W, W_T, n)];

    for (part_orig, part_tilde, len) in &parts_info {
        let one_poly = constant_poly::<R, N>(1);
        let (i, j) = if part_orig < part_tilde {
            (*part_orig, *part_tilde)
        } else {
            (*part_tilde, *part_orig)
        };
        let quad = vec![(i, j, one_poly.clone())];

        let mut phi: Vec<Vec<CyclotomicPolyRing<R, N>>> =
            vec![vec![zero.clone(); max_rank]; num_parts];
        for p in phi[*part_tilde].iter_mut().take(*len) {
            *p = -one_poly.clone();
        }

        let zero_b = CyclotomicPolyRing::<R, N>::zero();
        f_prime_functions.push(QuadraticFunction::from_parts(quad, phi, zero_b));
    }

    // 7e: Hadamard product check
    {
        let one_poly = constant_poly::<R, N>(1);
        let two_poly = constant_poly::<R, N>(2);
        let four_poly = constant_poly::<R, N>(4);

        let quad = vec![
            (A, A_T, one_poly.clone()),
            (A, B_T, one_poly.clone()),
            (A, C_T, -two_poly.clone()),
            (B, A_T, one_poly.clone()),
            (B, B_T, one_poly.clone()),
            (B, C_T, -two_poly.clone()),
            (C, A_T, -two_poly.clone()),
            (C, B_T, -two_poly.clone()),
            (C, C_T, four_poly),
        ];

        let mut phi: Vec<Vec<CyclotomicPolyRing<R, N>>> =
            vec![vec![zero.clone(); max_rank]; num_parts];
        // Linear: -<a+b-2c, 1> = <-a,1> + <-b,1> + <2c,1>
        #[allow(clippy::needless_range_loop)]
        for idx in 0..k {
            phi[A][idx] = -one_poly.clone();
            phi[B][idx] = -one_poly.clone();
            phi[C][idx] = two_poly.clone();
        }

        let zero_b = CyclotomicPolyRing::<R, N>::zero();
        f_prime_functions.push(QuadraticFunction::from_parts(quad, phi, zero_b));
    }

    // 7f: Verifier challenges (F2-linear combinations) from transcript
    let mut g_values: Vec<i64> = Vec::with_capacity(l);
    let challenges = sample_binary_challenges_transcript::<R, T>(k, l, transcript)?;
    for (alpha, beta, gamma) in challenges {
        let alpha: Vec<u64> = alpha.iter().map(|v| v.to_u64()).collect();
        let beta: Vec<u64> = beta.iter().map(|v| v.to_u64()).collect();
        let gamma: Vec<u64> = gamma.iter().map(|v| v.to_u64()).collect();

        // δ = (α * A^T + β * B^T + γ * C^T) mod 2
        let delta: Vec<u64> = (0..n)
            .map(|j| {
                let mut sum: u64 = 0;
                for i in 0..k {
                    sum = (sum + alpha[i] * instance.a_r1cs.get(i, j).to_u64()) % 2;
                    sum = (sum + beta[i] * instance.b_r1cs.get(i, j).to_u64()) % 2;
                    sum = (sum + gamma[i] * instance.c_r1cs.get(i, j).to_u64()) % 2;
                }
                sum
            })
            .collect();

        // Compute g_i
        let g_i: i128 = alpha
            .iter()
            .zip(a_vals.iter())
            .map(|(a, v)| a * v)
            .sum::<u64>() as i128
            + beta
                .iter()
                .zip(b_vals.iter())
                .map(|(a, v)| a * v)
                .sum::<u64>() as i128
            + gamma
                .iter()
                .zip(c_vals.iter())
                .map(|(a, v)| a * v)
                .sum::<u64>() as i128
            - delta
                .iter()
                .zip(witness.iter())
                .map(|(d, w)| d * w.to_u64())
                .sum::<u64>() as i128;
        debug_assert!(g_i % 2 == 0, "g_i should be even for honest prover");
        g_values.push(g_i as i64);

        let alpha_polys = pad_to_rank::<R, N>(&pack_to_poly::<R, N>(&alpha), max_rank);
        let beta_polys = pad_to_rank::<R, N>(&pack_to_poly::<R, N>(&beta), max_rank);
        let gamma_polys = pad_to_rank::<R, N>(&pack_to_poly::<R, N>(&gamma), max_rank);
        let delta_polys = pad_to_rank::<R, N>(&pack_to_poly::<R, N>(&delta), max_rank);

        let alpha_tilde: Vec<_> = alpha_polys.iter().map(conjugation).collect();
        let beta_tilde: Vec<_> = beta_polys.iter().map(conjugation).collect();
        let gamma_tilde: Vec<_> = gamma_polys.iter().map(conjugation).collect();
        let delta_tilde: Vec<_> = delta_polys.iter().map(conjugation).collect();

        let mut phi: Vec<Vec<CyclotomicPolyRing<R, N>>> =
            vec![vec![zero.clone(); max_rank]; num_parts];
        phi[A] = alpha_tilde;
        phi[B] = beta_tilde;
        phi[C] = gamma_tilde;
        for (p, d) in phi[W].iter_mut().zip(delta_tilde.iter()) {
            *p = -d.clone();
        }

        let q: u64 = R::modulus();
        let g_abs: u64 = if g_i < 0 { (-g_i) as u64 } else { g_i as u64 };
        let g_mod = g_abs % q;
        let g_sign: u64 = if g_i < 0 {
            if g_mod == 0 { 0 } else { q - g_mod }
        } else {
            g_mod
        };
        let b_val = constant_poly::<R, N>(g_sign);

        f_prime_functions.push(QuadraticFunction::from_parts(Vec::new(), phi, b_val));
    }

    let statement = LabradorStatement {
        f: f_functions,
        f_prime: f_prime_functions,
    };

    Ok(BinaryR1CSReduction {
        statement,
        witness: labrador_witness,
        commitment,
        l,
        g_values,
        k,
        n,
    })
}

/// Compute the F2 start index in F' for the binary reduction.
///
/// Paper §6 Figure 4: conjugacy applies to a(k), b(k), c(k), w(n) = 3k+n packed
/// entries. Each packed entry has N coefficients. So conjugacy uses N*(3k+n) F'.
/// F' layout: [N*(3k+n) conjugacy][4 binary][1 Hadamard][l F2]
pub fn binary_f2_start<const N: usize>(k: usize, n: usize) -> usize {
    N * (3 * k + n) + 4 + 1
}

/// Check that g-values from F2-challenges are all even.
///
/// Returns `Ok(())` if all g_i ≡ 0 (mod 2), or `Err(index)` of the first odd g_i.
pub fn check_g_even(g_values: &[i64]) -> Result<(), usize> {
    for (i, &g) in g_values.iter().enumerate() {
        if g % 2 != 0 {
            return Err(i);
        }
    }
    Ok(())
}

/// Verify a binary R1CS reduction: relation::verify + g_values parity check.
///
/// This is the complete verifier path per paper §6 Figure 4.
/// Extracts g_i from the F' b constants (last l F' functions) and checks
/// g_i ≡ 0 (mod 2), matching the proven statement values rather than side metadata.
/// Returns `Ok(())` if both the LaBRADOR relation holds and all g_i ≡ 0 (mod 2).
pub fn verify_binary_r1cs_reduction<R, const N: usize>(
    reduction: &BinaryR1CSReduction<R, N>,
    max_norm_bound: f64,
) -> Result<(), String>
where
    R: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    // Extract g_i from F' b constants (last l F' functions)
    // F' layout per paper §6 Figure 4: [N*(3k+n) conjugacy][4 binary][1 Hadamard][l F2]
    let f2_start = binary_f2_start::<N>(reduction.k, reduction.n);
    let f_prime = &reduction.statement.f_prime;
    if f_prime.len() < f2_start + reduction.l {
        return Err(format!(
            "F' has {} functions, need {} ({} F2 start + {} l)",
            f_prime.len(),
            f2_start + reduction.l,
            f2_start,
            reduction.l
        ));
    }

    // Extract g_i from F' b constants and check parity
    for i in 0..reduction.l {
        let g_mod = f_prime[f2_start + i].b().coeff(0).to_u64();
        let q = R::modulus();
        // Convert from mod q representation back to signed integer for parity check
        let g_signed: i64 = if g_mod > q / 2 {
            -((q - g_mod) as i64)
        } else {
            g_mod as i64
        };
        if g_signed % 2 != 0 {
            return Err(format!(
                "binary g_values[{}] is odd (parity check failed, g={} from F' b constant)",
                i, g_signed
            ));
        }
    }

    // Verify LaBRADOR relation
    verify(&reduction.statement, &reduction.witness, max_norm_bound)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::relation::verify;

    use grid_algebra::arith::prime::PrimeField;
    use grid_algebra::poly::ring::PolyRing;

    type F = PrimeField<17>;

    #[test]
    fn test_mat_vec_mod2() {
        let mat = RingMat::new(
            2,
            3,
            vec![
                F::from_u64(1),
                F::from_u64(0),
                F::from_u64(1),
                F::from_u64(0),
                F::from_u64(1),
                F::from_u64(1),
            ],
        );
        let w = vec![F::from_u64(1), F::from_u64(0), F::from_u64(1)];
        let result = mat_vec_mod2(&mat, &w);
        assert_eq!(result, vec![0, 1]);
    }

    #[test]
    fn test_pack_to_poly() {
        let polys = pack_to_poly::<F, 8>(&[1, 0, 1]);
        assert_eq!(polys.len(), 3);
        assert_eq!(polys[0].coeff(0).to_u64(), 1);
        assert_eq!(polys[1].coeff(0).to_u64(), 0);
        assert_eq!(polys[2].coeff(0).to_u64(), 1);
    }

    #[test]
    fn test_pad_to_rank() {
        let polys = pack_to_poly::<F, 8>(&[1, 2]);
        let padded = pad_to_rank::<F, 8>(&polys, 5);
        assert_eq!(padded.len(), 5);
        assert_eq!(padded[0].coeff(0).to_u64(), 1);
        assert_eq!(padded[1].coeff(0).to_u64(), 2);
        for p in padded.iter().skip(2) {
            assert!(p.is_zero());
        }
    }

    #[test]
    fn test_binary_check_constant_term() {
        let a_polys = pack_to_poly::<F, 8>(&[1, 0, 1]);
        let a_tilde: Vec<_> = a_polys.iter().map(conjugation).collect();

        let dot_ã_a = Ring::dot_product(&a_tilde, &a_polys);
        assert_eq!(dot_ã_a.coeff(0).to_u64(), 2);

        let ones = pack_to_poly::<F, 8>(&[1, 1, 1]);
        let dot_1_ã = Ring::dot_product(&ones, &a_tilde);
        assert_eq!(dot_1_ã.coeff(0).to_u64(), 2);

        let diff = dot_ã_a - dot_1_ã;
        assert!(diff.coeff(0).is_zero());
    }

    #[test]
    fn test_hadamard_binary_check() {
        let a = pack_to_poly::<F, 8>(&[1]);
        let b = pack_to_poly::<F, 8>(&[1]);
        let c = pack_to_poly::<F, 8>(&[1]);

        let v: Vec<_> = a
            .iter()
            .zip(b.iter())
            .zip(c.iter())
            .map(|((ai, bi), ci)| ai.clone() + bi.clone() - ci.clone() - ci.clone())
            .collect();
        let v_t: Vec<_> = v.iter().map(conjugation).collect();

        let dot_v = Ring::dot_product(&v_t, &v);
        let ones = pack_to_poly::<F, 8>(&[1]);
        let dot_ones = Ring::dot_product(&ones, &v);
        let diff = dot_v - dot_ones;
        assert!(diff.coeff(0).is_zero(), "ab=c: v=0, diff ct should be 0");

        let a2 = pack_to_poly::<F, 8>(&[1]);
        let b2 = pack_to_poly::<F, 8>(&[0]);
        let c2 = pack_to_poly::<F, 8>(&[1]);
        let v2: Vec<_> = a2
            .iter()
            .zip(b2.iter())
            .zip(c2.iter())
            .map(|((ai, bi), ci)| ai.clone() + bi.clone() - ci.clone() - ci.clone())
            .collect();
        let v2_t: Vec<_> = v2.iter().map(conjugation).collect();
        let dot_v2 = Ring::dot_product(&v2_t, &v2);
        let ones2 = pack_to_poly::<F, 8>(&[1]);
        let dot_ones2 = Ring::dot_product(&ones2, &v2);
        let diff2 = dot_v2 - dot_ones2;
        assert!(!diff2.coeff(0).is_zero(), "ab≠c: diff ct should be nonzero");
    }

    #[test]
    fn test_check_g_even() {
        assert!(check_g_even(&[0, 2, -4, 6]).is_ok());
        assert!(check_g_even(&[]).is_ok());
        assert_eq!(check_g_even(&[0, 1, 2]), Err(1));
    }

    #[test]
    fn test_full_binary_r1cs_reduction_smoke() {
        let k = 2;
        let n = 2;
        let kappa = 2;

        let a_r1cs = RingMat::new(
            k,
            n,
            vec![
                F::from_u64(1),
                F::from_u64(0),
                F::from_u64(0),
                F::from_u64(1),
            ],
        );
        let b_r1cs = RingMat::new(
            k,
            n,
            vec![
                F::from_u64(1),
                F::from_u64(1),
                F::from_u64(0),
                F::from_u64(1),
            ],
        );
        let c_r1cs = RingMat::new(
            k,
            n,
            vec![
                F::from_u64(1),
                F::from_u64(0),
                F::from_u64(0),
                F::from_u64(1),
            ],
        );

        let instance = BinaryR1CSInstance {
            a_r1cs,
            b_r1cs,
            c_r1cs,
        };
        let witness = vec![F::from_u64(1), F::from_u64(0)];

        let total_rank = 3 * k + n;
        let crs_a = RingMat::new(
            kappa,
            total_rank,
            (0..kappa * total_rank)
                .map(|i| constant_poly::<F, 8>((i % 17) as u64))
                .collect(),
        );

        let mut rng = grid_std::test_rng();
        let reduction =
            build_binary_r1cs_reduction::<F, _, 8>(&instance, &witness, &crs_a, &mut rng, 2)
                .unwrap();

        assert_eq!(reduction.witness.num_parts(), 8);
        assert_eq!(reduction.witness.rank(), 2); // max(k=2, n=2) = 2
        assert_eq!(reduction.commitment.len(), kappa);
        assert_eq!(reduction.l, 2);
        assert_eq!(reduction.statement.num_f(), kappa); // kappa Ajtai only
        assert_eq!(
            reduction.statement.num_f_prime(),
            8 * (3 * 2 + 2) + 4 + 1 + 2
        ); // N*(3k+n)=N*8 conjugacy + 4 binary + 1 Hadamard + l=2 F2

        // Verify all quad terms satisfy i <= j (dense: ij, sparse: ij_a)
        for f in reduction
            .statement
            .f
            .iter()
            .chain(reduction.statement.f_prime.iter())
        {
            match f {
                crate::relation::QuadraticFunction::Dense(d) => {
                    for &(i, j) in &d.ij {
                        assert!(i <= j, "quad index ({}, {}) violates i <= j", i, j);
                    }
                }
                crate::relation::QuadraticFunction::Sparse(s) => {
                    for &(i, j, _) in &s.ij_a {
                        assert!(i <= j, "quad index ({}, {}) violates i <= j", i, j);
                    }
                }
            }
        }
    }

    #[test]
    fn test_reduction_mismatched_k_n() {
        // k=3, n=2 → max_rank=3, w and w̃ padded to 3
        let k = 3;
        let n = 2;
        let kappa = 2;

        let a_r1cs = RingMat::new(
            k,
            n,
            vec![
                F::from_u64(1),
                F::from_u64(0),
                F::from_u64(0),
                F::from_u64(1),
                F::from_u64(0),
                F::from_u64(0),
            ],
        );
        let b_r1cs = RingMat::new(k, n, vec![F::from_u64(0); k * n]);
        let c_r1cs = RingMat::new(
            k,
            n,
            vec![
                F::from_u64(1),
                F::from_u64(0),
                F::from_u64(0),
                F::from_u64(1),
                F::from_u64(0),
                F::from_u64(0),
            ],
        );

        let instance = BinaryR1CSInstance {
            a_r1cs,
            b_r1cs,
            c_r1cs,
        };
        let witness = vec![F::from_u64(1), F::from_u64(0)];

        let total_rank = 3 * k + n;
        let crs_a = RingMat::new(
            kappa,
            total_rank,
            vec![constant_poly::<F, 8>(0); kappa * total_rank],
        );

        let mut rng = grid_std::test_rng();
        let reduction =
            build_binary_r1cs_reduction::<F, _, 8>(&instance, &witness, &crs_a, &mut rng, 1)
                .unwrap();

        // All parts should have rank = max(3, 2) = 3
        assert_eq!(reduction.witness.rank(), 3);
        for (i, part) in reduction.witness.parts.iter().enumerate() {
            assert_eq!(part.len(), 3, "part {} has wrong rank", i);
        }

        // All Dense phi vectors should have length = max_rank = 3
        // Sparse functions have individual entries, not full phi vectors
        for (fi, f) in reduction
            .statement
            .f
            .iter()
            .chain(reduction.statement.f_prime.iter())
            .enumerate()
        {
            if let crate::relation::QuadraticFunction::Dense(d) = f {
                for (pi, phi_i) in d.phi.iter().enumerate() {
                    assert_eq!(phi_i.len(), 3, "F[{}] phi[{}] has wrong length", fi, pi);
                }
            }
        }
    }

    #[test]
    fn test_reduction_verify_roundtrip() {
        // Test that relation::verify accepts the reduction output
        let k = 2;
        let n = 3; // k != n to test padding
        let kappa = 2;

        let a_r1cs = RingMat::new(
            k,
            n,
            vec![
                F::from_u64(1),
                F::from_u64(0),
                F::from_u64(0),
                F::from_u64(0),
                F::from_u64(1),
                F::from_u64(0),
            ],
        );
        let b_r1cs = RingMat::new(
            k,
            n,
            vec![
                F::from_u64(1),
                F::from_u64(1),
                F::from_u64(0),
                F::from_u64(0),
                F::from_u64(1),
                F::from_u64(0),
            ],
        );
        let c_r1cs = RingMat::new(
            k,
            n,
            vec![
                F::from_u64(1),
                F::from_u64(0),
                F::from_u64(0),
                F::from_u64(0),
                F::from_u64(1),
                F::from_u64(0),
            ],
        );

        let instance = BinaryR1CSInstance {
            a_r1cs,
            b_r1cs,
            c_r1cs,
        };
        let witness = vec![F::from_u64(1), F::from_u64(0), F::from_u64(1)];

        // w = [1,0,1], a = Aw = [1,0], b = Bw = [1,0], c = Cw = [1,0]
        // a∘b = [1,0] = c ✓

        let total_rank = 3 * k + n;
        let crs_a = RingMat::new(
            kappa,
            total_rank,
            vec![constant_poly::<F, 8>(0); kappa * total_rank],
        );

        let mut rng = grid_std::test_rng();
        let reduction =
            build_binary_r1cs_reduction::<F, _, 8>(&instance, &witness, &crs_a, &mut rng, 2)
                .unwrap();

        // Use a large beta since binary witnesses have small norms
        let result = verify(&reduction.statement, &reduction.witness, 1000.0);
        assert!(
            result.is_ok(),
            "reduction should pass relation::verify: {:?}",
            result
        );
    }

    #[test]
    fn test_sample_binary_challenges() {
        let mut rng = grid_std::test_rng();
        let challenges = sample_binary_challenges::<F, _>(3, 5, &mut rng);
        assert_eq!(challenges.len(), 5);
        for (alpha, beta, gamma) in &challenges {
            assert_eq!(alpha.len(), 3);
            for v in alpha.iter().chain(beta.iter()).chain(gamma.iter()) {
                let val = v.to_u64();
                assert!(val == 0 || val == 1);
            }
        }
    }

    #[test]
    fn test_conjugation_trick() {
        let alpha = pack_to_poly::<F, 8>(&[1, 0, 1]);
        let a = pack_to_poly::<F, 8>(&[1, 1, 0]);
        let alpha_t: Vec<_> = alpha.iter().map(conjugation).collect();

        let dot = Ring::dot_product(&alpha_t, &a);
        assert_eq!(dot.coeff(0).to_u64(), 1);
    }

    #[test]
    fn test_binary_reduction_transcript() {
        use grid_transcript::hash::ShakeTranscript;

        let k = 2;
        let n = 2;
        let kappa = 2;

        let a_r1cs = RingMat::new(
            k,
            n,
            vec![
                F::from_u64(1),
                F::from_u64(0),
                F::from_u64(0),
                F::from_u64(1),
            ],
        );
        let b_r1cs = RingMat::new(
            k,
            n,
            vec![
                F::from_u64(1),
                F::from_u64(1),
                F::from_u64(0),
                F::from_u64(1),
            ],
        );
        let c_r1cs = RingMat::new(
            k,
            n,
            vec![
                F::from_u64(1),
                F::from_u64(0),
                F::from_u64(0),
                F::from_u64(1),
            ],
        );

        let instance = BinaryR1CSInstance {
            a_r1cs,
            b_r1cs,
            c_r1cs,
        };
        let witness = vec![F::from_u64(1), F::from_u64(0)];

        let total_rank = 3 * k + n;
        let crs_a = RingMat::new(
            kappa,
            total_rank,
            (0..kappa * total_rank)
                .map(|i| constant_poly::<F, 8>((i % 17) as u64))
                .collect(),
        );

        let mut transcript = ShakeTranscript::default();
        let reduction = build_binary_r1cs_reduction_transcript::<F, _, 8>(
            &instance,
            &witness,
            &crs_a,
            &mut transcript,
            2,
        )
        .expect("transcript builder should succeed");

        assert_eq!(reduction.witness.num_parts(), 8);
        assert_eq!(reduction.witness.rank(), 2);
        assert_eq!(reduction.commitment.len(), kappa);
        assert_eq!(reduction.l, 2);
        assert_eq!(reduction.statement.num_f(), kappa); // kappa Ajtai only
        assert_eq!(
            reduction.statement.num_f_prime(),
            8 * (3 * 2 + 2) + 4 + 1 + 2
        ); // N*(3k+n)=N*8 conjugacy + 4 binary + 1 Hadamard + l=2 F2

        // Verify relation holds
        let result = verify(&reduction.statement, &reduction.witness, 1000.0);
        assert!(result.is_ok(), "reduction should pass verify: {:?}", result);
    }

    #[test]
    fn test_sample_binary_challenges_transcript() {
        use grid_transcript::hash::ShakeTranscript;

        let mut transcript = ShakeTranscript::default();
        let challenges =
            sample_binary_challenges_transcript::<F, _>(3, 5, &mut transcript).expect("ok");
        assert_eq!(challenges.len(), 5);
        for (alpha, beta, gamma) in &challenges {
            assert_eq!(alpha.len(), 3);
            for v in alpha.iter().chain(beta.iter()).chain(gamma.iter()) {
                let val = v.to_u64();
                assert!(val == 0 || val == 1);
            }
        }
    }
}
