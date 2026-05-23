//! Binary R1CS → R reduction (§6, Figure 4, Theorem 6.2).
//!
//! Reduces a binary R1CS instance (matrices over GF2) to a LaBRADOR principal
//! relation R instance. Soundness error: `2^(-l)` where `l` is the number of
//! F2-linear combination challenges.
//!
//! # Paper representation
//!
//! The paper packs N binary scalars into one ring element in
//! `R_q = Z_q[X]/(X^N+1)`, one bit per coefficient position. Vectors are
//! zero-padded to multiples of N before packing:
//!
//! ```text
//! k_padded = ceil(k/N)*N    n_padded = ceil(n/N)*N
//! k_packed = k_padded/N     n_packed = n_padded/N
//! pack(bits)[j] = Σ_{i=0}^{N-1} bits[j*N + i] · X^i
//! ```
//!
//! Missing entries (beyond the original k or n) are zero.
//!
//! # Witness layout (8 parts, rank = max(k_packed, n_packed))
//!
//! ```text
//! 0: a_pack          length k_packed
//! 1: b_pack          length k_packed
//! 2: c_pack          length k_packed
//! 3: w_pack          length n_packed
//! 4: σ₋₁(a)          length k_packed
//! 5: σ₋₁(b)          length k_packed
//! 6: σ₋₁(c)          length k_packed
//! 7: σ₋₁(w)          length n_packed
//! ```
//!
//! Commitment columns: `3·k_packed + n_packed`.

use alloc::vec;
use alloc::vec::Vec;

use grid_algebra::arith::gf2::GF2;
use grid_algebra::arith::ring::{IntegerRing, Ring};
use grid_algebra::lattice::types::{RingMat, RingVec};
use grid_algebra::poly::ring::{CyclotomicPolyRing, NegacyclicMulRing, PolyRing};
use grid_std::rand::RngExt;
use grid_transcript::TranscriptError;

use crate::error::LabradorError;
use crate::relation::{LabradorStatement, LabradorWitness, QuadraticFunction, conjugation};

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

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Triple of binary challenge vectors (alpha, beta, gamma).
pub type BinaryChallenges = (Vec<i64>, Vec<i64>, Vec<i64>);

/// Binary R1CS instance over GF2.
#[derive(Debug, Clone)]
pub struct BinaryR1CSInstance {
    pub a_r1cs: RingMat<GF2>,
    pub b_r1cs: RingMat<GF2>,
    pub c_r1cs: RingMat<GF2>,
}

/// Result of the binary R1CS → R reduction.
#[derive(Debug, Clone)]
pub struct BinaryR1CSReduction<P, const N: usize>
where
    P: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    pub statement: LabradorStatement<CyclotomicPolyRing<P, N>>,
    pub witness: LabradorWitness<CyclotomicPolyRing<P, N>>,
    pub commitment: RingVec<CyclotomicPolyRing<P, N>>,
    pub l: usize,
    pub g_values: Vec<i64>,
    pub k: usize,
    pub n: usize,
    pub k_padded: usize,
    pub n_padded: usize,
    pub k_packed: usize,
    pub n_packed: usize,
    /// Start index of F2 challenge constraints in f_prime.
    pub binary_f2_start: usize,
}

/// Witness and statement without the top-level Ajtai opening.
///
/// Exposed for mixed R1CS which replaces the standalone Ajtai opening with a
/// single combined commitment.
pub(crate) struct BinaryR1CSFragment<P, const N: usize>
where
    P: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    pub statement_without_opening: LabradorStatement<CyclotomicPolyRing<P, N>>,
    pub witness_parts: Vec<Vec<CyclotomicPolyRing<P, N>>>,
    pub a_pack: Vec<CyclotomicPolyRing<P, N>>,
    pub b_pack: Vec<CyclotomicPolyRing<P, N>>,
    pub c_pack: Vec<CyclotomicPolyRing<P, N>>,
    pub w_pack: Vec<CyclotomicPolyRing<P, N>>,
    pub k: usize,
    pub n: usize,
    pub k_padded: usize,
    pub n_padded: usize,
    pub k_packed: usize,
    pub n_packed: usize,
    pub l: usize,
    pub binary_f2_start: usize,
}

/// Challenge vectors for one F2 round.
pub(crate) struct BinaryChallengeSet {
    pub alpha: Vec<u64>,
    pub beta: Vec<u64>,
    pub gamma: Vec<u64>,
}

// ---------------------------------------------------------------------------
// Polynomial helpers
// ---------------------------------------------------------------------------

fn constant_poly<P, const N: usize>(val: P) -> CyclotomicPolyRing<P, N>
where
    P: IntegerRing + NegacyclicMulRing<N>,
{
    let mut poly = CyclotomicPolyRing::<P, N>::zero();
    poly.set_coeff(0, val);
    poly
}

fn monomial_poly<P, const N: usize>(exp: usize) -> CyclotomicPolyRing<P, N>
where
    P: IntegerRing + NegacyclicMulRing<N>,
{
    let mut poly = CyclotomicPolyRing::<P, N>::zero();
    poly.set_coeff(exp, P::one());
    poly
}

fn zero_poly<P, const N: usize>() -> CyclotomicPolyRing<P, N>
where
    P: IntegerRing + NegacyclicMulRing<N>,
{
    CyclotomicPolyRing::<P, N>::zero()
}

/// The polynomial `s` such that `ct(s · p) = Σᵢ coeffᵢ(p)` for all p.
///
/// `coeff_sum_selector = 1 - X - X² - ... - X^{N-1}`
///   `= σ₋₁(1 + X + ... + X^{N-1})`
///
/// Used in packed binaryity checks (§6 Fig 4, F₂).
fn coeff_sum_selector<P, const N: usize>() -> CyclotomicPolyRing<P, N>
where
    P: IntegerRing + NegacyclicMulRing<N>,
{
    let mut s = CyclotomicPolyRing::<P, N>::zero();
    s.set_coeff(0, P::one()); // coefficient 0 = 1
    for i in 1..N {
        s.set_coeff(i, -P::one()); // coefficients 1..N-1 = -1
    }
    s
}

/// Pack a slice of GF2 bits into `len/N` polynomials (one bit per coefficient).
///
/// `padded_len` must be a multiple of N. Missing entries beyond `bits.len()`
/// are zero.
fn pack_gf2<P, const N: usize>(bits: &[GF2], padded_len: usize) -> Vec<CyclotomicPolyRing<P, N>>
where
    P: IntegerRing + NegacyclicMulRing<N>,
{
    debug_assert!(padded_len.is_multiple_of(N));
    let packed = padded_len / N;
    let mut out = Vec::with_capacity(packed);
    for j in 0..packed {
        let mut poly = CyclotomicPolyRing::<P, N>::zero();
        for i in 0..N {
            let idx = j * N + i;
            if idx < bits.len() && !bits[idx].is_zero() {
                poly.set_coeff(i, P::one());
            }
        }
        out.push(poly);
    }
    out
}

/// Pack a slice of u64 (0 or 1) into `len/N` polynomials.
///
/// Same semantics as [`pack_gf2`] but reads from `&[u64]` where entries are
/// expected to be 0 or 1.
fn pack_u64<P, const N: usize>(bits: &[u64], padded_len: usize) -> Vec<CyclotomicPolyRing<P, N>>
where
    P: IntegerRing + NegacyclicMulRing<N>,
{
    debug_assert!(padded_len.is_multiple_of(N));
    let packed = padded_len / N;
    let mut out = Vec::with_capacity(packed);
    for j in 0..packed {
        let mut poly = CyclotomicPolyRing::<P, N>::zero();
        for i in 0..N {
            let idx = j * N + i;
            if idx < bits.len() && bits[idx] != 0 {
                poly.set_coeff(i, P::one());
            }
        }
        out.push(poly);
    }
    out
}

/// Pad a vector to `target` entries, filling with zero polynomials.
fn pad_entries<P, const N: usize>(
    entries: Vec<CyclotomicPolyRing<P, N>>,
    target: usize,
) -> Vec<CyclotomicPolyRing<P, N>>
where
    P: IntegerRing + NegacyclicMulRing<N>,
{
    let mut v = entries;
    v.resize_with(target, zero_poly::<P, N>);
    v
}

// ---------------------------------------------------------------------------
// Coefficients selector helpers for padding-zero F' constraints
// ---------------------------------------------------------------------------

/// Returns `selector(r)` where `ct(selector(r) · part[j]) = coeff_r(part[j])`.
///
/// `selector(0) = 1`, `selector(r) = -X^{N-r}` for r > 0.
fn coeff_selector<P, const N: usize>(r: usize) -> CyclotomicPolyRing<P, N>
where
    P: IntegerRing + NegacyclicMulRing<N>,
{
    if r == 0 {
        constant_poly::<P, N>(P::one())
    } else {
        // -X^{N-r}
        let mut p = CyclotomicPolyRing::<P, N>::zero();
        p.set_coeff(N - r, -P::one());
        p
    }
}

// ---------------------------------------------------------------------------
// Theorem 6.2 bound validation
// ---------------------------------------------------------------------------

fn validate_theorem_bounds<P>(k_padded: usize, n_padded: usize) -> Result<(), LabradorError>
where
    P: IntegerRing<Canonical = u64>,
{
    let q = P::modulus() as u128;
    let lhs = (n_padded + 3 * k_padded) as u128;
    bail!(lhs >= q, "n+3k ({lhs}) >= q ({q}), violates Theorem 6.2");
    bail!(
        6 * (k_padded as u128) >= q,
        "6k ({}) >= q ({}), violates Theorem 6.2",
        6 * k_padded,
        q
    );
    bail!(
        128 * lhs >= 15 * q,
        "128·(n+3k) ({}) >= 15q ({}), insufficient slack for main protocol",
        128 * lhs,
        15 * q
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Core builder: pre-F2 phase (everything before F2 challenge constraints)
// ---------------------------------------------------------------------------

/// Data produced by the pre-F2 phase of the binary reduction.
///
/// Carries everything needed to compute the commitment and then build the
/// F2 challenge constraints. Exposed `pub(crate)` so that the mixed R1CS
/// builder can compute a combined commitment before sampling challenges.
pub(crate) struct BinaryPreF2<P, const N: usize>
where
    P: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    pub(crate) a_pack: Vec<CyclotomicPolyRing<P, N>>,
    pub(crate) b_pack: Vec<CyclotomicPolyRing<P, N>>,
    pub(crate) c_pack: Vec<CyclotomicPolyRing<P, N>>,
    pub(crate) w_pack: Vec<CyclotomicPolyRing<P, N>>,
    pub(crate) witness_parts: Vec<Vec<CyclotomicPolyRing<P, N>>>,
    pub(crate) f_prime: Vec<QuadraticFunction<CyclotomicPolyRing<P, N>>>,
    pub(crate) f2_start: usize,
    pub(crate) a_bits: RingVec<GF2>,
    pub(crate) b_bits: RingVec<GF2>,
    pub(crate) c_bits: RingVec<GF2>,
    pub(crate) witness_bits: Vec<GF2>,
    pub(crate) k: usize,
    pub(crate) n: usize,
    pub(crate) k_padded: usize,
    pub(crate) n_padded: usize,
    pub(crate) k_packed: usize,
    pub(crate) n_packed: usize,
    pub(crate) rank: usize,
    pub(crate) num_parts: usize,
}

/// Build pre-F2 data: compute R1CS values, pack, conjugates, and all F'
/// constraints except the F2 challenge constraints.
///
/// Returns the packed values and partial statement. The caller must compute
/// the Ajtai commitment (and for the transcript path, absorb it) before
/// sampling F2 challenges and calling [`build_binary_f2_constraints`].
fn build_binary_pre_f2<P, const N: usize>(
    instance: &BinaryR1CSInstance,
    witness: &[GF2],
) -> Result<BinaryPreF2<P, N>, LabradorError>
where
    P: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    let k = instance.a_r1cs.rows();
    let n = instance.a_r1cs.cols();

    // --- Shape validation (was in old builder, restore) ---
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

    // --- Padded / packed dimensions ---
    let k_packed = k.div_ceil(N);
    let n_packed = n.div_ceil(N);
    let k_padded = k_packed * N;
    let n_padded = n_packed * N;
    let rank = k_packed.max(n_packed);

    // --- Step 1: Compute R1CS values (scalar GF2 vectors) ---
    let a_bits = instance.a_r1cs.mul_slice(witness);
    let b_bits = instance.b_r1cs.mul_slice(witness);
    let c_bits = instance.c_r1cs.mul_slice(witness);

    // --- Step 2: Pack into polynomials and pad to rank ---
    let a_pack_r = pad_entries(pack_gf2::<P, N>(a_bits.entries(), k_padded), rank);
    let b_pack_r = pad_entries(pack_gf2::<P, N>(b_bits.entries(), k_padded), rank);
    let c_pack_r = pad_entries(pack_gf2::<P, N>(c_bits.entries(), k_padded), rank);
    let w_pack_r = pad_entries(pack_gf2::<P, N>(witness, n_padded), rank);

    // Owned unpadded vectors for commitment construction
    let a_pack = a_pack_r[..k_packed].to_vec();
    let b_pack = b_pack_r[..k_packed].to_vec();
    let c_pack = c_pack_r[..k_packed].to_vec();
    let w_pack = w_pack_r[..n_packed].to_vec();

    // --- Step 3: Conjugates ---
    let a_tilde: Vec<_> = a_pack_r.iter().map(conjugation).collect();
    let b_tilde: Vec<_> = b_pack_r.iter().map(conjugation).collect();
    let c_tilde: Vec<_> = c_pack_r.iter().map(conjugation).collect();
    let w_tilde: Vec<_> = w_pack_r.iter().map(conjugation).collect();

    let num_parts: usize = 8;
    let witness_parts: Vec<Vec<CyclotomicPolyRing<P, N>>> = vec![
        a_pack_r, b_pack_r, c_pack_r, w_pack_r, a_tilde, b_tilde, c_tilde, w_tilde,
    ];

    const A: usize = 0;
    const B: usize = 1;
    const C: usize = 2;
    const W: usize = 3;
    const A_T: usize = 4;
    const B_T: usize = 5;
    const C_T: usize = 6;
    const W_T: usize = 7;

    let zero = zero_poly::<P, N>();

    // ------------------------------------------------------------------
    // F' constraints
    // ------------------------------------------------------------------
    let mut f_prime: Vec<QuadraticFunction<CyclotomicPolyRing<P, N>>> = Vec::new();

    // --- Conjugacy (N constraints per packed entry) ---
    {
        let parts: [(usize, usize, usize); 4] = [
            (A, A_T, k_packed),
            (B, B_T, k_packed),
            (C, C_T, k_packed),
            (W, W_T, n_packed),
        ];
        for (part, part_tilde, len) in parts {
            for j in 0..len {
                for coeff in 0..N {
                    let phi_entries = if coeff == 0 {
                        vec![
                            (part_tilde, j, constant_poly::<P, N>(P::one())),
                            (part, j, -constant_poly::<P, N>(P::one())),
                        ]
                    } else {
                        vec![
                            (part_tilde, j, monomial_poly::<P, N>(N - coeff)),
                            (part, j, monomial_poly::<P, N>(coeff)),
                        ]
                    };
                    f_prime.push(QuadraticFunction::from_sparse(
                        Vec::new(),
                        phi_entries,
                        zero.clone(),
                    ));
                }
            }
        }
    }

    // --- Padding-zero constraints ---
    let _padding_start = f_prime.len();
    {
        for scalar_idx in k..k_padded {
            let poly_idx = scalar_idx / N;
            let coeff = scalar_idx % N;
            let phi_entries = vec![(A, poly_idx, coeff_selector::<P, N>(coeff))];
            f_prime.push(QuadraticFunction::from_sparse(
                Vec::new(),
                phi_entries,
                zero.clone(),
            ));
        }
        for scalar_idx in k..k_padded {
            let poly_idx = scalar_idx / N;
            let coeff = scalar_idx % N;
            let phi_entries = vec![(B, poly_idx, coeff_selector::<P, N>(coeff))];
            f_prime.push(QuadraticFunction::from_sparse(
                Vec::new(),
                phi_entries,
                zero.clone(),
            ));
        }
        for scalar_idx in k..k_padded {
            let poly_idx = scalar_idx / N;
            let coeff = scalar_idx % N;
            let phi_entries = vec![(C, poly_idx, coeff_selector::<P, N>(coeff))];
            f_prime.push(QuadraticFunction::from_sparse(
                Vec::new(),
                phi_entries,
                zero.clone(),
            ));
        }
        for scalar_idx in n..n_padded {
            let poly_idx = scalar_idx / N;
            let coeff = scalar_idx % N;
            let phi_entries = vec![(W, poly_idx, coeff_selector::<P, N>(coeff))];
            f_prime.push(QuadraticFunction::from_sparse(
                Vec::new(),
                phi_entries,
                zero.clone(),
            ));
        }
    }
    let _padding_zero_count = f_prime.len() - _padding_start;

    // --- Binaryity checks ---
    {
        let selector = coeff_sum_selector::<P, N>();
        let parts_info: [(usize, usize, usize); 4] = [
            (A, A_T, k_packed),
            (B, B_T, k_packed),
            (C, C_T, k_packed),
            (W, W_T, n_packed),
        ];

        for (part_orig, part_tilde, len) in &parts_info {
            let one_poly = constant_poly::<P, N>(P::one());
            let qp = if part_orig < part_tilde {
                (*part_orig, *part_tilde)
            } else {
                (*part_tilde, *part_orig)
            };

            let mut phi: Vec<Vec<CyclotomicPolyRing<P, N>>> =
                vec![vec![zero.clone(); rank]; num_parts];
            for p in phi[*part_orig].iter_mut().take(*len) {
                *p = -selector.clone();
            }

            f_prime.push(QuadraticFunction::from_parts(
                vec![(qp.0, qp.1, one_poly)],
                phi,
                zero.clone(),
            ));
        }
    }

    // --- Hadamard ---
    {
        let one = constant_poly::<P, N>(P::one());
        let two = constant_poly::<P, N>(P::from_u64(2));
        let four = constant_poly::<P, N>(P::from_u64(4));
        let selector = coeff_sum_selector::<P, N>();

        let quad = vec![
            (A, A_T, one.clone()),
            (A, B_T, one.clone()),
            (A, C_T, -two.clone()),
            (B, A_T, one.clone()),
            (B, B_T, one.clone()),
            (B, C_T, -two.clone()),
            (C, A_T, -two.clone()),
            (C, B_T, -two.clone()),
            (C, C_T, four),
        ];

        let mut phi: Vec<Vec<CyclotomicPolyRing<P, N>>> = vec![vec![zero.clone(); rank]; num_parts];
        for idx in 0..k_packed {
            phi[A][idx] = -selector.clone();
            phi[B][idx] = -selector.clone();
            phi[C][idx] = selector.clone() + selector.clone();
        }

        f_prime.push(QuadraticFunction::from_parts(quad, phi, zero.clone()));
    }

    let f2_start = f_prime.len();

    Ok(BinaryPreF2 {
        a_pack,
        b_pack,
        c_pack,
        w_pack,
        witness_parts,
        f_prime,
        f2_start,
        a_bits,
        b_bits,
        c_bits,
        witness_bits: witness.to_vec(),
        k,
        n,
        k_padded,
        n_padded,
        k_packed,
        n_packed,
        rank,
        num_parts,
    })
}

/// Build F2 challenge constraints and append them to `f_prime`.
///
/// `challenges` provides `l` pre-sampled [`BinaryChallengeSet`]s.
/// The caller must have already computed and (for transcript paths) absorbed
/// the Ajtai commitment before sampling challenges.
fn build_binary_f2_constraints<P, const N: usize>(
    pre: &mut BinaryPreF2<P, N>,
    instance: &BinaryR1CSInstance,
    challenges: &[BinaryChallengeSet],
    g_values_out: &mut Vec<i64>,
) where
    P: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    #[allow(non_snake_case)]
    let (A, B, C, W): (usize, usize, usize, usize) = (0, 1, 2, 3);

    let zero = zero_poly::<P, N>();

    for challenge in challenges {
        // δ = Lift(α·A^T + β·B^T + γ·C^T mod 2) ∈ Z_q^{n_padded}
        let delta: Vec<u64> = (0..pre.n_padded)
            .map(|col| {
                let mut sum: u64 = 0;
                if col < pre.n {
                    for row in 0..pre.k {
                        sum = (sum + challenge.alpha[row] * instance.a_r1cs.get(row, col).to_u64())
                            % 2;
                        sum = (sum + challenge.beta[row] * instance.b_r1cs.get(row, col).to_u64())
                            % 2;
                        sum = (sum + challenge.gamma[row] * instance.c_r1cs.get(row, col).to_u64())
                            % 2;
                    }
                }
                sum
            })
            .collect();

        // g_i over padded scalar vectors (padded entries are zero)
        let g_i: i128 = {
            let mut acc: i128 = 0;
            for i in 0..pre.k {
                acc += challenge.alpha[i] as i128 * pre.a_bits.entries()[i].to_u64() as i128;
                acc += challenge.beta[i] as i128 * pre.b_bits.entries()[i].to_u64() as i128;
                acc += challenge.gamma[i] as i128 * pre.c_bits.entries()[i].to_u64() as i128;
            }
            for i in 0..pre.n {
                acc -= delta[i] as i128 * pre.witness_bits[i].to_u64() as i128;
            }
            acc
        };
        debug_assert!(g_i % 2 == 0, "g_i should be even for honest prover");
        g_values_out.push(g_i as i64);

        // Build F' constraint:
        // ct(<σ₋₁(pack(α)), a_pack> + <σ₋₁(pack(β)), b_pack>
        //   + <σ₋₁(pack(γ)), c_pack> - <σ₋₁(pack(δ)), w_pack>) = g_i
        let alpha_pack = pack_u64::<P, N>(&challenge.alpha, pre.k_padded);
        let beta_pack = pack_u64::<P, N>(&challenge.beta, pre.k_padded);
        let gamma_pack = pack_u64::<P, N>(&challenge.gamma, pre.k_padded);
        let delta_pack = pack_u64::<P, N>(&delta, pre.n_padded);

        let alpha_pack_r = pad_entries(alpha_pack, pre.rank);
        let beta_pack_r = pad_entries(beta_pack, pre.rank);
        let gamma_pack_r = pad_entries(gamma_pack, pre.rank);
        let delta_pack_r = pad_entries(delta_pack, pre.rank);

        let alpha_tilde: Vec<_> = alpha_pack_r.iter().map(conjugation).collect();
        let beta_tilde: Vec<_> = beta_pack_r.iter().map(conjugation).collect();
        let gamma_tilde: Vec<_> = gamma_pack_r.iter().map(conjugation).collect();
        let delta_tilde: Vec<_> = delta_pack_r.iter().map(conjugation).collect();

        let mut phi: Vec<Vec<CyclotomicPolyRing<P, N>>> =
            vec![vec![zero.clone(); pre.rank]; pre.num_parts];
        phi[A] = alpha_tilde;
        phi[B] = beta_tilde;
        phi[C] = gamma_tilde;
        for (p, d) in phi[W].iter_mut().zip(delta_tilde.iter()) {
            *p = -d.clone();
        }

        let q = P::modulus();
        let g_abs = g_i.unsigned_abs();
        let g_mod = (g_abs % q as u128) as u64;
        let g_canon = if g_i < 0 {
            if g_mod == 0 { 0 } else { q - g_mod }
        } else {
            g_mod
        };
        let b_val = constant_poly::<P, N>(P::from_u64(g_canon));

        pre.f_prime
            .push(QuadraticFunction::from_parts(Vec::new(), phi, b_val));
    }
}

// ---------------------------------------------------------------------------
// Fragment builder (for mixed R1CS reuse — two-phase API)
// ---------------------------------------------------------------------------

/// Phase 1: build pre-F2 data and return it alongside the packed vectors.
///
/// The caller must compute the (possibly combined) Ajtai commitment using
/// the returned `a_pack`/`b_pack`/`c_pack`/`w_pack`, absorb the commitment
/// into the transcript, then sample F2 challenges and call
/// [`finish_binary_fragment`].
pub(crate) fn build_binary_pre_f2_fragment<P, const N: usize>(
    instance: &BinaryR1CSInstance,
    witness: &[GF2],
) -> Result<BinaryPreF2<P, N>, LabradorError>
where
    P: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    build_binary_pre_f2::<P, N>(instance, witness)
}

/// Phase 2: finish the fragment by building F2 constraints.
///
/// `challenges` must have length `l` and each entry must have `k_padded`
/// coefficients. Returns the completed fragment (without Ajtai opening F).
pub(crate) fn finish_binary_fragment<P, const N: usize>(
    pre: &mut BinaryPreF2<P, N>,
    instance: &BinaryR1CSInstance,
    l: usize,
    challenges: &[BinaryChallengeSet],
    g_values_out: &mut Vec<i64>,
) -> Result<BinaryR1CSFragment<P, N>, LabradorError>
where
    P: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    bail!(
        challenges.len() != l,
        "binary fragment: challenges.len() ({}) != l ({})",
        challenges.len(),
        l
    );
    for (i, ch) in challenges.iter().enumerate() {
        bail!(
            ch.alpha.len() != pre.k_padded
                || ch.beta.len() != pre.k_padded
                || ch.gamma.len() != pre.k_padded,
            "binary fragment: challenge[{}] length mismatch (expected k_padded={})",
            i,
            pre.k_padded,
        );
    }

    build_binary_f2_constraints::<P, N>(pre, instance, challenges, g_values_out);

    let binary_f2_start = pre.f2_start;
    let statement = LabradorStatement {
        f: Vec::new(),
        f_prime: core::mem::take(&mut pre.f_prime),
    };

    Ok(BinaryR1CSFragment {
        statement_without_opening: statement,
        witness_parts: core::mem::take(&mut pre.witness_parts),
        a_pack: core::mem::take(&mut pre.a_pack),
        b_pack: core::mem::take(&mut pre.b_pack),
        c_pack: core::mem::take(&mut pre.c_pack),
        w_pack: core::mem::take(&mut pre.w_pack),
        k: pre.k,
        n: pre.n,
        k_padded: pre.k_padded,
        n_padded: pre.n_padded,
        k_packed: pre.k_packed,
        n_packed: pre.n_packed,
        l,
        binary_f2_start,
    })
}

/// Prepend Ajtai opening F constraints and compute commitment.
fn attach_opening<P, const N: usize>(
    statement: &mut LabradorStatement<CyclotomicPolyRing<P, N>>,
    a_pack: &[CyclotomicPolyRing<P, N>],
    b_pack: &[CyclotomicPolyRing<P, N>],
    c_pack: &[CyclotomicPolyRing<P, N>],
    w_pack: &[CyclotomicPolyRing<P, N>],
    crs_a: &RingMat<CyclotomicPolyRing<P, N>>,
    rank: usize,
) -> Result<RingVec<CyclotomicPolyRing<P, N>>, LabradorError>
where
    P: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    let k_packed = a_pack.len();
    let n_packed = w_pack.len();
    let total_cols = 3 * k_packed + n_packed;

    bail!(
        crs_a.cols() != total_cols,
        "binary CRS: cols ({}) != expected 3·k_packed + n_packed ({})",
        crs_a.cols(),
        total_cols
    );

    // Build concatenated vector: a_pack || b_pack || c_pack || w_pack
    let mut concat = Vec::with_capacity(total_cols);
    concat.extend_from_slice(a_pack);
    concat.extend_from_slice(b_pack);
    concat.extend_from_slice(c_pack);
    concat.extend_from_slice(w_pack);
    let commitment = crs_a.mul_slice(&concat);

    let kappa = crs_a.rows();
    let zero = zero_poly::<P, N>();
    let _num_parts = 8;
    let mut f: Vec<QuadraticFunction<CyclotomicPolyRing<P, N>>> = Vec::with_capacity(kappa);

    for j in 0..kappa {
        let row = crs_a.row(j);
        let row_entries = row.entries();
        let mut offset = 0usize;

        let slice_pad = |seg: &[CyclotomicPolyRing<P, N>]| {
            let mut v: Vec<CyclotomicPolyRing<P, N>> = seg.to_vec();
            while v.len() < rank {
                v.push(zero.clone());
            }
            v
        };

        let phi: Vec<Vec<CyclotomicPolyRing<P, N>>> = vec![
            slice_pad(&row_entries[offset..offset + k_packed]),
            {
                offset += k_packed;
                slice_pad(&row_entries[offset..offset + k_packed])
            },
            {
                offset += k_packed;
                slice_pad(&row_entries[offset..offset + k_packed])
            },
            {
                offset += k_packed;
                slice_pad(&row_entries[offset..offset + n_packed])
            },
            vec![zero.clone(); rank],
            vec![zero.clone(); rank],
            vec![zero.clone(); rank],
            vec![zero.clone(); rank],
        ];

        f.push(QuadraticFunction::from_parts(
            Vec::new(),
            phi,
            commitment.get(j).clone(),
        ));
    }

    // Prepend F to statement (F was empty)
    statement.f = f;
    Ok(commitment)
}

// ---------------------------------------------------------------------------
// Public builders
// ---------------------------------------------------------------------------

/// Build the binary R1CS → R reduction (RNG-based challenges).
pub fn build_binary_r1cs_reduction<P, Rng, const N: usize>(
    instance: &BinaryR1CSInstance,
    witness: &[GF2],
    crs_a: &RingMat<CyclotomicPolyRing<P, N>>,
    rng: &mut Rng,
    l: usize,
) -> Result<BinaryR1CSReduction<P, N>, LabradorError>
where
    P: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
    Rng: RngExt,
{
    let k = instance.a_r1cs.rows();
    let n = instance.a_r1cs.cols();

    validate_theorem_bounds::<P>(k.div_ceil(N) * N, n.div_ceil(N) * N)?;

    // Phase 1: build pre-F2 data
    let mut pre = build_binary_pre_f2::<P, N>(instance, witness)?;

    // Phase 2: compute commitment t = A·(a||b||c||w) (Figure 4 order)
    let rank = pre.k_packed.max(pre.n_packed);
    let mut statement = LabradorStatement {
        f: Vec::new(),
        f_prime: core::mem::take(&mut pre.f_prime),
    };
    let commitment = attach_opening::<P, N>(
        &mut statement,
        &pre.a_pack,
        &pre.b_pack,
        &pre.c_pack,
        &pre.w_pack,
        crs_a,
        rank,
    )?;
    // Restore f_prime for F2 building
    pre.f_prime = core::mem::take(&mut statement.f_prime);

    // Phase 3: sample F2 challenges via RNG
    let mut g_values = Vec::with_capacity(l);
    let challenges: Vec<BinaryChallengeSet> = (0..l)
        .map(|_| {
            let alpha: Vec<u64> = (0..pre.k_padded).map(|_| rng.random_range(0..2)).collect();
            let beta: Vec<u64> = (0..pre.k_padded).map(|_| rng.random_range(0..2)).collect();
            let gamma: Vec<u64> = (0..pre.k_padded).map(|_| rng.random_range(0..2)).collect();
            BinaryChallengeSet { alpha, beta, gamma }
        })
        .collect();

    // Phase 4: build F2 constraints
    build_binary_f2_constraints::<P, N>(&mut pre, instance, &challenges, &mut g_values);
    statement.f_prime = pre.f_prime;

    let witness = LabradorWitness::new(pre.witness_parts);

    Ok(BinaryR1CSReduction {
        statement,
        witness,
        commitment,
        l,
        g_values,
        k,
        n,
        k_padded: pre.k_padded,
        n_padded: pre.n_padded,
        k_packed: pre.k_packed,
        n_packed: pre.n_packed,
        binary_f2_start: pre.f2_start,
    })
}

/// Build the binary R1CS → R reduction (transcript-based challenges).
///
/// Follows the paper's Fiat-Shamir order (§6 Figure 4):
/// 1. Prover builds pre-F2 data
/// 2. Prover sends commitment t = A·(a||b||c||w)
/// 3. Transcript absorbs t
/// 4. Verifier samples α,β,γ
/// 5. Prover builds F2 constraints with the sampled challenges
pub fn build_binary_r1cs_reduction_transcript<P, T, const N: usize>(
    instance: &BinaryR1CSInstance,
    witness: &[GF2],
    crs_a: &RingMat<CyclotomicPolyRing<P, N>>,
    transcript: &mut T,
    l: usize,
) -> Result<BinaryR1CSReduction<P, N>, TranscriptError>
where
    P: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
    T: grid_transcript::Transcript,
{
    let k = instance.a_r1cs.rows();
    let n = instance.a_r1cs.cols();

    // Validate Theorem 6.2 bounds (all three checks)
    {
        let k_padded = k.div_ceil(N) * N;
        let n_padded = n.div_ceil(N) * N;
        let q = P::modulus() as u128;
        let lhs = (n_padded + 3 * k_padded) as u128;
        bail_t!(lhs >= q, "n+3k ({lhs}) >= q ({q}), violates Theorem 6.2");
        bail_t!(
            6 * (k_padded as u128) >= q,
            "6k ({}) >= q ({}), violates Theorem 6.2",
            6 * k_padded,
            q
        );
        bail_t!(
            128 * lhs >= 15 * q,
            "128·(n+3k) ({}) >= 15q ({}), insufficient slack for main protocol",
            128 * lhs,
            15 * q
        );
    }

    // Phase 1: build pre-F2 data
    let mut pre = build_binary_pre_f2::<P, N>(instance, witness)
        .map_err(|e| TranscriptError::InvalidInput(alloc::format!("{e:?}")))?;

    // Phase 2: compute commitment t = A·(a||b||c||w)
    let rank = pre.k_packed.max(pre.n_packed);
    let mut statement = LabradorStatement {
        f: Vec::new(),
        f_prime: core::mem::take(&mut pre.f_prime),
    };
    let commitment = attach_opening::<P, N>(
        &mut statement,
        &pre.a_pack,
        &pre.b_pack,
        &pre.c_pack,
        &pre.w_pack,
        crs_a,
        rank,
    )
    .map_err(|e| TranscriptError::InvalidInput(alloc::format!("{e:?}")))?;
    // Restore f_prime for F2 building
    pre.f_prime = core::mem::take(&mut statement.f_prime);

    // Phase 3: absorb commitment into transcript (paper FS order)
    transcript.append_serializable(b"labrador_binary_t", &commitment)?;

    // Phase 4: sample F2 challenges from transcript
    let challenges: Vec<BinaryChallengeSet> = (0..l)
        .map(|round| {
            let round_bytes = (round as u32).to_le_bytes();
            transcript.append_bytes(b"labrador_binary_f2_r", &round_bytes)?;
            let alpha: Vec<u64> = (0..pre.k_padded)
                .map(|_| {
                    transcript
                        .challenge_bytes(b"labrador_binary_f2_a", 1)
                        .map(|b| (b[0] % 2) as u64)
                })
                .collect::<Result<_, _>>()?;
            let beta: Vec<u64> = (0..pre.k_padded)
                .map(|_| {
                    transcript
                        .challenge_bytes(b"labrador_binary_f2_b", 1)
                        .map(|b| (b[0] % 2) as u64)
                })
                .collect::<Result<_, _>>()?;
            let gamma: Vec<u64> = (0..pre.k_padded)
                .map(|_| {
                    transcript
                        .challenge_bytes(b"labrador_binary_f2_c", 1)
                        .map(|b| (b[0] % 2) as u64)
                })
                .collect::<Result<_, _>>()?;
            Ok(BinaryChallengeSet { alpha, beta, gamma })
        })
        .collect::<Result<_, TranscriptError>>()?;

    // Phase 5: build F2 constraints
    let mut g_values = Vec::with_capacity(l);
    build_binary_f2_constraints::<P, N>(&mut pre, instance, &challenges, &mut g_values);

    // Reassemble statement with F2 constraints
    statement.f_prime = pre.f_prime;

    let witness = LabradorWitness::new(pre.witness_parts);

    Ok(BinaryR1CSReduction {
        statement,
        witness,
        commitment,
        l,
        g_values,
        k,
        n,
        k_padded: pre.k_padded,
        n_padded: pre.n_padded,
        k_packed: pre.k_packed,
        n_packed: pre.n_packed,
        binary_f2_start: pre.f2_start,
    })
}

// ---------------------------------------------------------------------------
// Challenge sampling (standalone API)
// ---------------------------------------------------------------------------

/// Sample binary challenges uniformly from F2.
pub fn sample_binary_challenges<Rng: RngExt>(
    k: usize,
    l: usize,
    rng: &mut Rng,
) -> Vec<BinaryChallenges> {
    (0..l)
        .map(|_| {
            let alpha: Vec<i64> = (0..k).map(|_| (rng.random::<u64>() % 2) as i64).collect();
            let beta: Vec<i64> = (0..k).map(|_| (rng.random::<u64>() % 2) as i64).collect();
            let gamma: Vec<i64> = (0..k).map(|_| (rng.random::<u64>() % 2) as i64).collect();
            (alpha, beta, gamma)
        })
        .collect()
}

/// Low-level helper: sample raw binary challenges from transcript.
///
/// **Note:** This function samples challenges under its own labels and does
/// **not** encode the paper Fiat-Shamir contract (absorb commitment first,
/// then sample). Prefer using [`build_binary_r1cs_reduction_transcript`]
/// which handles the full FS order internally. This helper is provided
/// for callers that manage the transcript state externally.
pub fn sample_binary_challenges_transcript<T: grid_transcript::Transcript>(
    k: usize,
    l: usize,
    transcript: &mut T,
) -> Result<Vec<BinaryChallenges>, TranscriptError> {
    (0..l)
        .map(|i| {
            let mut alpha = Vec::with_capacity(k);
            let mut beta = Vec::with_capacity(k);
            let mut gamma = Vec::with_capacity(k);
            let round_bytes = (i as u32).to_le_bytes();
            transcript.append_bytes(b"labrador_binary_chal", &round_bytes)?;
            for _ in 0..k {
                let b = transcript.challenge_bytes(b"labrador_binary_chal_a", 1)?;
                alpha.push((b[0] % 2) as i64);
            }
            for _ in 0..k {
                let b = transcript.challenge_bytes(b"labrador_binary_chal_b", 1)?;
                beta.push((b[0] % 2) as i64);
            }
            for _ in 0..k {
                let b = transcript.challenge_bytes(b"labrador_binary_chal_c", 1)?;
                gamma.push((b[0] % 2) as i64);
            }
            Ok((alpha, beta, gamma))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

/// Check all g-values are even.
pub fn check_g_even(g_values: &[i64]) -> bool {
    g_values.iter().all(|&g| g % 2 == 0)
}

/// Compute start index for binary F2 constraints in `f_prime`.
///
/// Layout:
/// ```text
/// N*(3·k_packed + n_packed)  conjugacy
/// + padding_zero_count       padding-zero constraints
/// + 4                        binaryity checks (a,b,c,w)
/// + 1                        Hadamard
/// ```
pub fn binary_f2_start<const N: usize>(
    k: usize,
    n: usize,
    k_padded: usize,
    n_padded: usize,
    k_packed: usize,
    n_packed: usize,
) -> usize {
    let padding_zero_count = 3 * (k_padded - k) + (n_padded - n);
    N * (3 * k_packed + n_packed) + padding_zero_count + 4 + 1
}

/// Validated parameters extracted from a binary reduction for the verifier.
pub(crate) struct VerifiedBinaryParams {
    pub f2_start: usize,
    pub max_pos: u64,
    pub min_neg: u64,
}

/// Checked validation of binary reduction metadata against Theorem 6.2.
///
/// Returns `None` if any dimension invariant or paper bound is violated.
pub(crate) fn check_binary_params<const N: usize>(
    k: usize,
    n: usize,
    k_padded: usize,
    n_padded: usize,
    k_packed: usize,
    n_packed: usize,
    l: usize,
    q: u64,
) -> Option<VerifiedBinaryParams> {
    // Dimension invariants: k_padded >= k, n_padded >= n
    if k_padded < k || n_padded < n {
        return None;
    }
    // k_padded == k_packed * N, n_padded == n_packed * N (checked)
    if k_packed.checked_mul(N)? != k_padded {
        return None;
    }
    if n_packed.checked_mul(N)? != n_padded {
        return None;
    }

    // Theorem 6.2 bounds
    let lhs = n_padded.checked_add(k_padded.checked_mul(3)?)?;
    let q128 = q as u128;
    if lhs as u128 >= q128 {
        return None;
    }
    if k_padded
        .checked_mul(6)
        .map_or(true, |v| (v as u128) >= q128)
    {
        return None;
    }
    let q15 = q128.checked_mul(15)?;
    if lhs.checked_mul(128).map_or(true, |v| v as u128 >= q15) {
        return None;
    }

    // Compute f2_start
    let padding = k_padded
        .checked_sub(k)?
        .checked_mul(3)?
        .checked_add(n_padded.checked_sub(n)?)?;
    let conjugacy = (k_packed.checked_mul(3)?)
        .checked_add(n_packed)?
        .checked_mul(N)?;
    let f2_start = conjugacy
        .checked_add(padding)?
        .checked_add(4)?
        .checked_add(1)?;

    // f2_start + l must not overflow
    let _total = f2_start.checked_add(l)?;

    let max_pos = (k_padded * 3) as u64;
    let min_neg = q.checked_sub(n_padded as u64)?;

    Some(VerifiedBinaryParams {
        f2_start,
        max_pos,
        min_neg,
    })
}

/// Verify a binary R1CS reduction.
///
/// 1. Validates metadata against Theorem 6.2 and dimension invariants.
/// 2. Extracts g_i from F' b-constants (does NOT trust side metadata).
/// 3. Checks g_i ≡ 0 (mod 2) using paper-aligned interval decoding.
/// 4. Verifies the LaBRADOR relation with the caller-provided norm bound.
pub fn verify_binary_r1cs_reduction<P, const N: usize>(
    reduction: &BinaryR1CSReduction<P, N>,
    max_norm_bound: f64,
) -> bool
where
    P: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    let q = P::modulus();
    let params = match check_binary_params::<N>(
        reduction.k,
        reduction.n,
        reduction.k_padded,
        reduction.n_padded,
        reduction.k_packed,
        reduction.n_packed,
        reduction.l,
        q,
    ) {
        Some(p) => p,
        None => return false,
    };

    let f_prime = &reduction.statement.f_prime;
    if f_prime.len() < params.f2_start + reduction.l {
        return false;
    }

    for i in 0..reduction.l {
        let b_poly = f_prime[params.f2_start + i].b();
        // Reject non-scalar b polynomials: honest g_i is a constant.
        for coeff_idx in 1..N {
            if !b_poly.coeff(coeff_idx).is_zero() {
                return false;
            }
        }
        let g_canon = b_poly.coeff(0).to_u64();
        let g_signed: i64 = if g_canon <= params.max_pos {
            g_canon as i64
        } else if g_canon >= params.min_neg {
            -((q - g_canon) as i64)
        } else {
            return false;
        };
        if g_signed % 2 != 0 {
            return false;
        }
    }

    crate::relation::verify(&reduction.statement, &reduction.witness, max_norm_bound).is_ok()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use grid_algebra::arith::prime::PrimeField;
    use grid_algebra::lattice::types::RingMat;
    use grid_algebra::poly::ring::PolyRing;

    type P = PrimeField<17>;
    type PL = PrimeField<12289>; // large enough for Theorem 6.2 bounds with packing

    fn gf2(v: u8) -> GF2 {
        GF2::new(v)
    }

    // ------------------------------------------------------------------
    // Packing / coefficient helpers
    // ------------------------------------------------------------------

    #[test]
    fn test_pack_gf2_basic() {
        // 8 bits → one polynomial with N=8
        let bits: Vec<GF2> = vec![
            gf2(1),
            gf2(0),
            gf2(1),
            gf2(0),
            gf2(1),
            gf2(0),
            gf2(1),
            gf2(0),
        ];
        let pack = pack_gf2::<P, 8>(&bits, 8);
        assert_eq!(pack.len(), 1);
        let p = &pack[0];
        for i in 0..8 {
            assert_eq!(p.coeff(i), if i % 2 == 0 { P::one() } else { P::zero() });
        }
    }

    #[test]
    fn test_pack_gf2_padded() {
        // 3 bits padded to 8
        let bits: Vec<GF2> = vec![gf2(1), gf2(1), gf2(0)];
        let pack = pack_gf2::<P, 8>(&bits, 8);
        assert_eq!(pack.len(), 1);
        let p = &pack[0];
        assert_eq!(p.coeff(0), P::one());
        assert_eq!(p.coeff(1), P::one());
        assert_eq!(p.coeff(2), P::zero());
        // padded entries are zero
        for i in 3..8 {
            assert_eq!(p.coeff(i), P::zero());
        }
    }

    #[test]
    fn test_pack_gf2_multi_poly() {
        // 10 bits padded to 16 (2 polynomials with N=8)
        let bits: Vec<GF2> = (0..10).map(|i| gf2((i % 2) as u8)).collect();
        let pack = pack_gf2::<P, 8>(&bits, 16);
        assert_eq!(pack.len(), 2);
    }

    #[test]
    fn test_coeff_sum_selector() {
        let s = coeff_sum_selector::<P, 8>();
        assert_eq!(s.coeff(0), P::one());
        for i in 1..8 {
            assert_eq!(s.coeff(i), -P::one());
        }
    }

    #[test]
    fn test_coeff_sum_selector_ct() {
        // Verify ct(coeff_sum_selector * p) = sum of all coeffs of p
        let s = coeff_sum_selector::<P, 8>();
        let mut p = CyclotomicPolyRing::<P, 8>::zero();
        p.set_coeff(0, P::one());
        p.set_coeff(1, P::one());
        p.set_coeff(3, P::one());
        // sum = 3
        let prod = &s * &p;
        assert_eq!(prod.coeff(0), P::from_u64(3));
    }

    #[test]
    fn test_coeff_selector() {
        // selector(0) = 1
        let s0 = coeff_selector::<P, 8>(0);
        assert_eq!(s0.coeff(0), P::one());

        // selector(1) = -X^{7}
        let s1 = coeff_selector::<P, 8>(1);
        assert_eq!(s1.coeff(7), -P::one());

        // ct(s1 * X) = coeff_1 of X = ...
        let mut x = CyclotomicPolyRing::<P, 8>::zero();
        x.set_coeff(1, P::one());
        let prod = &s1 * &x;
        // s1 = -X^7, X^1 * (-X^7) = -X^8 = -(-1) = 1
        assert_eq!(prod.coeff(0), P::one());
    }

    // ------------------------------------------------------------------
    // Theorem 6.2 bounds
    // ------------------------------------------------------------------

    #[test]
    fn test_theorem_bounds_pass() {
        // PL has q = 12289. With k_padded=8, n_padded=8, N=4:
        // n+3k = 32 < 12289 ✓, 6k = 48 < 12289 ✓, 128·32 = 4096 < 15·12289 ✓
        validate_theorem_bounds::<PL>(8, 8).unwrap();
    }

    #[test]
    fn test_theorem_bounds_fail_large_k() {
        type Q = PrimeField<17>;
        let k_padded: usize = 16; // 6*k = 96 > 17
        let result = validate_theorem_bounds::<Q>(k_padded, 0);
        assert!(result.is_err());
    }

    // ------------------------------------------------------------------
    // Smoke test with N dividing k,n
    // ------------------------------------------------------------------

    #[test]
    fn test_binary_packed_smoke_exact_multiples() {
        // N=4, k=4, n=4 — exact multiples
        let k = 4;
        let n = 4;
        let l = 2;

        let a = RingMat::new(
            k,
            n,
            vec![
                gf2(1),
                gf2(0),
                gf2(0),
                gf2(0),
                gf2(0),
                gf2(1),
                gf2(0),
                gf2(0),
                gf2(0),
                gf2(0),
                gf2(1),
                gf2(0),
                gf2(0),
                gf2(0),
                gf2(0),
                gf2(1),
            ],
        );
        let b = a.clone();
        let c = a.clone();
        let instance = BinaryR1CSInstance {
            a_r1cs: a,
            b_r1cs: b,
            c_r1cs: c,
        };
        let witness = vec![gf2(1), gf2(1), gf2(0), gf2(0)];

        // Packed dimensions: k_packed=1, n_packed=1
        // CRS cols = 3*1 + 1 = 4
        let kappa = 2;
        let crs_a = RingMat::new(
            kappa,
            4,
            (0..kappa * 4)
                .map(|_| CyclotomicPolyRing::<PL, 4>::zero())
                .collect(),
        );

        let mut rng = grid_std::test_rng();
        let reduction =
            build_binary_r1cs_reduction::<PL, _, 4>(&instance, &witness, &crs_a, &mut rng, l)
                .unwrap();

        assert_eq!(reduction.k, 4);
        assert_eq!(reduction.n, 4);
        assert_eq!(reduction.k_padded, 4);
        assert_eq!(reduction.n_padded, 4);
        assert_eq!(reduction.k_packed, 1);
        assert_eq!(reduction.n_packed, 1);
        assert_eq!(reduction.witness.num_parts(), 8);
        assert_eq!(reduction.l, l);

        // Verify
        assert_eq!(
            reduction.binary_f2_start,
            binary_f2_start::<4>(
                reduction.k,
                reduction.n,
                reduction.k_padded,
                reduction.n_padded,
                reduction.k_packed,
                reduction.n_packed,
            ),
            "binary_f2_start should match computed value"
        );
        assert!(verify_binary_r1cs_reduction::<PL, 4>(
            &reduction,
            reduction.witness.l2_norm_squared().sqrt().max(1.0)
        ));
    }

    // ------------------------------------------------------------------
    // Padding test: k not a multiple of N
    // ------------------------------------------------------------------

    #[test]
    fn test_binary_packed_with_padding() {
        // N=4, k=3 (not multiple of 4), n=4
        let k = 3;
        let n = 4;
        let l = 2;

        let a = RingMat::new(
            k,
            n,
            vec![
                gf2(1),
                gf2(0),
                gf2(0),
                gf2(0),
                gf2(0),
                gf2(1),
                gf2(0),
                gf2(0),
                gf2(0),
                gf2(0),
                gf2(1),
                gf2(0),
            ],
        );
        let b = a.clone();
        let c = a.clone();
        let instance = BinaryR1CSInstance {
            a_r1cs: a,
            b_r1cs: b,
            c_r1cs: c,
        };
        let witness = vec![gf2(1), gf2(1), gf2(0), gf2(0)];

        // k_padded=4, n_padded=4, k_packed=1, n_packed=1
        let kappa = 2;
        let crs_a = RingMat::new(
            kappa,
            4,
            (0..kappa * 4)
                .map(|_| CyclotomicPolyRing::<PL, 4>::zero())
                .collect(),
        );

        let mut rng = grid_std::test_rng();
        let reduction =
            build_binary_r1cs_reduction::<PL, _, 4>(&instance, &witness, &crs_a, &mut rng, l)
                .unwrap();

        assert_eq!(reduction.k_padded, 4); // padded from 3
        assert_eq!(reduction.k_packed, 1);

        // Verify passes (padded entries constrained to zero)
        assert!(verify_binary_r1cs_reduction::<PL, 4>(
            &reduction,
            reduction.witness.l2_norm_squared().sqrt().max(1.0)
        ));
    }

    // ------------------------------------------------------------------
    // Roundtrip verification test
    // ------------------------------------------------------------------

    #[test]
    fn test_binary_reduction_verify_roundtrip() {
        // N=8, k=8, n=8
        let k = 8;
        let n = 8;
        let l = 2;

        let mut entries = vec![gf2(0); k * n];
        for i in 0..k {
            entries[i * n + i] = gf2(1);
        }
        let a = RingMat::new(k, n, entries.clone());
        let b = a.clone();
        let c = a.clone();
        let instance = BinaryR1CSInstance {
            a_r1cs: a,
            b_r1cs: b,
            c_r1cs: c,
        };
        let witness = vec![
            gf2(1),
            gf2(0),
            gf2(0),
            gf2(0),
            gf2(0),
            gf2(0),
            gf2(0),
            gf2(0),
        ];

        // k_packed=1, n_packed=1, CRS cols=4
        let kappa = 2;
        let crs_a = RingMat::new(
            kappa,
            4,
            (0..kappa * 4)
                .map(|_| CyclotomicPolyRing::<PL, 8>::zero())
                .collect(),
        );

        let mut rng = grid_std::test_rng();
        let reduction =
            build_binary_r1cs_reduction::<PL, _, 8>(&instance, &witness, &crs_a, &mut rng, l)
                .unwrap();

        let g_ok = check_g_even(&reduction.g_values);
        assert!(
            g_ok,
            "check_g_even failed: g_values={:?}",
            reduction.g_values
        );
        assert!(verify_binary_r1cs_reduction::<PL, 8>(
            &reduction,
            reduction.witness.l2_norm_squared().sqrt().max(1.0)
        ));
    }

    // ------------------------------------------------------------------
    // Tamper tests
    // ------------------------------------------------------------------

    #[test]
    fn test_tampered_packed_coeff_fails() {
        let k = 8;
        let n = 8;
        let l = 2;
        let mut entries = vec![gf2(0); k * n];
        for i in 0..k {
            entries[i * n + i] = gf2(1);
        }
        let a = RingMat::new(k, n, entries);
        let instance = BinaryR1CSInstance {
            a_r1cs: a.clone(),
            b_r1cs: a.clone(),
            c_r1cs: a,
        };
        let witness = vec![
            gf2(1),
            gf2(0),
            gf2(0),
            gf2(0),
            gf2(0),
            gf2(0),
            gf2(0),
            gf2(0),
        ];

        let kappa = 2;
        let crs_a = RingMat::new(
            kappa,
            4,
            (0..kappa * 4)
                .map(|_| CyclotomicPolyRing::<PL, 8>::zero())
                .collect(),
        );

        let mut rng = grid_std::test_rng();
        let mut reduction =
            build_binary_r1cs_reduction::<PL, _, 8>(&instance, &witness, &crs_a, &mut rng, l)
                .unwrap();

        // Tamper: flip one coefficient of w_pack[0] (coefficient 1 → 0 becomes 1)
        reduction.witness.parts[3][0].set_coeff(1, PL::one());
        assert!(!verify_binary_r1cs_reduction::<PL, 8>(
            &reduction,
            reduction.witness.l2_norm_squared().sqrt().max(1.0)
        ));
    }

    #[test]
    fn test_f2_b_constant_tampered_fails() {
        let k = 8;
        let n = 8;
        let l = 2;
        let mut entries = vec![gf2(0); k * n];
        for i in 0..k {
            entries[i * n + i] = gf2(1);
        }
        let a = RingMat::new(k, n, entries);
        let instance = BinaryR1CSInstance {
            a_r1cs: a.clone(),
            b_r1cs: a.clone(),
            c_r1cs: a,
        };
        let witness = vec![
            gf2(1),
            gf2(0),
            gf2(0),
            gf2(0),
            gf2(0),
            gf2(0),
            gf2(0),
            gf2(0),
        ];

        let kappa = 2;
        let crs_a = RingMat::new(
            kappa,
            4,
            (0..kappa * 4)
                .map(|_| CyclotomicPolyRing::<PL, 8>::zero())
                .collect(),
        );

        let mut rng = grid_std::test_rng();
        let mut reduction =
            build_binary_r1cs_reduction::<PL, _, 8>(&instance, &witness, &crs_a, &mut rng, l)
                .unwrap();

        // Tamper: clone the existing F2 b, preserve coeff(0), add coeff(1)=1.
        // The non-scalar guard should reject it before even checking parity.
        let f2s = binary_f2_start::<8>(
            reduction.k,
            reduction.n,
            reduction.k_padded,
            reduction.n_padded,
            reduction.k_packed,
            reduction.n_packed,
        );
        let orig_b = reduction.statement.f_prime[f2s].b().clone();
        let mut tampered = orig_b.clone();
        tampered.set_coeff(1, PL::one());
        match &mut reduction.statement.f_prime[f2s] {
            QuadraticFunction::Dense(d) => d.b = tampered.clone(),
            QuadraticFunction::Sparse(s) => s.b = tampered,
        }
        assert!(!verify_binary_r1cs_reduction::<PL, 8>(
            &reduction,
            reduction.witness.l2_norm_squared().sqrt().max(1.0)
        ));
    }

    // ------------------------------------------------------------------
    // Legacy compatibility tests (adapted for packed format)
    // ------------------------------------------------------------------

    #[test]
    fn test_pad_to_rank() {
        let p1 = constant_poly::<P, 8>(P::one());
        let p2 = constant_poly::<P, 8>(P::from_u64(2));
        let padded = pad_entries::<P, 8>(vec![p1.clone(), p2.clone()], 5);
        assert_eq!(padded.len(), 5);
        assert_eq!(padded[0].coeff(0), P::one());
        assert_eq!(padded[1].coeff(0), P::from_u64(2));
        for p in padded.iter().skip(2) {
            assert!(p.is_zero());
        }
    }

    #[test]
    fn test_check_g_even() {
        assert!(check_g_even(&[0, 2, -4, 6]));
        assert!(check_g_even(&[]));
        assert!(!check_g_even(&[0, 1, 2]));
    }

    #[test]
    fn test_sample_binary_challenges() {
        let mut rng = grid_std::test_rng();
        let challenges = sample_binary_challenges(3, 4, &mut rng);
        assert_eq!(challenges.len(), 4);
        for (alpha, beta, gamma) in &challenges {
            assert_eq!(alpha.len(), 3);
            assert_eq!(beta.len(), 3);
            assert_eq!(gamma.len(), 3);
            for v in alpha.iter().chain(beta.iter()).chain(gamma.iter()) {
                assert!(*v == 0 || *v == 1);
            }
        }
    }
}
