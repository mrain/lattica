//! Mixed R1CS → R reduction (§6, prose after Figure 5).
//!
//! Combines binary R1CS and arithmetic R1CS into a single LaBRADOR relation R
//! instance using the paper's construction: one binary witness encodes the
//! arithmetic witness and one mixed Ajtai commitment covers all 7 committed
//! segments.
//!
//! # Witness layout
//!
//! ```text
//! 0: packed binary A*w output     4: σ₋₁(part 0)
//! 1: packed binary B*w output     5: σ₋₁(part 1)
//! 2: packed binary C*w output     6: σ₋₁(part 2)
//! 3: shared packed witness w_pack  7: σ₋₁(part 3)
//! 8: arithmetic Enc(A*w)          9: arithmetic Enc(B*w)
//! 10: arithmetic Enc(C*w)         11..: arithmetic d_1..d_l
//! ```
//!
//! Part 3 is the single shared witness referenced by both the binary and
//! arithmetic fragments. There is no separate arithmetic w part.

use alloc::vec;
use alloc::vec::Vec;

use grid_algebra::arith::gf2::GF2;
use grid_algebra::arith::ring::{IntegerRing, Ring};
use grid_algebra::lattice::types::{RingMat, RingVec};
use grid_algebra::poly::ring::{CyclotomicPolyRing, NegacyclicMulRing, PolyRing};
use grid_std::rand::RngExt;
use grid_transcript::TranscriptError;

use crate::error::LabradorError;
use crate::reduction::app_ring::AppModRing;
use crate::reduction::arith_r1cs::{
    ArithR1CSInstance, check_divisible_by_x_minus_2, verify_naf_coeffs,
};
use crate::reduction::binary_r1cs::{
    BinaryChallengeSet, BinaryPreF2, BinaryR1CSFragment, BinaryR1CSInstance,
    build_binary_pre_f2_fragment, finish_binary_fragment,
};
use crate::relation::{LabradorStatement, LabradorWitness, QuadraticFunction};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MixedR1CSInstance<A: AppModRing> {
    pub binary: BinaryR1CSInstance,
    pub arithmetic: ArithR1CSInstance<A>,
}

#[derive(Debug, Clone)]
pub struct MixedR1CSReduction<A, P, const N: usize>
where
    A: AppModRing,
    P: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    pub statement: LabradorStatement<CyclotomicPolyRing<P, N>>,
    pub witness: LabradorWitness<CyclotomicPolyRing<P, N>>,
    pub commitment_mixed_t: RingVec<CyclotomicPolyRing<P, N>>,
    pub commitment_arith_td: RingVec<CyclotomicPolyRing<P, N>>,
    pub n_arith: usize,
    pub k_bin: usize,
    pub k_bin_padded: usize,
    pub k_bin_packed: usize,
    pub n_bin: usize,
    pub n_bin_padded: usize,
    pub n_bin_packed: usize,
    pub k_arith: usize,
    pub l_binary: usize,
    pub l_arithmetic: usize,
    pub binary_f2_start: usize,
    pub arith_agg_start: usize,
    pub _app: core::marker::PhantomData<A>,
}

/// One round of arithmetic aggregation challenges: (α, β, γ, δ).
pub type ArithAggChallenge<A> = (Vec<A>, Vec<A>, Vec<A>, Vec<A>);

// ---------------------------------------------------------------------------
// Arithmetic staging helpers
// ---------------------------------------------------------------------------

struct ArithPreData<A, P, const N: usize>
where
    A: AppModRing,
    P: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    a_enc: Vec<CyclotomicPolyRing<P, N>>,
    b_enc: Vec<CyclotomicPolyRing<P, N>>,
    c_enc: Vec<CyclotomicPolyRing<P, N>>,
    a_field: RingVec<A>,
    k: usize,
    n: usize,
}

fn build_arith_pre<A, P, const N: usize>(
    instance: &ArithR1CSInstance<A>,
    witness: &[A],
) -> Result<ArithPreData<A, P, N>, LabradorError>
where
    A: AppModRing,
    P: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    use crate::reduction::arith_r1cs::encode_app_naf;
    let k = instance.a_r1cs.rows();
    let n = instance.a_r1cs.cols();
    // Shape validation (mirrors standalone arith_r1cs checks).
    if instance.b_r1cs.rows() != k {
        return Err(LabradorError::InvalidInput(alloc::format!(
            "b_r1cs rows ({}) != a_r1cs rows ({k})",
            instance.b_r1cs.rows()
        )));
    }
    if instance.b_r1cs.cols() != n {
        return Err(LabradorError::InvalidInput(alloc::format!(
            "b_r1cs cols ({}) != a_r1cs cols ({n})",
            instance.b_r1cs.cols()
        )));
    }
    if instance.c_r1cs.rows() != k {
        return Err(LabradorError::InvalidInput(alloc::format!(
            "c_r1cs rows ({}) != a_r1cs rows ({k})",
            instance.c_r1cs.rows()
        )));
    }
    if instance.c_r1cs.cols() != n {
        return Err(LabradorError::InvalidInput(alloc::format!(
            "c_r1cs cols ({}) != a_r1cs cols ({n})",
            instance.c_r1cs.cols()
        )));
    }
    if witness.len() != n {
        return Err(LabradorError::InvalidInput(alloc::format!(
            "witness len ({}) != expected ({n})",
            witness.len()
        )));
    }
    let a_field = instance.a_r1cs.mul_slice(witness);
    let b_field = instance.b_r1cs.mul_slice(witness);
    let c_field = instance.c_r1cs.mul_slice(witness);
    let a_enc = a_field
        .entries()
        .iter()
        .map(|v| encode_app_naf::<A, P, N>(v))
        .collect::<Result<_, _>>()?;
    let b_enc = b_field
        .entries()
        .iter()
        .map(|v| encode_app_naf::<A, P, N>(v))
        .collect::<Result<_, _>>()?;
    let c_enc = c_field
        .entries()
        .iter()
        .map(|v| encode_app_naf::<A, P, N>(v))
        .collect::<Result<_, _>>()?;
    Ok(ArithPreData {
        a_enc,
        b_enc,
        c_enc,
        a_field,
        k,
        n,
    })
}

struct ArithDData<P, const N: usize>
where
    P: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    d_segments_for_td: Vec<CyclotomicPolyRing<P, N>>,
    phi_challenges: Vec<Vec<CyclotomicPolyRing<P, N>>>,
}

fn build_arith_d<P, A, const N: usize>(
    pre: &ArithPreData<A, P, N>,
    _instance: &ArithR1CSInstance<A>,
    phi_challenges: &[Vec<A>],
) -> Result<ArithDData<P, N>, LabradorError>
where
    A: AppModRing,
    P: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    use crate::reduction::arith_r1cs::encode_app_naf;
    let k = pre.k;
    let l = phi_challenges.len();
    for (i, phi) in phi_challenges.iter().enumerate() {
        if phi.len() != k {
            return Err(LabradorError::InvalidInput(alloc::format!(
                "phi_challenges[{i}] len {} != k ({k})",
                phi.len()
            )));
        }
    }
    let mut d_segments = Vec::with_capacity(l * k);
    for phi in phi_challenges {
        for col in 0..k {
            let val = phi[col].clone() * pre.a_field.entries()[col].clone();
            d_segments.push(encode_app_naf::<A, P, N>(&val)?);
        }
    }
    let phi_enc: Vec<Vec<CyclotomicPolyRing<P, N>>> = phi_challenges
        .iter()
        .map(|phi| {
            phi.iter()
                .map(|v| encode_app_naf::<A, P, N>(v))
                .collect::<Result<_, _>>()
        })
        .collect::<Result<_, _>>()?;
    Ok(ArithDData {
        d_segments_for_td: d_segments,
        phi_challenges: phi_enc,
    })
}

// ---------------------------------------------------------------------------
// F' padding helper
// ---------------------------------------------------------------------------

fn pad_f_prime_to_mixed<P, const N: usize>(
    f: QuadraticFunction<CyclotomicPolyRing<P, N>>,
    rank: usize,
    num_parts: usize,
) -> QuadraticFunction<CyclotomicPolyRing<P, N>>
where
    P: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    let zero = CyclotomicPolyRing::<P, N>::zero();
    match f {
        QuadraticFunction::Dense(d) => {
            let old_rank = d.phi.first().map(|v| v.len()).unwrap_or(0);
            let mut phi = d.phi;
            if old_rank < rank {
                for vec in phi.iter_mut() {
                    vec.resize(rank, zero.clone());
                }
            }
            phi.resize_with(num_parts, || vec![zero.clone(); rank]);
            QuadraticFunction::Dense(crate::relation::DenseQuadraticFunction {
                a: d.a,
                ij: d.ij,
                phi,
                b: d.b,
            })
        }
        QuadraticFunction::Sparse(s) => QuadraticFunction::Sparse(s),
    }
}

// ---------------------------------------------------------------------------
// Two-phase core builder
// ---------------------------------------------------------------------------

struct MixedPhase1<A, P, const N: usize>
where
    A: AppModRing,
    P: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    bin_pre: BinaryPreF2<P, N>,
    arith_pre: ArithPreData<A, P, N>,
    commitment_mixed_t: RingVec<CyclotomicPolyRing<P, N>>,
    rank: usize,
    num_parts: usize,
    k_bin_packed: usize,
    n_bin_packed: usize,
    k_arith: usize,
}

#[allow(non_snake_case)]
fn build_mixed_phase1<A, P, const N: usize>(
    instance: &MixedR1CSInstance<A>,
    witness: &[GF2],
    crs_a_mixed: &RingMat<CyclotomicPolyRing<P, N>>,
    l_binary: usize,
    l_arithmetic: usize,
    beta: f64,
) -> Result<MixedPhase1<A, P, N>, LabradorError>
where
    A: AppModRing,
    P: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    if N == 0 {
        return Err(LabradorError::InvalidInput(alloc::string::String::from(
            "mixed R1CS requires N > 0",
        )));
    }
    if !A::is_fermat_modulus_for_degree::<N>() {
        return Err(LabradorError::InvalidInput(alloc::format!(
            "mixed R1CS requires app modulus = 2^{N}+1"
        )));
    }
    let bit_w = A::binary_encoding_width::<N>().unwrap();
    debug_assert_eq!(bit_w, N);
    let k_bin = instance.binary.a_r1cs.rows();
    let n_bin = instance.binary.a_r1cs.cols();
    let k_arith = instance.arithmetic.a_r1cs.rows();
    let n_arith = instance.arithmetic.a_r1cs.cols();
    if n_bin != n_arith * bit_w {
        return Err(LabradorError::InvalidInput(alloc::format!(
            "n_bin ({n_bin}) != n_arith ({n_arith}) * bit_w ({bit_w})"
        )));
    }
    if witness.len() != n_bin {
        return Err(LabradorError::InvalidInput(alloc::format!(
            "witness len ({}) != n_bin ({n_bin})",
            witness.len()
        )));
    }
    if l_binary == 0 || l_arithmetic == 0 {
        return Err(LabradorError::InvalidInput(alloc::string::String::from(
            "l_binary and l_arithmetic must be > 0",
        )));
    }
    let w_arith: Vec<A> = (0..n_arith)
        .map(|j| {
            A::try_decode_from_gf2_bits::<N>(&witness[j * bit_w..(j + 1) * bit_w]).ok_or_else(
                || {
                    LabradorError::InvalidInput(alloc::format!(
                        "failed to decode arithmetic variable {j}"
                    ))
                },
            )
        })
        .collect::<Result<_, _>>()?;
    let k_bin_packed = k_bin.div_ceil(N);
    let n_bin_packed = n_bin.div_ceil(N);
    let rank = k_bin_packed.max(n_bin_packed).max(k_arith).max(n_arith);
    crate::reduction::arith_r1cs::validate_theorem_6_3_bounds::<P>(
        k_arith,
        n_arith,
        l_arithmetic,
        N,
        beta,
    )?;
    // Binary Theorem 6.2 bounds on padded dimensions.
    {
        let k_bin_padded = k_bin_packed * N;
        let n_bin_padded = n_bin_packed * N;
        let q = P::modulus();
        let lhs = n_bin_padded + 3 * k_bin_padded;
        if lhs as u128 >= q as u128 {
            return Err(LabradorError::InvalidInput(alloc::format!(
                "binary Theorem 6.2: n+3k ({lhs}) >= q ({q})"
            )));
        }
        if 6 * k_bin_padded as u128 >= q as u128 {
            return Err(LabradorError::InvalidInput(alloc::format!(
                "binary Theorem 6.2: 6k ({}) >= q ({q})",
                6 * k_bin_padded
            )));
        }
        if (lhs as u128) * 128 >= (q as u128) * 15 {
            return Err(LabradorError::InvalidInput(alloc::string::String::from(
                "binary Theorem 6.2: insufficient slack for main protocol",
            )));
        }
    }
    let bin_pre = build_binary_pre_f2_fragment::<P, N>(&instance.binary, witness)?;
    let arith_pre = build_arith_pre::<A, P, N>(&instance.arithmetic, &w_arith)?;
    let mixed_cols = 3 * k_bin_packed + 3 * k_arith + n_bin_packed;
    if crs_a_mixed.cols() != mixed_cols {
        return Err(LabradorError::InvalidInput(alloc::format!(
            "mixed CRS cols ({}) != expected ({mixed_cols})",
            crs_a_mixed.cols()
        )));
    }
    let mut concat = Vec::with_capacity(mixed_cols);
    concat.extend_from_slice(&bin_pre.a_pack[..k_bin_packed]);
    concat.extend_from_slice(&bin_pre.b_pack[..k_bin_packed]);
    concat.extend_from_slice(&bin_pre.c_pack[..k_bin_packed]);
    concat.extend_from_slice(&arith_pre.a_enc);
    concat.extend_from_slice(&arith_pre.b_enc);
    concat.extend_from_slice(&arith_pre.c_enc);
    concat.extend_from_slice(&bin_pre.w_pack[..n_bin_packed]);
    let commitment_mixed_t = crs_a_mixed.mul_slice(&concat);
    let num_parts = 11 + l_arithmetic;
    Ok(MixedPhase1 {
        bin_pre,
        arith_pre,
        commitment_mixed_t,
        rank,
        num_parts,
        k_bin_packed,
        n_bin_packed,
        k_arith,
    })
}

/// Intermediate result of phase 2a: binary F2 built, d_i + td computed.
struct MixedPhase2a<A, P, const N: usize>
where
    A: AppModRing,
    P: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    commitment_mixed_t: RingVec<CyclotomicPolyRing<P, N>>,
    commitment_arith_td: RingVec<CyclotomicPolyRing<P, N>>,
    bin_frag: BinaryR1CSFragment<P, N>,
    arith_pre: ArithPreData<A, P, N>,
    arith_d: ArithDData<P, N>,
    binary_f2_start: usize,
    rank: usize,
    num_parts: usize,
    k_bin_packed: usize,
    n_bin_packed: usize,
    k_arith: usize,
    n_arith: usize,
    k_bin: usize,
    n_bin: usize,
    k_bin_padded: usize,
    n_bin_padded: usize,
}

#[allow(non_snake_case)]
fn build_mixed_phase2a<A, P, const N: usize>(
    instance: &MixedR1CSInstance<A>,
    ph1: MixedPhase1<A, P, N>,
    crs_b_arith_d: &RingMat<CyclotomicPolyRing<P, N>>,
    l_binary: usize,
    l_arithmetic: usize,
    binary_f2_challenges: Vec<BinaryChallengeSet>,
    arith_phi_challenges: Vec<Vec<A>>,
) -> Result<MixedPhase2a<A, P, N>, LabradorError>
where
    A: AppModRing,
    P: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    let MixedPhase1 {
        mut bin_pre,
        arith_pre,
        commitment_mixed_t,
        rank,
        num_parts,
        k_bin_packed,
        n_bin_packed,
        k_arith: _,
    } = ph1;
    let k_arith_val = instance.arithmetic.a_r1cs.rows();
    let n_arith_val = instance.arithmetic.a_r1cs.cols();
    let k_bin = instance.binary.a_r1cs.rows();
    let n_bin = instance.binary.a_r1cs.cols();
    let k_bin_padded = k_bin.div_ceil(N) * N;
    let n_bin_padded = n_bin.div_ceil(N) * N;

    // Finish binary F2
    let mut bin_g_values = Vec::with_capacity(l_binary);
    let bin_frag = finish_binary_fragment::<P, N>(
        &mut bin_pre,
        &instance.binary,
        l_binary,
        &binary_f2_challenges,
        &mut bin_g_values,
    )?;
    let binary_f2_start = bin_frag.binary_f2_start;

    // Build arithmetic d_i and td
    if arith_phi_challenges.len() != l_arithmetic {
        return Err(LabradorError::InvalidInput(alloc::format!(
            "arith_phi_challenges len ({}) != l_arithmetic ({l_arithmetic})",
            arith_phi_challenges.len()
        )));
    }
    let arith_d =
        build_arith_d::<P, A, N>(&arith_pre, &instance.arithmetic, &arith_phi_challenges)?;
    if crs_b_arith_d.cols() != l_arithmetic * k_arith_val {
        return Err(LabradorError::InvalidInput(alloc::format!(
            "mixed CRS B cols ({}) != l*k ({})",
            crs_b_arith_d.cols(),
            l_arithmetic * k_arith_val
        )));
    }
    let commitment_arith_td = crs_b_arith_d.mul_slice(&arith_d.d_segments_for_td);

    Ok(MixedPhase2a {
        commitment_mixed_t,
        commitment_arith_td,
        bin_frag,
        arith_pre,
        arith_d,
        binary_f2_start,
        rank,
        num_parts,
        k_bin_packed,
        n_bin_packed,
        k_arith: k_arith_val,
        n_arith: n_arith_val,
        k_bin,
        n_bin,
        k_bin_padded,
        n_bin_padded,
    })
}

#[allow(non_snake_case)]
fn build_mixed_phase2b<A, P, const N: usize>(
    instance: &MixedR1CSInstance<A>,
    ph2a: MixedPhase2a<A, P, N>,
    crs_a_mixed: &RingMat<CyclotomicPolyRing<P, N>>,
    crs_b_arith_d: &RingMat<CyclotomicPolyRing<P, N>>,
    l_binary: usize,
    l_arithmetic: usize,
    arith_phi_challenges: Vec<Vec<A>>,
    arith_agg_challenges: Vec<ArithAggChallenge<A>>,
) -> Result<MixedR1CSReduction<A, P, N>, LabradorError>
where
    A: AppModRing,
    P: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    let MixedPhase2a {
        commitment_mixed_t,
        commitment_arith_td,
        bin_frag,
        arith_pre,
        arith_d,
        binary_f2_start,
        rank,
        num_parts,
        k_bin_packed,
        n_bin_packed,
        k_arith,
        n_arith,
        k_bin,
        n_bin,
        k_bin_padded,
        n_bin_padded,
    } = ph2a;
    let zero = CyclotomicPolyRing::<P, N>::zero();

    // --- Build arithmetic aggregation ---
    let enc = |a: &A| -> Result<CyclotomicPolyRing<P, N>, LabradorError> {
        crate::reduction::arith_r1cs::encode_app_naf::<A, P, N>(a)
    };
    let pad = |v: &[CyclotomicPolyRing<P, N>]| -> Vec<CyclotomicPolyRing<P, N>> {
        let mut p = v.to_vec();
        p.resize(rank, zero.clone());
        p
    };
    let arith_a_pad = pad(&arith_pre.a_enc);
    let arith_b_pad = pad(&arith_pre.b_enc);
    let arith_c_pad = pad(&arith_pre.c_enc);

    const ARITH_A: usize = 8;
    const ARITH_B: usize = 9;
    const ARITH_C: usize = 10;
    const W_PACK: usize = 3;
    const ARITH_D_BASE: usize = 11;

    let mut arith_f_agg: Vec<QuadraticFunction<CyclotomicPolyRing<P, N>>> =
        Vec::with_capacity(l_arithmetic);

    // Validate challenge vector shapes before indexing.
    if arith_phi_challenges.len() != l_arithmetic {
        return Err(LabradorError::InvalidInput(alloc::format!(
            "arith_phi_challenges len {} != l ({l_arithmetic})",
            arith_phi_challenges.len()
        )));
    }
    for (i, phi) in arith_phi_challenges.iter().enumerate() {
        if phi.len() != k_arith {
            return Err(LabradorError::InvalidInput(alloc::format!(
                "arith_phi_challenges[{i}] len {} != k_arith ({k_arith})",
                phi.len()
            )));
        }
    }
    if arith_agg_challenges.len() != l_arithmetic {
        return Err(LabradorError::InvalidInput(alloc::format!(
            "arith_agg_challenges len {} != l ({l_arithmetic})",
            arith_agg_challenges.len()
        )));
    }
    for (j, (alpha, beta, gamma, delta)) in arith_agg_challenges.iter().enumerate() {
        if alpha.len() != k_arith {
            return Err(LabradorError::InvalidInput(alloc::format!(
                "agg[{j}].alpha len {} != k_arith ({k_arith})",
                alpha.len()
            )));
        }
        if beta.len() != k_arith {
            return Err(LabradorError::InvalidInput(alloc::format!(
                "agg[{j}].beta len {} != k_arith ({k_arith})",
                beta.len()
            )));
        }
        if gamma.len() != k_arith {
            return Err(LabradorError::InvalidInput(alloc::format!(
                "agg[{j}].gamma len {} != k_arith ({k_arith})",
                gamma.len()
            )));
        }
        if delta.len() != l_arithmetic * k_arith {
            return Err(LabradorError::InvalidInput(alloc::format!(
                "agg[{j}].delta len {} != l*k ({})",
                delta.len(),
                l_arithmetic * k_arith
            )));
        }
    }

    // Pre-build the aggregation witness once (identical across all rounds).
    let mut agg_witness_parts: Vec<Vec<CyclotomicPolyRing<P, N>>> = Vec::with_capacity(num_parts);
    for part in bin_frag.witness_parts.iter() {
        let mut p = part.clone();
        p.resize(rank, zero.clone());
        agg_witness_parts.push(p);
    }
    agg_witness_parts.push(arith_a_pad.clone());
    agg_witness_parts.push(arith_b_pad.clone());
    agg_witness_parts.push(arith_c_pad.clone());
    for i in 0..l_arithmetic {
        agg_witness_parts.push(pad(
            &arith_d.d_segments_for_td[i * k_arith..(i + 1) * k_arith]
        ));
    }
    let agg_witness = LabradorWitness::new(agg_witness_parts);

    for j in 0..l_arithmetic {
        let (ref alpha_j, ref beta_j, ref gamma_j, ref delta_j) = arith_agg_challenges[j];
        let mut phi: Vec<Vec<CyclotomicPolyRing<P, N>>> = vec![vec![zero.clone(); rank]; num_parts];

        for (part_a, challenge, matrix) in [
            (ARITH_A, alpha_j.as_slice(), &instance.arithmetic.a_r1cs),
            (ARITH_B, beta_j.as_slice(), &instance.arithmetic.b_r1cs),
            (ARITH_C, gamma_j.as_slice(), &instance.arithmetic.c_r1cs),
        ] {
            let (left, right) = phi.split_at_mut(part_a);
            let phi_w = &mut left[W_PACK];
            let phi_a = &mut right[0];
            for idx in 0..k_arith {
                phi_a[idx] = -enc(&challenge[idx])?;
            }
            for col in 0..n_arith {
                let mut acc = A::zero();
                for i in 0..k_arith {
                    acc += challenge[i].clone() * matrix.get(i, col).clone();
                }
                phi_w[col] += enc(&acc)?;
            }
        }
        for idx in 0..k_arith {
            phi[ARITH_C][idx] -= enc(&arith_phi_challenges[j][idx])?;
        }
        for col in 0..k_arith {
            let mut acc = A::zero();
            for i in 0..l_arithmetic {
                acc += delta_j[i * k_arith + col].clone() * arith_phi_challenges[i][col].clone();
            }
            phi[ARITH_A][col] += enc(&acc)?;
        }
        for i in 0..l_arithmetic {
            let d_idx = ARITH_D_BASE + i;
            for col in 0..k_arith {
                phi[d_idx][col] = -enc(&delta_j[i * k_arith + col])?;
            }
        }
        let idx_b = ARITH_B;
        let d_j_idx = ARITH_D_BASE + j;
        let qp = if idx_b < d_j_idx {
            (idx_b, d_j_idx)
        } else {
            (d_j_idx, idx_b)
        };
        let one = CyclotomicPolyRing::<P, N>::one();
        let f_j = QuadraticFunction::from_parts(vec![(qp.0, qp.1, one)], phi, zero.clone());
        let g_j = f_j.evaluate(&agg_witness);
        let f_constraint = match f_j {
            QuadraticFunction::Dense(d) => {
                QuadraticFunction::Dense(crate::relation::DenseQuadraticFunction {
                    a: d.a,
                    ij: d.ij,
                    phi: d.phi,
                    b: g_j,
                })
            }
            QuadraticFunction::Sparse(s) => {
                QuadraticFunction::Sparse(crate::relation::SparseQuadraticFunction {
                    ij_a: s.ij_a,
                    phi: s.phi,
                    b: g_j,
                })
            }
        };
        arith_f_agg.push(f_constraint);
    }

    // --- Assemble F ---
    let kappa = crs_a_mixed.rows();
    let kappa_b = crs_b_arith_d.rows();
    let mut mixed_f: Vec<QuadraticFunction<CyclotomicPolyRing<P, N>>> =
        Vec::with_capacity(kappa + kappa_b + l_arithmetic);
    for j in 0..kappa {
        let row = crs_a_mixed.row(j);
        let row_entries = row.entries();
        let slice = |s: usize, l: usize| -> Vec<CyclotomicPolyRing<P, N>> {
            let mut v = row_entries[s..s + l].to_vec();
            v.resize(rank, zero.clone());
            v
        };
        let mut phi: Vec<Vec<_>> = vec![
            slice(0, k_bin_packed),
            slice(k_bin_packed, k_bin_packed),
            slice(2 * k_bin_packed, k_bin_packed),
            slice(3 * k_bin_packed + 3 * k_arith, n_bin_packed),
            vec![zero.clone(); rank],
            vec![zero.clone(); rank],
            vec![zero.clone(); rank],
            vec![zero.clone(); rank],
            slice(3 * k_bin_packed, k_arith),
            slice(3 * k_bin_packed + k_arith, k_arith),
            slice(3 * k_bin_packed + 2 * k_arith, k_arith),
        ];
        phi.resize_with(num_parts, || vec![zero.clone(); rank]);
        mixed_f.push(QuadraticFunction::from_parts(
            Vec::new(),
            phi,
            commitment_mixed_t.get(j).clone(),
        ));
    }
    for j in 0..kappa_b {
        let row = crs_b_arith_d.row(j);
        let row_entries = row.entries();
        let mut phi: Vec<Vec<_>> = vec![vec![zero.clone(); rank]; num_parts];
        for i in 0..l_arithmetic {
            let mut v = row_entries[i * k_arith..(i + 1) * k_arith].to_vec();
            v.resize(rank, zero.clone());
            phi[ARITH_D_BASE + i] = v;
        }
        mixed_f.push(QuadraticFunction::from_parts(
            Vec::new(),
            phi,
            commitment_arith_td.get(j).clone(),
        ));
    }
    let arith_agg_start = mixed_f.len();
    mixed_f.extend(arith_f_agg);

    // --- Assemble F' ---
    let bin_f_prime: Vec<QuadraticFunction<_>> = bin_frag
        .statement_without_opening
        .f_prime
        .into_iter()
        .map(|f| pad_f_prime_to_mixed::<P, N>(f, rank, num_parts))
        .collect();

    // --- Assemble witness ---
    let mut full_witness_parts: Vec<Vec<_>> = Vec::with_capacity(num_parts);
    for part in bin_frag.witness_parts.iter() {
        let mut p = part.clone();
        p.resize(rank, zero.clone());
        full_witness_parts.push(p);
    }
    full_witness_parts.push(arith_a_pad);
    full_witness_parts.push(arith_b_pad);
    full_witness_parts.push(arith_c_pad);
    for i in 0..l_arithmetic {
        full_witness_parts.push(pad(
            &arith_d.d_segments_for_td[i * k_arith..(i + 1) * k_arith]
        ));
    }
    let witness = LabradorWitness::new(full_witness_parts);
    let statement = LabradorStatement {
        f: mixed_f,
        f_prime: bin_f_prime,
    };

    Ok(MixedR1CSReduction {
        statement,
        witness,
        commitment_mixed_t,
        commitment_arith_td,
        n_arith,
        k_bin,
        k_bin_padded,
        k_bin_packed,
        n_bin,
        n_bin_padded,
        n_bin_packed,
        k_arith,
        l_binary,
        l_arithmetic,
        binary_f2_start,
        arith_agg_start,
        _app: core::marker::PhantomData,
    })
}

// ---------------------------------------------------------------------------
// Public builders
// ---------------------------------------------------------------------------

/// Build the mixed R1CS → R reduction (RNG-based challenges).
pub fn build_mixed_r1cs_reduction<A, P, Rng, const N: usize>(
    instance: &MixedR1CSInstance<A>,
    witness: &[GF2],
    crs_a_mixed: &RingMat<CyclotomicPolyRing<P, N>>,
    crs_b_arith_d: &RingMat<CyclotomicPolyRing<P, N>>,
    rng: &mut Rng,
    l_binary: usize,
    l_arithmetic: usize,
    beta: f64,
) -> Result<MixedR1CSReduction<A, P, N>, LabradorError>
where
    A: AppModRing,
    P: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
    Rng: RngExt,
{
    let k_arith = instance.arithmetic.a_r1cs.rows();
    let k_bin_padded = instance.binary.a_r1cs.rows().div_ceil(N) * N;
    let binary_f2_challenges: Vec<BinaryChallengeSet> = (0..l_binary)
        .map(|_| BinaryChallengeSet {
            alpha: (0..k_bin_padded).map(|_| rng.random_range(0..2)).collect(),
            beta: (0..k_bin_padded).map(|_| rng.random_range(0..2)).collect(),
            gamma: (0..k_bin_padded).map(|_| rng.random_range(0..2)).collect(),
        })
        .collect();
    let arith_phi: Vec<Vec<A>> = (0..l_arithmetic)
        .map(|_| (0..k_arith).map(|_| A::rand(rng)).collect())
        .collect();
    let arith_agg: Vec<ArithAggChallenge<A>> = (0..l_arithmetic)
        .map(|_| {
            (
                (0..k_arith).map(|_| A::rand(rng)).collect(),
                (0..k_arith).map(|_| A::rand(rng)).collect(),
                (0..k_arith).map(|_| A::rand(rng)).collect(),
                (0..l_arithmetic * k_arith).map(|_| A::rand(rng)).collect(),
            )
        })
        .collect();

    let ph1 = build_mixed_phase1::<A, P, N>(
        instance,
        witness,
        crs_a_mixed,
        l_binary,
        l_arithmetic,
        beta,
    )?;
    let ph2a = build_mixed_phase2a::<A, P, N>(
        instance,
        ph1,
        crs_b_arith_d,
        l_binary,
        l_arithmetic,
        binary_f2_challenges,
        arith_phi.clone(),
    )?;
    build_mixed_phase2b::<A, P, N>(
        instance,
        ph2a,
        crs_a_mixed,
        crs_b_arith_d,
        l_binary,
        l_arithmetic,
        arith_phi,
        arith_agg,
    )
}

/// Build the mixed R1CS → R reduction (transcript-based challenges).
///
/// Follows the paper FS order (§6): compute t_mixed → absorb → sample
/// binary F2 + φ_i → compute d_i + td → absorb td → sample aggregation
/// challenges → build F2 + g_j.
pub fn build_mixed_r1cs_reduction_transcript<A, P, T, const N: usize>(
    instance: &MixedR1CSInstance<A>,
    witness: &[GF2],
    crs_a_mixed: &RingMat<CyclotomicPolyRing<P, N>>,
    crs_b_arith_d: &RingMat<CyclotomicPolyRing<P, N>>,
    transcript: &mut T,
    l_binary: usize,
    l_arithmetic: usize,
    beta: f64,
) -> Result<MixedR1CSReduction<A, P, N>, TranscriptError>
where
    A: AppModRing,
    P: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
    T: grid_transcript::Transcript,
{
    let k_arith = instance.arithmetic.a_r1cs.rows();
    let k_bin_padded = instance.binary.a_r1cs.rows().div_ceil(N) * N;

    // Phase 1: compute t_mixed.
    let ph1 =
        build_mixed_phase1::<A, P, N>(instance, witness, crs_a_mixed, l_binary, l_arithmetic, beta)
            .map_err(|e| TranscriptError::InvalidInput(alloc::format!("{e:?}")))?;

    // Absorb t_mixed.
    transcript
        .append_serializable(b"mx_t_mixed", &ph1.commitment_mixed_t)
        .map_err(|e| TranscriptError::InvalidInput(alloc::format!("{e:?}")))?;

    // Sample binary F2 challenges.
    let binary_f2_challenges: Vec<BinaryChallengeSet> = (0..l_binary)
        .map(|_| -> Result<_, TranscriptError> {
            Ok(BinaryChallengeSet {
                alpha: (0..k_bin_padded)
                    .map(|_| {
                        transcript
                            .challenge_bytes(b"mx_bin_f2_a", 1)
                            .map(|b| (b[0] % 2) as u64)
                    })
                    .collect::<Result<_, _>>()?,
                beta: (0..k_bin_padded)
                    .map(|_| {
                        transcript
                            .challenge_bytes(b"mx_bin_f2_b", 1)
                            .map(|b| (b[0] % 2) as u64)
                    })
                    .collect::<Result<_, _>>()?,
                gamma: (0..k_bin_padded)
                    .map(|_| {
                        transcript
                            .challenge_bytes(b"mx_bin_f2_c", 1)
                            .map(|b| (b[0] % 2) as u64)
                    })
                    .collect::<Result<_, _>>()?,
            })
        })
        .collect::<Result<_, _>>()?;

    // Sample arithmetic φ_i challenges (bound to t_mixed).
    let arith_phi: Vec<Vec<A>> = (0..l_arithmetic)
        .map(|_| {
            (0..k_arith)
                .map(|_| {
                    A::sample_from_transcript(transcript, b"mx_arith_phi")
                        .map_err(|e| TranscriptError::InvalidInput(alloc::format!("{e:?}")))
                })
                .collect::<Result<_, _>>()
        })
        .collect::<Result<_, _>>()?;

    // Phase 2a: finish binary F2, compute d_i and td.
    let ph2a = build_mixed_phase2a::<A, P, N>(
        instance,
        ph1,
        crs_b_arith_d,
        l_binary,
        l_arithmetic,
        binary_f2_challenges,
        arith_phi.clone(),
    )
    .map_err(|e| TranscriptError::InvalidInput(alloc::format!("{e:?}")))?;

    // Absorb td.
    transcript
        .append_serializable(b"mx_arith_td", &ph2a.commitment_arith_td)
        .map_err(|e| TranscriptError::InvalidInput(alloc::format!("{e:?}")))?;

    // Sample arithmetic aggregation challenges (bound to t_mixed and td).
    let arith_agg: Vec<ArithAggChallenge<A>> = (0..l_arithmetic)
        .map(|_| -> Result<_, TranscriptError> {
            Ok((
                (0..k_arith)
                    .map(|_| {
                        A::sample_from_transcript(transcript, b"mx_arith_agg_alpha")
                            .map_err(|e| TranscriptError::InvalidInput(alloc::format!("{e:?}")))
                    })
                    .collect::<Result<_, _>>()?,
                (0..k_arith)
                    .map(|_| {
                        A::sample_from_transcript(transcript, b"mx_arith_agg_beta")
                            .map_err(|e| TranscriptError::InvalidInput(alloc::format!("{e:?}")))
                    })
                    .collect::<Result<_, _>>()?,
                (0..k_arith)
                    .map(|_| {
                        A::sample_from_transcript(transcript, b"mx_arith_agg_gamma")
                            .map_err(|e| TranscriptError::InvalidInput(alloc::format!("{e:?}")))
                    })
                    .collect::<Result<_, _>>()?,
                (0..l_arithmetic * k_arith)
                    .map(|_| {
                        A::sample_from_transcript(transcript, b"mx_arith_agg_delta")
                            .map_err(|e| TranscriptError::InvalidInput(alloc::format!("{e:?}")))
                    })
                    .collect::<Result<_, _>>()?,
            ))
        })
        .collect::<Result<_, _>>()?;

    // Phase 2b: build aggregation constraints and assemble final statement/witness.
    build_mixed_phase2b::<A, P, N>(
        instance,
        ph2a,
        crs_a_mixed,
        crs_b_arith_d,
        l_binary,
        l_arithmetic,
        arith_phi,
        arith_agg,
    )
    .map_err(|e| TranscriptError::InvalidInput(alloc::format!("{e:?}")))
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

/// Verify a mixed R1CS reduction.
pub fn verify_mixed_r1cs_reduction<A, P, const N: usize>(
    reduction: &MixedR1CSReduction<A, P, N>,
    max_norm_bound: f64,
) -> Result<(), alloc::string::String>
where
    A: AppModRing,
    P: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    crate::relation::verify(&reduction.statement, &reduction.witness, max_norm_bound)?;

    // Reject degenerate degree.
    if N == 0 {
        return Err(alloc::string::String::from("N must be > 0"));
    }
    // Enforce the paper's restricted mixed-encoding shape independently.
    if !A::is_fermat_modulus_for_degree::<N>() {
        return Err(alloc::string::String::from(
            "mixed R1CS requires app modulus = 2^N+1",
        ));
    }
    if A::binary_encoding_width::<N>() != Some(N) {
        return Err(alloc::string::String::from(
            "mixed R1CS requires binary encoding width = N",
        ));
    }
    if reduction.n_bin != reduction.n_arith * N {
        return Err(alloc::format!(
            "n_bin ({}) != n_arith ({}) * N ({N})",
            reduction.n_bin,
            reduction.n_arith
        ));
    }

    // Reject vacuous soundness parameters.
    if reduction.l_binary == 0 || reduction.l_arithmetic == 0 {
        return Err(alloc::string::String::from(
            "l_binary and l_arithmetic must be > 0",
        ));
    }

    // Validate binary dimensions and Theorem 6.2 bounds via checked helper.
    let q = P::modulus();
    let Some(bp) = crate::reduction::binary_r1cs::check_binary_params::<N>(
        reduction.k_bin,
        reduction.n_bin,
        reduction.k_bin_padded,
        reduction.n_bin_padded,
        reduction.k_bin_packed,
        reduction.n_bin_packed,
        reduction.l_binary,
        q,
    ) else {
        return Err(alloc::string::String::from(
            "binary dimensions fail invariants or Theorem 6.2",
        ));
    };
    if bp.f2_start != reduction.binary_f2_start {
        return Err(alloc::format!(
            "binary_f2_start mismatch: struct says {}, computed {}",
            reduction.binary_f2_start,
            bp.f2_start,
        ));
    }

    // Validate arith_agg_start: must lie exactly at the commit-opening boundary.
    let f = &reduction.statement.f;
    let expected_agg_start =
        reduction.commitment_mixed_t.len() + reduction.commitment_arith_td.len();
    if reduction.arith_agg_start != expected_agg_start
        || reduction.arith_agg_start + reduction.l_arithmetic != f.len()
    {
        return Err(alloc::format!(
            "arith_agg_start {} + l_arithmetic {} != f.len {} (expected agg_start {expected_agg_start})",
            reduction.arith_agg_start,
            reduction.l_arithmetic,
            f.len(),
        ));
    }

    let f_prime = &reduction.statement.f_prime;
    let bf2 = reduction.binary_f2_start;
    if f_prime.len() < bf2 + reduction.l_binary {
        return Err(alloc::format!(
            "F' has {} functions, need {} for binary F2",
            f_prime.len(),
            bf2 + reduction.l_binary
        ));
    }
    let max_pos = bp.max_pos;
    let min_neg = bp.min_neg;
    for i in 0..reduction.l_binary {
        let b_poly = f_prime[bf2 + i].b();
        for coeff_idx in 1..N {
            if !b_poly.coeff(coeff_idx).is_zero() {
                return Err(alloc::format!(
                    "mixed binary F2 b[{i}] has non-zero coefficient at {coeff_idx}"
                ));
            }
        }
        let g_canon = b_poly.coeff(0).to_u64();
        let g_signed: i64 = if g_canon <= max_pos {
            g_canon as i64
        } else if g_canon >= min_neg {
            -((q - g_canon) as i64)
        } else {
            return Err(alloc::format!(
                "mixed binary g[{i}] out of range (g_canon={g_canon})"
            ));
        };
        if g_signed % 2 != 0 {
            return Err(alloc::format!("mixed binary g[{i}] is odd (g={g_signed})"));
        }
    }

    for j in 0..reduction.l_arithmetic {
        if !check_divisible_by_x_minus_2::<A, P, N>(f[reduction.arith_agg_start + j].b()) {
            return Err(alloc::format!("arithmetic g_{j}(2) not 0 mod app modulus"));
        }
    }

    const BINARY_PARTS: usize = 8;
    let arith_total = 3 + reduction.l_arithmetic;
    for part_idx in BINARY_PARTS..BINARY_PARTS + arith_total {
        if part_idx >= reduction.witness.parts.len() {
            return Err(alloc::format!(
                "witness has {} parts, need {}",
                reduction.witness.parts.len(),
                part_idx + 1
            ));
        }
        for (poly_idx, poly) in reduction.witness.parts[part_idx].iter().enumerate() {
            if verify_naf_coeffs::<P, N>(poly).is_none() {
                return Err(alloc::format!(
                    "arithmetic part {part_idx} NAF coeff out of bounds at poly {poly_idx}"
                ));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use grid_algebra::arith::gf2::GF2;
    use grid_algebra::arith::prime::PrimeField;
    use grid_algebra::arith::zm::Zm;
    use grid_algebra::lattice::types::RingMat;
    use grid_algebra::poly::ring::PolyRing;

    type P = PrimeField<12289>;
    type A257 = Zm<257>;

    fn gf2(v: u8) -> GF2 {
        GF2::new(v)
    }

    #[test]
    fn test_mixed_r1cs_reduction_smoke() {
        let k_bin = 8;
        let n_arith = 1;
        let n_bin = n_arith * 8;
        let l_bin = 1;
        let l_arith = 1;
        let mut bin_entries = vec![gf2(0); k_bin * n_bin];
        for i in 0..k_bin.min(n_bin) {
            bin_entries[i * n_bin + i] = gf2(1);
        }
        let binary = BinaryR1CSInstance {
            a_r1cs: RingMat::new(k_bin, n_bin, bin_entries.clone()),
            b_r1cs: RingMat::new(k_bin, n_bin, bin_entries.clone()),
            c_r1cs: RingMat::new(k_bin, n_bin, bin_entries),
        };
        let k_arith = 1;
        let arithmetic = ArithR1CSInstance {
            a_r1cs: RingMat::new(k_arith, n_arith, vec![A257::from_u64(1)]),
            b_r1cs: RingMat::new(k_arith, n_arith, vec![A257::from_u64(1)]),
            c_r1cs: RingMat::new(k_arith, n_arith, vec![A257::from_u64(1)]),
        };
        let instance = MixedR1CSInstance { binary, arithmetic };
        let mut witness = vec![gf2(0); n_bin];
        witness[0] = gf2(1);
        let mixed_cols = 3 * (k_bin.div_ceil(8)) + 3 * k_arith + (n_bin.div_ceil(8));
        let crs_a_mixed = RingMat::new(
            1,
            mixed_cols,
            (0..mixed_cols)
                .map(|_| CyclotomicPolyRing::<P, 8>::zero())
                .collect(),
        );
        let crs_b_arith_d = RingMat::new(
            1,
            l_arith * k_arith,
            (0..l_arith * k_arith)
                .map(|_| CyclotomicPolyRing::<P, 8>::zero())
                .collect(),
        );
        let mut rng = grid_std::test_rng();
        let reduction = build_mixed_r1cs_reduction::<A257, P, _, 8>(
            &instance,
            &witness,
            &crs_a_mixed,
            &crs_b_arith_d,
            &mut rng,
            l_bin,
            l_arith,
            10.0,
        )
        .unwrap();
        assert_eq!(reduction.witness.num_parts(), 8 + 3 + l_arith);
        assert_eq!(reduction.n_arith, n_arith);
        let l2 = reduction.witness.l2_norm_squared().sqrt();
        match verify_mixed_r1cs_reduction::<A257, P, 8>(&reduction, l2 + 1.0) {
            Ok(()) => {}
            Err(e) => panic!("verify failed: {e}"),
        }
    }

    #[test]
    fn test_mixed_r1cs_verify_positive() {
        let k_bin = 8;
        let n_arith = 1;
        let n_bin = 8;
        let l = 1;
        let mut bin_entries = vec![gf2(0); k_bin * n_bin];
        for i in 0..k_bin.min(n_bin) {
            bin_entries[i * n_bin + i] = gf2(1);
        }
        let binary = BinaryR1CSInstance {
            a_r1cs: RingMat::new(k_bin, n_bin, bin_entries.clone()),
            b_r1cs: RingMat::new(k_bin, n_bin, bin_entries.clone()),
            c_r1cs: RingMat::new(k_bin, n_bin, bin_entries),
        };
        let k_arith = 1;
        let arithmetic = ArithR1CSInstance {
            a_r1cs: RingMat::new(k_arith, n_arith, vec![A257::from_u64(1)]),
            b_r1cs: RingMat::new(k_arith, n_arith, vec![A257::from_u64(1)]),
            c_r1cs: RingMat::new(k_arith, n_arith, vec![A257::from_u64(1)]),
        };
        let instance = MixedR1CSInstance { binary, arithmetic };
        let mut witness = vec![gf2(0); n_bin];
        witness[0] = gf2(1);
        let mixed_cols = 3 * (k_bin.div_ceil(8)) + 3 * k_arith + (n_bin.div_ceil(8));
        let crs_a_mixed = RingMat::new(
            1,
            mixed_cols,
            (0..mixed_cols)
                .map(|_| CyclotomicPolyRing::<P, 8>::zero())
                .collect(),
        );
        let crs_b_arith_d = RingMat::new(
            1,
            l * k_arith,
            (0..l * k_arith)
                .map(|_| CyclotomicPolyRing::<P, 8>::zero())
                .collect(),
        );
        let mut rng = grid_std::test_rng();
        let reduction = build_mixed_r1cs_reduction::<A257, P, _, 8>(
            &instance,
            &witness,
            &crs_a_mixed,
            &crs_b_arith_d,
            &mut rng,
            l,
            l,
            10.0,
        )
        .unwrap();
        let l2 = reduction.witness.l2_norm_squared().sqrt();
        assert!(verify_mixed_r1cs_reduction::<A257, P, 8>(&reduction, l2 + 1.0).is_ok());
    }

    #[test]
    fn test_mixed_r1cs_tampered_non_scalar_f2_b_rejected() {
        let k_bin = 8;
        let n_arith = 1;
        let n_bin = 8;
        let l = 1;
        let mut bin_entries = vec![gf2(0); k_bin * n_bin];
        for i in 0..k_bin.min(n_bin) {
            bin_entries[i * n_bin + i] = gf2(1);
        }
        let binary = BinaryR1CSInstance {
            a_r1cs: RingMat::new(k_bin, n_bin, bin_entries.clone()),
            b_r1cs: RingMat::new(k_bin, n_bin, bin_entries.clone()),
            c_r1cs: RingMat::new(k_bin, n_bin, bin_entries),
        };
        let k_arith = 1;
        let arithmetic = ArithR1CSInstance {
            a_r1cs: RingMat::new(k_arith, n_arith, vec![A257::from_u64(1)]),
            b_r1cs: RingMat::new(k_arith, n_arith, vec![A257::from_u64(1)]),
            c_r1cs: RingMat::new(k_arith, n_arith, vec![A257::from_u64(1)]),
        };
        let instance = MixedR1CSInstance { binary, arithmetic };
        let mut witness = vec![gf2(0); n_bin];
        witness[0] = gf2(1);
        let mixed_cols = 3 * (k_bin.div_ceil(8)) + 3 * k_arith + (n_bin.div_ceil(8));
        let crs_a_mixed = RingMat::new(
            1,
            mixed_cols,
            (0..mixed_cols)
                .map(|_| CyclotomicPolyRing::<P, 8>::zero())
                .collect(),
        );
        let crs_b_arith_d = RingMat::new(
            1,
            l * k_arith,
            (0..l * k_arith)
                .map(|_| CyclotomicPolyRing::<P, 8>::zero())
                .collect(),
        );
        let mut rng = grid_std::test_rng();
        let mut reduction = build_mixed_r1cs_reduction::<A257, P, _, 8>(
            &instance,
            &witness,
            &crs_a_mixed,
            &crs_b_arith_d,
            &mut rng,
            l,
            l,
            10.0,
        )
        .unwrap();
        let bf2 = reduction.binary_f2_start;
        let orig_b = reduction.statement.f_prime[bf2].b().clone();
        let mut tampered = orig_b.clone();
        tampered.set_coeff(1, P::one());
        match &mut reduction.statement.f_prime[bf2] {
            QuadraticFunction::Dense(d) => d.b = tampered.clone(),
            QuadraticFunction::Sparse(s) => s.b = tampered,
        }
        let l2 = reduction.witness.l2_norm_squared().sqrt();
        let err = verify_mixed_r1cs_reduction::<A257, P, 8>(&reduction, l2 + 1.0)
            .expect_err("non-scalar F2 b should be rejected");
        assert!(
            err.contains("non-zero coefficient"),
            "expected 'non-zero coefficient' in error, got: {err}"
        );
    }

    #[test]
    fn test_mixed_r1cs_tampered_w_pack_rejected() {
        let k_bin = 8;
        let n_arith = 1;
        let n_bin = 8;
        let l = 1;
        let mut bin_entries = vec![gf2(0); k_bin * n_bin];
        for i in 0..k_bin.min(n_bin) {
            bin_entries[i * n_bin + i] = gf2(1);
        }
        let binary = BinaryR1CSInstance {
            a_r1cs: RingMat::new(k_bin, n_bin, bin_entries.clone()),
            b_r1cs: RingMat::new(k_bin, n_bin, bin_entries.clone()),
            c_r1cs: RingMat::new(k_bin, n_bin, bin_entries),
        };
        let k_arith = 1;
        let arithmetic = ArithR1CSInstance {
            a_r1cs: RingMat::new(k_arith, n_arith, vec![A257::from_u64(1)]),
            b_r1cs: RingMat::new(k_arith, n_arith, vec![A257::from_u64(1)]),
            c_r1cs: RingMat::new(k_arith, n_arith, vec![A257::from_u64(1)]),
        };
        let instance = MixedR1CSInstance { binary, arithmetic };
        let mut witness = vec![gf2(0); n_bin];
        witness[0] = gf2(1);
        let mixed_cols = 3 * (k_bin.div_ceil(8)) + 3 * k_arith + (n_bin.div_ceil(8));
        let crs_a_mixed = RingMat::new(
            1,
            mixed_cols,
            (0..mixed_cols)
                .map(|_| CyclotomicPolyRing::<P, 8>::zero())
                .collect(),
        );
        let crs_b_arith_d = RingMat::new(
            1,
            l * k_arith,
            (0..l * k_arith)
                .map(|_| CyclotomicPolyRing::<P, 8>::zero())
                .collect(),
        );
        let mut rng = grid_std::test_rng();
        let mut reduction = build_mixed_r1cs_reduction::<A257, P, _, 8>(
            &instance,
            &witness,
            &crs_a_mixed,
            &crs_b_arith_d,
            &mut rng,
            l,
            l,
            10.0,
        )
        .unwrap();

        // Tamper with coefficient 0 of w_pack (witness part 3) — set it to 2
        // instead of a binary value. This breaks the binaryity constraint.
        let w_pack = &mut reduction.witness.parts[3][0];
        w_pack.set_coeff(0, P::from_u64(2));

        let l2 = reduction.witness.l2_norm_squared().sqrt();
        assert!(
            verify_mixed_r1cs_reduction::<A257, P, 8>(&reduction, l2 + 1.0).is_err(),
            "tampered w_pack coefficient should cause verification failure"
        );
    }
}
