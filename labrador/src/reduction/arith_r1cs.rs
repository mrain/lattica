//! R1CS mod 2^d+1 → R reduction (§6, Figure 5, Theorem 6.3).
//!
//! Reduces an R1CS instance over `Z_{2^d+1}` to a LaBRADOR principal relation R
//! instance. Uses NAF encoding to embed field elements as polynomials with
//! small coefficients, and the morphism `φ: X → 2` to verify arithmetic.
//!
//! Soundness error: `2 * p^(-l)` where `p` is the smallest prime factor of
//! `2^d + 1`, and `l` is the number of aggregation rounds.
//!
//! # Encoding
//!
//! Unlike the binary reduction (which packs N bits per polynomial), arithmetic
//! R1CS uses one NAF-encoded polynomial per scalar value. Each app-ring element
//! `a ∈ Z_{2^N+1}` is encoded as `Enc(a) ∈ R_q` via [`AppModRing::to_naf_digits`].
//!
//! # Witness padding
//!
//! Figure 5's logical vectors have varying lengths (a,b,c,d_i = k; w = n).
//! The LaBRADOR relation R requires uniform vector rank, so these are embedded
//! into rank-`max(k,n)` vectors by zero-padding. Padding entries are never
//! referenced by aggregation or opening phi terms — all phi entries beyond the
//! logical vector length are fixed to zero.
//!
//! # Modulus constraint
//!
//! Theorem 6.3 is for R1CS over `Z_{2^d+1}` specifically. Builders gate on
//! [`AppModRing::is_fermat_modulus_for_degree`] and reject non-matching moduli.
//!
//! **IMPORTANT**: Each aggregation round produces `g_j` (the evaluated F b-constant).
//! `g_j` MUST be hashed into the transcript at the reduction layer and also appear
//! in the downstream LaBRADOR statement. If a future refactor decouples this module
//! from the LaBRADOR statement-hashing flow, the transcript binding here provides
//! the soundness anchor — removing it silently breaks composition.

use alloc::vec;
use alloc::vec::Vec;

use grid_algebra::arith::ring::{IntegerRing, Ring};
use grid_algebra::lattice::types::{RingMat, RingVec};
use grid_algebra::poly::ring::{CyclotomicPolyRing, NegacyclicMulRing, PolyRing};
use grid_serialize::{CanonicalDeserialize, CanonicalSerialize};
use grid_std::rand::RngExt;
use grid_transcript::TranscriptError;

use crate::error::LabradorError;
use crate::reduction::app_ring::AppModRing;
use crate::relation::{LabradorStatement, LabradorWitness, QuadraticFunction};

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

/// Arithmetic R1CS instance over `Z_{M}` where `M = 2^d + 1`.
///
/// The R1CS is defined over the application ring `A`. The morphism `φ: X → 2`
/// maps polynomials back to app-ring elements via evaluation.
#[derive(Debug, Clone)]
pub struct ArithR1CSInstance<A: AppModRing> {
    pub a_r1cs: RingMat<A>,
    pub b_r1cs: RingMat<A>,
    pub c_r1cs: RingMat<A>,
}

/// Result of the arithmetic R1CS → R reduction.
///
/// `A` — application ring (e.g. `Zm<257>`, `FermatRing64`)
/// `P` — proof-ring coefficient type (e.g. `PrimeField<12289>`)
#[derive(Debug, Clone)]
pub struct ArithR1CSReduction<A, P, const N: usize>
where
    A: AppModRing,
    P: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    pub statement: LabradorStatement<CyclotomicPolyRing<P, N>>,
    pub witness: LabradorWitness<CyclotomicPolyRing<P, N>>,
    pub commitment_t: RingVec<CyclotomicPolyRing<P, N>>,
    pub commitment_td: RingVec<CyclotomicPolyRing<P, N>>,
    pub k: usize,
    pub n: usize,
    pub l: usize,
    pub alpha_challenges: Vec<Vec<A>>,
    pub beta_challenges: Vec<Vec<A>>,
    pub gamma_challenges: Vec<Vec<A>>,
    pub delta_challenges: Vec<Vec<A>>,
    /// Redundant builder output — verifier reads from `statement`.
    pub g_polys: Vec<CyclotomicPolyRing<P, N>>,
    pub phi_challenges: Vec<Vec<A>>,
    pub _app: core::marker::PhantomData<A>,
}

// ---------------------------------------------------------------------------
// Theorem 6.3 bound validation
// ---------------------------------------------------------------------------

/// Validate Theorem 6.3 slack conditions (paper line 429).
///
/// Checks:
/// 1. `sqrt((n + (3+l)k) * d / 2) * β + β²/2 < q` (overflow guarantee)
/// 2. `(n + (3+l)k) * d < 0.3 * q` (composition slack)
pub(crate) fn validate_theorem_6_3_bounds<P>(
    k: usize,
    n: usize,
    l: usize,
    d: usize,
    beta: f64,
) -> Result<(), LabradorError>
where
    P: IntegerRing<Canonical = u64>,
{
    if !beta.is_finite() || beta <= 0.0 {
        return Err(LabradorError::InvalidInput(alloc::format!(
            "β ({beta}) must be finite and positive"
        )));
    }
    if l == 0 {
        return Err(LabradorError::InvalidInput(alloc::string::String::from(
            "l must be > 0; no aggregation rounds means the a∘b=c check disappears",
        )));
    }

    let total = n
        .checked_add(
            k.checked_mul(3usize.checked_add(l).ok_or_else(|| {
                LabradorError::InvalidInput(alloc::string::String::from("l overflow"))
            })?)
            .ok_or_else(|| {
                LabradorError::InvalidInput(alloc::string::String::from("k*(3+l) overflow"))
            })?,
        )
        .ok_or_else(|| {
            LabradorError::InvalidInput(alloc::string::String::from("n+(3+l)k overflow"))
        })?;

    let scalar_bound = total.checked_mul(d).ok_or_else(|| {
        LabradorError::InvalidInput(alloc::format!(
            "(n+(3+l)k)*d overflow: total={total}, d={d}"
        ))
    })?;

    let q = P::modulus();

    // Overflow guarantee: sqrt(scalar_bound/2) * β + β²/2 < q
    let rhs = (scalar_bound as f64 / 2.0).sqrt() * beta + beta * beta / 2.0;
    bail!(
        rhs >= q as f64,
        "Theorem 6.3 overflow: sqrt((n+(3+l)k)d/2)*β + β²/2 = {rhs:.1} >= q ({q})"
    );

    // Composition slack: (n+(3+l)k)*d < 0.3 * q
    bail!(
        (scalar_bound as u128) * 10 >= (q as u128) * 3,
        "Theorem 6.3 slack: (n+(3+l)k)d = {scalar_bound} >= 0.3·q ({})",
        (q as f64 * 0.3) as u64
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// NAF encoding helpers
// ---------------------------------------------------------------------------

/// NAF-encode an app-ring element as a proof-ring polynomial.
pub(crate) fn encode_app_naf<A, P, const N: usize>(
    a: &A,
) -> Result<CyclotomicPolyRing<P, N>, LabradorError>
where
    A: AppModRing,
    P: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    let digits = a.to_naf_digits::<N>()?;
    let coeffs = core::array::from_fn(|i| match digits[i] {
        1 => P::one(),
        -1 => -P::one(),
        _ => P::zero(),
    });
    Ok(CyclotomicPolyRing::from_array(coeffs))
}

/// Verify all coefficients of a NAF polynomial are in {-1, 0, 1}.
pub fn verify_naf_coeffs<P, const N: usize>(poly: &CyclotomicPolyRing<P, N>) -> Option<usize>
where
    P: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    let q = P::modulus();
    let minus_one = q.wrapping_sub(1);
    let mut norm_sq = 0usize;
    for c in poly.coeffs() {
        let v = c.to_u64();
        if v == 0 {
        } else if v == 1 || v == minus_one {
            norm_sq += 1;
        } else {
            return None;
        }
    }
    Some(norm_sq)
}

/// Verify that all NAF-encoded witness parts have coefficients in {-1, 0, 1}.
pub fn verify_naf_witness<A, P, const N: usize>(
    reduction: &ArithR1CSReduction<A, P, N>,
) -> Result<(), (usize, usize)>
where
    A: AppModRing,
    P: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    let witness = &reduction.witness;
    for (part_idx, part) in witness.parts.iter().enumerate() {
        for (poly_idx, poly) in part.iter().enumerate() {
            if verify_naf_coeffs(poly).is_none() {
                return Err((part_idx, poly_idx));
            }
        }
    }
    Ok(())
}

/// Extract NAF digits from a proof-ring polynomial, then evaluate in the app
/// ring and check the result is zero.
/// Check if a proof-ring polynomial evaluates to 0 in the app ring at X=2.
///
/// Centers each coefficient into `[-q/2, q/2)`, converts to an A element,
/// and accumulates `∑ c_i · 2^i` directly in A.  Works for all app rings
/// including multi-limb moduli.
pub fn check_divisible_by_x_minus_2<A, P, const N: usize>(poly: &CyclotomicPolyRing<P, N>) -> bool
where
    A: AppModRing,
    P: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    let q = P::modulus();
    let half_q = q / 2;
    let mut acc: A = A::zero();
    let mut weight: A = A::one();
    for c in poly.coeffs() {
        let raw = c.to_u64();
        if raw != 0 {
            let coeff_a: A = if raw > half_q {
                // centered = raw - q  (negative)
                let neg_val = q - raw;
                -A::from_u64(neg_val)
            } else {
                A::from_u64(raw)
            };
            acc += coeff_a * weight.clone();
        }
        weight = weight.clone() + weight;
    }
    acc.is_zero()
}

/// Compute Hadamard (element-wise) product of two vectors in the app ring.
fn hadamard_product<A: AppModRing>(a: &[A], b: &[A]) -> Vec<A> {
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| ai.clone() * bi.clone())
        .collect()
}

// ---------------------------------------------------------------------------
// Aggregation function builder
// ---------------------------------------------------------------------------

/// Build the F (full-vanishing) quadratic function for arithmetic aggregation.
///
/// Paper-aligned construction (§6, Figure 5).
/// Returns `Ok((f_j_constraint, g_j))` or an error if NAF encoding fails.
#[allow(clippy::needless_range_loop)]
fn build_f_aggregation_rq<A, P, const N: usize>(
    alpha_j: &[A],
    beta_j: &[A],
    gamma_j: &[A],
    delta_j: &[A],
    phi_challenges: &[Vec<A>],
    j: usize,
    l: usize,
    k: usize,
    n: usize,
    max_rank: usize,
    num_parts: usize,
    a_r1cs: &RingMat<A>,
    b_r1cs: &RingMat<A>,
    c_r1cs: &RingMat<A>,
    witness_parts: &[Vec<CyclotomicPolyRing<P, N>>],
) -> Result<
    (
        QuadraticFunction<CyclotomicPolyRing<P, N>>,
        CyclotomicPolyRing<P, N>,
    ),
    LabradorError,
>
where
    A: AppModRing,
    P: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    let zero = CyclotomicPolyRing::<P, N>::zero();
    let one = CyclotomicPolyRing::<P, N>::one();

    let idx_a: usize = 0;
    let idx_b: usize = 1;
    let idx_c: usize = 2;
    let idx_w: usize = 3;

    let enc = |a: &A| encode_app_naf::<A, P, N>(a);

    // --- Linear terms (phi vectors) ---
    let mut phi: Vec<Vec<CyclotomicPolyRing<P, N>>> = vec![vec![zero.clone(); max_rank]; num_parts];

    // Term 1: ⟨α_j, A·w - a⟩
    for idx in 0..k {
        phi[idx_a][idx] = -enc(&alpha_j[idx])?;
    }
    for col in 0..n {
        let mut acc = A::zero();
        for i in 0..k {
            acc += alpha_j[i].clone() * a_r1cs.get(i, col).clone();
        }
        phi[idx_w][col] = enc(&acc)?;
    }

    // Term 2: ⟨β_j, B·w - b⟩
    for idx in 0..k {
        phi[idx_b][idx] = -enc(&beta_j[idx])?;
    }
    for col in 0..n {
        let mut acc = A::zero();
        for i in 0..k {
            acc += beta_j[i].clone() * b_r1cs.get(i, col).clone();
        }
        phi[idx_w][col] += enc(&acc)?;
    }

    // Term 3: ⟨γ_j, C·w - c⟩
    for idx in 0..k {
        phi[idx_c][idx] = -enc(&gamma_j[idx])?;
    }
    for col in 0..n {
        let mut acc = A::zero();
        for i in 0..k {
            acc += gamma_j[i].clone() * c_r1cs.get(i, col).clone();
        }
        phi[idx_w][col] += enc(&acc)?;
    }

    // Term 4 (product check): ⟨d_j, b⟩ - ⟨φ_j, c⟩
    for idx in 0..k {
        phi[idx_c][idx] -= enc(&phi_challenges[j][idx])?;
    }

    // Term 5 (Hadamard): Σ_i ⟨δ_i^(j), φ_i ∘ a - d_i⟩
    for col in 0..k {
        let mut acc = A::zero();
        for i in 0..l {
            acc += delta_j[i * k + col].clone() * phi_challenges[i][col].clone();
        }
        phi[idx_a][col] += enc(&acc)?;
    }
    for i in 0..l {
        let d_idx = 4 + i;
        for col in 0..k {
            phi[d_idx][col] = -enc(&delta_j[i * k + col])?;
        }
    }

    // --- Quadratic terms ---
    // ⟨d_j, b⟩: a_{d_j,b} = 1
    let d_j_idx = 4 + j;
    let (qi, qj) = if idx_b < d_j_idx {
        (idx_b, d_j_idx)
    } else {
        (d_j_idx, idx_b)
    };
    let quad = vec![(qi, qj, one)];

    // Debug: verify padding entries are never referenced.
    debug_assert!(phi[idx_a].iter().skip(k).all(|p| p.is_zero()));
    debug_assert!(phi[idx_b].iter().skip(k).all(|p| p.is_zero()));
    debug_assert!(phi[idx_c].iter().skip(k).all(|p| p.is_zero()));
    debug_assert!(phi[idx_w].iter().skip(n).all(|p| p.is_zero()));
    for i in 0..l {
        debug_assert!(phi[4 + i].iter().skip(k).all(|p| p.is_zero()));
    }

    let f_j = QuadraticFunction::from_parts(quad, phi, zero.clone());

    let temp_witness = LabradorWitness::new(witness_parts.to_vec());
    let g_j = f_j.evaluate(&temp_witness);

    let f_constraint = match f_j {
        QuadraticFunction::Dense(d) => {
            QuadraticFunction::Dense(crate::relation::DenseQuadraticFunction {
                a: d.a,
                ij: d.ij,
                phi: d.phi,
                b: g_j.clone(),
            })
        }
        QuadraticFunction::Sparse(s) => {
            QuadraticFunction::Sparse(crate::relation::SparseQuadraticFunction {
                ij_a: s.ij_a,
                phi: s.phi,
                b: g_j.clone(),
            })
        }
    };

    Ok((f_constraint, g_j))
}

// ---------------------------------------------------------------------------
// Builder (RNG-based)
// ---------------------------------------------------------------------------

/// Build the arithmetic R1CS → R reduction.
pub fn build_arith_r1cs_reduction<A, P, Rng, const N: usize>(
    instance: &ArithR1CSInstance<A>,
    witness: &[A],
    crs_a: &RingMat<CyclotomicPolyRing<P, N>>,
    crs_b: &RingMat<CyclotomicPolyRing<P, N>>,
    rng: &mut Rng,
    l: usize,
    beta: f64,
) -> Result<ArithR1CSReduction<A, P, N>, LabradorError>
where
    A: AppModRing,
    P: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
    Rng: RngExt,
{
    let k = instance.a_r1cs.rows();
    let n = instance.a_r1cs.cols();

    bail!(
        instance.b_r1cs.rows() != k,
        "b_r1cs rows ({}) != a_r1cs rows ({})",
        instance.b_r1cs.rows(),
        k
    );
    bail!(
        instance.b_r1cs.cols() != n,
        "b_r1cs cols ({}) != a_r1cs cols ({})",
        instance.b_r1cs.cols(),
        n
    );
    bail!(
        instance.c_r1cs.rows() != k,
        "c_r1cs rows ({}) != a_r1cs rows ({})",
        instance.c_r1cs.rows(),
        k
    );
    bail!(
        instance.c_r1cs.cols() != n,
        "c_r1cs cols ({}) != a_r1cs cols ({})",
        instance.c_r1cs.cols(),
        n
    );
    bail!(
        witness.len() != n,
        "witness len ({}) != expected ({})",
        witness.len(),
        n
    );

    // Phase 0: enforce paper modulus shape (Z_{2^N+1}).
    bail!(
        !A::is_fermat_modulus_for_degree::<N>(),
        "app modulus is not Z_{{2^{N}+1}}; arithmetic reduction requires 2^N+1 modulus"
    );

    // Phase 3: Theorem 6.3 bound validation.
    validate_theorem_6_3_bounds::<P>(k, n, l, N, beta)?;

    let max_rank = k.max(n);
    let zero = CyclotomicPolyRing::<P, N>::zero();

    // Helper: NAF-encode app-ring values and pad to max_rank.
    let encode_pad = |vals: &[A]| -> Result<Vec<CyclotomicPolyRing<P, N>>, LabradorError> {
        let mut polys = Vec::with_capacity(vals.len());
        for v in vals {
            polys.push(encode_app_naf::<A, P, N>(v)?);
        }
        while polys.len() < max_rank {
            polys.push(zero.clone());
        }
        Ok(polys)
    };

    // Step 1: Compute a = A*w, b = B*w, c = C*w in the app ring.
    let a_field = instance.a_r1cs.mul_slice(witness);
    let b_field = instance.b_r1cs.mul_slice(witness);
    let c_field = instance.c_r1cs.mul_slice(witness);

    // Step 2: Encode as NAF polynomials, padded to max_rank.
    let a_polys = encode_pad(a_field.entries())?;
    let b_polys = encode_pad(b_field.entries())?;
    let c_polys = encode_pad(c_field.entries())?;
    let w_polys = encode_pad(witness)?;

    // Step 3: Ajtai commitment t = A * (a||b||c||w)
    let total_rank_a = 3 * k + n;
    let mut concat_a = Vec::with_capacity(total_rank_a);
    concat_a.extend_from_slice(&a_polys[..k]);
    concat_a.extend_from_slice(&b_polys[..k]);
    concat_a.extend_from_slice(&c_polys[..k]);
    concat_a.extend_from_slice(&w_polys[..n]);
    bail!(
        crs_a.cols() != total_rank_a,
        "crs_a cols ({}) != expected total rank ({})",
        crs_a.cols(),
        total_rank_a
    );
    let commitment_t = crs_a.mul_slice(&concat_a);

    // Step 4: Sample φ_i ∈ A^k (app-ring challenges).
    let phi_challenges: Vec<Vec<A>> = (0..l)
        .map(|_| (0..k).map(|_| A::rand(rng)).collect())
        .collect();

    // Step 5: Compute d_i = φ_i ∘ a (Hadamard product in app ring).
    let mut d_polys_list: Vec<Vec<CyclotomicPolyRing<P, N>>> = Vec::with_capacity(l);
    let mut d_concat: Vec<CyclotomicPolyRing<P, N>> = Vec::with_capacity(l * k);
    for phi_i in &phi_challenges {
        let d_i_field = hadamard_product(phi_i, a_field.entries());
        let d_i_polys = encode_pad(&d_i_field)?;
        d_polys_list.push(d_i_polys.clone());
        d_concat.extend_from_slice(&d_i_polys[..k]);
    }

    // Step 6: Commitment t_d = B * (d_1||...||d_l)
    bail!(
        crs_b.cols() != l * k,
        "crs_b cols ({}) != expected ({} * {})",
        crs_b.cols(),
        l,
        k
    );
    let commitment_td = crs_b.mul_slice(&d_concat);

    // Step 6.5: Compute aggregation polynomials g_j and build F constraints.
    let num_parts = 4 + l;

    let mut f_aggregation: Vec<QuadraticFunction<CyclotomicPolyRing<P, N>>> = Vec::with_capacity(l);
    let mut alpha_challenges: Vec<Vec<A>> = Vec::with_capacity(l);
    let mut beta_challenges: Vec<Vec<A>> = Vec::with_capacity(l);
    let mut gamma_challenges: Vec<Vec<A>> = Vec::with_capacity(l);
    let mut delta_challenges: Vec<Vec<A>> = Vec::with_capacity(l);
    let mut g_polys: Vec<CyclotomicPolyRing<P, N>> = Vec::with_capacity(l);

    let temp_witness_parts: Vec<Vec<CyclotomicPolyRing<P, N>>> = {
        let mut parts = vec![
            a_polys.clone(),
            b_polys.clone(),
            c_polys.clone(),
            w_polys.clone(),
        ];
        for d_i in &d_polys_list {
            parts.push(d_i.clone());
        }
        parts
    };

    for j in 0..l {
        let alpha_j: Vec<A> = (0..k).map(|_| A::rand(rng)).collect();
        let beta_j: Vec<A> = (0..k).map(|_| A::rand(rng)).collect();
        let gamma_j: Vec<A> = (0..k).map(|_| A::rand(rng)).collect();
        let delta_j: Vec<A> = (0..l * k).map(|_| A::rand(rng)).collect();

        alpha_challenges.push(alpha_j.clone());
        beta_challenges.push(beta_j.clone());
        gamma_challenges.push(gamma_j.clone());
        delta_challenges.push(delta_j.clone());

        let (f_j, g_j) = build_f_aggregation_rq::<A, P, N>(
            &alpha_j,
            &beta_j,
            &gamma_j,
            &delta_j,
            &phi_challenges,
            j,
            l,
            k,
            n,
            max_rank,
            num_parts,
            &instance.a_r1cs,
            &instance.b_r1cs,
            &instance.c_r1cs,
            &temp_witness_parts,
        )?;

        g_polys.push(g_j);
        f_aggregation.push(f_j);
    }

    // Step 7: Build LaBRADOR witness: (a, b, c, w, d_1, ..., d_l)
    let mut witness_parts: Vec<Vec<CyclotomicPolyRing<P, N>>> =
        vec![a_polys, b_polys, c_polys, w_polys];
    for d_i in &d_polys_list {
        witness_parts.push(d_i.clone());
    }
    let labrador_witness = LabradorWitness::new(witness_parts);

    // Step 8: Build F family — Ajtai opening + aggregation constraints.
    let kappa_a = crs_a.rows();
    let kappa_b = crs_b.rows();
    let mut f_functions: Vec<QuadraticFunction<CyclotomicPolyRing<P, N>>> =
        Vec::with_capacity(kappa_a + kappa_b + l);

    // t opening
    for j in 0..kappa_a {
        let row = crs_a.row(j);
        let row_entries = row.entries();

        let mut phi: Vec<Vec<CyclotomicPolyRing<P, N>>> = Vec::with_capacity(num_parts);

        let push_padded = |start: usize, len: usize| {
            let mut v: Vec<CyclotomicPolyRing<P, N>> = row_entries[start..start + len].to_vec();
            while v.len() < max_rank {
                v.push(zero.clone());
            }
            v
        };

        phi.push(push_padded(0, k));
        phi.push(push_padded(k, k));
        phi.push(push_padded(2 * k, k));
        phi.push(push_padded(3 * k, n));
        for _ in 0..l {
            phi.push(vec![zero.clone(); max_rank]);
        }

        f_functions.push(QuadraticFunction::from_parts(
            Vec::new(),
            phi,
            commitment_t.get(j).clone(),
        ));
    }

    // t_d opening
    for j in 0..kappa_b {
        let row = crs_b.row(j);
        let row_entries = row.entries();
        let mut offset = 0usize;

        let mut phi: Vec<Vec<CyclotomicPolyRing<P, N>>> = Vec::with_capacity(num_parts);
        for _ in 0..4 {
            phi.push(vec![zero.clone(); max_rank]);
        }
        for _di in 0..l {
            let mut seg: Vec<CyclotomicPolyRing<P, N>> = row_entries[offset..offset + k].to_vec();
            while seg.len() < max_rank {
                seg.push(zero.clone());
            }
            phi.push(seg);
            offset += k;
        }

        f_functions.push(QuadraticFunction::from_parts(
            Vec::new(),
            phi,
            commitment_td.get(j).clone(),
        ));
    }

    for f_j in f_aggregation {
        f_functions.push(f_j);
    }

    let statement = LabradorStatement {
        f: f_functions,
        f_prime: Vec::new(),
    };

    Ok(ArithR1CSReduction {
        statement,
        witness: labrador_witness,
        commitment_t,
        commitment_td,
        k,
        n,
        l,
        alpha_challenges,
        beta_challenges,
        gamma_challenges,
        delta_challenges,
        g_polys,
        phi_challenges,
        _app: core::marker::PhantomData,
    })
}

// ---------------------------------------------------------------------------
// Builder (transcript-based)
// ---------------------------------------------------------------------------

/// Build the arithmetic R1CS → R reduction using transcript for Fiat-Shamir.
pub fn build_arith_r1cs_reduction_transcript<A, P, T, const N: usize>(
    instance: &ArithR1CSInstance<A>,
    witness: &[A],
    crs_a: &RingMat<CyclotomicPolyRing<P, N>>,
    crs_b: &RingMat<CyclotomicPolyRing<P, N>>,
    transcript: &mut T,
    l: usize,
    beta: f64,
) -> Result<ArithR1CSReduction<A, P, N>, TranscriptError>
where
    A: AppModRing,
    P: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
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

    // Phase 0: enforce paper modulus shape.
    bail_t!(
        !A::is_fermat_modulus_for_degree::<N>(),
        "app modulus is not Z_{{2^{N}+1}}; arithmetic reduction requires 2^N+1 modulus"
    );

    // Phase 3: Theorem 6.3 bound validation.
    validate_theorem_6_3_bounds::<P>(k, n, l, N, beta)
        .map_err(|e| TranscriptError::InvalidInput(alloc::format!("{e:?}")))?;

    let max_rank = k.max(n);
    let zero = CyclotomicPolyRing::<P, N>::zero();

    let encode_pad = |vals: &[A]| -> Result<Vec<CyclotomicPolyRing<P, N>>, TranscriptError> {
        let mut polys = Vec::with_capacity(vals.len());
        for v in vals {
            polys.push(
                encode_app_naf::<A, P, N>(v)
                    .map_err(|e| TranscriptError::InvalidInput(alloc::format!("{e}")))?,
            );
        }
        while polys.len() < max_rank {
            polys.push(zero.clone());
        }
        Ok(polys)
    };

    let a_field = instance.a_r1cs.mul_slice(witness);
    let b_field = instance.b_r1cs.mul_slice(witness);
    let c_field = instance.c_r1cs.mul_slice(witness);

    let a_polys = encode_pad(a_field.entries())?;
    let b_polys = encode_pad(b_field.entries())?;
    let c_polys = encode_pad(c_field.entries())?;
    let w_polys = encode_pad(witness)?;

    let total_rank_a = 3 * k + n;
    let mut concat_a = Vec::with_capacity(total_rank_a);
    concat_a.extend_from_slice(&a_polys[..k]);
    concat_a.extend_from_slice(&b_polys[..k]);
    concat_a.extend_from_slice(&c_polys[..k]);
    concat_a.extend_from_slice(&w_polys[..n]);
    bail_t!(
        crs_a.cols() != total_rank_a,
        "crs_a cols ({}) != expected total rank ({})",
        crs_a.cols(),
        total_rank_a
    );
    let commitment_t = crs_a.mul_slice(&concat_a);
    transcript.append_serializable(b"labrador_arith_t", &commitment_t)?;

    let phi_challenges: Vec<Vec<A>> = (0..l)
        .map(|i| {
            let round_bytes = (i as u32).to_le_bytes();
            transcript.append_bytes(b"labrador_arith_phi", &round_bytes)?;
            (0..k)
                .map(|_| A::sample_from_transcript(transcript, b"labrador_arith_phi_c"))
                .collect::<Result<_, _>>()
        })
        .collect::<Result<_, _>>()?;

    let mut d_polys_list: Vec<Vec<CyclotomicPolyRing<P, N>>> = Vec::with_capacity(l);
    let mut d_concat: Vec<CyclotomicPolyRing<P, N>> = Vec::with_capacity(l * k);
    for phi_i in &phi_challenges {
        let d_i_field = hadamard_product(phi_i, a_field.entries());
        let d_i_polys = encode_pad(&d_i_field)?;
        d_polys_list.push(d_i_polys.clone());
        d_concat.extend_from_slice(&d_i_polys[..k]);
    }

    bail_t!(
        crs_b.cols() != l * k,
        "crs_b cols ({}) != expected ({} * {})",
        crs_b.cols(),
        l,
        k
    );
    let commitment_td = crs_b.mul_slice(&d_concat);
    transcript.append_serializable(b"labrador_arith_td", &commitment_td)?;

    let num_parts = 4 + l;
    let mut f_aggregation: Vec<QuadraticFunction<CyclotomicPolyRing<P, N>>> = Vec::with_capacity(l);
    let mut alpha_challenges: Vec<Vec<A>> = Vec::with_capacity(l);
    let mut beta_challenges: Vec<Vec<A>> = Vec::with_capacity(l);
    let mut gamma_challenges: Vec<Vec<A>> = Vec::with_capacity(l);
    let mut delta_challenges: Vec<Vec<A>> = Vec::with_capacity(l);
    let mut g_polys: Vec<CyclotomicPolyRing<P, N>> = Vec::with_capacity(l);

    let temp_witness_parts: Vec<Vec<CyclotomicPolyRing<P, N>>> = {
        let mut parts = vec![
            a_polys.clone(),
            b_polys.clone(),
            c_polys.clone(),
            w_polys.clone(),
        ];
        for d_i in &d_polys_list {
            parts.push(d_i.clone());
        }
        parts
    };

    for j in 0..l {
        let round_bytes = (j as u32).to_le_bytes();
        transcript.append_bytes(b"labrador_arith_agg", &round_bytes)?;
        let alpha_j: Vec<A> = (0..k)
            .map(|_| A::sample_from_transcript(transcript, b"labrador_arith_agg_alpha"))
            .collect::<Result<_, _>>()?;
        let beta_j: Vec<A> = (0..k)
            .map(|_| A::sample_from_transcript(transcript, b"labrador_arith_agg_beta"))
            .collect::<Result<_, _>>()?;
        let gamma_j: Vec<A> = (0..k)
            .map(|_| A::sample_from_transcript(transcript, b"labrador_arith_agg_gamma"))
            .collect::<Result<_, _>>()?;
        let delta_j: Vec<A> = (0..(l * k))
            .map(|_| A::sample_from_transcript(transcript, b"labrador_arith_agg_delta"))
            .collect::<Result<_, _>>()?;

        alpha_challenges.push(alpha_j.clone());
        beta_challenges.push(beta_j.clone());
        gamma_challenges.push(gamma_j.clone());
        delta_challenges.push(delta_j.clone());

        let (f_j, g_j) = build_f_aggregation_rq::<A, P, N>(
            &alpha_j,
            &beta_j,
            &gamma_j,
            &delta_j,
            &phi_challenges,
            j,
            l,
            k,
            n,
            max_rank,
            num_parts,
            &instance.a_r1cs,
            &instance.b_r1cs,
            &instance.c_r1cs,
            &temp_witness_parts,
        )
        .map_err(|e| TranscriptError::InvalidInput(alloc::format!("{e}")))?;

        f_aggregation.push(f_j);
        g_polys.push(g_j.clone());
        transcript.append_serializable(b"labrador_arith_g", &g_j)?;
    }

    let f_functions: Vec<QuadraticFunction<CyclotomicPolyRing<P, N>>> = {
        let kappa_a = crs_a.rows();
        let kappa_b = crs_b.rows();
        let mut f_funcs = Vec::with_capacity(kappa_a + kappa_b + l);

        for j in 0..kappa_a {
            let row = crs_a.row(j);
            let row_entries = row.entries();
            let mut phi: Vec<Vec<CyclotomicPolyRing<P, N>>> = Vec::with_capacity(num_parts);
            let push_padded = |start: usize, len: usize| {
                let mut v: Vec<CyclotomicPolyRing<P, N>> = row_entries[start..start + len].to_vec();
                while v.len() < max_rank {
                    v.push(zero.clone());
                }
                v
            };
            phi.push(push_padded(0, k));
            phi.push(push_padded(k, k));
            phi.push(push_padded(2 * k, k));
            phi.push(push_padded(3 * k, n));
            for _ in 0..l {
                phi.push(vec![zero.clone(); max_rank]);
            }
            f_funcs.push(QuadraticFunction::from_parts(
                Vec::new(),
                phi,
                commitment_t.get(j).clone(),
            ));
        }

        for j in 0..kappa_b {
            let row = crs_b.row(j);
            let row_entries = row.entries();
            let mut phi: Vec<Vec<CyclotomicPolyRing<P, N>>> = Vec::with_capacity(num_parts);
            for _ in 0..4 {
                phi.push(vec![zero.clone(); max_rank]);
            }
            let mut offset = 0usize;
            for _di in 0..l {
                let mut seg: Vec<CyclotomicPolyRing<P, N>> =
                    row_entries[offset..offset + k].to_vec();
                while seg.len() < max_rank {
                    seg.push(zero.clone());
                }
                phi.push(seg);
                offset += k;
            }
            f_funcs.push(QuadraticFunction::from_parts(
                Vec::new(),
                phi,
                commitment_td.get(j).clone(),
            ));
        }

        for f_j in f_aggregation {
            f_funcs.push(f_j);
        }
        f_funcs
    };

    let statement = LabradorStatement {
        f: f_functions,
        f_prime: Vec::new(),
    };

    let labrador_witness = LabradorWitness {
        parts: temp_witness_parts,
    };

    Ok(ArithR1CSReduction {
        statement,
        witness: labrador_witness,
        commitment_t,
        commitment_td,
        k,
        n,
        l,
        alpha_challenges,
        beta_challenges,
        gamma_challenges,
        delta_challenges,
        g_polys,
        phi_challenges,
        _app: core::marker::PhantomData,
    })
}

// ---------------------------------------------------------------------------
// Verifier helpers
// ---------------------------------------------------------------------------

/// Verify that all aggregation polynomials g_j are divisible by (X-2).
pub fn verify_aggregation<A, P, const N: usize>(reduction: &ArithR1CSReduction<A, P, N>) -> bool
where
    A: AppModRing,
    P: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    if reduction.l == 0 || reduction.g_polys.len() != reduction.l {
        return false;
    }
    reduction
        .g_polys
        .iter()
        .all(|g_j| check_divisible_by_x_minus_2::<A, P, N>(g_j))
}

/// Verify arithmetic aggregation constraints (paper-aligned, §6 Figure 5).
///
/// Three checks:
/// 1. NAF coefficients valid.
/// 2. F constraints vanish: f̃_j(w) - g_j = 0 in R_q.
/// 3. g_j(2) = 0 mod M (via `check_divisible_by_x_minus_2`).
pub fn verify_aggregation_rq<A, P, const N: usize>(reduction: &ArithR1CSReduction<A, P, N>) -> bool
where
    A: AppModRing,
    P: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    if reduction.l == 0 || reduction.statement.f.len() < reduction.l {
        return false;
    }
    if verify_naf_witness(reduction).is_err() {
        return false;
    }

    let f = &reduction.statement.f;
    let witness = &reduction.witness;
    let l = reduction.l;

    let agg_start = f.len().saturating_sub(l);
    for f_j in f.iter().skip(agg_start) {
        let val = f_j.evaluate(witness);
        if !val.is_zero() {
            return false;
        }
        if !check_divisible_by_x_minus_2::<A, P, N>(f_j.b()) {
            return false;
        }
    }
    true
}

/// Paper-aligned verifier helper for arithmetic R1CS aggregation.
///
/// For each round j, rebuilds the F constraint from stored challenges,
/// evaluates on the witness, and checks divisibility by (X-2).
pub fn recompute_and_verify<A, P, const N: usize>(
    instance: &ArithR1CSInstance<A>,
    reduction: &ArithR1CSReduction<A, P, N>,
) -> bool
where
    A: AppModRing,
    P: IntegerRing<Canonical = u64>
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize,
{
    let k = instance.a_r1cs.rows();
    let n = instance.a_r1cs.cols();
    let l = reduction.l;
    let max_rank = k.max(n);
    let num_parts = 4 + l;

    for j in 0..l {
        let result = build_f_aggregation_rq::<A, P, N>(
            &reduction.alpha_challenges[j],
            &reduction.beta_challenges[j],
            &reduction.gamma_challenges[j],
            &reduction.delta_challenges[j],
            &reduction.phi_challenges,
            j,
            l,
            k,
            n,
            max_rank,
            num_parts,
            &instance.a_r1cs,
            &instance.b_r1cs,
            &instance.c_r1cs,
            &reduction.witness.parts,
        );
        let (_f_j, g_j_recomputed) = match result {
            Ok(pair) => pair,
            Err(_) => return false,
        };
        if g_j_recomputed != reduction.g_polys[j] {
            return false;
        }
        if !check_divisible_by_x_minus_2::<A, P, N>(&g_j_recomputed) {
            return false;
        }
    }
    true
}

/// Sample verifier challenges for arithmetic R1CS reduction.
pub fn sample_arith_challenges<A, Rng>(
    k: usize,
    l: usize,
    rng: &mut Rng,
) -> (Vec<Vec<A>>, Vec<Vec<A>>)
where
    A: AppModRing,
    Rng: RngExt,
{
    let phi_challenges: Vec<Vec<A>> = (0..l)
        .map(|_| (0..k).map(|_| A::rand(rng)).collect())
        .collect();

    let challenge_len = k * (l + 3);
    let agg_challenges: Vec<Vec<A>> = (0..l)
        .map(|_| (0..challenge_len).map(|_| A::rand(rng)).collect())
        .collect();

    (phi_challenges, agg_challenges)
}

/// Sample verifier challenges from transcript (Fiat-Shamir).
#[allow(clippy::type_complexity)]
pub fn sample_arith_challenges_transcript<A, P, T>(
    k: usize,
    l: usize,
    transcript: &mut T,
) -> Result<(Vec<Vec<A>>, Vec<Vec<A>>), TranscriptError>
where
    A: AppModRing,
    P: IntegerRing<Canonical = u64>,
    T: grid_transcript::Transcript,
{
    let mut phi_challenges = Vec::with_capacity(l);
    for i in 0..l {
        let round_bytes = (i as u32).to_le_bytes();
        transcript.append_bytes(b"labrador_arith_phi", &round_bytes)?;
        let phi_i: Vec<A> = (0..k)
            .map(|_| A::sample_from_transcript(transcript, b"labrador_arith_phi_c"))
            .collect::<Result<_, _>>()?;
        phi_challenges.push(phi_i);
    }

    let mut agg_challenges = Vec::with_capacity(l);
    for j in 0..l {
        let round_bytes = (j as u32).to_le_bytes();
        transcript.append_bytes(b"labrador_arith_agg", &round_bytes)?;
        let alpha_j: Vec<A> = (0..k)
            .map(|_| A::sample_from_transcript(transcript, b"labrador_arith_agg_alpha"))
            .collect::<Result<_, _>>()?;
        let beta_j: Vec<A> = (0..k)
            .map(|_| A::sample_from_transcript(transcript, b"labrador_arith_agg_beta"))
            .collect::<Result<_, _>>()?;
        let gamma_j: Vec<A> = (0..k)
            .map(|_| A::sample_from_transcript(transcript, b"labrador_arith_agg_gamma"))
            .collect::<Result<_, _>>()?;
        let delta_j: Vec<A> = (0..(l * k))
            .map(|_| A::sample_from_transcript(transcript, b"labrador_arith_agg_delta"))
            .collect::<Result<_, _>>()?;
        let mut agg_j = Vec::with_capacity(k * (l + 3));
        agg_j.extend(alpha_j);
        agg_j.extend(beta_j);
        agg_j.extend(gamma_j);
        agg_j.extend(delta_j);
        agg_challenges.push(agg_j);
    }

    Ok((phi_challenges, agg_challenges))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    use grid_algebra::arith::FermatRing64;
    use grid_algebra::arith::prime::PrimeField;
    use grid_algebra::arith::zm::Zm;
    use grid_algebra::poly::ring::PolyRing;

    type F = PrimeField<12289>;
    type A257 = Zm<257>;

    #[test]
    fn test_hadamard_product() {
        let a = vec![A257::from_u64(3), A257::from_u64(5), A257::from_u64(7)];
        let b = vec![A257::from_u64(2), A257::from_u64(4), A257::from_u64(6)];
        let result = hadamard_product(&a, &b);
        assert_eq!(result[0].to_u64(), 6);
        assert_eq!(result[1].to_u64(), 20);
        assert_eq!(result[2].to_u64(), 42);
    }

    #[test]
    fn test_check_divisible_by_x_minus_2() {
        let zero: CyclotomicPolyRing<F, 8> = CyclotomicPolyRing::zero();
        assert!(check_divisible_by_x_minus_2::<A257, F, 8>(&zero));

        let mut f = CyclotomicPolyRing::<F, 8>::zero();
        f.set_coeff(1, F::from_u64(2));
        f.set_coeff(0, -F::from_u64(4));
        assert!(check_divisible_by_x_minus_2::<A257, F, 8>(&f));

        let mut g = CyclotomicPolyRing::<F, 8>::zero();
        g.set_coeff(1, F::from_u64(1));
        g.set_coeff(0, -F::from_u64(3));
        assert!(!check_divisible_by_x_minus_2::<A257, F, 8>(&g));
    }

    #[test]
    fn test_full_arith_r1cs_reduction_smoke() {
        let k = 2;
        let n = 2;
        let l = 2;

        let a_r1cs = RingMat::new(
            k,
            n,
            vec![
                A257::from_u64(1),
                A257::from_u64(0),
                A257::from_u64(0),
                A257::from_u64(1),
            ],
        );
        let b_r1cs = RingMat::new(
            k,
            n,
            vec![
                A257::from_u64(1),
                A257::from_u64(1),
                A257::from_u64(0),
                A257::from_u64(1),
            ],
        );
        let c_r1cs = RingMat::new(
            k,
            n,
            vec![
                A257::from_u64(1),
                A257::from_u64(0),
                A257::from_u64(0),
                A257::from_u64(1),
            ],
        );

        let instance = ArithR1CSInstance {
            a_r1cs,
            b_r1cs,
            c_r1cs,
        };
        let witness = vec![A257::from_u64(1), A257::from_u64(0)];

        let kappa_a = 2;
        let kappa_b = 2;
        let total_rank_a = 3 * k + n;

        let crs_a = RingMat::new(
            kappa_a,
            total_rank_a,
            (0..kappa_a * total_rank_a)
                .map(|_| CyclotomicPolyRing::<F, 8>::zero())
                .collect(),
        );
        let crs_b = RingMat::new(
            kappa_b,
            l * k,
            (0..kappa_b * l * k)
                .map(|_| CyclotomicPolyRing::<F, 8>::zero())
                .collect(),
        );

        let mut rng = grid_std::test_rng();
        let reduction = build_arith_r1cs_reduction::<A257, F, _, 8>(
            &instance, &witness, &crs_a, &crs_b, &mut rng, l, 1.0,
        )
        .unwrap();

        assert_eq!(reduction.witness.num_parts(), 4 + l);
        assert_eq!(reduction.l, l);
        assert_eq!(reduction.statement.num_f(), kappa_a + kappa_b + l);
        assert_eq!(reduction.statement.num_f_prime(), 0);

        let rank = reduction.witness.rank();
        for (i, part) in reduction.witness.parts.iter().enumerate() {
            assert_eq!(part.len(), rank, "part {} rank mismatch", i);
        }
    }

    #[test]
    fn test_sample_arith_challenges() {
        let mut rng = grid_std::test_rng();
        let (phi, agg) = sample_arith_challenges::<A257, _>(3, 4, &mut rng);
        assert_eq!(phi.len(), 4);
        assert_eq!(agg.len(), 4);
        for v in &phi {
            assert_eq!(v.len(), 3);
            for e in v {
                assert!(e.try_to_u64().unwrap() < 257);
            }
        }
    }

    #[test]
    fn test_rq_aggregation_verify() {
        let k = 2;
        let n = 2;
        let l = 2;

        let a_r1cs = RingMat::new(
            k,
            n,
            vec![
                A257::from_u64(1),
                A257::from_u64(0),
                A257::from_u64(0),
                A257::from_u64(1),
            ],
        );
        let b_r1cs = RingMat::new(
            k,
            n,
            vec![
                A257::from_u64(1),
                A257::from_u64(1),
                A257::from_u64(0),
                A257::from_u64(1),
            ],
        );
        let c_r1cs = RingMat::new(
            k,
            n,
            vec![
                A257::from_u64(1),
                A257::from_u64(0),
                A257::from_u64(0),
                A257::from_u64(1),
            ],
        );

        let instance = ArithR1CSInstance {
            a_r1cs,
            b_r1cs,
            c_r1cs,
        };
        let witness = vec![A257::from_u64(1), A257::from_u64(0)];

        let kappa_a = 2;
        let kappa_b = 2;
        let total_rank_a = 3 * k + n;

        let crs_a = RingMat::new(
            kappa_a,
            total_rank_a,
            (0..kappa_a * total_rank_a)
                .map(|_| CyclotomicPolyRing::<F, 8>::zero())
                .collect(),
        );
        let crs_b = RingMat::new(
            kappa_b,
            l * k,
            (0..kappa_b * l * k)
                .map(|_| CyclotomicPolyRing::<F, 8>::zero())
                .collect(),
        );

        let mut rng = grid_std::test_rng();
        let reduction = build_arith_r1cs_reduction::<A257, F, _, 8>(
            &instance, &witness, &crs_a, &crs_b, &mut rng, l, 1.0,
        )
        .unwrap();

        assert!(verify_aggregation::<A257, F, 8>(&reduction));
        assert!(verify_aggregation_rq::<A257, F, 8>(&reduction));

        let f = &reduction.statement.f;
        let agg_start = f.len().saturating_sub(l);
        for (j, f_j) in f.iter().skip(agg_start).enumerate() {
            let val = f_j.evaluate(&reduction.witness);
            assert!(val.is_zero(), "F aggregation[{}] does not vanish in R_q", j);
            assert!(
                check_divisible_by_x_minus_2::<A257, F, 8>(f_j.b()),
                "F aggregation[{}] g_j(2) != 0 mod M",
                j
            );
        }
    }

    #[test]
    fn test_naf_encode_decode_roundtrip() {
        // Verify that NAF round-trips through the app ring.
        for a in 0..257u64 {
            let val = A257::from_u64(a);
            let digits = val.to_naf_digits::<16>().unwrap();
            let decoded = A257::eval_naf_digits(&digits);
            assert_eq!(decoded.to_u64(), a, "NAF round-trip failed for a={}", a);
        }
    }

    // --- FermatRing64 (multi-limb, M = 2^64 + 1) tests ---

    type P64 = PrimeField<12289>; // large enough for Theorem 6.3 with N=64
    const N64: usize = 64;

    #[test]
    fn test_fermat64_naf_roundtrip() {
        // NAF round-trip for multi-limb modulus 2^64 + 1.
        let zero = FermatRing64::zero();
        let digits = zero.to_naf_digits::<N64>().unwrap();
        let decoded = FermatRing64::eval_naf_digits(&digits);
        assert!(decoded.is_zero());

        let one = FermatRing64::one();
        let digits = one.to_naf_digits::<N64>().unwrap();
        let decoded = FermatRing64::eval_naf_digits(&digits);
        assert_eq!(decoded, one);

        // 2^64 ≡ -1 mod (2^64+1), so canonical value is [0, 1].
        let two_to_64 = FermatRing64::from_canonical(&grid_algebra::arith::bigint::BigUint::<2> {
            limbs: [0, 1],
        });
        let digits = two_to_64.to_naf_digits::<N64>().unwrap();
        let decoded = FermatRing64::eval_naf_digits(&digits);
        assert_eq!(decoded, two_to_64, "2^64 NAF round-trip mismatch");
    }

    #[test]
    fn test_fermat64_check_divisible() {
        // Zero polynomial should always pass.
        let zero = CyclotomicPolyRing::<P64, N64>::zero();
        assert!(check_divisible_by_x_minus_2::<FermatRing64, P64, N64>(
            &zero
        ));

        // g(X) = X - 2: coeff[0]=-2, coeff[1]=1.  g(2) = 2 - 2 = 0.
        let mut g = CyclotomicPolyRing::<P64, N64>::zero();
        g.set_coeff(0, -P64::from_u64(2)); // -2 mod 17 = 15
        g.set_coeff(1, P64::from_u64(1));
        assert!(
            check_divisible_by_x_minus_2::<FermatRing64, P64, N64>(&g),
            "X-2 should be divisible by X-2"
        );
    }

    #[test]
    fn test_fermat64_arith_r1cs_reduction_smoke() {
        let k = 1;
        let n = 1;
        let l = 2;

        let a_r1cs = RingMat::new(k, n, vec![FermatRing64::from_u64(1)]);
        let b_r1cs = RingMat::new(k, n, vec![FermatRing64::from_u64(1)]);
        let c_r1cs = RingMat::new(k, n, vec![FermatRing64::from_u64(1)]);

        let instance = ArithR1CSInstance {
            a_r1cs,
            b_r1cs,
            c_r1cs,
        };
        let witness = vec![FermatRing64::from_u64(1)];

        let total_rank_a = 3 * k + n;
        let crs_a = RingMat::new(
            2,
            total_rank_a,
            (0..2 * total_rank_a)
                .map(|_| CyclotomicPolyRing::<P64, N64>::zero())
                .collect(),
        );
        let crs_b = RingMat::new(
            2,
            l * k,
            (0..2 * l * k)
                .map(|_| CyclotomicPolyRing::<P64, N64>::zero())
                .collect(),
        );

        let mut rng = grid_std::test_rng();
        let reduction = build_arith_r1cs_reduction::<FermatRing64, P64, _, N64>(
            &instance, &witness, &crs_a, &crs_b, &mut rng, l, 1.0,
        )
        .unwrap();

        assert_eq!(reduction.witness.num_parts(), 4 + l);
        assert_eq!(reduction.l, l);
    }

    #[test]
    fn test_fermat64_arith_r1cs_reduction_transcript_smoke() {
        // Fake transcript for deterministic challenge testing.
        struct FakeTranscript {
            counter: usize,
        }
        impl grid_transcript::Transcript for FakeTranscript {
            fn append_preframed_bytes(
                &mut self,
                _bytes: &[u8],
            ) -> Result<(), grid_transcript::TranscriptError> {
                Ok(())
            }
            fn challenge_bytes(
                &mut self,
                _label: &'static [u8],
                out_len: usize,
            ) -> Result<alloc::vec::Vec<u8>, grid_transcript::TranscriptError> {
                self.counter += 1;
                let val = (self.counter as u64).to_le_bytes();
                let mut out = alloc::vec![0u8; out_len];
                for i in 0..out_len {
                    out[i] = val[i % val.len()];
                }
                Ok(out)
            }
        }

        let k = 1;
        let n = 1;
        let l = 2;

        let a_r1cs = RingMat::new(k, n, vec![FermatRing64::from_u64(1)]);
        let b_r1cs = RingMat::new(k, n, vec![FermatRing64::from_u64(1)]);
        let c_r1cs = RingMat::new(k, n, vec![FermatRing64::from_u64(1)]);

        let instance = ArithR1CSInstance {
            a_r1cs,
            b_r1cs,
            c_r1cs,
        };
        let witness = vec![FermatRing64::from_u64(1)];

        let total_rank_a = 3 * k + n;
        let crs_a = RingMat::new(
            2,
            total_rank_a,
            (0..2 * total_rank_a)
                .map(|_| CyclotomicPolyRing::<P64, N64>::zero())
                .collect(),
        );
        let crs_b = RingMat::new(
            2,
            l * k,
            (0..2 * l * k)
                .map(|_| CyclotomicPolyRing::<P64, N64>::zero())
                .collect(),
        );

        let mut transcript = FakeTranscript { counter: 0 };
        let reduction = build_arith_r1cs_reduction_transcript::<FermatRing64, P64, _, N64>(
            &instance,
            &witness,
            &crs_a,
            &crs_b,
            &mut transcript,
            l,
            1.0,
        )
        .unwrap();

        assert_eq!(reduction.l, l);
    }

    #[test]
    fn test_modulus_shape_rejected() {
        // Zm<251> with N=8: 251 ≠ 2^8+1 = 257 → Phase 0 should reject.
        type A251 = Zm<251>;
        let instance = ArithR1CSInstance {
            a_r1cs: RingMat::new(1, 1, vec![A251::from_u64(1)]),
            b_r1cs: RingMat::new(1, 1, vec![A251::from_u64(1)]),
            c_r1cs: RingMat::new(1, 1, vec![A251::from_u64(1)]),
        };
        let crs_a = RingMat::new(1, 4, vec![CyclotomicPolyRing::<F, 8>::zero(); 4]);
        let crs_b = RingMat::new(1, 1, vec![CyclotomicPolyRing::<F, 8>::zero()]);
        let mut rng = grid_std::test_rng();
        let result = build_arith_r1cs_reduction::<A251, F, _, 8>(
            &instance,
            &[A251::from_u64(1)],
            &crs_a,
            &crs_b,
            &mut rng,
            1,
            1.0,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_asymmetric_k_gt_n_padding() {
        // k=3, n=1: a,b,c,d_i have 3 entries, w has 1. Padded to max_rank=3.
        let instance = ArithR1CSInstance {
            a_r1cs: RingMat::new(
                3,
                1,
                vec![A257::from_u64(1), A257::from_u64(0), A257::from_u64(0)],
            ),
            b_r1cs: RingMat::new(
                3,
                1,
                vec![A257::from_u64(1), A257::from_u64(0), A257::from_u64(0)],
            ),
            c_r1cs: RingMat::new(
                3,
                1,
                vec![A257::from_u64(1), A257::from_u64(0), A257::from_u64(0)],
            ),
        };
        let crs_a = RingMat::new(
            1,
            10, // 3k + n = 10
            vec![CyclotomicPolyRing::<F, 8>::zero(); 10],
        );
        let crs_b = RingMat::new(1, 3, vec![CyclotomicPolyRing::<F, 8>::zero(); 3]);
        let mut rng = grid_std::test_rng();
        let reduction = build_arith_r1cs_reduction::<A257, F, _, 8>(
            &instance,
            &[A257::from_u64(1)],
            &crs_a,
            &crs_b,
            &mut rng,
            1,
            1.0,
        )
        .unwrap();
        // Verify aggregation passes with asymmetric dimensions.
        assert!(verify_aggregation_rq::<A257, F, 8>(&reduction));
    }

    #[test]
    fn test_theorem_6_3_rejects_invalid_beta() {
        let result = validate_theorem_6_3_bounds::<F>(1, 1, 1, 8, -1.0);
        assert!(result.is_err());
        let result = validate_theorem_6_3_bounds::<F>(1, 1, 1, 8, 0.0);
        assert!(result.is_err());
        let result = validate_theorem_6_3_bounds::<F>(1, 1, 1, 8, f64::NAN);
        assert!(result.is_err());
    }

    #[test]
    fn test_theorem_6_3_rejects_l_zero() {
        let result = validate_theorem_6_3_bounds::<F>(1, 1, 0, 8, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_naf_max_residue_roundtrip() {
        // Encode modulus-1 (largest canonical residue = 2^N) and verify roundtrip.
        let max_val = A257::from_u64(256); // 257-1 = 256 = 2^8
        let digits = max_val.to_naf_digits::<8>().unwrap();
        let decoded = A257::eval_naf_digits(&digits);
        assert_eq!(decoded, max_val);
    }
}
