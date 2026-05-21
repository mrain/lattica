//! R1CS mod 2^d+1 → R reduction (§6, Figure 5, Theorem 6.3).
//!
//! Reduces an R1CS instance over `Z_{2^d+1}` to a LaBRADOR principal relation R
//! instance. Uses NAF encoding to embed field elements as polynomials with
//! small coefficients, and the morphism `φ: X → 2` to verify arithmetic.
//!
//! Soundness error: `2 * p^(-l)` where `p` is the smallest prime factor of
//! `2^d + 1`, and `l` is the number of aggregation rounds.
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
/// The R1CS is defined over a large field. The morphism `φ: X → 2` maps
/// polynomials to field elements via evaluation. Soundness depends on the
/// smallest prime factor p of M (not M itself), per Theorem 6.3.
#[derive(Debug, Clone)]
pub struct ArithR1CSInstance<R: IntegerRing<Canonical = u64>> {
    pub a_r1cs: RingMat<R>,
    pub b_r1cs: RingMat<R>,
    pub c_r1cs: RingMat<R>,
    /// Modulus M = 2^d + 1 (NOT the soundness prime p).
    pub modulus_m: u64,
}

/// Result of the arithmetic R1CS → R reduction.
#[derive(Debug, Clone)]
pub struct ArithR1CSReduction<R, const N: usize>
where
    R: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    /// LaBRADOR statement (F: Ajtai opening + aggregation constraints)
    pub statement: LabradorStatement<CyclotomicPolyRing<R, N>>,
    /// LaBRADOR witness: (a, b, c, w, d_1, ..., d_l)
    pub witness: LabradorWitness<CyclotomicPolyRing<R, N>>,
    /// Ajtai commitment `t = A * (a||b||c||w)`
    pub commitment_t: RingVec<CyclotomicPolyRing<R, N>>,
    /// Ajtai commitment `t_d = B * (d_1||...||d_l)`
    pub commitment_td: RingVec<CyclotomicPolyRing<R, N>>,
    /// R1CS matrix row count (constraint count)
    pub k: usize,
    /// R1CS matrix column count (witness dimension)
    pub n: usize,
    /// Number of aggregation rounds (soundness: 2*p^(-l) where p = smallest_prime_factor(modulus_m))
    pub l: usize,
    /// Aggregation challenges α_j per round (each length k)
    pub alpha_challenges: Vec<Vec<R>>,
    /// Aggregation challenges β_j per round (each length k)
    pub beta_challenges: Vec<Vec<R>>,
    /// Aggregation challenges γ_j per round (each length k)
    pub gamma_challenges: Vec<Vec<R>>,
    /// Aggregation challenges δ_j per round (each length l*k)
    pub delta_challenges: Vec<Vec<R>>,
    /// Aggregation polynomials g_j = f̃_j(witness) over R_q (one per round)
    pub g_polys: Vec<CyclotomicPolyRing<R, N>>,
    /// Hadamard challenges φ_i per round (each length k)
    pub phi_challenges: Vec<Vec<R>>,
    /// Modulus M = 2^d + 1 (NOT the soundness prime). Used for g_j(2) ≡ 0 mod M check.
    pub m: u64,
}

/// Encode a single field element `a ∈ Z_M` as a polynomial in `R_q` using NAF.
fn encode_naf<R, const N: usize>(a: u64, m: u64) -> CyclotomicPolyRing<R, N>
where
    R: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    // Reduce into [0, M) then center into [-(M-1)/2, (M-1)/2].
    // For M = 2^N + 1 this guarantees the signed value has magnitude
    // at most (2^N)/2 = 2^(N-1), so NAF fits in N coefficients.
    // Use i128 to avoid overflow for large M (e.g., N >= 63).
    let half_m = (m - 1) / 2;
    let raw = a % m;
    let centered: i128 = if raw > half_m {
        (raw as i128) - (m as i128)
    } else {
        raw as i128
    };

    let mut coeffs = core::array::from_fn(|_| R::zero());
    let mut val = centered;

    let mut i: usize = 0;
    while val != 0 && i < N {
        if val & 1 != 0 {
            // val is odd: subtract +/-1 to make it even
            let rem = val.rem_euclid(4);
            if rem == 1 {
                coeffs[i] = R::from_u64(1);
                val -= 1;
            } else {
                coeffs[i] = -R::from_u64(1);
                val += 1;
            }
        }
        val /= 2;
        i += 1;
    }

    CyclotomicPolyRing::from_array(coeffs)
}

/// Verify all coefficients of a NAF polynomial are in {-1, 0, 1}.
/// Returns the squared L2 norm (number of nonzero coefficients for valid NAF).
/// Returns `None` if any coefficient is outside {-1, 0, 1}.
pub fn verify_naf_coeffs<R, const N: usize>(poly: &CyclotomicPolyRing<R, N>) -> Option<usize>
where
    R: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    let q = R::modulus();
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

/// Verify that all NAF-encoded witness parts in the arithmetic reduction
/// have coefficients in {-1, 0, 1}.
///
/// Checks all parts (a, b, c, w, d_i) since all are NAF-encoded.
/// Returns the (part_idx, poly_idx) of the first invalid polynomial,
/// or `Ok(())` if all valid.
pub fn verify_naf_witness<R, const N: usize>(
    reduction: &ArithR1CSReduction<R, N>,
) -> Result<(), (usize, usize)>
where
    R: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    let witness = &reduction.witness;
    // All parts in the arithmetic reduction are NAF-encoded:
    // parts 0..3: a, b, c, w
    // parts 4..: d_i for each round
    for (part_idx, part) in witness.parts.iter().enumerate() {
        for (poly_idx, poly) in part.iter().enumerate() {
            if verify_naf_coeffs(poly).is_none() {
                return Err((part_idx, poly_idx));
            }
        }
    }
    Ok(())
}

/// Decode a NAF polynomial back to field element: evaluate at X=2, then mod M.
/// Generic version without CanonicalSerialize/Deserialize bounds (for builder path).
fn decode_naf_field<R, const N: usize>(poly: &CyclotomicPolyRing<R, N>, p: u64) -> u64
where
    R: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    let coeffs = poly.coeffs();
    let q: u64 = R::modulus();

    let mut result: i128 = 0;
    let mut power_of_2: i128 = 1;
    for coeff in coeffs.iter().take(N) {
        let raw = coeff.to_u64();
        let centered = if raw > q / 2 {
            raw as i128 - q as i128
        } else {
            raw as i128
        };
        result += centered * power_of_2;
        power_of_2 *= 2;
    }

    let p_i128 = p as i128;
    let mut r = result % p_i128;
    if r < 0 {
        r += p_i128;
    }
    r as u64
}

/// Decode a NAF polynomial back to field element: evaluate at X=2, then mod M.
/// Version requiring CanonicalSerialize/Deserialize (used by recompute_and_verify).
fn decode_naf<R, const N: usize>(poly: &CyclotomicPolyRing<R, N>, p: u64) -> u64
where
    R: IntegerRing<Canonical = u64>
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize,
{
    decode_naf_field::<R, N>(poly, p)
}

/// Compute Hadamard product of two vectors of field elements.
fn hadamard_product<R: IntegerRing<Canonical = u64>>(a: &[R], b: &[R], m: u64) -> Vec<R> {
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| {
            let prod = ai.to_u64() as i128 * bi.to_u64() as i128;
            let mut r = prod % m as i128;
            if r < 0 {
                r += m as i128;
            }
            R::from_u64(r as u64)
        })
        .collect()
}

/// Check if a polynomial is divisible by (X - 2).
///
/// Evaluates the polynomial at X=2 using centered coefficients, then checks
/// `f(2) ≡ 0 (mod M)`.
///
/// For the arithmetic R1CS reduction, `g_j = f̃_j(witness)` is the full R_q polynomial.
/// The verifier checks `g_j(2) ≡ 0 (mod M)` where `M = 2^d + 1`. For an honest
/// prover, `f̃_j` vanishes on the witness so `g_j(2) = 0 (mod M)`.
///
/// Soundness: a cheating prover has probability at most `N/p` of making a non-zero
/// polynomial of degree ≤ N pass the check.
pub fn check_divisible_by_x_minus_2<R, const N: usize>(
    poly: &CyclotomicPolyRing<R, N>,
    m: u64,
) -> bool
where
    R: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    let coeffs = poly.coeffs();
    let q: u64 = R::modulus();
    let mut eval: i128 = 0;
    let mut power: i128 = 1;
    for c in coeffs {
        let raw = c.to_u64();
        // Center coefficient: represent in [-q/2, q/2)
        let centered = if raw > q / 2 {
            raw as i128 - q as i128
        } else {
            raw as i128
        };
        eval += centered * power;
        power *= 2;
    }
    // Check eval ≡ 0 (mod M)
    let r = eval % m as i128;
    r == 0
}

/// Compute matrix-vector product mod M.
fn mat_vec_mod_m<R: IntegerRing<Canonical = u64>>(mat: &RingMat<R>, w: &[R], m: u64) -> Vec<R> {
    let m_i128 = m as i128;
    let mut result = Vec::with_capacity(mat.rows());
    for i in 0..mat.rows() {
        let mut sum: i128 = 0;
        for (j, wi) in w.iter().enumerate().take(mat.cols()) {
            sum = (sum + mat.get(i, j).to_u64() as i128 * wi.to_u64() as i128) % m_i128;
        }
        let mut r = sum;
        if r < 0 {
            r += m_i128;
        }
        result.push(R::from_u64(r as u64));
    }
    result
}

/// Build the F (full-vanishing) quadratic function for arithmetic aggregation.
///
/// Paper-aligned construction (§6, Figure 5):
/// 1. Build f̃_j(s) in R_q by replacing all scalar coefficients with NAF encodings.
/// 2. Compute g_j = f̃_j(witness) as the full R_q polynomial (not Enc(field_scalar)).
/// 3. Set F_j(s) = f̃_j(s) - g_j, which vanishes in R_q by definition.
/// 4. Verifier separately checks g_j(2) = 0 mod M (M = 2^d + 1).
///
/// Returns the F function and the computed g_j polynomial.
#[allow(clippy::needless_range_loop)]
fn build_f_aggregation_rq<R, const N: usize>(
    alpha_j: &[R],
    beta_j: &[R],
    gamma_j: &[R],
    delta_j: &[R],
    phi_challenges: &[Vec<R>],
    j: usize,
    l: usize,
    k: usize,
    n: usize,
    m: u64,
    max_rank: usize,
    num_parts: usize,
    a_r1cs: &RingMat<R>,
    b_r1cs: &RingMat<R>,
    c_r1cs: &RingMat<R>,
    witness_parts: &[Vec<CyclotomicPolyRing<R, N>>],
) -> (
    QuadraticFunction<CyclotomicPolyRing<R, N>>,
    CyclotomicPolyRing<R, N>,
)
where
    R: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    let zero = CyclotomicPolyRing::<R, N>::zero();
    let one = CyclotomicPolyRing::<R, N>::one();

    // Witness part indices
    let idx_a: usize = 0;
    let idx_b: usize = 1;
    let idx_c: usize = 2;
    let idx_w: usize = 3;
    // d_i parts start at index 4

    // Helper: NAF-encode a scalar as a polynomial in R_q
    let enc = |v: u64| -> CyclotomicPolyRing<R, N> { encode_naf::<R, N>(v, m) };

    // --- Linear terms (phi vectors) ---
    let mut phi: Vec<Vec<CyclotomicPolyRing<R, N>>> = vec![vec![zero.clone(); max_rank]; num_parts];

    // Term 1: ⟨α_j, A·w - a⟩
    for idx in 0..k {
        phi[idx_a][idx] = -enc(alpha_j[idx].to_u64());
    }
    for col in 0..n {
        let mut sum: i128 = 0;
        for i in 0..k {
            sum += alpha_j[i].to_u64() as i128 * a_r1cs.get(i, col).to_u64() as i128;
        }
        let mut r = sum % m as i128;
        if r < 0 {
            r += m as i128;
        }
        phi[idx_w][col] = enc(r as u64);
    }

    // Term 2: ⟨β_j, B·w - b⟩
    for idx in 0..k {
        phi[idx_b][idx] = -enc(beta_j[idx].to_u64());
    }
    for col in 0..n {
        let mut sum: i128 = 0;
        for i in 0..k {
            sum += beta_j[i].to_u64() as i128 * b_r1cs.get(i, col).to_u64() as i128;
        }
        let mut r = sum % m as i128;
        if r < 0 {
            r += m as i128;
        }
        phi[idx_w][col] += enc(r as u64);
    }

    // Term 3: ⟨γ_j, C·w - c⟩
    for idx in 0..k {
        phi[idx_c][idx] = -enc(gamma_j[idx].to_u64());
    }
    for col in 0..n {
        let mut sum: i128 = 0;
        for i in 0..k {
            sum += gamma_j[i].to_u64() as i128 * c_r1cs.get(i, col).to_u64() as i128;
        }
        let mut r = sum % m as i128;
        if r < 0 {
            r += m as i128;
        }
        phi[idx_w][col] += enc(r as u64);
    }

    // Term 4 (product check): ⟨d_j, b⟩ - ⟨φ_j, c⟩
    // ⟨d_j, b⟩: quadratic term (added below)
    // ⟨φ_j, c⟩: linear in c
    for idx in 0..k {
        phi[idx_c][idx] -= enc(phi_challenges[j][idx].to_u64());
    }

    // Term 5 (Hadamard): Σ_i ⟨δ_i^(j), φ_i ∘ a - d_i⟩
    for col in 0..k {
        let mut sum: i128 = 0;
        for i in 0..l {
            sum += delta_j[i * k + col].to_u64() as i128 * phi_challenges[i][col].to_u64() as i128;
        }
        let mut r = sum % m as i128;
        if r < 0 {
            r += m as i128;
        }
        phi[idx_a][col] += enc(r as u64);
    }
    for i in 0..l {
        let d_idx = 4 + i;
        for col in 0..k {
            phi[d_idx][col] = -enc(delta_j[i * k + col].to_u64());
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

    // Build F function with b = 0 first, then evaluate to get g_j
    let f_j = QuadraticFunction::from_parts(quad, phi, zero.clone());

    // Evaluate f̃_j on witness to get g_j as full R_q polynomial
    let temp_witness = LabradorWitness::new(witness_parts.to_vec());
    let g_j = f_j.evaluate(&temp_witness);

    // F constraint: f̃_j(w) - g_j = 0, so b = g_j
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

    (f_constraint, g_j)
}

/// Build the arithmetic R1CS → R reduction.
pub fn build_arith_r1cs_reduction<R, Rng, const N: usize>(
    instance: &ArithR1CSInstance<R>,
    witness: &[R],
    crs_a: &RingMat<CyclotomicPolyRing<R, N>>,
    crs_b: &RingMat<CyclotomicPolyRing<R, N>>,
    rng: &mut Rng,
    l: usize,
) -> Result<ArithR1CSReduction<R, N>, LabradorError>
where
    R: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
    Rng: RngExt,
{
    let k = instance.a_r1cs.rows();
    let n = instance.a_r1cs.cols();
    let m = instance.modulus_m;

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

    // Validate modulus M = 2^N + 1 (morphism X→2 into Z_M).
    // N must be < 63 to keep signed/NAF arithmetic within supported range.
    // CyclotomicPolyRing<N> is only defined for N = power of 2 up to 32768,
    // but practical N is <= 62 for i64-based signed coefficient handling.
    bail!(
        N >= 63,
        "ArithR1CS ring degree N={} must be < 63 (exceeds supported range for signed/NAF arithmetic)",
        N
    );
    let two_pow_n: u64 = 1 << N;
    bail!(
        m != two_pow_n.wrapping_add(1),
        "ArithR1CS modulus_m={} must equal 2^N+1={} for ring degree N={}",
        m,
        two_pow_n.wrapping_add(1),
        N
    );

    // All witness parts share the same rank
    let max_rank = k.max(n);
    let zero = CyclotomicPolyRing::<R, N>::zero();

    // Helper: encode field values, then pad to max_rank
    let encode_pad = |vals: &[R]| -> Vec<CyclotomicPolyRing<R, N>> {
        let polys: Vec<CyclotomicPolyRing<R, N>> = vals
            .iter()
            .map(|v| encode_naf::<R, N>(v.to_u64(), m))
            .collect();
        let mut padded = polys;
        while padded.len() < max_rank {
            padded.push(zero.clone());
        }
        padded
    };

    // Step 1: Compute a = A*w mod m, b = B*w mod m, c = C*w mod m
    let a_field = mat_vec_mod_m(&instance.a_r1cs, witness, m);
    let b_field = mat_vec_mod_m(&instance.b_r1cs, witness, m);
    let c_field = mat_vec_mod_m(&instance.c_r1cs, witness, m);

    // Step 2: Encode as NAF polynomials, padded to max_rank
    let a_polys = encode_pad(&a_field);
    let b_polys = encode_pad(&b_field);
    let c_polys = encode_pad(&c_field);
    let w_polys = encode_pad(witness);

    // Part indices
    const A: usize = 0;
    const B_IDX: usize = 1;
    const C: usize = 2;
    const W: usize = 3;
    // d_i parts start at index 4

    // Step 3: Ajtai commitment t = A * (a[k]||b[k]||c[k]||w[n])
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

    // Step 4: Verifier sends l challenge vectors φ_i ∈ Z_M^k
    let phi_challenges: Vec<Vec<R>> = (0..l)
        .map(|_| {
            (0..k)
                .map(|_| R::from_u64(rng.random_range(0..m)))
                .collect()
        })
        .collect();

    // Step 5: Prover computes d_i = φ_i ∘ a (Hadamard product over Z_M)
    let mut d_polys_list: Vec<Vec<CyclotomicPolyRing<R, N>>> = Vec::with_capacity(l);
    let mut d_concat: Vec<CyclotomicPolyRing<R, N>> = Vec::with_capacity(l * k);
    for phi_i in &phi_challenges {
        let d_i_field = hadamard_product(phi_i, &a_field, m);
        let mut d_i_polys: Vec<CyclotomicPolyRing<R, N>> = d_i_field
            .iter()
            .map(|v| encode_naf::<R, N>(v.to_u64(), m))
            .collect();
        while d_i_polys.len() < max_rank {
            d_i_polys.push(zero.clone());
        }
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

    // Step 6.5: Compute aggregation polynomials g_j and build F (full-vanishing) constraints.
    //
    // Paper-aligned construction (§6, Figure 5):
    // 1. Build f̃_j(s) in R_q by replacing scalar coefficients with NAF encodings.
    // 2. Compute g_j = f̃_j(witness) as the full R_q polynomial.
    // 3. F_j(s) = f̃_j(s) - g_j vanishes in R_q by definition.
    // 4. Verifier separately checks g_j(2) = 0 mod M (M = 2^d + 1).
    //
    // This replaces the field-level NAF(g_j) approach. g_j is now the actual R_q
    // polynomial output, and the F constraint ties it to the witness.
    let num_parts = 4 + l;

    let mut f_aggregation: Vec<QuadraticFunction<CyclotomicPolyRing<R, N>>> = Vec::with_capacity(l);
    let mut alpha_challenges: Vec<Vec<R>> = Vec::with_capacity(l);
    let mut beta_challenges: Vec<Vec<R>> = Vec::with_capacity(l);
    let mut gamma_challenges: Vec<Vec<R>> = Vec::with_capacity(l);
    let mut delta_challenges: Vec<Vec<R>> = Vec::with_capacity(l);
    let mut g_polys: Vec<CyclotomicPolyRing<R, N>> = Vec::with_capacity(l);

    // Temporary witness parts for evaluation (before ownership transfer)
    let temp_witness_parts: Vec<Vec<CyclotomicPolyRing<R, N>>> = {
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
        let alpha_j: Vec<R> = (0..k)
            .map(|_| R::from_u64(rng.random_range(0..m)))
            .collect();
        let beta_j: Vec<R> = (0..k)
            .map(|_| R::from_u64(rng.random_range(0..m)))
            .collect();
        let gamma_j: Vec<R> = (0..k)
            .map(|_| R::from_u64(rng.random_range(0..m)))
            .collect();
        let delta_j: Vec<R> = (0..l * k)
            .map(|_| R::from_u64(rng.random_range(0..m)))
            .collect();

        alpha_challenges.push(alpha_j.clone());
        beta_challenges.push(beta_j.clone());
        gamma_challenges.push(gamma_j.clone());
        delta_challenges.push(delta_j.clone());

        // Build F constraint and compute g_j as full R_q polynomial
        let (f_j, g_j) = build_f_aggregation_rq::<R, N>(
            &alpha_j,
            &beta_j,
            &gamma_j,
            &delta_j,
            &phi_challenges,
            j,
            l,
            k,
            n,
            m,
            max_rank,
            num_parts,
            &instance.a_r1cs,
            &instance.b_r1cs,
            &instance.c_r1cs,
            &temp_witness_parts,
        );

        // Verify g_j(2) = 0 mod m (honest prover check — debug only, not protocol-critical)
        debug_assert!(
            check_divisible_by_x_minus_2::<R, N>(&g_j, m),
            "Aggregation polynomial g_{} not divisible by (X-2)",
            j
        );

        g_polys.push(g_j);
        f_aggregation.push(f_j);
    }

    // Step 7: Build LaBRADOR witness: (a, b, c, w, d_1, ..., d_l)
    let mut witness_parts: Vec<Vec<CyclotomicPolyRing<R, N>>> =
        vec![a_polys, b_polys, c_polys, w_polys];
    for d_i in &d_polys_list {
        witness_parts.push(d_i.clone());
    }
    let labrador_witness = LabradorWitness::new(witness_parts);

    // Step 8: F family — Ajtai opening + arithmetic aggregation constraints
    // For t: kappa_a rows, each gives <row_j, [a||b||c||w]> = t_j
    // For t_d: kappa_b rows, each gives <row_j, [d_1||...||d_l]> = td_j
    // For aggregation: l functions f̃_j(w) - g_j = 0 (paper §6, Figure 5)
    let kappa_a = crs_a.rows();
    let kappa_b = crs_b.rows();
    let mut f_functions: Vec<QuadraticFunction<CyclotomicPolyRing<R, N>>> =
        Vec::with_capacity(kappa_a + kappa_b + l);

    // t opening
    for j in 0..kappa_a {
        let row = crs_a.row(j);
        let row_entries = row.entries();

        let mut phi: Vec<Vec<CyclotomicPolyRing<R, N>>> = Vec::with_capacity(num_parts);

        // Helper inline: slice row segment, pad to max_rank
        let push_padded = |start: usize, len: usize| {
            let mut v: Vec<CyclotomicPolyRing<R, N>> = row_entries[start..start + len].to_vec();
            while v.len() < max_rank {
                v.push(zero.clone());
            }
            v
        };

        phi.push(push_padded(0, k)); // a
        phi.push(push_padded(k, k)); // b
        phi.push(push_padded(2 * k, k)); // c
        phi.push(push_padded(3 * k, n)); // w
        // d parts not in t commitment
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

        let mut phi: Vec<Vec<CyclotomicPolyRing<R, N>>> = Vec::with_capacity(num_parts);
        // Base parts not in t_d commitment
        for _ in 0..4 {
            phi.push(vec![zero.clone(); max_rank]);
        }
        // d parts: slice from row, pad to max_rank
        for _di in 0..l {
            let mut seg: Vec<CyclotomicPolyRing<R, N>> = row_entries[offset..offset + k].to_vec();
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

    // Arithmetic aggregation: f̃_j(w) - g_j = 0 for each round j
    for f_j in f_aggregation {
        f_functions.push(f_j);
    }

    // F' is empty for arithmetic R1CS — aggregation goes into F, not F'
    let f_prime_functions: Vec<QuadraticFunction<CyclotomicPolyRing<R, N>>> = Vec::new();

    let statement = LabradorStatement {
        f: f_functions,
        f_prime: f_prime_functions,
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
        m,
    })
}

/// Build the arithmetic R1CS → R reduction using transcript for Fiat-Shamir.
///
/// Same as [`build_arith_r1cs_reduction`] but challenges are derived from the
/// transcript rather than a local RNG. The prover appends commitments to the
/// transcript before sampling challenges, binding them to the Fiat-Shamir
/// transcript.
pub fn build_arith_r1cs_reduction_transcript<R, T, const N: usize>(
    instance: &ArithR1CSInstance<R>,
    witness: &[R],
    crs_a: &RingMat<CyclotomicPolyRing<R, N>>,
    crs_b: &RingMat<CyclotomicPolyRing<R, N>>,
    transcript: &mut T,
    l: usize,
) -> Result<ArithR1CSReduction<R, N>, TranscriptError>
where
    R: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
    T: grid_transcript::Transcript,
{
    let k = instance.a_r1cs.rows();
    let n = instance.a_r1cs.cols();
    let m = instance.modulus_m;

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
    bail_t!(N >= 63, "ArithR1CS ring degree N={} must be < 63", N);
    let two_pow_n: u64 = 1 << N;
    bail_t!(
        m != two_pow_n.wrapping_add(1),
        "ArithR1CS modulus_m={} must equal 2^N+1={} for ring degree N={}",
        m,
        two_pow_n.wrapping_add(1),
        N
    );

    let max_rank = k.max(n);
    let zero = CyclotomicPolyRing::<R, N>::zero();

    let encode_pad = |vals: &[R]| -> Vec<CyclotomicPolyRing<R, N>> {
        let polys: Vec<CyclotomicPolyRing<R, N>> = vals
            .iter()
            .map(|v| encode_naf::<R, N>(v.to_u64(), m))
            .collect();
        let mut padded = polys;
        while padded.len() < max_rank {
            padded.push(zero.clone());
        }
        padded
    };

    let a_field = mat_vec_mod_m(&instance.a_r1cs, witness, m);
    let b_field = mat_vec_mod_m(&instance.b_r1cs, witness, m);
    let c_field = mat_vec_mod_m(&instance.c_r1cs, witness, m);

    let a_polys = encode_pad(&a_field);
    let b_polys = encode_pad(&b_field);
    let c_polys = encode_pad(&c_field);
    let w_polys = encode_pad(witness);

    const A: usize = 0;
    const B_IDX: usize = 1;
    const C: usize = 2;
    const W: usize = 3;

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

    // Append commitment to transcript before sampling phi challenges
    transcript.append_serializable(b"labrador_arith_t", &commitment_t)?;

    // Sample phi challenges from transcript (uniformly below m)
    let phi_challenges: Vec<Vec<R>> = (0..l)
        .map(|i| {
            let round_bytes = (i as u32).to_le_bytes();
            transcript.append_bytes(b"labrador_arith_phi", &round_bytes)?;
            (0..k)
                .map(|_| challenge_mod_m::<R, _>(transcript, b"labrador_arith_phi_c", m))
                .collect::<Result<_, _>>()
        })
        .collect::<Result<_, _>>()?;

    let mut d_polys_list: Vec<Vec<CyclotomicPolyRing<R, N>>> = Vec::with_capacity(l);
    let mut d_concat: Vec<CyclotomicPolyRing<R, N>> = Vec::with_capacity(l * k);
    for phi_i in &phi_challenges {
        let d_i_field = hadamard_product(phi_i, &a_field, m);
        let mut d_i_polys: Vec<CyclotomicPolyRing<R, N>> = d_i_field
            .iter()
            .map(|v| encode_naf::<R, N>(v.to_u64(), m))
            .collect();
        while d_i_polys.len() < max_rank {
            d_i_polys.push(zero.clone());
        }
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
    let mut f_aggregation: Vec<QuadraticFunction<CyclotomicPolyRing<R, N>>> = Vec::with_capacity(l);
    let mut alpha_challenges: Vec<Vec<R>> = Vec::with_capacity(l);
    let mut beta_challenges: Vec<Vec<R>> = Vec::with_capacity(l);
    let mut gamma_challenges: Vec<Vec<R>> = Vec::with_capacity(l);
    let mut delta_challenges: Vec<Vec<R>> = Vec::with_capacity(l);
    let mut g_polys: Vec<CyclotomicPolyRing<R, N>> = Vec::with_capacity(l);

    let temp_witness_parts: Vec<Vec<CyclotomicPolyRing<R, N>>> = {
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
        // Sample aggregation challenges from transcript (uniformly below m)
        let round_bytes = (j as u32).to_le_bytes();
        transcript.append_bytes(b"labrador_arith_agg", &round_bytes)?;
        let alpha_j: Vec<R> = (0..k)
            .map(|_| challenge_mod_m::<R, _>(transcript, b"labrador_arith_agg_alpha", m))
            .collect::<Result<_, _>>()?;
        let beta_j: Vec<R> = (0..k)
            .map(|_| challenge_mod_m::<R, _>(transcript, b"labrador_arith_agg_beta", m))
            .collect::<Result<_, _>>()?;
        let gamma_j: Vec<R> = (0..k)
            .map(|_| challenge_mod_m::<R, _>(transcript, b"labrador_arith_agg_gamma", m))
            .collect::<Result<_, _>>()?;
        let delta_j: Vec<R> = (0..(l * k))
            .map(|_| challenge_mod_m::<R, _>(transcript, b"labrador_arith_agg_delta", m))
            .collect::<Result<_, _>>()?;

        alpha_challenges.push(alpha_j.clone());
        beta_challenges.push(beta_j.clone());
        gamma_challenges.push(gamma_j.clone());
        delta_challenges.push(delta_j.clone());

        let (f_j, g_j) = build_f_aggregation_rq::<R, N>(
            &alpha_j,
            &beta_j,
            &gamma_j,
            &delta_j,
            &phi_challenges,
            j,
            l,
            k,
            n,
            m,
            max_rank,
            num_parts,
            &instance.a_r1cs,
            &instance.b_r1cs,
            &instance.c_r1cs,
            &temp_witness_parts,
        );

        f_aggregation.push(f_j);
        g_polys.push(g_j.clone());
        transcript.append_serializable(b"labrador_arith_g", &g_j)?;
    }

    let f_functions: Vec<QuadraticFunction<CyclotomicPolyRing<R, N>>> = {
        let kappa_a = crs_a.rows();
        let kappa_b = crs_b.rows();
        let mut f_funcs = Vec::with_capacity(kappa_a + kappa_b + l);

        for j in 0..kappa_a {
            let row = crs_a.row(j);
            let row_entries = row.entries();
            let mut phi: Vec<Vec<CyclotomicPolyRing<R, N>>> = Vec::with_capacity(num_parts);
            let push_padded = |start: usize, len: usize| {
                let mut v: Vec<CyclotomicPolyRing<R, N>> = row_entries[start..start + len].to_vec();
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
            let mut phi: Vec<Vec<CyclotomicPolyRing<R, N>>> = Vec::with_capacity(num_parts);
            for _ in 0..4 {
                phi.push(vec![zero.clone(); max_rank]);
            }
            let mut offset = 0usize;
            for _di in 0..l {
                let mut seg: Vec<CyclotomicPolyRing<R, N>> =
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

    let f_prime_functions: Vec<QuadraticFunction<CyclotomicPolyRing<R, N>>> = Vec::new();

    let statement = LabradorStatement {
        f: f_functions,
        f_prime: f_prime_functions,
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
        m,
    })
}

/// Verify that all aggregation polynomials g_j are divisible by (X-2).
///
/// g_j = f̃_j(witness) evaluated at X=2 equals f̃_j(field_values), which is 0
/// for an honest prover. A cheating prover has probability at most 2/p per
/// round of passing this check.
pub fn verify_aggregation<R, const N: usize>(reduction: &ArithR1CSReduction<R, N>) -> bool
where
    R: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    reduction
        .g_polys
        .iter()
        .all(|g_j| check_divisible_by_x_minus_2::<R, N>(g_j, reduction.m))
}

/// Verify arithmetic aggregation constraints (paper-aligned, §6 Figure 5).
///
/// Three checks:
/// 1. NAF coefficients valid: all NAF-encoded witness parts have coefficients in {-1, 0, 1}
/// 2. F constraints vanish: f̃_j(w) - g_j = 0 in R_q (verified by relation::verify)
/// 3. g_j(2) = 0 mod M externally (M = 2^d + 1)
///
/// The aggregation functions are the last `l` functions in statement.f
/// (after Ajtai opening constraints).
pub fn verify_aggregation_rq<R, const N: usize>(reduction: &ArithR1CSReduction<R, N>) -> bool
where
    R: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    // 1. Verify NAF coefficients are in {-1, 0, 1}
    if verify_naf_witness(reduction).is_err() {
        return false;
    }

    let m = reduction.m;
    let f = &reduction.statement.f;
    let witness = &reduction.witness;
    let l = reduction.l;

    // The last l F functions are the aggregation constraints
    let agg_start = f.len().saturating_sub(l);

    for f_j in f.iter().skip(agg_start) {
        // 2. F constraint: f̃_j(w) - g_j = 0
        // Evaluate: should be zero in R_q
        let val = f_j.evaluate(witness);
        if !val.is_zero() {
            return false;
        }
        // 3. External check: g_j(2) = 0 mod M
        // g_j is stored as the b term of the F function
        if !check_divisible_by_x_minus_2::<R, N>(f_j.b(), m) {
            return false;
        }
    }
    true
}

/// Paper-aligned verifier helper for arithmetic R1CS aggregation.
///
/// For each round j, rebuilds the F constraint f̃_j(s) in R_q from stored
/// challenges, evaluates on the witness to get g_j_recomputed, then checks:
/// 1. g_j_recomputed == stored g_polys[j]
/// 2. g_j_recomputed(2) = 0 mod M (divisibility by X-2)
pub fn recompute_and_verify<R, const N: usize>(
    instance: &ArithR1CSInstance<R>,
    reduction: &ArithR1CSReduction<R, N>,
) -> bool
where
    R: IntegerRing<Canonical = u64>
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize,
{
    let k = instance.a_r1cs.rows();
    let n = instance.a_r1cs.cols();
    let m = reduction.m;
    let l = reduction.l;
    let max_rank = k.max(n);
    let num_parts = 4 + l;

    for j in 0..l {
        // Rebuild f̃_j(s) in R_q and evaluate on witness to get g_j
        let (_f_j, g_j_recomputed) = build_f_aggregation_rq::<R, N>(
            &reduction.alpha_challenges[j],
            &reduction.beta_challenges[j],
            &reduction.gamma_challenges[j],
            &reduction.delta_challenges[j],
            &reduction.phi_challenges,
            j,
            l,
            k,
            n,
            m,
            max_rank,
            num_parts,
            &instance.a_r1cs,
            &instance.b_r1cs,
            &instance.c_r1cs,
            &reduction.witness.parts,
        );

        // Recomputed g_j must match stored value
        if g_j_recomputed != reduction.g_polys[j] {
            return false;
        }
        // g_j(2) = 0 mod m
        if !check_divisible_by_x_minus_2::<R, N>(&g_j_recomputed, m) {
            return false;
        }
    }

    true
}

/// Sample verifier challenges for arithmetic R1CS reduction.
pub fn sample_arith_challenges<R, Rng>(
    k: usize,
    l: usize,
    m: u64,
    rng: &mut Rng,
) -> (Vec<Vec<R>>, Vec<Vec<R>>)
where
    R: IntegerRing<Canonical = u64>,
    Rng: RngExt,
{
    let phi_challenges: Vec<Vec<R>> = (0..l)
        .map(|_| {
            (0..k)
                .map(|_| R::from_u64(rng.random_range(0..m)))
                .collect()
        })
        .collect();

    let challenge_len = k * (l + 3);
    let agg_challenges: Vec<Vec<R>> = (0..l)
        .map(|_| {
            (0..challenge_len)
                .map(|_| R::from_u64(rng.random_range(0..m)))
                .collect()
        })
        .collect();

    (phi_challenges, agg_challenges)
}

/// Sample a challenge scalar uniformly below M from transcript.
///
/// Uses rejection sampling to avoid bias when q is not a multiple of M.
fn challenge_mod_m<R, T>(
    transcript: &mut T,
    label: &'static [u8],
    m: u64,
) -> Result<R, TranscriptError>
where
    R: IntegerRing<Canonical = u64>,
    T: grid_transcript::Transcript,
{
    // Compute rejection threshold: largest multiple of M that fits in u64
    let threshold = u64::MAX - (u64::MAX % m);

    loop {
        let bytes = transcript.challenge_bytes(label, 8)?;
        let val = u64::from_le_bytes(bytes.try_into().unwrap()); // always 8 bytes
        if val < threshold {
            return Ok(R::from_u64(val % m));
        }
        // Reject and resample
    }
}

/// Sample verifier challenges from transcript (Fiat-Shamir).
///
/// Domain-separates each round by appending the round index as bytes,
/// then uses static labels for the actual challenges.
#[allow(clippy::type_complexity)]
pub fn sample_arith_challenges_transcript<R, T>(
    k: usize,
    l: usize,
    m: u64,
    transcript: &mut T,
) -> Result<(Vec<Vec<R>>, Vec<Vec<R>>), TranscriptError>
where
    R: IntegerRing<Canonical = u64>,
    T: grid_transcript::Transcript,
{
    let mut phi_challenges = Vec::with_capacity(l);

    for i in 0..l {
        let round_bytes = (i as u32).to_le_bytes();
        transcript.append_bytes(b"labrador_arith_phi", &round_bytes)?;
        let phi_i: Vec<R> = (0..k)
            .map(|_| challenge_mod_m::<R, _>(transcript, b"labrador_arith_phi_c", m))
            .collect::<Result<_, _>>()?;
        phi_challenges.push(phi_i);
    }

    let mut agg_challenges = Vec::with_capacity(l);

    for j in 0..l {
        let round_bytes = (j as u32).to_le_bytes();
        transcript.append_bytes(b"labrador_arith_agg", &round_bytes)?;
        let alpha_j: Vec<R> = (0..k)
            .map(|_| challenge_mod_m::<R, _>(transcript, b"labrador_arith_agg_alpha", m))
            .collect::<Result<_, _>>()?;
        let beta_j: Vec<R> = (0..k)
            .map(|_| challenge_mod_m::<R, _>(transcript, b"labrador_arith_agg_beta", m))
            .collect::<Result<_, _>>()?;
        let gamma_j: Vec<R> = (0..k)
            .map(|_| challenge_mod_m::<R, _>(transcript, b"labrador_arith_agg_gamma", m))
            .collect::<Result<_, _>>()?;
        let delta_j: Vec<R> = (0..(l * k))
            .map(|_| challenge_mod_m::<R, _>(transcript, b"labrador_arith_agg_delta", m))
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

#[cfg(test)]
mod tests {
    use super::*;

    use grid_algebra::arith::prime::PrimeField;
    use grid_algebra::poly::ring::PolyRing;

    type F = PrimeField<12289>;

    #[test]
    fn test_encode_decode_naf() {
        let p = 257u64;

        for a in 0..257u64 {
            let poly = encode_naf::<F, 16>(a, p);
            let decoded = decode_naf::<F, 16>(&poly, p);
            assert_eq!(
                decoded, a,
                "NAF encode/decode mismatch for a={} (decoded={})",
                a, decoded
            );
        }

        let poly = encode_naf::<F, 16>(200, p);
        let q = F::modulus();
        let half = q / 2;
        for i in 0..16 {
            let raw = poly.coeff(i).to_u64();
            let centered = if raw > half {
                raw as i64 - q as i64
            } else {
                raw as i64
            };
            assert!(
                centered == 0 || centered == 1 || centered == -1,
                "NAF coefficient {} at position {} not in {{-1,0,1}}",
                centered,
                i
            );
        }
    }

    #[test]
    fn test_hadamard_product() {
        let p = 257u64;
        let a = vec![F::from_u64(3), F::from_u64(5), F::from_u64(7)];
        let b = vec![F::from_u64(2), F::from_u64(4), F::from_u64(6)];
        let result = hadamard_product(&a, &b, p);
        assert_eq!(result[0].to_u64(), 6);
        assert_eq!(result[1].to_u64(), 20);
        assert_eq!(result[2].to_u64(), 42);
    }

    #[test]
    fn test_mat_vec_mod_m() {
        let p = 257u64;
        let mat = RingMat::new(
            2,
            2,
            vec![
                F::from_u64(1),
                F::from_u64(2),
                F::from_u64(3),
                F::from_u64(4),
            ],
        );
        let w = vec![F::from_u64(5), F::from_u64(6)];
        let result = mat_vec_mod_m(&mat, &w, p);
        assert_eq!(result[0].to_u64(), 17);
        assert_eq!(result[1].to_u64(), 39);
    }

    #[test]
    fn test_check_divisible_by_x_minus_2() {
        let p = 257u64;
        let zero: CyclotomicPolyRing<F, 8> = CyclotomicPolyRing::zero();
        assert!(check_divisible_by_x_minus_2::<F, 8>(&zero, p));

        let mut f = CyclotomicPolyRing::<F, 8>::zero();
        f.set_coeff(1, F::from_u64(2));
        f.set_coeff(0, -F::from_u64(4));
        assert!(check_divisible_by_x_minus_2::<F, 8>(&f, p));

        let mut g = CyclotomicPolyRing::<F, 8>::zero();
        g.set_coeff(1, F::from_u64(1));
        g.set_coeff(0, -F::from_u64(3));
        assert!(!check_divisible_by_x_minus_2::<F, 8>(&g, p));
    }

    #[test]
    fn test_full_arith_r1cs_reduction_smoke() {
        let m = 257u64;
        let k = 2;
        let n = 2;
        let l = 2;

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

        let instance = ArithR1CSInstance {
            a_r1cs,
            b_r1cs,
            c_r1cs,
            modulus_m: m,
        };
        let witness = vec![F::from_u64(1), F::from_u64(0)];

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
        let reduction =
            build_arith_r1cs_reduction::<F, _, 8>(&instance, &witness, &crs_a, &crs_b, &mut rng, l)
                .unwrap();

        assert_eq!(reduction.witness.num_parts(), 4 + l);
        assert_eq!(reduction.l, l);
        // F: Ajtai openings (kappa_a + kappa_b) + aggregation (l)
        assert_eq!(reduction.statement.num_f(), kappa_a + kappa_b + l);
        // F': empty (aggregation moved to F as full-vanishing constraints)
        assert_eq!(reduction.statement.num_f_prime(), 0);

        // All parts have same rank
        let rank = reduction.witness.rank();
        for (i, part) in reduction.witness.parts.iter().enumerate() {
            assert_eq!(part.len(), rank, "part {} rank mismatch", i);
        }
    }

    #[test]
    fn test_sample_arith_challenges() {
        let p = 257u64;
        let mut rng = grid_std::test_rng();
        let (phi, agg) = sample_arith_challenges::<F, _>(3, 4, p, &mut rng);
        assert_eq!(phi.len(), 4);
        assert_eq!(agg.len(), 4);
        for v in &phi {
            assert_eq!(v.len(), 3);
            for e in v {
                assert!(e.to_u64() < p);
            }
        }
    }

    #[test]
    fn test_rq_aggregation_verify() {
        let m = 257u64;
        let k = 2;
        let n = 2;
        let l = 2;

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

        let instance = ArithR1CSInstance {
            a_r1cs,
            b_r1cs,
            c_r1cs,
            modulus_m: m,
        };
        let witness = vec![F::from_u64(1), F::from_u64(0)];

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
        let reduction =
            build_arith_r1cs_reduction::<F, _, 8>(&instance, &witness, &crs_a, &crs_b, &mut rng, l)
                .unwrap();

        // Verify stored g_polys are divisible by (X-2)
        assert!(
            verify_aggregation::<F, 8>(&reduction),
            "aggregation divisibility failed"
        );

        // Verify R_q F constraints vanish and g_j(2) = 0 mod M (paper-aligned C1 path)
        assert!(
            verify_aggregation_rq::<F, 8>(&reduction),
            "R_q F aggregation failed"
        );

        // Verify each F aggregation constraint vanishes in R_q
        let f = &reduction.statement.f;
        let agg_start = f.len().saturating_sub(l);
        for (j, f_j) in f.iter().skip(agg_start).enumerate() {
            let val = f_j.evaluate(&reduction.witness);
            assert!(val.is_zero(), "F aggregation[{}] does not vanish in R_q", j);
            // g_j (stored as b term) is divisible by (X-2)
            assert!(
                check_divisible_by_x_minus_2::<F, 8>(f_j.b(), m),
                "F aggregation[{}] g_j(2) != 0 mod M",
                j
            );
        }
    }

    #[test]
    #[should_panic(expected = "must equal 2^N+1")]
    fn test_invalid_modulus_m() {
        // 19 is not of the form 2^d + 1 (2^4=16, 2^5=32)
        let m = 19u64;
        let k = 2;
        let n = 2;

        let a_r1cs = RingMat::new(k, n, vec![F::from_u64(1); k * n]);
        let b_r1cs = RingMat::new(k, n, vec![F::from_u64(1); k * n]);
        let c_r1cs = RingMat::new(k, n, vec![F::from_u64(1); k * n]);
        let instance = ArithR1CSInstance {
            a_r1cs,
            b_r1cs,
            c_r1cs,
            modulus_m: m,
        };
        let witness = vec![F::from_u64(1), F::from_u64(0)];

        let kappa_a = 2;
        let kappa_b = 2;
        let l = 2;
        let max_rank = 3 * k + n;

        let crs_a = RingMat::new(
            kappa_a,
            max_rank,
            (0..kappa_a * max_rank)
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
        let _ =
            build_arith_r1cs_reduction::<F, _, 8>(&instance, &witness, &crs_a, &crs_b, &mut rng, l)
                .unwrap();
    }
}
