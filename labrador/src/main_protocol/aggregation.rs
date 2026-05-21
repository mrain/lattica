//! Two-step aggregation (§5.2 steps 6-11).
//!
//! Step 1: Compress L F' functions + 256 JL constraints into ℓ batches of b''.
//! Step 2: Compress F functions + aggregated b'' + JL φ terms into final function.

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use grid_algebra::arith::ring::IntegerRing;
use grid_algebra::arith::ring::Ring;
use grid_algebra::poly::ring::{CyclotomicPolyRing, NegacyclicMulRing, PolyRing};
use grid_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::main_protocol::{garbage_count, garbage_index, garbage_index_inv};

use crate::main_protocol::Transcript;
use crate::relation::QuadraticFunction;

/// Final aggregated quadratic function coefficients.
///
/// Produced by [`aggregate_step2_flat`]. Determines the φ_i used in h_ij computation.
#[derive(Debug, Clone)]
pub struct AggregatedFunction<R, const N: usize>
where
    R: IntegerRing<Canonical = u64>
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize,
{
    /// Aggregated a_ij: sparse upper-triangular quadratic terms.
    pub a_ij: Vec<(usize, usize, CyclotomicPolyRing<R, N>)>,
    /// Aggregated φ_i: linear terms per witness part.
    /// `phi[i][j]` is the j-th polynomial of φ_i (length n per part).
    pub phi: Vec<Vec<CyclotomicPolyRing<R, N>>>,
    /// Aggregated b: constant term.
    pub b: CyclotomicPolyRing<R, N>,
}

/// Number of first-step aggregation batches.
/// ℓ = ⌈128 / log₂ q⌉, computed via integer arithmetic to avoid cross-platform f64 drift.
/// Uses ⌊log₂ q⌋ in the denominator, which is ≤ float log₂(q), so the result is ≥ the
/// float formula and never undershoots soundness.
pub fn num_agg_batches(q: u64) -> usize {
    debug_assert!(q > 1, "modulus must be > 1 for aggregation");
    let floor_log2 = (63 - q.leading_zeros()) as usize;
    debug_assert!(
        floor_log2 > 0,
        "modulus must have at least 1 significant bit"
    );
    128_usize.div_ceil(floor_log2)
}

/// Step 1: Sample ℓ batches of (ψ, ω) challenges from transcript.
///
/// ψ^(k): one scalar per F' function (length L).
/// ω^(k): one scalar per JL constraint entry (length 256).
#[allow(clippy::type_complexity)]
pub fn sample_step1_challenges<R, T>(
    num_f_prime: usize,
    num_jl_entries: usize,
    q: u64,
    transcript: &mut T,
) -> Result<(Vec<Vec<R>>, Vec<Vec<R>>), T::Error>
where
    R: IntegerRing<Canonical = u64>,
    T: Transcript,
{
    let ell = num_agg_batches(q);

    let mut psi = Vec::with_capacity(ell);
    let mut omega = Vec::with_capacity(ell);

    for k in 0..ell {
        let k_bytes = (k as u32).to_le_bytes();
        transcript.append_bytes(b"labrador_agg_k", &k_bytes)?;

        let psi_k: Vec<_> = (0..num_f_prime)
            .map(|_| transcript.challenge_scalar::<R>(b"labrador_psi"))
            .collect::<Result<_, _>>()?;

        let omega_k: Vec<_> = (0..num_jl_entries)
            .map(|_| transcript.challenge_scalar::<R>(b"labrador_omega"))
            .collect::<Result<_, _>>()?;

        psi.push(psi_k);
        omega.push(omega_k);
    }

    Ok((psi, omega))
}

/// Prover computes `b''^(k) = Σ_l ψ_l^(k) · (f'_l(s) + f'_l.b) + Σ_m ω_m^(k) · jl_eval[m]`.
///
/// **Paper structure (§5.2 p.198,202):** The paper constructs `b''^(k)` by adding
/// the public constant term f'.b back to the evaluation f'(s) = Q + L - b,
/// giving f'(s) + b = Q + L. This ensures ct(b'') = Σ ψ · ct(f'.b) + ⟨ω, p⟩
/// where ct(f'.b) = f'.b.coeff(0) is the public subtracted constant.
///
/// `jl_eval[m] = Σ_i ⟨σ₋₁(π_i^(m)), s_i⟩` is the full polynomial JL evaluation.
/// For honest witness: `ct(jl_eval[m]) = p[m]` (JL projection scalar).
pub fn compute_b_double_prime<R, const N: usize>(
    f_prime_evals: &[CyclotomicPolyRing<R, N>],
    f_prime_b: &[CyclotomicPolyRing<R, N>],
    jl_evals: &[CyclotomicPolyRing<R, N>],
    psi: &[Vec<R>],
    omega: &[Vec<R>],
) -> Vec<CyclotomicPolyRing<R, N>>
where
    R: IntegerRing<Canonical = u64>
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize,
{
    let ell = psi.len();
    let mut b_dp = Vec::with_capacity(ell);

    for k in 0..ell {
        let mut sum = CyclotomicPolyRing::<R, N>::zero();

        // Σ_l ψ_l^(k) · (f'_l(s) + f'_l.b)  = Σ ψ · (Q + L)
        for (l, f_eval) in f_prime_evals.iter().enumerate() {
            sum += CyclotomicPolyRing::add_ref(f_eval, &f_prime_b[l]).scalar_mul(&psi[k][l]);
        }

        // Σ_m ω_m^(k) · jl_eval[m]  (full polynomial JL contribution)
        for (m, jl_eval) in jl_evals.iter().enumerate() {
            sum += jl_eval.scalar_mul(&omega[k][m]);
        }

        b_dp.push(sum);
    }

    b_dp
}

/// Verifier checks ct(b''^(k)) = Σ_l ψ_l^(k) · b0'^(l) + ⟨ω^(k), p⟩.
///
/// b0'^(l) = f'_l.b.coeff(0) is the public subtracted constant term from the
/// statement. The paper notation has functions of the form Q + L - b, so b0'
/// is the positive public constant (NOT its negation).
pub fn verify_b_double_prime<R, const N: usize>(
    b_double_prime: &[CyclotomicPolyRing<R, N>],
    psi: &[Vec<R>],
    omega: &[Vec<R>],
    p: &[R],
    b0_primes: &[R],
) -> Result<(), String>
where
    R: IntegerRing<Canonical = u64>
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize,
{
    let q = R::modulus();

    for k in 0..b_double_prime.len() {
        let mut expected_ct: u128 = 0;
        let q128 = q as u128;

        for l in 0..b0_primes.len() {
            let prod = psi[k][l].to_u64() as u128 * b0_primes[l].to_u64() as u128 % q128;
            expected_ct = (expected_ct + prod) % q128;
        }
        for m in 0..p.len() {
            let prod = omega[k][m].to_u64() as u128 * p[m].to_u64() as u128 % q128;
            expected_ct = (expected_ct + prod) % q128;
        }

        let actual_ct = b_double_prime[k].coeff(0).to_u64() as u128;
        if actual_ct != expected_ct {
            return Err(format!(
                "b''[{}] constant term mismatch: expected {}, got {}",
                k, expected_ct, actual_ct
            ));
        }
    }

    Ok(())
}

/// Flat-JL variant of step 2 aggregation.
#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
pub fn aggregate_step2_flat<R, const N: usize>(
    f: &[QuadraticFunction<CyclotomicPolyRing<R, N>>],
    f_prime: &[QuadraticFunction<CyclotomicPolyRing<R, N>>],
    alpha: &[CyclotomicPolyRing<R, N>],
    beta: &[CyclotomicPolyRing<R, N>],
    b_double_prime: &[CyclotomicPolyRing<R, N>],
    psi: &[Vec<R>],
    omega: &[Vec<R>],
    jl_rows_poly: &super::JlRowsFlat<CyclotomicPolyRing<R, N>>,
) -> AggregatedFunction<R, N>
where
    R: IntegerRing<Canonical = u64>
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize,
{
    let num_jl = jl_rows_poly.num_rows();
    let r = jl_rows_poly.num_parts();
    let n = jl_rows_poly.num_polys();
    let mut agg_a_ij: Vec<CyclotomicPolyRing<R, N>> =
        vec![CyclotomicPolyRing::<R, N>::zero(); garbage_count(r)];
    let mut agg_phi: Vec<Vec<CyclotomicPolyRing<R, N>>> =
        vec![vec![CyclotomicPolyRing::<R, N>::zero(); n]; r];
    let mut agg_b = CyclotomicPolyRing::<R, N>::zero();

    for (k, func) in f.iter().enumerate() {
        scale_and_add_poly(&mut agg_a_ij, &mut agg_phi, &mut agg_b, func, &alpha[k], r);
    }

    for l in 0..f_prime.len() {
        let scale: CyclotomicPolyRing<R, N> = (0..beta.len())
            .map(|k| beta[k].scalar_mul(&psi[k][l]))
            .fold(CyclotomicPolyRing::<R, N>::zero(), |acc, x| acc + x);
        scale_and_add_a_phi_only(&mut agg_a_ij, &mut agg_phi, &f_prime[l], &scale, r);
    }

    for m in 0..num_jl {
        let jl_scale: CyclotomicPolyRing<R, N> = (0..beta.len())
            .map(|k| beta[k].scalar_mul(&omega[k][m]))
            .fold(CyclotomicPolyRing::<R, N>::zero(), |acc, x| acc + x);
        for i in 0..r {
            for j in 0..n {
                agg_phi[i][j] += &jl_scale * jl_rows_poly.get(m, i, j);
            }
        }
    }

    for k in 0..beta.len() {
        agg_b += &beta[k] * &b_double_prime[k];
    }

    let mut a_ij_sparse = Vec::with_capacity(garbage_count(r));
    for (gi, entry) in agg_a_ij.iter().enumerate() {
        if !entry.is_zero() {
            let (i, j) = garbage_index_inv(r, gi);
            a_ij_sparse.push((i, j, entry.clone()));
        }
    }

    AggregatedFunction {
        a_ij: a_ij_sparse,
        phi: agg_phi,
        b: agg_b,
    }
}

/// Scale a function by a polynomial and add into accumulator (full a_ij + φ + b).
fn scale_and_add_poly<R, const N: usize>(
    acc_a_ij: &mut [CyclotomicPolyRing<R, N>],
    acc_phi: &mut Vec<Vec<CyclotomicPolyRing<R, N>>>,
    acc_b: &mut CyclotomicPolyRing<R, N>,
    func: &QuadraticFunction<CyclotomicPolyRing<R, N>>,
    scale: &CyclotomicPolyRing<R, N>,
    r: usize,
) where
    R: IntegerRing<Canonical = u64>
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize,
{
    func.for_each_quad(|(i, j, coeff)| {
        let scaled = coeff * scale;
        acc_a_ij[garbage_index(r, i, j)] += scaled;
    });
    func.for_each_nonzero_phi(|(pi, ei, coeff)| {
        debug_assert!(pi < r, "phi index {} must be < r ({})", pi, r);
        while acc_phi.len() <= pi {
            acc_phi.push(Vec::new());
        }
        while acc_phi[pi].len() <= ei {
            acc_phi[pi].push(CyclotomicPolyRing::<R, N>::zero());
        }
        acc_phi[pi][ei] += coeff * scale;
    });
    *acc_b += func.b() * scale;
}

/// Scale F' function by a polynomial: add ONLY a_ij and φ (NOT b).
fn scale_and_add_a_phi_only<R, const N: usize>(
    acc_a_ij: &mut [CyclotomicPolyRing<R, N>],
    acc_phi: &mut Vec<Vec<CyclotomicPolyRing<R, N>>>,
    func: &QuadraticFunction<CyclotomicPolyRing<R, N>>,
    scale: &CyclotomicPolyRing<R, N>,
    r: usize,
) where
    R: IntegerRing<Canonical = u64>
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize,
{
    func.for_each_quad(|(i, j, coeff)| {
        let scaled = coeff * scale;
        acc_a_ij[garbage_index(r, i, j)] += scaled;
    });
    func.for_each_nonzero_phi(|(pi, ei, coeff)| {
        while acc_phi.len() <= pi {
            acc_phi.push(Vec::new());
        }
        while acc_phi[pi].len() <= ei {
            acc_phi[pi].push(CyclotomicPolyRing::<R, N>::zero());
        }
        acc_phi[pi][ei] += coeff * scale;
    });
    // NOTE: func.b is NOT added — it is already inside b'' from the prover.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::main_protocol::JlRowsFlat;
    use alloc::vec;

    use grid_algebra::arith::prime::PrimeField;
    use grid_algebra::arith::ring::Ring;

    type F = PrimeField<12289>;

    #[test]
    fn test_num_agg_batches() {
        // q = 12289, floor_log2 = 13, ℓ = ceil(128/13) = 10
        assert_eq!(num_agg_batches(12289), 10);
        // q = 2^32, floor_log2 = 32, ℓ = ceil(128/32) = 4
        assert_eq!(num_agg_batches(4294967296), 4);
        // q = 2^18, floor_log2 = 18, ℓ = ceil(128/18) = 8
        assert_eq!(num_agg_batches(262144), 8);
    }

    #[test]
    fn test_empty_aggregation() {
        let r = 3;
        let n = 4;
        let num_rows = 1;
        let data = vec![CyclotomicPolyRing::<F, 64>::zero(); num_rows * r * n];
        let jl_rows_poly = JlRowsFlat::new(data, num_rows, r, n);

        let agg = aggregate_step2_flat::<F, 64>(&[], &[], &[], &[], &[], &[], &[], &jl_rows_poly);

        assert!(agg.a_ij.is_empty());
        assert_eq!(agg.phi.len(), r);
        for phi_i in &agg.phi {
            assert_eq!(phi_i.len(), n);
            for p in phi_i {
                assert!(p.is_zero());
            }
        }
        assert!(agg.b.is_zero());
    }

    #[test]
    fn test_scale_and_add_poly_zero() {
        let r = 2;
        let mut agg_a_ij: Vec<CyclotomicPolyRing<F, 64>> =
            vec![CyclotomicPolyRing::<F, 64>::zero(); crate::main_protocol::garbage_count(r)];
        let n = 4;
        let mut agg_phi: Vec<Vec<CyclotomicPolyRing<F, 64>>> =
            vec![vec![CyclotomicPolyRing::<F, 64>::zero(); n]; r];
        let mut agg_b = CyclotomicPolyRing::<F, 64>::zero();

        let func = QuadraticFunction::from_parts(
            Vec::new(),
            vec![
                vec![CyclotomicPolyRing::<F, 64>::zero(); n],
                vec![CyclotomicPolyRing::<F, 64>::zero(); n],
            ],
            CyclotomicPolyRing::<F, 64>::zero(),
        );
        let scale = CyclotomicPolyRing::<F, 64>::one();

        scale_and_add_poly(&mut agg_a_ij, &mut agg_phi, &mut agg_b, &func, &scale, r);

        assert!(agg_a_ij.iter().all(|p| p.is_zero()));
        assert!(agg_b.is_zero());
    }

    #[test]
    fn test_empty_input_behavior() {
        let bdp = compute_b_double_prime::<F, 64>(&[], &[], &[], &[], &[]);
        assert!(bdp.is_empty());

        assert!(verify_b_double_prime::<F, 64>(&[], &[], &[], &[], &[]).is_ok());
    }

    #[test]
    fn test_aggregated_function_shapes() {
        let r = 2;
        let n = 4;
        let ell = 2;

        // Simple non-empty case with one F function
        let f_func = QuadraticFunction::from_parts(
            vec![(0, 0, CyclotomicPolyRing::<F, 64>::one())],
            vec![
                vec![CyclotomicPolyRing::<F, 64>::zero(); n],
                vec![CyclotomicPolyRing::<F, 64>::zero(); n],
            ],
            CyclotomicPolyRing::<F, 64>::zero(),
        );

        let num_rows = 1;
        let data = vec![CyclotomicPolyRing::<F, 64>::zero(); num_rows * r * n];
        let jl_rows_poly = JlRowsFlat::new(data, num_rows, r, n);

        let alpha = vec![CyclotomicPolyRing::<F, 64>::one()];
        let beta = vec![CyclotomicPolyRing::<F, 64>::zero(); ell];
        let psi: Vec<Vec<F>> = Vec::new();
        let omega: Vec<Vec<F>> = (0..ell).map(|_| vec![F::zero()]).collect();
        let bdp = vec![CyclotomicPolyRing::<F, 64>::zero(); ell];

        let agg = aggregate_step2_flat::<F, 64>(
            &[f_func],
            &[],
            &alpha,
            &beta,
            &bdp,
            &psi,
            &omega,
            &jl_rows_poly,
        );

        assert_eq!(agg.phi.len(), r);
        for phi_i in &agg.phi {
            assert_eq!(phi_i.len(), n);
        }
        assert!(!agg.a_ij.is_empty());
    }
}
