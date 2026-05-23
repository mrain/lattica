//! Verification equations (§5.2).
//!
//! Checks all four verification equations. Critical: t_vecs are reconstructed
//! from decomposed limbs, NOT trusted from the prover.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use grid_algebra::arith::ring::IntegerRing;
use grid_algebra::arith::ring::Ring;
use grid_algebra::lattice::types::RingVec;
use grid_algebra::poly::ring::{CyclotomicPolyRing, NegacyclicMulRing, PolyRing};
use grid_serialize::CanonicalDeserialize;
use grid_serialize::CanonicalSerialize;
use grid_std::UniformRand;

use crate::crs::CommitKey;
use crate::main_protocol::aggregation::AggregatedFunction;
use crate::main_protocol::{DecomposedPolys, garbage_index, reconstruct_t_vecs};
use crate::params::LabradorParams;

/// Validate decomposed limb coefficients are short (within centered bounds).
///
/// Each limb coefficient is stored unsigned in [0, q). It represents a centered
/// value in [-base/2, base/2]. A coefficient `v` is short if either:
/// - `v <= base/2` (positive centered value), or
/// - `v >= q - base/2` (negative centered value encoded as `q - |v|`).
///   Long limbs indicate a malicious decomposition outside the intended relation.
fn validate_decomposed_limbs<R, const N: usize>(
    decomposed: &DecomposedPolys<CyclotomicPolyRing<R, N>>,
    label: &str,
    q: u64,
) -> Result<(), String>
where
    R: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    let base = decomposed.base;
    let half_base = base / 2;
    let neg_threshold = q.wrapping_sub(half_base);

    for (idx, limb_poly) in decomposed.flat.iter().enumerate() {
        for i in 0..N {
            let v = limb_poly.coeff(i).to_u64();
            let is_short = v <= half_base || v >= neg_threshold;
            if !is_short {
                return Err(format!(
                    "limb shortness: {} limb[{}].coeff({})={} exceeds centered bound [0,{}] U [{}-half_base, q)",
                    label, idx, i, v, half_base, q
                ));
            }
        }
    }
    Ok(())
}

/// Validate all proof vector shapes before any indexing.
/// Returns Err for malformed proofs instead of panicking.
fn validate_proof_shapes<R, const N: usize>(
    z: &[CyclotomicPolyRing<R, N>],
    challenges: &[CyclotomicPolyRing<R, N>],
    t_decomposed: &DecomposedPolys<CyclotomicPolyRing<R, N>>,
    g_decomposed: &DecomposedPolys<CyclotomicPolyRing<R, N>>,
    h_decomposed: &DecomposedPolys<CyclotomicPolyRing<R, N>>,
    params: &LabradorParams,
) -> Result<(), String>
where
    R: IntegerRing<Canonical = u64>
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize,
{
    validate_proof_shapes_common(challenges, t_decomposed, g_decomposed, h_decomposed, params)?;
    if z.len() != params.n {
        return Err(format!(
            "shape: z.len()={} expected n={}",
            z.len(),
            params.n
        ));
    }
    Ok(())
}

fn validate_proof_shapes_common<R, const N: usize>(
    challenges: &[CyclotomicPolyRing<R, N>],
    t_decomposed: &DecomposedPolys<CyclotomicPolyRing<R, N>>,
    g_decomposed: &DecomposedPolys<CyclotomicPolyRing<R, N>>,
    h_decomposed: &DecomposedPolys<CyclotomicPolyRing<R, N>>,
    params: &LabradorParams,
) -> Result<(), String>
where
    R: IntegerRing<Canonical = u64>
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize,
{
    validate_challenges_len(challenges, params)?;
    validate_decomposed_shapes(t_decomposed, g_decomposed, h_decomposed, params)?;
    Ok(())
}

/// Validate challenges length only (domain-agnostic).
fn validate_challenges_len<T>(challenges: &[T], params: &LabradorParams) -> Result<(), String> {
    if challenges.len() != params.r {
        return Err(format!(
            "shape: challenges.len()={} expected r={}",
            challenges.len(),
            params.r
        ));
    }
    Ok(())
}

/// Validate decomposed polynomial shapes (domain-agnostic, no challenge dependency).
fn validate_decomposed_shapes<R, const N: usize>(
    t_decomposed: &DecomposedPolys<CyclotomicPolyRing<R, N>>,
    g_decomposed: &DecomposedPolys<CyclotomicPolyRing<R, N>>,
    h_decomposed: &DecomposedPolys<CyclotomicPolyRing<R, N>>,
    params: &LabradorParams,
) -> Result<(), String>
where
    R: IntegerRing<Canonical = u64>
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize,
{
    let garbage_len = crate::main_protocol::garbage_count(params.r);

    let expected_t_polys = params.r * params.kappa;
    if t_decomposed.num_polys != expected_t_polys {
        return Err(format!(
            "shape: t_decomposed.num_polys={} expected r*kappa={}",
            t_decomposed.num_polys, expected_t_polys
        ));
    }
    if t_decomposed.num_limbs != params.t1 {
        return Err(format!(
            "shape: t_decomposed.num_limbs={} expected t1={}",
            t_decomposed.num_limbs, params.t1
        ));
    }
    if t_decomposed.flat.len() != expected_t_polys * params.t1 {
        return Err(format!(
            "shape: t_decomposed.flat.len()={} expected {}",
            t_decomposed.flat.len(),
            expected_t_polys * params.t1
        ));
    }
    if t_decomposed.base != params.b1 {
        return Err(format!(
            "shape: t_decomposed.base={} expected b1={}",
            t_decomposed.base, params.b1
        ));
    }

    if g_decomposed.num_polys != garbage_len {
        return Err(format!(
            "shape: g_decomposed.num_polys={} expected garbage_count(r)={}",
            g_decomposed.num_polys, garbage_len
        ));
    }
    if g_decomposed.num_limbs != params.t2 {
        return Err(format!(
            "shape: g_decomposed.num_limbs={} expected t2={}",
            g_decomposed.num_limbs, params.t2
        ));
    }
    if g_decomposed.flat.len() != garbage_len * params.t2 {
        return Err(format!(
            "shape: g_decomposed.flat.len()={} expected {}",
            g_decomposed.flat.len(),
            garbage_len * params.t2
        ));
    }
    if g_decomposed.base != params.b2 {
        return Err(format!(
            "shape: g_decomposed.base={} expected b2={}",
            g_decomposed.base, params.b2
        ));
    }

    if h_decomposed.num_polys != garbage_len {
        return Err(format!(
            "shape: h_decomposed.num_polys={} expected garbage_count(r)={}",
            h_decomposed.num_polys, garbage_len
        ));
    }
    if h_decomposed.num_limbs != params.t1 {
        return Err(format!(
            "shape: h_decomposed.num_limbs={} expected t1={}",
            h_decomposed.num_limbs, params.t1
        ));
    }
    if h_decomposed.flat.len() != garbage_len * params.t1 {
        return Err(format!(
            "shape: h_decomposed.flat.len()={} expected {}",
            h_decomposed.flat.len(),
            garbage_len * params.t1
        ));
    }
    if h_decomposed.base != params.b1 {
        return Err(format!(
            "shape: h_decomposed.base={} expected b1={}",
            h_decomposed.base, params.b1
        ));
    }

    Ok(())
}

/// Compute squared L2 norm of a vector of polynomials.
///
/// Uses u128 accumulation (exact before f64 conversion) with overflow guards.
/// For large q overflow returns f64::INFINITY (conservative rejection).
fn polys_l2_norm_squared_u128<R, const N: usize>(
    polys: &[CyclotomicPolyRing<R, N>],
) -> Result<u128, String>
where
    R: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    let coeffs: Vec<_> = polys.iter().flat_map(|p| p.coeffs()).cloned().collect();
    crate::main_protocol::squared_l2_norm_u128(coeffs).map_err(String::from)
}

/// Verify L2 norm bounds for the outer openings (§5.2).
///
/// Current CRS binding: u1 = B·t + C·g, u2 = D·h.
/// Parameter derivation: gamma1 covers t+g combined, gamma2 covers h alone.
/// Coefficient shortness alone is insufficient — worst-case centered limbs
/// can reach b^2/4 per coefficient, while gamma1/gamma2 use b^2/12 heuristic variance.
fn verify_opening_norms<R, const N: usize>(
    t_decomposed: &DecomposedPolys<CyclotomicPolyRing<R, N>>,
    g_decomposed: &DecomposedPolys<CyclotomicPolyRing<R, N>>,
    h_decomposed: &DecomposedPolys<CyclotomicPolyRing<R, N>>,
    gamma1_sq: u128,
    gamma2_sq: u128,
) -> Result<(), String>
where
    R: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    // ||t||^2 + ||g||^2 <= gamma1^2 (t+g bound for u1 = B·t + C·g)
    let t_norm_sq = polys_l2_norm_squared_u128(&t_decomposed.flat)?;
    let g_norm_sq = polys_l2_norm_squared_u128(&g_decomposed.flat)?;
    let tg_combined = t_norm_sq
        .checked_add(g_norm_sq)
        .ok_or("norm overflow: ||t||² + ||g||² exceeds u128")?;
    if tg_combined > gamma1_sq {
        return Err(format!(
            "opening norm: ||t||² + ||g||² ({}) exceeds γ₁² ({})",
            tg_combined, gamma1_sq
        ));
    }

    // ||h|| <= gamma2 (h bound for u2 = D·h)
    let h_norm_sq = polys_l2_norm_squared_u128(&h_decomposed.flat)?;
    if h_norm_sq > gamma2_sq {
        return Err(format!(
            "opening norm: ||h||² ({}) exceeds γ₂² ({})",
            h_norm_sq, gamma2_sq
        ));
    }

    Ok(())
}

/// Verify all four equations.
///
/// CRITICAL: t_vecs are reconstructed from t_decomposed, NOT trusted.
/// Shape validation runs before any indexing to return Err instead of panicking.
#[allow(clippy::too_many_arguments)]
pub fn verify_all<R, const N: usize>(
    key: &CommitKey<R, N>,
    u1: &RingVec<CyclotomicPolyRing<R, N>>,
    u2: &RingVec<CyclotomicPolyRing<R, N>>,
    z: &[CyclotomicPolyRing<R, N>],
    challenges: &[CyclotomicPolyRing<R, N>],
    t_decomposed: &DecomposedPolys<CyclotomicPolyRing<R, N>>,
    g_decomposed: &DecomposedPolys<CyclotomicPolyRing<R, N>>,
    h_decomposed: &DecomposedPolys<CyclotomicPolyRing<R, N>>,
    aggregated: &AggregatedFunction<R, N>,
    params: &LabradorParams,
) -> Result<(), String>
where
    R: IntegerRing<Canonical = u64>
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize
        + UniformRand,
{
    // ── Shape validation (fail with Err, never panic) ──
    validate_proof_shapes(
        z,
        challenges,
        t_decomposed,
        g_decomposed,
        h_decomposed,
        params,
    )?;

    // ── Limb shortness validation (reject long decompositions) ──
    let q = R::modulus();
    validate_decomposed_limbs(t_decomposed, "t", q)?;
    validate_decomposed_limbs(g_decomposed, "g", q)?;
    validate_decomposed_limbs(h_decomposed, "h", q)?;

    // ── Opening L2 norm bounds (gamma1 for t, gamma2 for g/h) ──
    verify_opening_norms(
        t_decomposed,
        g_decomposed,
        h_decomposed,
        params.gamma1_sq,
        params.gamma2_sq,
    )?;

    // Reconstruct t_vecs from decomposed limbs (NOT trusted)
    let t_vecs = reconstruct_t_vecs(t_decomposed, params.r, params.kappa, params.t1);

    // Reconstruct g_ij, h_ij from decomposed limbs
    let g_polys = g_decomposed.reconstruct();
    let h_polys = h_decomposed.reconstruct();

    // Equation (1): u1 = B·t + C·g
    verify_eq_u1(key, u1, t_decomposed, g_decomposed)?;

    // Equation (2a): A·z = Σ c_i · t_i  (using RECONSTRUCTED t_vecs)
    verify_eq_2a(key, z, challenges, &t_vecs)?;

    // Equation (2b): ||z|| ≤ γ
    verify_eq_2b(z, params.gamma)?;

    // Equation (3a): ⟨z, z⟩ = Σ g_ij·c_i·c_j
    verify_eq_3a(z, &g_polys, challenges)?;

    // Equation (3b): Σ⟨φ_i, z⟩c_i = Σ h_ij·c_i·c_j
    verify_eq_3b(z, &h_polys, challenges, &aggregated.phi)?;

    // Equation (3c): Σ a_ij·g_ij + Σ h_ii - b = 0
    verify_eq_3c(&g_polys, &h_polys, &aggregated.a_ij, &aggregated.b)?;

    // Equation (4): u2 = D·h
    verify_eq_u2(key, u2, h_decomposed)?;

    Ok(())
}

/// Equation (1): u1 = B·t + C·g
fn verify_eq_u1<R, const N: usize>(
    key: &CommitKey<R, N>,
    u1: &RingVec<CyclotomicPolyRing<R, N>>,
    t_decomposed: &DecomposedPolys<CyclotomicPolyRing<R, N>>,
    g_decomposed: &DecomposedPolys<CyclotomicPolyRing<R, N>>,
) -> Result<(), String>
where
    R: IntegerRing<Canonical = u64>
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize
        + UniformRand,
{
    let expected = key.outer_commit_u1_slice(&t_decomposed.flat, &g_decomposed.flat);
    if u1 != &expected {
        return Err("equation (1): u1 ≠ B·t + C·g".into());
    }
    Ok(())
}

/// Equation (2a): A·z = Σ c_i · t_i
fn verify_eq_2a<R, const N: usize>(
    key: &CommitKey<R, N>,
    z: &[CyclotomicPolyRing<R, N>],
    challenges: &[CyclotomicPolyRing<R, N>],
    t_vecs: &[RingVec<CyclotomicPolyRing<R, N>>],
) -> Result<(), String>
where
    R: IntegerRing<Canonical = u64>
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize
        + UniformRand,
{
    let left = key.a.mul_slice(z);

    let mut right = RingVec::zero(t_vecs[0].len());
    for (c_i, t_i) in challenges.iter().zip(t_vecs.iter()) {
        for (r_entry, t_entry) in right.entries_mut().iter_mut().zip(t_i.entries().iter()) {
            *r_entry += c_i * t_entry;
        }
    }

    if left != right {
        return Err("equation (2a): A·z ≠ Σ c_i·t_i".into());
    }
    Ok(())
}

/// Equation (2b): ||z|| ≤ γ
fn verify_eq_2b<R, const N: usize>(z: &[CyclotomicPolyRing<R, N>], gamma: f64) -> Result<(), String>
where
    R: IntegerRing<Canonical = u64>
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize,
{
    let coeffs: Vec<_> = z.iter().flat_map(|p| p.coeffs()).cloned().collect();
    let norm_sq = crate::main_protocol::squared_l2_norm::<R>(coeffs).map_err(String::from)?;
    if norm_sq > gamma * gamma {
        return Err(format!(
            "equation (2b): ||z||² ({}) exceeds γ² ({})",
            norm_sq,
            gamma * gamma
        ));
    }
    Ok(())
}

/// Equation (3a): ⟨z, z⟩ = Σ_{i,j} g_ij · c_i · c_j
fn verify_eq_3a<R, const N: usize>(
    z: &[CyclotomicPolyRing<R, N>],
    g_polys: &[CyclotomicPolyRing<R, N>],
    challenges: &[CyclotomicPolyRing<R, N>],
) -> Result<(), String>
where
    R: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    let r = challenges.len();

    // Left: ⟨z, z⟩
    let left = Ring::dot_product(z, z);

    // Right: Σ_{i,j} g_ij · c_i · c_j (upper-triangular, off-diagonal doubled)
    let mut right = CyclotomicPolyRing::<R, N>::zero();
    let mut gi = 0;
    for i in 0..r {
        for j in i..r {
            let term = &g_polys[gi] * &challenges[i] * &challenges[j];
            if i == j {
                right += term;
            } else {
                right += &term;
                right += &term;
            }
            gi += 1;
        }
    }

    if left != right {
        return Err("equation (3a): ⟨z,z⟩ ≠ Σ g_ij·c_i·c_j".into());
    }
    Ok(())
}

/// Equation (3b): Σ_i ⟨φ_i, z⟩ · c_i = Σ_{i,j} h_ij · c_i · c_j
fn verify_eq_3b<R, const N: usize>(
    z: &[CyclotomicPolyRing<R, N>],
    h_polys: &[CyclotomicPolyRing<R, N>],
    challenges: &[CyclotomicPolyRing<R, N>],
    aggregated_phis: &[Vec<CyclotomicPolyRing<R, N>>],
) -> Result<(), String>
where
    R: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    let r = challenges.len();

    // Left: Σ_i ⟨φ_i, z⟩ · c_i
    let mut left = CyclotomicPolyRing::<R, N>::zero();
    for (i, phi_i) in aggregated_phis.iter().enumerate() {
        let dot = Ring::dot_product(phi_i, z);
        left += dot * &challenges[i];
    }

    // Right: Σ_{i,j} h_ij · c_i · c_j (upper-triangular, off-diagonal doubled)
    let mut right = CyclotomicPolyRing::<R, N>::zero();
    let mut hi = 0;
    for i in 0..r {
        for j in i..r {
            let term = &h_polys[hi] * &challenges[i] * &challenges[j];
            if i == j {
                right += term;
            } else {
                right += &term;
                right += &term;
            }
            hi += 1;
        }
    }

    if left != right {
        return Err("equation (3b): Σ⟨φ_i,z⟩c_i ≠ Σ h_ij·c_i·c_j".into());
    }
    Ok(())
}

/// Equation (3c): Σ_{i,j} a_ij · g_ij + Σ_i h_ii - b = 0
fn verify_eq_3c<R, const N: usize>(
    g_polys: &[CyclotomicPolyRing<R, N>],
    h_polys: &[CyclotomicPolyRing<R, N>],
    a_ij: &[(usize, usize, CyclotomicPolyRing<R, N>)],
    b: &CyclotomicPolyRing<R, N>,
) -> Result<(), String>
where
    R: IntegerRing<Canonical = u64>
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize,
{
    let r = challenges_count_from_g(g_polys.len());
    let mut sum = CyclotomicPolyRing::<R, N>::zero();

    // Σ_{i,j} a_ij · g_ij
    for &(i, j, ref a_coeff) in a_ij {
        let g_idx = garbage_index(r, i, j);
        sum += a_coeff * &g_polys[g_idx];
    }

    // Σ_i h_ii (diagonal h entries only)
    for i in 0..r {
        let h_ii_idx = garbage_index(r, i, i);
        sum += &h_polys[h_ii_idx];
    }

    // Subtract b
    sum -= b;

    if !sum.is_zero() {
        return Err("equation (3c): Σ a_ij·g_ij + Σ h_ii - b ≠ 0".into());
    }
    Ok(())
}

/// Equation (4): u2 = D·h
fn verify_eq_u2<R, const N: usize>(
    key: &CommitKey<R, N>,
    u2: &RingVec<CyclotomicPolyRing<R, N>>,
    h_decomposed: &DecomposedPolys<CyclotomicPolyRing<R, N>>,
) -> Result<(), String>
where
    R: IntegerRing<Canonical = u64>
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize
        + UniformRand,
{
    let expected = key.outer_commit_u2_slice(&h_decomposed.flat);
    if u2 != &expected {
        return Err("equation (4): u2 ≠ D·h".into());
    }
    Ok(())
}

/// Helper: derive r from garbage count.
fn challenges_count_from_g(garbage_len: usize) -> usize {
    // Solve r*(r+1)/2 = garbage_len for integer r.
    // r = (-1 + sqrt(1 + 8*n)) / 2, using integer arithmetic.
    let disc = 1 + 8 * garbage_len as u64;
    (disc.isqrt().saturating_sub(1) / 2) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    use grid_algebra::arith::prime::PrimeField;
    use grid_algebra::arith::ring::Ring;

    type F = PrimeField<12289>;

    #[test]
    fn test_challenges_count_from_g() {
        assert_eq!(challenges_count_from_g(1), 1);
        assert_eq!(challenges_count_from_g(3), 2);
        assert_eq!(challenges_count_from_g(6), 3);
        assert_eq!(challenges_count_from_g(10), 4);
        assert_eq!(challenges_count_from_g(55), 10);
    }

    #[test]
    fn test_verify_eq_2b_zero() {
        let z: Vec<CyclotomicPolyRing<F, 64>> = vec![CyclotomicPolyRing::<F, 64>::zero(); 4];
        let result = verify_eq_2b::<F, 64>(&z, 1.0);
        assert!(result.is_ok(), "zero z should pass norm check");
    }

    #[test]
    fn test_verify_eq_3a_zero() {
        let r = 3;
        let z: Vec<CyclotomicPolyRing<F, 64>> = vec![CyclotomicPolyRing::<F, 64>::zero(); 4];
        let g_polys: Vec<CyclotomicPolyRing<F, 64>> =
            vec![CyclotomicPolyRing::<F, 64>::zero(); r * (r + 1) / 2];
        let challenges: Vec<CyclotomicPolyRing<F, 64>> =
            vec![CyclotomicPolyRing::<F, 64>::one(); r];
        let result = verify_eq_3a::<F, 64>(&z, &g_polys, &challenges);
        assert!(result.is_ok(), "zero z and zero g should pass (3a)");
    }
}
