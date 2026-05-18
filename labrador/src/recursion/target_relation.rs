//! Target relation G for the next recursion level (§5.3).
//!
//! Builds the family of K' = κ + κ₁ + κ₂ + 3 dot product constraints
//! from the verification equations of the current level. The constraints
//! encode: inner commitment check, outer commitment checks, and the
//! three separate garbage equations (3a), (3b), (3c).
//!
//! # Witness layout
//!
//! Parts 0..ν: z⁰ split into ν chunks of rank n'
//! Parts ν..2ν: z¹ split into ν chunks of rank n'
//! Parts 2ν..(2ν+μ): v = t‖g‖h split into μ chunks of rank n'
//!
//! Within v (flat offset):
//! - [0 .. r·κ·t₁]: t limbs
//! - [r·κ·t₁ .. r·κ·t₁ + garbage_len·t₂]: g limbs
//! - [r·κ·t₁ + garbage_len·t₂ .. r·κ·t₁ + garbage_len·(t₂+t₁)]: h limbs

use alloc::vec;
use alloc::vec::Vec;

use grid_algebra::arith::ring::IntegerRing;
use grid_algebra::arith::ring::Ring;
use grid_algebra::lattice::types::{RingMat, RingVec};
use grid_algebra::poly::ring::PolyRing;
use grid_algebra::poly::ring::{CyclotomicPolyRing, NegacyclicMulRing};
use grid_std::UniformRand;

use crate::crs::CommitKey;
use crate::error::LabradorError;
use crate::main_protocol::aggregation::AggregatedFunction;
use crate::main_protocol::{garbage_count, garbage_index};
use crate::params::LabradorParams;
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

/// The target relation G for the next recursion level.
#[derive(Debug, Clone)]
pub struct RecursiveTarget<R, const N: usize>
where
    R: IntegerRing<Uint = u64> + NegacyclicMulRing<N> + UniformRand,
{
    /// Family G: K' constraints (all fully vanishing, no F').
    pub statement: LabradorStatement<CyclotomicPolyRing<R, N>>,
    /// Consolidated norm bound β' = √(2/b²·γ² + γ₁² + γ₂²).
    pub beta_prime: f64,
    /// Next-level witness (split into r' parts of rank n').
    pub witness: LabradorWitness<CyclotomicPolyRing<R, N>>,
    /// Next-level multiplicity r' = 2ν + μ.
    pub r_prime: usize,
    /// Next-level rank n'.
    pub n_prime: usize,
}

/// Build the target relation G from main protocol outputs.
///
/// Constructs K' = κ + κ₁ + κ₂ + 3 quadratic constraints:
///
/// 1. **Inner commitment** (κ): A·(z⁰+bz¹) - Σcᵢ·Σl limb_l·b1^l = 0
/// 2. **Outer commitment u1** (κ₁): B·t + C·g - u1 = 0
/// 3. **Outer commitment u2** (κ₂): D·h - u2 = 0
/// 4. **Garbage (3a)** (1): ⟨z,z⟩ - Σgᵢⱼcᵢcⱼ = 0
/// 5. **Garbage (3b)** (1): Σ⟨φᵢ,z⟩cᵢ - Σhᵢⱼcᵢcⱼ = 0
/// 6. **Garbage (3c)** (1): Σaᵢⱼgᵢⱼ + Σhᵢᵢ - b = 0
#[allow(clippy::too_many_arguments)]
pub fn build_target_relation<R, const N: usize>(
    key: &CommitKey<R, N>,
    u1: &RingVec<CyclotomicPolyRing<R, N>>,
    u2: &RingVec<CyclotomicPolyRing<R, N>>,
    challenges: &[CyclotomicPolyRing<R, N>],
    aggregated: &AggregatedFunction<R, N>,
    params: &LabradorParams,
    witness: LabradorWitness<CyclotomicPolyRing<R, N>>,
    r_prime: usize,
    n_prime: usize,
) -> RecursiveTarget<R, N>
where
    R: IntegerRing<Uint = u64> + NegacyclicMulRing<N> + UniformRand,
{
    let nu = params.nu;
    let mu = params.mu;
    let kappa = params.kappa;
    let kappa1 = params.kappa1;
    let kappa2 = params.kappa2;
    let b = params.b;
    let r = params.r;
    let t1 = params.t1;
    let t2 = params.t2;
    let b1 = params.b1;
    let b2 = params.b2;
    let garbage_len = garbage_count(r);

    let t_end = r * kappa * t1;
    let g_len = garbage_len * t2;
    let g_end = t_end + g_len;
    let h_len = garbage_len * t1;
    let m = t_end + g_len + h_len; // total v length: t‖g‖h
    let v_per_part = m.div_ceil(mu);

    let zero_poly = CyclotomicPolyRing::<R, N>::zero();
    let b1_powers = scalar_powers(&R::from_u64(b1), t1);
    let b2_powers = scalar_powers(&R::from_u64(b2), t2);
    let mut f_functions = Vec::with_capacity(kappa + kappa1 + kappa2 + 3);

    // --- 1. Inner commitment (κ constraints) ---
    for d in 0..kappa {
        let phi = build_inner_commitment_phi(
            key, challenges, d, nu, mu, r, kappa, t1, v_per_part, n_prime, b, &b1_powers,
        );
        f_functions.push(QuadraticFunction::from_parts(
            Vec::new(),
            phi,
            zero_poly.clone(),
        ));
    }

    // --- 2. Outer commitment u1 (κ₁ constraints) ---
    for d in 0..kappa1 {
        let phi = build_outer_u1_phi(
            key,
            d,
            nu,
            r_prime,
            n_prime,
            t_end,
            g_len,
            v_per_part,
            2 * nu,
        );
        let b_const = u1.entries()[d].clone();
        f_functions.push(QuadraticFunction::from_parts(Vec::new(), phi, b_const));
    }

    // --- 3. Outer commitment u2 (κ₂ constraints) ---
    for d in 0..kappa2 {
        let phi = build_outer_u2_phi(
            key,
            d,
            nu,
            r_prime,
            n_prime,
            g_end,
            h_len,
            v_per_part,
            2 * nu,
        );
        let b_const = u2.entries()[d].clone();
        f_functions.push(QuadraticFunction::from_parts(Vec::new(), phi, b_const));
    }

    // --- 4. Garbage (3a): ⟨z,z⟩ = Σgᵢⱼcᵢcⱼ ---
    let (quad_3a, phi_3a, b_3a) = build_garbage_3a(
        challenges, nu, r_prime, r, t_end, t2, v_per_part, n_prime, b, &b2_powers,
    );
    f_functions.push(QuadraticFunction::from_parts(quad_3a, phi_3a, b_3a));

    // --- 5. Garbage (3b): Σ⟨φᵢ,z⟩cᵢ = Σhᵢⱼcᵢcⱼ ---
    let (quad_3b, phi_3b, b_3b) = build_garbage_3b(
        challenges, aggregated, nu, r_prime, r, t1, g_end, v_per_part, n_prime, b, &b1_powers,
    );
    f_functions.push(QuadraticFunction::from_parts(quad_3b, phi_3b, b_3b));

    // --- 6. Garbage (3c): Σaᵢⱼgᵢⱼ + Σhᵢᵢ - b = 0 ---
    let (quad_3c, phi_3c, b_3c) = build_garbage_3c(
        aggregated,
        nu,
        r_prime,
        r,
        t1,
        t2,
        t_end,
        g_end,
        v_per_part,
        n_prime,
        &b1_powers,
        &b2_powers,
        2 * nu,
    );
    f_functions.push(QuadraticFunction::from_parts(quad_3c, phi_3c, b_3c));

    let statement = LabradorStatement {
        f: f_functions,
        f_prime: vec![],
    };

    RecursiveTarget {
        statement,
        beta_prime: params.beta_prime,
        witness,
        r_prime,
        n_prime,
    }
}

/// Build φ for inner commitment constraint dimension d.
///
/// A·(z⁰+bz¹) - Σcᵢtᵢ = 0, where tᵢ is reconstructed from base-b1 limbs.
/// Limb l of t_{wi,dim}: coefficient -cᵢ * b1^l.
#[allow(clippy::too_many_arguments)]
fn build_inner_commitment_phi<R, const N: usize>(
    key: &CommitKey<R, N>,
    challenges: &[CyclotomicPolyRing<R, N>],
    dim: usize,
    nu: usize,
    mu: usize,
    r: usize,
    kappa: usize,
    t1: usize,
    v_per_part: usize,
    n_prime: usize,
    b: u64,
    b1_powers: &[R],
) -> Vec<Vec<CyclotomicPolyRing<R, N>>>
where
    R: IntegerRing<Uint = u64> + NegacyclicMulRing<N> + UniformRand,
{
    let r_prime = 2 * nu + mu;
    let n = key.a.cols();
    let zero = CyclotomicPolyRing::<R, N>::zero();
    let mut phi = vec![vec![zero.clone(); n_prime]; r_prime];

    let b_scalar = R::from_u64(b);
    let z_per_part = n.div_ceil(nu);

    #[allow(clippy::needless_range_loop)]
    for part_i in 0..nu {
        let start_k = part_i * z_per_part;
        let end_k = (start_k + z_per_part).min(n);
        for (local_j, k) in (start_k..end_k).enumerate() {
            phi[part_i][local_j] = key.a.entries()[dim * n + k].clone();
        }
    }

    for part_i in 0..nu {
        let w_part = nu + part_i;
        let start_k = part_i * z_per_part;
        let end_k = (start_k + z_per_part).min(n);
        for (local_j, k) in (start_k..end_k).enumerate() {
            phi[w_part][local_j] = key.a.entries()[dim * n + k].scalar_mul(&b_scalar);
        }
    }

    for (wi, c_i) in challenges.iter().enumerate().take(r) {
        for (l, base_pow) in b1_powers.iter().enumerate() {
            let limb_flat_idx = wi * kappa * t1 + dim * t1 + l;
            let (w_part, local_j) = v_offset_to_witness_part(limb_flat_idx, nu, v_per_part);
            debug_assert!(
                phi[w_part][local_j].is_zero(),
                "collision in phi write: w_part={}, local_j={} (stride kappa*t1={t1}, l={l})",
                w_part,
                local_j
            );
            phi[w_part][local_j] = -c_i.scalar_mul(base_pow);
        }
    }

    phi
}

/// Build φ for outer u1 constraint row d.
#[allow(clippy::too_many_arguments)]
fn build_outer_u1_phi<R, const N: usize>(
    key: &CommitKey<R, N>,
    dim: usize,
    _nu: usize,
    r_prime: usize,
    n_prime: usize,
    t_end: usize,
    g_len: usize,
    v_per_part: usize,
    v_start: usize,
) -> Vec<Vec<CyclotomicPolyRing<R, N>>>
where
    R: IntegerRing<Uint = u64> + NegacyclicMulRing<N> + UniformRand,
{
    let zero = CyclotomicPolyRing::<R, N>::zero();
    let mut phi = vec![vec![zero.clone(); n_prime]; r_prime];

    let cols_b = key.b.cols();
    let cols_c = key.c.cols();
    assert!(
        cols_b == t_end,
        "B columns ({cols_b}) != expected t limbs ({t_end})"
    );
    assert!(
        cols_c == g_len,
        "C columns ({cols_c}) != expected g limbs ({g_len})"
    );

    for l in 0..t_end {
        let (w_part, local_j) = v_offset_to_witness_part_base(l, v_start, v_per_part);
        phi[w_part][local_j] = key.b.entries()[dim * cols_b + l].clone();
    }

    for l in 0..g_len {
        let v_offset = t_end + l;
        let (w_part, local_j) = v_offset_to_witness_part_base(v_offset, v_start, v_per_part);
        phi[w_part][local_j] = key.c.entries()[dim * cols_c + l].clone();
    }

    phi
}

/// Build φ for outer u2 constraint row d.
#[allow(clippy::too_many_arguments)]
fn build_outer_u2_phi<R, const N: usize>(
    key: &CommitKey<R, N>,
    dim: usize,
    _nu: usize,
    r_prime: usize,
    n_prime: usize,
    g_end: usize,
    h_len: usize,
    v_per_part: usize,
    v_start: usize,
) -> Vec<Vec<CyclotomicPolyRing<R, N>>>
where
    R: IntegerRing<Uint = u64> + NegacyclicMulRing<N> + UniformRand,
{
    let zero = CyclotomicPolyRing::<R, N>::zero();
    let mut phi = vec![vec![zero.clone(); n_prime]; r_prime];

    let cols_d = key.d.cols();
    assert!(
        cols_d == h_len,
        "D columns ({cols_d}) != expected h limbs ({h_len})"
    );

    for l in 0..h_len {
        let v_offset = g_end + l;
        let (w_part, local_j) = v_offset_to_witness_part_base(v_offset, v_start, v_per_part);
        phi[w_part][local_j] = key.d.entries()[dim * cols_d + l].clone();
    }

    phi
}

/// Build garbage equation (3a): ⟨z,z⟩ - Σgᵢⱼcᵢcⱼ = 0
///
/// After z = z⁰ + bz¹:
/// ⟨z⁰,z⁰⟩ + 2b·⟨z⁰,z¹⟩ + b²·⟨z¹,z¹⟩ - Σgᵢⱼcᵢcⱼ = 0
///
/// Quadratic: z⁰ diagonal, z⁰/z¹ corresponding-chunk cross, z¹ diagonal
/// Linear: -g limbs (off-diagonal doubled for symmetric sum), each limb k weighted by b2^k
/// Constant: 0
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn build_garbage_3a<R, const N: usize>(
    challenges: &[CyclotomicPolyRing<R, N>],
    nu: usize,
    r_prime: usize,
    r: usize,
    t_end: usize,
    t2: usize,
    v_per_part: usize,
    n_prime: usize,
    b: u64,
    b2_powers: &[R],
) -> (
    Vec<(usize, usize, CyclotomicPolyRing<R, N>)>,
    Vec<Vec<CyclotomicPolyRing<R, N>>>,
    CyclotomicPolyRing<R, N>,
)
where
    R: IntegerRing<Uint = u64> + NegacyclicMulRing<N> + UniformRand,
{
    let zero = CyclotomicPolyRing::<R, N>::zero();
    let mut quad = Vec::new();
    let mut phi = vec![vec![zero.clone(); n_prime]; r_prime];

    let b_scalar = R::from_u64(b);
    let two_b = R::from_u64(2) * b_scalar.clone();
    let b_sq = b_scalar.clone() * b_scalar.clone();

    // ⟨z⁰,z⁰⟩: diagonal-only
    for part_i in 0..nu {
        quad.push((part_i, part_i, CyclotomicPolyRing::<R, N>::one()));
    }

    // 2b·⟨z⁰,z¹⟩: corresponding-chunk cross terms
    for p in 0..nu {
        quad.push((p, nu + p, {
            let mut cp = CyclotomicPolyRing::<R, N>::zero();
            cp.set_coeff(0, two_b.clone());
            cp
        }));
    }

    // b²·⟨z¹,z¹⟩: diagonal-only
    for part_i in 0..nu {
        quad.push((nu + part_i, nu + part_i, {
            let mut cp = CyclotomicPolyRing::<R, N>::zero();
            cp.set_coeff(0, b_sq.clone());
            cp
        }));
    }

    // Linear: -g limbs (off-diagonal doubled), each limb k weighted by b2^k
    for i in 0..r {
        for j in i..r {
            let g_idx = garbage_index(r, i, j);
            let c_prod = &challenges[i] * &challenges[j];

            let base_coeff: CyclotomicPolyRing<R, N> = if i == j {
                -c_prod
            } else {
                -(&c_prod + &c_prod)
            };

            for (k, base_pow) in b2_powers.iter().enumerate() {
                let v_offset = t_end + g_idx * t2 + k;
                let (w_part, local_j) = v_offset_to_witness_part(v_offset, nu, v_per_part);
                phi[w_part][local_j] += base_coeff.scalar_mul(base_pow);
            }
        }
    }

    (quad, phi, zero)
}

/// Build garbage equation (3b): Σ⟨φᵢ,z⟩cᵢ - Σhᵢⱼcᵢcⱼ = 0
///
/// No quadratic terms. Linear in z parts (via aggregated φ scaled by challenges and b),
/// linear in h limbs (off-diagonal doubled), each limb k weighted by b1^k.
/// Constant: 0
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn build_garbage_3b<R, const N: usize>(
    challenges: &[CyclotomicPolyRing<R, N>],
    aggregated: &AggregatedFunction<R, N>,
    nu: usize,
    r_prime: usize,
    r: usize,
    t1: usize,
    g_end: usize,
    v_per_part: usize,
    n_prime: usize,
    b: u64,
    b1_powers: &[R],
) -> (
    Vec<(usize, usize, CyclotomicPolyRing<R, N>)>,
    Vec<Vec<CyclotomicPolyRing<R, N>>>,
    CyclotomicPolyRing<R, N>,
)
where
    R: IntegerRing<Uint = u64> + NegacyclicMulRing<N> + UniformRand,
{
    let zero = CyclotomicPolyRing::<R, N>::zero();
    assert!(
        !aggregated.phi.is_empty(),
        "aggregated.phi must be non-empty"
    );
    let n = aggregated.phi[0].len();
    let z_per_part = n.div_ceil(nu);

    let quad: Vec<(usize, usize, CyclotomicPolyRing<R, N>)> = Vec::new();
    let mut phi = vec![vec![zero.clone(); n_prime]; r_prime];

    let b_scalar = R::from_u64(b);

    // Linear: -h limbs (off-diagonal doubled), each limb k weighted by b1^k
    for i in 0..r {
        for j in i..r {
            let h_idx = garbage_index(r, i, j);
            let c_prod = &challenges[i] * &challenges[j];

            let base_coeff: CyclotomicPolyRing<R, N> = if i == j {
                -c_prod
            } else {
                -(&c_prod + &c_prod)
            };

            for (k, base_pow) in b1_powers.iter().enumerate() {
                let v_offset = g_end + h_idx * t1 + k;
                let (w_part, local_j) = v_offset_to_witness_part(v_offset, nu, v_per_part);
                phi[w_part][local_j] += base_coeff.scalar_mul(base_pow);
            }
        }
    }

    // Linear: aggregated φ terms Σ⟨φᵢ,z⟩cᵢ
    // ⟨φᵢ, z⟩ = ⟨φᵢ, z⁰ + bz¹⟩ = ⟨φᵢ, z⁰⟩ + b·⟨φᵢ, z¹⟩
    for (c_i, phi_i) in challenges.iter().take(r).zip(aggregated.phi.iter().take(r)) {
        let phi_i_scaled: Vec<CyclotomicPolyRing<R, N>> = phi_i.iter().map(|p| p * c_i).collect();
        let phi_i_scaled_b: Vec<CyclotomicPolyRing<R, N>> = phi_i_scaled
            .iter()
            .map(|p| p.scalar_mul(&b_scalar))
            .collect();

        for (j, phi_coeff) in phi_i_scaled.iter().enumerate().take(n) {
            let part_i = j / z_per_part;
            let local_j = j % z_per_part;
            if part_i < nu {
                phi[part_i][local_j] += phi_coeff;
            }
        }

        for (j, phi_coeff) in phi_i_scaled_b.iter().enumerate().take(n) {
            let part_i = nu + j / z_per_part;
            let local_j = j % z_per_part;
            if part_i < 2 * nu {
                phi[part_i][local_j] += phi_coeff;
            }
        }
    }

    (quad, phi, zero)
}

/// Build garbage equation (3c): Σaᵢⱼgᵢⱼ + Σhᵢᵢ - b = 0
///
/// No quadratic terms. Linear in g limbs (via aᵢⱼ, each limb k weighted by b2^k),
/// linear in h diagonal limbs (each limb k weighted by b1^k).
/// Constant: b (QuadraticFunction form subtracts b, so field = +b for equation - b)
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn build_garbage_3c<R, const N: usize>(
    aggregated: &AggregatedFunction<R, N>,
    _nu: usize,
    r_prime: usize,
    r: usize,
    t1: usize,
    t2: usize,
    t_end: usize,
    g_end: usize,
    v_per_part: usize,
    n_prime: usize,
    b1_powers: &[R],
    b2_powers: &[R],
    v_start: usize,
) -> (
    Vec<(usize, usize, CyclotomicPolyRing<R, N>)>,
    Vec<Vec<CyclotomicPolyRing<R, N>>>,
    CyclotomicPolyRing<R, N>,
)
where
    R: IntegerRing<Uint = u64> + NegacyclicMulRing<N> + UniformRand,
{
    let zero = CyclotomicPolyRing::<R, N>::zero();

    let quad: Vec<(usize, usize, CyclotomicPolyRing<R, N>)> = Vec::new();
    let mut phi = vec![vec![zero.clone(); n_prime]; r_prime];

    // Linear: g limbs Σaᵢⱼgᵢⱼ, each limb k weighted by b2^k
    for &(i, j, ref a_coeff) in &aggregated.a_ij {
        let g_idx = garbage_index(r, i, j);
        for (k, base_pow) in b2_powers.iter().enumerate() {
            let v_offset = t_end + g_idx * t2 + k;
            let (w_part, local_j) = v_offset_to_witness_part_base(v_offset, v_start, v_per_part);
            phi[w_part][local_j] += a_coeff.scalar_mul(base_pow);
        }
    }

    // Linear: h diagonal Σhᵢᵢ, each limb k weighted by b1^k
    for i in 0..r {
        let h_ii_idx = garbage_index(r, i, i);
        for (k, base_pow) in b1_powers.iter().enumerate() {
            let v_offset = g_end + h_ii_idx * t1 + k;
            let (w_part, local_j) = v_offset_to_witness_part_base(v_offset, v_start, v_per_part);
            phi[w_part][local_j] += {
                let mut cp = CyclotomicPolyRing::<R, N>::zero();
                cp.set_coeff(0, base_pow.clone());
                cp
            };
        }
    }

    // Constant: b (QuadraticFunction subtracts b, so field = +b)
    let b_const = aggregated.b.clone();

    (quad, phi, b_const)
}

/// Map a flat v offset to (witness_part_index, local_index_within_part).
///
/// Uses `v_per_part` (the actual chunk size v was split into), NOT `n_prime`.
/// split_witness divides v into `v_per_part = m.div_ceil(mu)` chunks and then
/// pads each to `n_prime`. Using `n_prime` here would misalign coefficients
/// when `n_prime > v_per_part` (i.e., when the z side dominates).
#[inline]
fn v_offset_to_witness_part(v_offset: usize, nu: usize, v_per_part: usize) -> (usize, usize) {
    v_offset_to_witness_part_base(v_offset, 2 * nu, v_per_part)
}

/// Map a flat v offset to (witness_part_index, local_index_within_part) for the
/// **last level**. Unlike the regular target, the last level has z NOT decomposed,
/// so v-parts start at `nu` (not `2*nu`).
#[inline]
fn v_offset_to_last_level_part(v_offset: usize, nu: usize, v_per_part: usize) -> (usize, usize) {
    v_offset_to_witness_part_base(v_offset, nu, v_per_part)
}

/// General v-offset to witness-part mapper with configurable base.
#[inline]
fn v_offset_to_witness_part_base(
    v_offset: usize,
    v_start: usize,
    v_per_part: usize,
) -> (usize, usize) {
    let w_part = v_start + v_offset / v_per_part;
    let local_j = v_offset % v_per_part;
    (w_part, local_j)
}

/// Precompute [base^0, base^1, ..., base^{max_pow - 1}] as scalars.
fn scalar_powers<R>(base: &R, max_pow: usize) -> Vec<R>
where
    R: Ring + UniformRand,
{
    let mut powers = Vec::with_capacity(max_pow);
    let mut cur = R::one();
    for _ in 0..max_pow {
        powers.push(cur.clone());
        cur *= base;
    }
    powers
}

/// Build the target relation G for the **last** recursion level (§5.6).
///
/// Same K' = κ + κ₁ + κ₂ + 3 constraint structure as regular target,
/// but z is NOT decomposed: r = ν + μ (not 2ν + μ). This simplifies:
/// - Inner commitment φ: no b-scaled z¹ copy
/// - Garbage (3a): only ν diagonal terms (no z⁰/z¹ cross)
/// - Garbage (3b): no b-scaled φ terms
///
/// Norm bound β'' = √(γ² + γ₁² + γ₂²) — consolidated from previous level
/// (no 2/b² factor since z isn't decomposed).
/// Builds the target relation for the last recursion level (§5.6).
///
/// `source_crs` provides A/B/C/D for the target's inner/outer constraints.
/// The inner commitment uses `source_crs.a` (from the step where t limbs were
/// committed), since the witness's v-parts contain the source step's t limbs.
/// `a_last` (the last-level protocol's own inner commitment matrix) is NOT part
/// of this target relation; it is only used by the final protocol's Check (1).
///
/// # Preconditions
///
/// - `r_last == nu + mu` (witness has ν z-parts and μ v-parts)
/// - `n_last > 0` (polynomial degree is positive)
/// - `challenges.len() == source_r` (matches previous level's challenge count)
///
/// # Panics
///
/// Panics if any precondition is violated.
#[allow(clippy::too_many_arguments)]
pub fn build_last_level_target<R, const N: usize>(
    source_key: &CommitKey<R, N>,
    u1: &RingVec<CyclotomicPolyRing<R, N>>,
    u2: &RingVec<CyclotomicPolyRing<R, N>>,
    challenges: &[CyclotomicPolyRing<R, N>],
    aggregated: &AggregatedFunction<R, N>,
    params: &LabradorParams,
    witness: LabradorWitness<CyclotomicPolyRing<R, N>>,
    r_last: usize,
    n_last: usize,
    source_r: usize,
) -> Result<RecursiveTarget<R, N>, LabradorError>
where
    R: IntegerRing<Uint = u64> + NegacyclicMulRing<N> + UniformRand,
{
    let nu = params.nu;
    let mu = params.mu;
    bail!(
        r_last != nu + mu,
        "r_last ({}) must equal nu + mu ({})",
        r_last,
        nu + mu
    );
    bail!(n_last == 0, "n_last must be > 0");
    bail!(
        challenges.len() != source_r,
        "challenges length ({}) != source_r ({})",
        challenges.len(),
        source_r
    );
    let kappa = params.kappa;
    let kappa1 = params.kappa1;
    let kappa2 = params.kappa2;
    let t1 = params.t1;
    let t2 = params.t2;
    let b1 = params.b1;
    let b2 = params.b2;

    // t/g/h limb layout is determined by the PREVIOUS step's challenge count (source_r),
    // NOT the last-level target witness count (r_last = ν + μ).
    let garbage_len = garbage_count(source_r);

    let t_end = source_r * kappa * t1;
    let g_len = garbage_len * t2;
    let g_end = t_end + g_len;
    let h_len = garbage_len * t1;
    let m = t_end + g_len + h_len;
    let v_per_part = m.div_ceil(mu);

    let zero_poly = CyclotomicPolyRing::<R, N>::zero();
    let b1_powers = scalar_powers(&R::from_u64(b1), t1);
    let b2_powers = scalar_powers(&R::from_u64(b2), t2);
    let mut f_functions = Vec::with_capacity(kappa + kappa1 + kappa2 + 3);

    // --- 1. Inner commitment (κ constraints) ---
    // Simplified: no z⁰/z¹ split, just A·z = Σcᵢtᵢ
    // Uses source_crs.a (from the step where t limbs were committed), since the
    // witness's v-parts contain the source step's t limbs, not a_last-based ones.
    for d in 0..kappa {
        let phi = build_last_level_inner_phi(
            &source_key.a,
            challenges,
            d,
            nu,
            mu,
            source_r,
            kappa,
            t1,
            v_per_part,
            n_last,
            &b1_powers,
        );
        f_functions.push(QuadraticFunction::from_parts(
            Vec::new(),
            phi,
            zero_poly.clone(),
        ));
    }

    // --- 2. Outer commitment u1 (κ₁ constraints) ---
    // Uses source_crs.b and source_crs.c (from the step where u1 was computed).
    for d in 0..kappa1 {
        let phi = build_outer_u1_phi(
            source_key, d, nu, r_last, n_last, t_end, g_len, v_per_part, nu,
        );
        let b_const = u1.entries()[d].clone();
        f_functions.push(QuadraticFunction::from_parts(Vec::new(), phi, b_const));
    }

    // --- 3. Outer commitment u2 (κ₂ constraints) ---
    // Uses source_crs.d (from the step where u2 was computed).
    for d in 0..kappa2 {
        let phi = build_outer_u2_phi(
            source_key, d, nu, r_last, n_last, g_end, h_len, v_per_part, nu,
        );
        let b_const = u2.entries()[d].clone();
        f_functions.push(QuadraticFunction::from_parts(Vec::new(), phi, b_const));
    }

    // --- 4. Garbage (3a): ⟨z,z⟩ = Σgᵢⱼcᵢcⱼ ---
    // Simplified: only diagonal z terms (no decomposition)
    let (quad_3a, phi_3a, b_3a) = build_last_level_garbage_3a(
        challenges, nu, r_last, source_r, t_end, t2, v_per_part, n_last, &b2_powers,
    );
    f_functions.push(QuadraticFunction::from_parts(quad_3a, phi_3a, b_3a));

    // --- 5. Garbage (3b): Σ⟨φᵢ,z⟩cᵢ = Σhᵢⱼcᵢcⱼ ---
    // Simplified: no b scaling on φ terms
    let (quad_3b, phi_3b, b_3b) = build_last_level_garbage_3b(
        challenges, aggregated, nu, r_last, source_r, t1, g_end, v_per_part, n_last, &b1_powers,
    );
    f_functions.push(QuadraticFunction::from_parts(quad_3b, phi_3b, b_3b));

    // --- 6. Garbage (3c): Σaᵢⱼgᵢⱼ + Σhᵢᵢ - b = 0 ---
    // Same as regular target
    let (quad_3c, phi_3c, b_3c) = build_garbage_3c(
        aggregated, nu, r_last, source_r, t1, t2, t_end, g_end, v_per_part, n_last, &b1_powers,
        &b2_powers, nu,
    );
    f_functions.push(QuadraticFunction::from_parts(quad_3c, phi_3c, b_3c));

    let statement = LabradorStatement {
        f: f_functions,
        f_prime: vec![],
    };

    // β'' = √(γ² + γ₁² + γ₂²) — no 2/b² factor vs regular β'
    // The last-level protocol does not decompose z (witness used directly),
    // so the recursed witness norm is bounded by √(γ² + γ₁² + γ₂²)
    // rather than √(2/b²·γ² + γ₁² + γ₂²) which accounts for z = z⁰ + b·z¹ decomposition.
    let beta_double_prime = grid_std::sqrt(
        params.gamma * params.gamma + params.gamma1_sq as f64 + params.gamma2_sq as f64,
    );

    Ok(RecursiveTarget {
        statement,
        beta_prime: beta_double_prime,
        witness,
        r_prime: r_last,
        n_prime: n_last,
    })
}

/// Build φ for inner commitment in last level.
///
/// Simplified from regular target: z is not decomposed, so only
/// ν z-parts contribute (not 2ν). A·z = Σcᵢtᵢ.
/// `a` is the inner commitment matrix with κ × n columns (generated separately).
#[allow(clippy::too_many_arguments)]
fn build_last_level_inner_phi<R, const N: usize>(
    a: &RingMat<CyclotomicPolyRing<R, N>>,
    challenges: &[CyclotomicPolyRing<R, N>],
    dim: usize,
    nu: usize,
    mu: usize,
    r: usize,
    kappa: usize,
    t1: usize,
    v_per_part: usize,
    n_prime: usize,
    b1_powers: &[R],
) -> Vec<Vec<CyclotomicPolyRing<R, N>>>
where
    R: IntegerRing<Uint = u64> + NegacyclicMulRing<N> + UniformRand,
{
    let n = a.cols();
    let zero = CyclotomicPolyRing::<R, N>::zero();
    let mut phi = vec![vec![zero.clone(); n_prime]; nu + mu];

    let z_per_part = n.div_ceil(nu);

    // z-parts (indices 0..nu): A rows
    #[allow(clippy::needless_range_loop)]
    for part_i in 0..nu {
        let start_k = part_i * z_per_part;
        let end_k = (start_k + z_per_part).min(n);
        for (local_j, k) in (start_k..end_k).enumerate() {
            phi[part_i][local_j] = a.entries()[dim * n + k].clone();
        }
    }

    // t limbs in v-parts (indices nu..nu+mu): -cᵢ·b1^l
    for (wi, c_i) in challenges.iter().enumerate().take(r) {
        for (l, base_pow) in b1_powers.iter().enumerate() {
            let limb_flat_idx = wi * kappa * t1 + dim * t1 + l;
            let (w_part, local_j) = v_offset_to_last_level_part(limb_flat_idx, nu, v_per_part);
            debug_assert!(
                phi[w_part][local_j].is_zero(),
                "collision in phi write: w_part={}, local_j={} (stride kappa*t1={t1}, l={l})",
                w_part,
                local_j
            );
            phi[w_part][local_j] = -c_i.scalar_mul(base_pow);
        }
    }

    phi
}

/// Build garbage (3a) for last level: ⟨z,z⟩ = Σgᵢⱼcᵢcⱼ.
///
/// Simplified: no z decomposition, so only ν diagonal terms.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn build_last_level_garbage_3a<R, const N: usize>(
    challenges: &[CyclotomicPolyRing<R, N>],
    nu: usize,
    r_last: usize,
    r: usize,
    t_end: usize,
    t2: usize,
    v_per_part: usize,
    n_prime: usize,
    b2_powers: &[R],
) -> (
    Vec<(usize, usize, CyclotomicPolyRing<R, N>)>,
    Vec<Vec<CyclotomicPolyRing<R, N>>>,
    CyclotomicPolyRing<R, N>,
)
where
    R: IntegerRing<Uint = u64> + NegacyclicMulRing<N> + UniformRand,
{
    let zero = CyclotomicPolyRing::<R, N>::zero();
    let mut quad = Vec::new();
    let mut phi = vec![vec![zero.clone(); n_prime]; r_last];

    // ⟨z,z⟩: only diagonal for z-parts (no decomposition)
    for part_i in 0..nu {
        quad.push((part_i, part_i, CyclotomicPolyRing::<R, N>::one()));
    }

    // Linear: -g limbs (off-diagonal doubled), each limb k weighted by b2^k
    for i in 0..r {
        for j in i..r {
            let g_idx = garbage_index(r, i, j);
            let c_prod = &challenges[i] * &challenges[j];

            let base_coeff: CyclotomicPolyRing<R, N> = if i == j {
                -c_prod
            } else {
                -(&c_prod + &c_prod)
            };

            for (k, base_pow) in b2_powers.iter().enumerate() {
                let v_offset = t_end + g_idx * t2 + k;
                let (w_part, local_j) = v_offset_to_last_level_part(v_offset, nu, v_per_part);
                phi[w_part][local_j] += base_coeff.scalar_mul(base_pow);
            }
        }
    }

    (quad, phi, zero)
}

/// Build garbage (3b) for last level: Σ⟨φᵢ,z⟩cᵢ = Σhᵢⱼcᵢcⱼ.
///
/// Simplified: no b scaling on φ terms (z is not decomposed).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn build_last_level_garbage_3b<R, const N: usize>(
    challenges: &[CyclotomicPolyRing<R, N>],
    aggregated: &AggregatedFunction<R, N>,
    nu: usize,
    r_last: usize,
    r: usize,
    t1: usize,
    g_end: usize,
    v_per_part: usize,
    n_prime: usize,
    b1_powers: &[R],
) -> (
    Vec<(usize, usize, CyclotomicPolyRing<R, N>)>,
    Vec<Vec<CyclotomicPolyRing<R, N>>>,
    CyclotomicPolyRing<R, N>,
)
where
    R: IntegerRing<Uint = u64> + NegacyclicMulRing<N> + UniformRand,
{
    let zero = CyclotomicPolyRing::<R, N>::zero();
    assert!(
        !aggregated.phi.is_empty(),
        "aggregated.phi must be non-empty"
    );
    let n = aggregated.phi[0].len();
    let z_per_part = n.div_ceil(nu);

    let quad: Vec<(usize, usize, CyclotomicPolyRing<R, N>)> = Vec::new();
    let mut phi = vec![vec![zero.clone(); n_prime]; r_last];

    // Linear: -h limbs (off-diagonal doubled), each limb k weighted by b1^k
    for i in 0..r {
        for j in i..r {
            let h_idx = garbage_index(r, i, j);
            let c_prod = &challenges[i] * &challenges[j];

            let base_coeff: CyclotomicPolyRing<R, N> = if i == j {
                -c_prod
            } else {
                -(&c_prod + &c_prod)
            };

            for (k, base_pow) in b1_powers.iter().enumerate() {
                let v_offset = g_end + h_idx * t1 + k;
                let (w_part, local_j) = v_offset_to_last_level_part(v_offset, nu, v_per_part);
                phi[w_part][local_j] += base_coeff.scalar_mul(base_pow);
            }
        }
    }

    // Linear: aggregated φ terms Σ⟨φᵢ,z⟩cᵢ (no b scaling)
    for (c_i, phi_i) in challenges.iter().take(r).zip(aggregated.phi.iter().take(r)) {
        let phi_i_scaled: Vec<CyclotomicPolyRing<R, N>> = phi_i.iter().map(|p| p * c_i).collect();

        for (j, phi_coeff) in phi_i_scaled.iter().enumerate().take(n) {
            let part_i = j / z_per_part;
            let local_j = j % z_per_part;
            if part_i < nu {
                phi[part_i][local_j] += phi_coeff;
            }
        }
    }

    (quad, phi, zero)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    use crate::params::{ChallengeProfile, JLProfile};
    use crate::recursion::decompose::{bundle_v, decompose_z};
    use crate::recursion::split::{compute_next_level_shape, split_witness};
    use grid_algebra::arith::prime::PrimeField;
    use grid_algebra::arith::ring::Ring;
    use grid_algebra::lattice::types::RingMat;
    use grid_algebra::poly::ring::PolyRing;

    type F = PrimeField<12289>;

    fn make_poly<const D: usize>(v: u64) -> CyclotomicPolyRing<F, D> {
        let mut p = CyclotomicPolyRing::<F, D>::zero();
        p.set_coeff(0, F::from_u64(v));
        p
    }

    fn fake_params(nu: usize, mu: usize) -> LabradorParams {
        use crate::params::{ChallengeProfile, JLProfile};
        LabradorParams {
            jl: JLProfile::default(),
            challenge: ChallengeProfile::paper_default(),
            security_bits: 8,
            soundness_error: 0.0,
            l: 1,
            arith_p: 274177,
            n: 4,
            r: 2,
            beta: 100.0,
            d: 64,
            q: 12289.0,
            sigma: 1.0,
            b: 16,
            b1: 16,
            b2: 16,
            t1: 2,
            t2: 2,
            kappa: 2,
            kappa1: 2,
            kappa2: 2,
            gamma: 100.0,
            gamma1_sq: 10_000,
            gamma2_sq: 10_000,
            beta_prime: 200.0,
            nu,
            mu,
            num_levels: 1,
        }
    }

    /// Shared params for last-level target tests where source_r (2*nu+mu) differs
    /// from r_last (nu+mu). Returns (params, r_last, source_r).
    fn fake_last_level_params(kappa1: usize, kappa2: usize) -> (LabradorParams, usize, usize) {
        let nu = 1;
        let mu = 1;
        let r_last = nu + mu; // 2
        let source_r = 2 * nu + mu; // 3 (regular step's r')
        let params = LabradorParams {
            jl: JLProfile::default(),
            challenge: ChallengeProfile::paper_default(),
            security_bits: 8,
            soundness_error: 0.0,
            l: 1,
            arith_p: 274177,
            n: 4,
            r: source_r,
            beta: 1000.0,
            d: 64,
            q: 12289.0,
            sigma: 1.0,
            b: 2,
            b1: 16,
            b2: 16,
            t1: 4,
            t2: 4,
            kappa: 2,
            kappa1,
            kappa2,
            gamma: 1000.0,
            gamma1_sq: 4_000_000,
            gamma2_sq: 4_000_000,
            beta_prime: 3000.0,
            nu,
            mu,
            num_levels: 1,
        };
        (params, r_last, source_r)
    }

    /// Comprehensive smoke test for build_target_relation (nu=1, mu=1).
    /// Merges: shapes, garbage 3a quadratic terms, inner/outer commitment phi nonzero.
    #[test]
    fn test_build_target_relation_smoke() {
        const N: usize = 64;
        let params = fake_params(1, 1);
        let nu = params.nu;
        let mu = params.mu;
        let r_prime = 2 * nu + mu;
        let b = params.b;

        let z: Vec<CyclotomicPolyRing<F, N>> = (0..params.n)
            .map(|i| make_poly::<N>(i as u64 + 1))
            .collect();
        let t_limbs: Vec<_> = (0..params.r * params.kappa * params.t1)
            .map(|i| make_poly::<N>((100 + i) as u64))
            .collect();
        let g_limbs: Vec<_> = (0..garbage_count(params.r) * params.t2)
            .map(|i| make_poly::<N>((200 + i) as u64))
            .collect();
        let h_limbs: Vec<_> = (0..garbage_count(params.r) * params.t1)
            .map(|i| make_poly::<N>((300 + i) as u64))
            .collect();

        let z_parts = decompose_z(&z, params.b);
        let v = bundle_v(&t_limbs, &g_limbs, &h_limbs);
        let m = v.len();
        let (_, n_prime) = compute_next_level_shape(params.n, m, nu, mu);
        let witness = split_witness(&z_parts, &v, params.n, nu, mu, n_prime);

        let aggregated = AggregatedFunction {
            a_ij: vec![],
            phi: (0..params.r)
                .map(|_| (0..params.n).map(|_| make_poly::<N>(0)).collect())
                .collect(),
            b: make_poly::<N>(0),
        };

        let mut rng = grid_std::test_rng();
        let key = CommitKey::<F, N>::generate_from_params(&mut rng, &params);

        let challenges: Vec<_> = (0..params.r)
            .map(|i| make_poly::<N>((10 + i) as u64))
            .collect();
        let u1 = RingVec::new(
            (0..params.kappa1)
                .map(|i| make_poly::<N>((1000 + i) as u64))
                .collect(),
        );
        let u2 = RingVec::new(
            (0..params.kappa2)
                .map(|i| make_poly::<N>((2000 + i) as u64))
                .collect(),
        );

        let target = build_target_relation(
            &key,
            &u1,
            &u2,
            &challenges,
            &aggregated,
            &params,
            witness,
            r_prime,
            n_prime,
        );

        // --- Structural: K' count, f_prime empty, r', beta', phi dimensions ---
        assert_eq!(
            target.statement.f.len(),
            params.kappa + params.kappa1 + params.kappa2 + 3,
            "K' = κ + κ₁ + κ₂ + 3"
        );
        assert!(target.statement.f_prime.is_empty());
        assert_eq!(target.r_prime, r_prime);
        assert_eq!(target.beta_prime, params.beta_prime);
        for f in &target.statement.f {
            let dense = f.expect_dense();
            assert_eq!(dense.phi.len(), r_prime, "phi length must match r'");
            for phi_part in &dense.phi {
                assert_eq!(phi_part.len(), n_prime, "phi[i] length must match n'");
            }
        }

        // --- Garbage 3a: quadratic terms for nu=1 ---
        let idx_3a = params.kappa + params.kappa1 + params.kappa2;
        let dense_3a = target.statement.f[idx_3a].expect_dense();
        assert_eq!(dense_3a.a.len(), 3, "3 quadratic terms for ν=1 (3a)");

        let z0_term = dense_3a
            .a
            .iter()
            .zip(dense_3a.ij.iter())
            .find(|&(_, ij)| *ij == (0, 0))
            .expect("z⁰ diagonal term (0,0) should exist");
        assert!(z0_term.0.coeff(0).is_one(), "z⁰ coeff should be 1");

        let cross_term = dense_3a
            .a
            .iter()
            .zip(dense_3a.ij.iter())
            .find(|&(_, ij)| *ij == (0, 1))
            .expect("cross term (0,1) should exist");
        let expected_cross = (2u64 * b) % F::modulus();
        assert_eq!(
            cross_term.0.coeff(0).to_u64(),
            expected_cross,
            "cross term should be 2b"
        );

        let z1_term = dense_3a
            .a
            .iter()
            .zip(dense_3a.ij.iter())
            .find(|&(_, ij)| *ij == (1, 1))
            .expect("z¹ diagonal term (1,1) should exist");
        let expected_b2 = b.wrapping_mul(b) % F::modulus();
        assert_eq!(
            z1_term.0.coeff(0).to_u64(),
            expected_b2,
            "z¹ coeff should be b²"
        );

        // --- Inner commitment: phi[0], phi[1] nonzero for each kappa dim ---
        for d in 0..params.kappa {
            let dense = target.statement.f[d].expect_dense();
            assert!(
                dense.phi[0].iter().any(|p| !p.is_zero()),
                "inner commit {}: z⁰ φ should be nonzero",
                d
            );
            assert!(
                dense.phi[1].iter().any(|p| !p.is_zero()),
                "inner commit {}: z¹ φ should be nonzero",
                d
            );
        }

        // --- Outer commitment: phi[2] nonzero, b constants nonzero ---
        for d in 0..params.kappa1 {
            let dense = target.statement.f[params.kappa + d].expect_dense();
            assert!(
                dense.phi[2].iter().any(|p| !p.is_zero()),
                "u1 {}: v φ should be nonzero",
                d
            );
            assert!(
                !target.statement.f[params.kappa + d].b().is_zero(),
                "u1 {}: b constant should be nonzero",
                d
            );
        }
        for d in 0..params.kappa2 {
            let dense = target.statement.f[params.kappa + params.kappa1 + d].expect_dense();
            assert!(
                dense.phi[2].iter().any(|p| !p.is_zero()),
                "u2 {}: v φ should be nonzero",
                d
            );
            assert!(
                !target.statement.f[params.kappa + params.kappa1 + d]
                    .b()
                    .is_zero(),
                "u2 {}: b constant should be nonzero",
                d
            );
        }
    }

    #[test]
    fn test_build_last_level_target_shapes() {
        const N: usize = 64;
        let params = fake_params(1, 1);
        let nu = params.nu;
        let mu = params.mu;
        let r_last = nu + mu;

        let z: Vec<CyclotomicPolyRing<F, N>> = (0..params.n)
            .map(|i| make_poly::<N>(i as u64 + 1))
            .collect();
        let t_limbs: Vec<_> = (0..params.r * params.kappa * params.t1)
            .map(|i| make_poly::<N>((100 + i) as u64))
            .collect();
        let g_limbs: Vec<_> = (0..garbage_count(params.r) * params.t2)
            .map(|i| make_poly::<N>((200 + i) as u64))
            .collect();
        let h_limbs: Vec<_> = (0..garbage_count(params.r) * params.t1)
            .map(|i| make_poly::<N>((300 + i) as u64))
            .collect();

        let v = bundle_v(&t_limbs, &g_limbs, &h_limbs);
        let m = v.len();
        let (_, n_prime) = compute_next_level_shape(params.n, m, nu, mu);

        // For last level: z is NOT decomposed, just split into nu parts
        let z_per_part = params.n.div_ceil(nu);
        let mut z_parts = Vec::with_capacity(nu);
        for i in 0..nu {
            let start = i * z_per_part;
            let end = (start + z_per_part).min(params.n);
            z_parts.push(z[start..end].to_vec());
        }
        // Pad z parts and v parts to n_prime
        let v_per_part = m.div_ceil(mu);
        let mut v_parts = Vec::with_capacity(mu);
        for i in 0..mu {
            let start = i * v_per_part;
            let end = (start + v_per_part).min(m);
            let mut part = v[start..end].to_vec();
            while part.len() < n_prime {
                part.push(CyclotomicPolyRing::<F, N>::zero());
            }
            v_parts.push(part);
        }
        for part in &mut z_parts {
            while part.len() < n_prime {
                part.push(CyclotomicPolyRing::<F, N>::zero());
            }
        }

        let witness = LabradorWitness::new([z_parts, v_parts].concat());
        let aggregated = AggregatedFunction {
            a_ij: vec![],
            phi: (0..params.r)
                .map(|_| (0..params.n).map(|_| make_poly::<N>(0)).collect())
                .collect(),
            b: make_poly::<N>(0),
        };

        let mut rng = grid_std::test_rng();
        let key = CommitKey::<F, N>::generate_from_params(&mut rng, &params);
        let target = build_last_level_target(
            &key,
            &RingVec::new((0..params.kappa1).map(|_| make_poly::<N>(0)).collect()),
            &RingVec::new((0..params.kappa2).map(|_| make_poly::<N>(0)).collect()),
            &{ (0..params.r).map(|_| make_poly::<N>(1)).collect::<Vec<_>>() },
            &aggregated,
            &params,
            witness,
            r_last,
            n_prime,
            params.r, // source_r = params.r (previous step's challenge count)
        )
        .unwrap();

        assert_eq!(
            target.statement.f.len(),
            params.kappa + params.kappa1 + params.kappa2 + 3,
            "K' = κ + κ₁ + κ₂ + 3"
        );
        assert!(target.statement.f_prime.is_empty());
        assert_eq!(target.r_prime, r_last);
        assert_eq!(target.n_prime, n_prime);

        // Verify beta_prime is sqrt(gamma² + gamma1² + gamma2²)
        let expected_beta =
            (params.gamma * params.gamma + params.gamma1_sq as f64 + params.gamma2_sq as f64)
                .sqrt();
        assert!((target.beta_prime - expected_beta).abs() < 1e-9);

        for f in &target.statement.f {
            let dense = f.expect_dense();
            assert_eq!(dense.phi.len(), r_last, "phi length must match r_last");
            for phi_part in &dense.phi {
                assert_eq!(phi_part.len(), n_prime, "phi[i] part length must match n'");
            }
        }
    }

    /// Smoke test: source_r (previous challenge count) differs from r_last.
    /// Uses zero CRS + zero witness so all F functions evaluate to zero,
    /// enabling relation::verify to confirm correctness of target construction.
    #[test]
    fn test_build_last_level_target_smoke_zero_case() {
        const N: usize = 64;
        let (params, r_last, source_r) = fake_last_level_params(2, 2);
        let nu = params.nu;
        let mu = params.mu;

        let t_end = source_r * params.kappa * params.t1;
        let g_len = garbage_count(source_r) * params.t2;
        let h_len = garbage_count(source_r) * params.t1;
        let m = t_end + g_len + h_len;
        let (_, n_prime) = compute_next_level_shape(params.n, m, nu, mu);

        // Zero key: all matrices are zero, so all phi entries are zero
        let key = CommitKey {
            a: RingMat::new(
                params.kappa,
                params.n,
                vec![CyclotomicPolyRing::<F, N>::zero(); params.kappa * params.n],
            ),
            b: RingMat::new(
                params.kappa1,
                t_end,
                vec![CyclotomicPolyRing::<F, N>::zero(); params.kappa1 * t_end],
            ),
            c: RingMat::new(
                params.kappa1,
                g_len,
                vec![CyclotomicPolyRing::<F, N>::zero(); params.kappa1 * g_len],
            ),
            d: RingMat::new(
                params.kappa2,
                h_len,
                vec![CyclotomicPolyRing::<F, N>::zero(); params.kappa2 * h_len],
            ),
        };

        // Zero witness: all parts are zero
        let witness = LabradorWitness {
            parts: (0..r_last)
                .map(|_| vec![CyclotomicPolyRing::<F, N>::zero(); n_prime])
                .collect(),
        };

        // Zero aggregated: no quadratic terms, zero phi, zero b
        let aggregated = AggregatedFunction {
            a_ij: vec![],
            phi: (0..source_r)
                .map(|_| (0..params.n).map(|_| make_poly::<N>(0)).collect())
                .collect(),
            b: make_poly::<N>(0),
        };

        // Zero challenges: avoids nonzero phi from challenge-dependent terms
        let challenges: Vec<_> = (0..source_r).map(|_| make_poly::<N>(0)).collect();

        // Zero u1/u2: b constants for outer commitments are zero
        let u1 = RingVec::new((0..params.kappa1).map(|_| make_poly::<N>(0)).collect());
        let u2 = RingVec::new((0..params.kappa2).map(|_| make_poly::<N>(0)).collect());

        let target = build_last_level_target(
            &key,
            &u1,
            &u2,
            &challenges,
            &aggregated,
            &params,
            witness,
            r_last,
            n_prime,
            source_r,
        )
        .unwrap();

        // Structural checks
        assert_eq!(target.r_prime, r_last, "target r' = r_last = nu + mu");
        assert_eq!(target.witness.num_parts(), r_last);
        for (fi, f) in target.statement.f.iter().enumerate() {
            assert_eq!(
                f.expect_dense().phi.len(),
                r_last,
                "F[{}] phi length must be r_last",
                fi
            );
        }

        // Semantic check: all F functions evaluate to zero
        // (zero CRS → zero phi; zero challenges → zero garbage phi; zero b)
        let result = crate::relation::verify(&target.statement, &target.witness, target.beta_prime);
        assert!(
            result.is_ok(),
            "target relation should be satisfied with zero CRS+witness: {:?}",
            result.err()
        );
    }

    /// Verify limb offsets use source_r (not r_last) when source_r > r_last.
    /// The g/h garbage regions exist beyond the r_last witness parts, and
    /// the CRS b/c/d matrices must be indexed by source_r-based offsets.
    #[test]
    fn test_build_last_level_target_limb_offsets_source_r() {
        const N: usize = 64;
        let (params, r_last, source_r) = fake_last_level_params(1, 1);
        let nu = params.nu;
        let mu = params.mu;

        let t_end = source_r * params.kappa * params.t1; // 3*2*4 = 24
        let g_len = garbage_count(source_r) * params.t2; // 6*4 = 24 (garbage_count(3)=6)
        let h_len = garbage_count(source_r) * params.t1; // 6*4 = 24 (garbage_count(3)=6)
        let m = t_end + g_len + h_len; // 72 (t_end=24, g_len=24, h_len=24)
        let (_, n_prime) = compute_next_level_shape(params.n, m, nu, mu); // n_prime = 72

        // Nonzero key: b/c/d matrices have columns matching source_r layout
        // b: kappa1 x t_end (24), c: kappa1 x g_len (20), d: kappa2 x h_len (20)
        let key = CommitKey {
            a: RingMat::new(
                params.kappa,
                params.n,
                (0..params.kappa * params.n)
                    .map(|i| make_poly::<N>((1 + i) as u64))
                    .collect(),
            ),
            b: RingMat::new(
                params.kappa1,
                t_end,
                (0..params.kappa1 * t_end)
                    .map(|i| make_poly::<N>((100 + i) as u64))
                    .collect(),
            ),
            c: RingMat::new(
                params.kappa1,
                g_len,
                (0..params.kappa1 * g_len)
                    .map(|i| make_poly::<N>((200 + i) as u64))
                    .collect(),
            ),
            d: RingMat::new(
                params.kappa2,
                h_len,
                (0..params.kappa2 * h_len)
                    .map(|i| make_poly::<N>((300 + i) as u64))
                    .collect(),
            ),
        };

        // Nonzero witness: r_last parts
        let witness = LabradorWitness {
            parts: (0..r_last)
                .map(|i| {
                    (0..n_prime)
                        .map(|j| make_poly::<N>((10 + i * n_prime + j) as u64))
                        .collect()
                })
                .collect(),
        };

        // Nonzero aggregated: source_r phi entries
        let aggregated = AggregatedFunction {
            a_ij: vec![],
            phi: (0..source_r)
                .map(|i| {
                    (0..params.n)
                        .map(|j| make_poly::<N>((1000 + i * params.n + j) as u64))
                        .collect()
                })
                .collect(),
            b: make_poly::<N>(500),
        };

        // Nonzero challenges: source_r values (including last index beyond r_last)
        let challenges: Vec<_> = (0..source_r)
            .map(|i| make_poly::<N>((50 + i) as u64))
            .collect();

        // Nonzero u1/u2
        let u1 = RingVec::new(
            (0..params.kappa1)
                .map(|i| make_poly::<N>((4000 + i) as u64))
                .collect(),
        );
        let u2 = RingVec::new(
            (0..params.kappa2)
                .map(|i| make_poly::<N>((5000 + i) as u64))
                .collect(),
        );

        let target = build_last_level_target(
            &key,
            &u1,
            &u2,
            &challenges,
            &aggregated,
            &params,
            witness,
            r_last,
            n_prime,
            source_r,
        )
        .unwrap();

        // Structural: phi dimensions are r_last, not source_r
        assert_eq!(target.r_prime, r_last);
        assert_eq!(target.witness.num_parts(), r_last);
        for f in &target.statement.f {
            let dense = f.expect_dense();
            assert_eq!(
                dense.phi.len(),
                r_last,
                "phi length must be r_last, not source_r"
            );
            for phi_part in &dense.phi {
                assert_eq!(phi_part.len(), n_prime);
            }
        }

        // Key: source_r > r_last means extra challenge-dependent garbage limbs exist.
        // The CRS b/c/d matrices are indexed by source_r-based offsets (t_end, g_len, h_len),
        // but the target phi is only r_last-wide. The garbage challenge-dependent terms
        // end up in the 3 extra check functions (inner h, outer g, JL), not in phi.
        assert_eq!(
            target.statement.f.len(),
            params.kappa + params.kappa1 + params.kappa2 + 3,
            "3 extra check functions carry the source_r challenge terms"
        );

        // The CRS column counts must match source_r-based layout (not r_last)
        assert_eq!(key.b.cols(), t_end, "b columns = source_r * kappa * t1");
        assert_eq!(
            key.c.cols(),
            g_len,
            "c columns = garbage_count(source_r) * t2"
        );
        assert_eq!(
            key.d.cols(),
            h_len,
            "d columns = garbage_count(source_r) * t1"
        );

        // The last source_r challenge index (source_r - 1 = 2) maps to garbage limbs,
        // not to witness. If source_r were incorrectly used as r_last, the phi would
        // be 3-wide instead of 2-wide, and the witness part count would be 3.
        assert!(
            target.witness.num_parts() < source_r,
            "witness parts (r_last={}) must be less than source_r ({})",
            r_last,
            source_r
        );

        // --- Coefficient assertions: verify phi placement is correct ---
        // Inner commitment F[0] (dim=0): z-part phi[0] = source_crs.a row 0 entries
        // source_crs.a values start at 1, shape: kappa(2) x n(4)
        let f_inner0 = &target.statement.f[0];
        let dense_inner0 = f_inner0.expect_dense();
        assert_eq!(
            dense_inner0.phi[0][0].coeff(0).to_u64(),
            1u64,
            "inner F[0] z-part phi[0][0] = source_crs.a[0,0] = 1"
        );
        assert_eq!(
            dense_inner0.phi[0][1].coeff(0).to_u64(),
            2u64,
            "inner F[0] z-part phi[0][1] = source_crs.a[0,1] = 2"
        );
        assert_eq!(
            dense_inner0.phi[0][2].coeff(0).to_u64(),
            3u64,
            "inner F[0] z-part phi[0][2] = source_crs.a[0,2] = 3"
        );
        assert_eq!(
            dense_inner0.phi[0][3].coeff(0).to_u64(),
            4u64,
            "inner F[0] z-part phi[0][3] = source_crs.a[0,3] = 4"
        );

        // Inner commitment F[0] (dim=0): t-limb phi[1] = -c_i * b1^l
        // c_0 = 50, b1 = 16, limbs 0..3 at phi[1][0..3]
        assert_eq!(
            dense_inner0.phi[1][0].coeff(0).to_u64(),
            12289 - 50,
            "inner F[0] t-limb phi[1][0] = -c_0 * b1^0 = -50"
        );
        assert_eq!(
            dense_inner0.phi[1][1].coeff(0).to_u64(),
            12289 - 50 * 16,
            "inner F[0] t-limb phi[1][1] = -c_0 * b1^1 = -800"
        );

        // Inner commitment F[1] (dim=1): z-part phi[0] = source_crs.a row 1 entries
        // source_crs.a shape: kappa(2) x n(4), row 1 starts at offset n=4
        // source_crs.a[1,0] = 1 + 4 + 0 = 5
        // source_crs.a[1,3] = 1 + 4 + 3 = 8
        let f_inner1 = &target.statement.f[1];
        let dense_inner1 = f_inner1.expect_dense();
        assert_eq!(
            dense_inner1.phi[0][0].coeff(0).to_u64(),
            5u64,
            "inner F[1] z-part phi[0][0] = source_crs.a[1,0] = 5"
        );
        assert_eq!(
            dense_inner1.phi[0][3].coeff(0).to_u64(),
            8u64,
            "inner F[1] z-part phi[0][3] = source_crs.a[1,3] = 8"
        );

        // Challenge index 2 (source_r-1) maps to t limbs at phi[1][8..11]
        // c_2 = 52, limb_flat_idx = 2*2*4 + 0*4 + l = 16+l
        // v_offset_to_last_level_part(16+l, nu=1, v_per_part) = (1, 16+l)
        assert_eq!(
            dense_inner0.phi[1][16].coeff(0).to_u64(),
            12289 - 52,
            "inner F[0] last source_r challenge c_2 at limb offset 16"
        );

        // Outer commitment F[kappa] = F[2]: b_const = u1[0]
        let f_outer_u1 = &target.statement.f[params.kappa];
        assert_eq!(
            f_outer_u1.b().coeff(0).to_u64(),
            4000u64,
            "outer u1 b constant = u1[0] = 4000"
        );

        // Outer commitment F[kappa+kappa1] = F[3]: b_const = u2[0]
        let f_outer_u2 = &target.statement.f[params.kappa + params.kappa1];
        assert_eq!(
            f_outer_u2.b().coeff(0).to_u64(),
            5000u64,
            "outer u2 b constant = u2[0] = 5000"
        );

        // Garbage (3a) F[kappa+kappa1+kappa2] = F[4]: b constant = 0
        let f_garbage_3a = &target.statement.f[params.kappa + params.kappa1 + params.kappa2];
        assert!(f_garbage_3a.b().is_zero(), "garbage 3a b constant is zero");

        // Garbage (3b) F[kappa+kappa1+kappa2+1] = F[5]: b constant = 0
        let f_garbage_3b = &target.statement.f[params.kappa + params.kappa1 + params.kappa2 + 1];
        assert!(f_garbage_3b.b().is_zero(), "garbage 3b b constant is zero");

        // Garbage (3c) F[kappa+kappa1+kappa2+2] = F[6]: b constant = aggregated.b
        let f_garbage_3c = &target.statement.f[params.kappa + params.kappa1 + params.kappa2 + 2];
        assert_eq!(
            f_garbage_3c.b().coeff(0).to_u64(),
            500u64,
            "garbage 3c b constant = aggregated.b = 500"
        );
    }

    #[test]
    fn test_last_level_garbage_3a_simplified() {
        const N: usize = 64;
        let params = fake_params(2, 1);
        let nu = 2;
        let mu = 1;
        let r_last = nu + mu;

        let challenges: Vec<_> = (0..params.r)
            .map(|i| make_poly::<N>((10 + i) as u64))
            .collect();

        let t_end = params.r * params.kappa * params.t1;
        let v_per_part =
            (t_end + garbage_count(params.r) * params.t2 + garbage_count(params.r) * params.t1)
                .div_ceil(mu);
        let (_, n_prime) = compute_next_level_shape(params.n, v_per_part * mu, nu, mu);

        let b2_powers = scalar_powers(&F::from_u64(params.b2), params.t2);
        let (quad, phi, b_const) = build_last_level_garbage_3a::<F, N>(
            &challenges,
            nu,
            r_last,
            params.r,
            t_end,
            params.t2,
            v_per_part,
            n_prime,
            &b2_powers,
        );

        // Only diagonal z terms (no decomposition): ν terms
        assert_eq!(quad.len(), nu, "should have exactly ν diagonal terms");
        for &(i, j, _) in &quad {
            assert_eq!(i, j, "all terms should be diagonal");
            assert!(i < nu, "all terms should be z-parts");
        }
        assert!(b_const.is_zero());
        assert_eq!(phi.len(), r_last);
    }

    #[test]
    fn test_last_level_garbage_3b_no_b_scaling() {
        const N: usize = 64;
        let params = fake_params(1, 1);
        let nu = 1;
        let mu = 1;
        let r_last = nu + mu;

        let n = params.n;
        let phi_vec: Vec<CyclotomicPolyRing<F, N>> =
            (0..n).map(|i| make_poly::<N>((i + 1) as u64)).collect();
        let aggregated = AggregatedFunction {
            a_ij: vec![],
            phi: vec![phi_vec.clone(), phi_vec.clone()],
            b: make_poly::<N>(0),
        };

        let challenges: Vec<_> = (0..params.r)
            .map(|i| make_poly::<N>((10 + i) as u64))
            .collect();

        let t_end = params.r * params.kappa * params.t1;
        let g_end = t_end + garbage_count(params.r) * params.t2;
        let v_per_part = (g_end + garbage_count(params.r) * params.t1).div_ceil(mu);
        let (_, n_prime) = compute_next_level_shape(params.n, v_per_part * mu, nu, mu);

        let b1_powers = scalar_powers(&F::from_u64(params.b1), params.t1);
        let (quad, phi, b_const) = build_last_level_garbage_3b::<F, N>(
            &challenges,
            &aggregated,
            nu,
            r_last,
            params.r,
            params.t1,
            g_end,
            v_per_part,
            n_prime,
            &b1_powers,
        );

        // No quadratic terms in (3b)
        assert!(quad.is_empty());
        assert!(b_const.is_zero());
        assert_eq!(phi.len(), r_last);

        // Check that φ is populated for z-parts (not scaled by b)
        let has_z0 = phi[0].iter().any(|p| !p.is_zero());
        assert!(has_z0, "z⁰ φ should be nonzero");
    }
}
