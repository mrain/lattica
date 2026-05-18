//! Principal relation R (§5.1).
//!
//! LaBRADOR proves knowledge of a short solution to a system of dot product
//! constraints over `R_q = Z_q[X]/(X^d + 1)`.
//!
//! # Relation
//!
//! A statement is `((F, F', β))` where:
//! - `F` — family of K quadratic functions that must **fully vanish** in R_q
//! - `F'` — family of L quadratic functions whose **constant term** must vanish in Z_q
//! - `β` — norm bound: Σ ‖sᵢ‖₂² ≤ β²
//!
//! A witness is `(s₁, …, sᵣ)` with `sᵢ ∈ R_qⁿ`.
//!
//! Each quadratic function has the form:
//! ```text
//! f(s₁,…,sᵣ) = Σᵢⱼ aᵢⱼ · ⟨sᵢ, sⱼ⟩ + Σᵢ ⟨φᵢ, sᵢ⟩ - b
//! ```
//!
//! # Automorphism σ₋₁
//!
//! The paper uses `<a, b> = ct(⟨σ₋₁(a), b⟩)` to expose coefficient-level dot
//! products through ring elements. The conjugation `σ₋₁(X) = X^(-1)` is applied
//! by the caller when constructing constraints (e.g., JL projection rows).
//! The relation itself treats all functions as opaque quadratic forms.

use alloc::format;
use alloc::vec::Vec;

use crate::String;
use grid_algebra::arith::ring::{IntegerRing, Ring};
use grid_algebra::poly::ring::PolyRing;

/// Dense quadratic dot-product constraint function.
///
/// Represents `f(s₁,…,sᵣ) = Σᵢⱼ aᵢⱼ · ⟨sᵢ, sⱼ⟩ + Σᵢ ⟨φᵢ, sᵢ⟩ - b`.
///
/// Sparse upper-triangular storage: only `(i, j, aᵢⱼ)` triples with `i ≤ j`
/// are kept. `evaluate()` sums each pair **once** (no doubling). Callers
/// deriving constraints from a full symmetric matrix should pre-double
/// off-diagonal entries before constructing the function.
///
/// `a`: quadratic coefficients `aᵢⱼ ∈ R_q` for each `(i, j)` pair
/// `ij`: upper-triangular index pairs with `0 ≤ i ≤ j < r`
/// `phi`: linear terms — `phi[i]` is `φᵢ ∈ R_qⁿ` for witness part `i`
/// `b`: constant term `b ∈ R_q`
#[derive(Debug, Clone)]
pub struct DenseQuadraticFunction<Rq> {
    pub a: Vec<Rq>,
    pub ij: Vec<(usize, usize)>,
    pub phi: Vec<Vec<Rq>>,
    pub b: Rq,
}

/// Sparse quadratic dot-product constraint function.
///
/// Same semantics as `DenseQuadraticFunction` but stores phi and quadratic
/// terms as explicit (index, coefficient) triples. Used for conjugacy F'
/// constraints where phi has only 2 non-zero entries out of num_parts × rank.
///
/// `ij_a`: upper-triangular quadratic terms `(i, j, aᵢⱼ)` with `i ≤ j`
/// `phi`: sparse linear terms `(part_idx, entry_idx, coeff)`
/// `b`: constant term `b ∈ R_q`
#[derive(Debug, Clone)]
pub struct SparseQuadraticFunction<Rq> {
    pub ij_a: Vec<(usize, usize, Rq)>,
    pub phi: Vec<(usize, usize, Rq)>,
    pub b: Rq,
}

/// Quadratic dot-product constraint function (dense or sparse).
///
/// Use `Dense` for Ajtai openings and constraints with full phi vectors.
/// Use `Sparse` for conjugacy checks where phi has few non-zero entries.
#[derive(Debug, Clone)]
pub enum QuadraticFunction<Rq> {
    Dense(DenseQuadraticFunction<Rq>),
    Sparse(SparseQuadraticFunction<Rq>),
}

impl<Rq> QuadraticFunction<Rq> {
    /// Get the dense variant (panics if sparse).
    pub fn expect_dense(&self) -> &DenseQuadraticFunction<Rq> {
        match self {
            Self::Dense(d) => d,
            Self::Sparse(_) => panic!("QuadraticFunction is Sparse, not Dense"),
        }
    }

    /// Get mutable access to the dense variant (panics if sparse).
    pub fn expect_dense_mut(&mut self) -> &mut DenseQuadraticFunction<Rq> {
        match self {
            Self::Dense(d) => d,
            Self::Sparse(_) => panic!("QuadraticFunction is Sparse, not Dense"),
        }
    }
}

/// Public statement for the principal relation.
///
/// `F` — functions that must fully vanish: `f(s) = 0` in R_q
/// `F_prime` — functions whose constant term must vanish: `ct(f'(s)) = 0` in Z_q
#[derive(Debug, Clone)]
pub struct LabradorStatement<Rq> {
    pub f: Vec<QuadraticFunction<Rq>>,
    pub f_prime: Vec<QuadraticFunction<Rq>>,
}

/// Witness for the principal relation.
///
/// `parts[i]` is the vector `sᵢ ∈ R_qⁿ`.
#[derive(Debug, Clone)]
pub struct LabradorWitness<Rq> {
    pub parts: Vec<Vec<Rq>>,
}

// --- QuadraticFunction ---

impl<Rq> QuadraticFunction<Rq> {
    /// Create a dense quadratic function from sparse upper-triangular quadratic terms.
    ///
    /// `quad`: list of `(i, j, aᵢⱼ)` triples with `0 ≤ i ≤ j < r`
    /// `phi`: linear coefficients `φᵢ ∈ R_qⁿ` for each witness part
    /// `b`: constant term
    pub fn from_parts(quad: Vec<(usize, usize, Rq)>, phi: Vec<Vec<Rq>>, b: Rq) -> Self {
        let mut a = Vec::with_capacity(quad.len());
        let mut ij = Vec::with_capacity(quad.len());
        for (i, j, coeff) in quad {
            debug_assert!(
                i <= j,
                "quadratic indices must satisfy i <= j, got ({}, {})",
                i,
                j
            );
            ij.push((i, j));
            a.push(coeff);
        }
        Self::Dense(DenseQuadraticFunction { a, ij, phi, b })
    }

    /// Create a sparse quadratic function from explicit triples.
    ///
    /// `ij_a`: upper-triangular quadratic terms `(i, j, aᵢⱼ)` with `i ≤ j`
    /// `phi`: sparse linear terms `(part_idx, entry_idx, coeff)`
    /// `b`: constant term
    pub fn from_sparse(ij_a: Vec<(usize, usize, Rq)>, phi: Vec<(usize, usize, Rq)>, b: Rq) -> Self {
        Self::Sparse(SparseQuadraticFunction { ij_a, phi, b })
    }

    /// Get the constant term `b` (works for both dense and sparse variants).
    pub fn b(&self) -> &Rq {
        match self {
            Self::Dense(d) => &d.b,
            Self::Sparse(s) => &s.b,
        }
    }

    /// Get quadratic term indices and coefficients as `[(i, j, &coeff)]`.
    pub fn quads(&self) -> Vec<(usize, usize, &Rq)> {
        match self {
            Self::Dense(d) => {
                d.ij.iter()
                    .zip(d.a.iter())
                    .map(|(&(i, j), coeff)| (i, j, coeff))
                    .collect()
            }
            Self::Sparse(s) => s
                .ij_a
                .iter()
                .map(|&(i, j, ref coeff)| (i, j, coeff))
                .collect(),
        }
    }

    /// Get non-zero linear (phi) entries as `[(part_idx, entry_idx, &coeff)]`.
    pub fn nonzero_phi(&self) -> Vec<(usize, usize, &Rq)>
    where
        Rq: PolyRing,
    {
        match self {
            Self::Dense(d) => {
                let mut result = Vec::new();
                for (pi, phi_i) in d.phi.iter().enumerate() {
                    for (ei, coeff) in phi_i.iter().enumerate() {
                        if !coeff.is_zero() {
                            result.push((pi, ei, coeff));
                        }
                    }
                }
                result
            }
            Self::Sparse(s) => s
                .phi
                .iter()
                .map(|&(part, entry, ref coeff)| (part, entry, coeff))
                .collect(),
        }
    }

    /// Evaluate the quadratic function on the witness.
    ///
    /// Returns `Σᵢⱼ aᵢⱼ · ⟨sᵢ, sⱼ⟩ + Σᵢ ⟨φᵢ, sᵢ⟩ - b` in R_q.
    pub fn evaluate(&self, witness: &LabradorWitness<Rq>) -> Rq
    where
        Rq: PolyRing,
    {
        match self {
            Self::Dense(d) => d.evaluate(witness),
            Self::Sparse(s) => s.evaluate(witness),
        }
    }

    /// Evaluate and return only the constant term (coefficient of X⁰).
    ///
    /// Used for F' constraints where only the constant coefficient matters.
    pub fn evaluate_constant_term(&self, witness: &LabradorWitness<Rq>) -> Rq::Coeff
    where
        Rq: PolyRing,
    {
        let val = self.evaluate(witness);
        val.coeff(0)
    }

    /// Iterate over quadratic terms `(i, j, coeff)` with `i <= j`.
    ///
    /// For dense: yields from `ij`/`a` parallel vectors.
    /// For sparse: yields from `ij_a` triples.
    pub fn for_each_quad<Fn: FnMut((usize, usize, &Rq))>(&self, mut f: Fn) {
        match self {
            Self::Dense(d) => {
                for (&(i, j), coeff) in d.ij.iter().zip(d.a.iter()) {
                    f((i, j, coeff));
                }
            }
            Self::Sparse(s) => {
                for &(i, j, ref coeff) in &s.ij_a {
                    f((i, j, coeff));
                }
            }
        }
    }

    /// Iterate over linear (phi) entries as `(part_idx, entry_idx, coeff)`.
    ///
    /// For dense: expands the full `phi[part][entry]` grid.
    /// For sparse: yields only the stored `(part_idx, entry_idx, coeff)` triples.
    pub fn for_each_phi<Fn: FnMut((usize, usize, &Rq))>(&self, mut f: Fn) {
        match self {
            Self::Dense(d) => {
                for (pi, phi_i) in d.phi.iter().enumerate() {
                    for (ei, coeff) in phi_i.iter().enumerate() {
                        f((pi, ei, coeff));
                    }
                }
            }
            Self::Sparse(s) => {
                for &(part, entry, ref coeff) in &s.phi {
                    f((part, entry, coeff));
                }
            }
        }
    }

    /// Iterate over only non-zero linear (phi) entries as `(part_idx, entry_idx, coeff)`.
    ///
    /// For dense: expands the full grid but skips zero coefficients.
    /// For sparse: yields stored triples (assumed non-zero by construction).
    pub fn for_each_nonzero_phi<Fn: FnMut((usize, usize, &Rq))>(&self, mut f: Fn)
    where
        Rq: PolyRing,
    {
        match self {
            Self::Dense(d) => {
                for (pi, phi_i) in d.phi.iter().enumerate() {
                    for (ei, coeff) in phi_i.iter().enumerate() {
                        if !coeff.is_zero() {
                            f((pi, ei, coeff));
                        }
                    }
                }
            }
            Self::Sparse(s) => {
                for &(part, entry, ref coeff) in &s.phi {
                    f((part, entry, coeff));
                }
            }
        }
    }
}

impl<Rq: PolyRing> DenseQuadraticFunction<Rq> {
    /// Evaluate the dense quadratic function on the witness.
    fn evaluate(&self, witness: &LabradorWitness<Rq>) -> Rq {
        let parts = &witness.parts;

        debug_assert!(
            !parts.is_empty() || self.a.is_empty() && self.phi.is_empty(),
            "witness has no parts but function has quadratic or linear terms"
        );

        if !parts.is_empty() {
            let n = parts[0].len();
            debug_assert!(
                parts.iter().skip(1).all(|p| p.len() == n),
                "witness parts have inconsistent lengths (r={}, n={})",
                parts.len(),
                n
            );
            debug_assert_eq!(
                self.phi.len(),
                parts.len(),
                "phi count ({}) must match witness parts count ({})",
                self.phi.len(),
                parts.len()
            );
            for (pi, (phi_i, part)) in self.phi.iter().zip(parts.iter()).enumerate() {
                debug_assert_eq!(
                    phi_i.len(),
                    part.len(),
                    "phi[{}] length ({}) must match parts[{}] length ({})",
                    pi,
                    phi_i.len(),
                    pi,
                    part.len()
                );
            }
        }

        debug_assert_eq!(
            self.a.len(),
            self.ij.len(),
            "a and ij must have equal length"
        );

        let mut result = Rq::zero();

        for (coeff, &(i, j)) in self.a.iter().zip(self.ij.iter()) {
            let dot = Ring::dot_product(&parts[i], &parts[j]);
            result += Ring::mul_ref(coeff, &dot);
        }

        for (phi_i, part) in self.phi.iter().zip(parts.iter()) {
            result += Ring::dot_product(phi_i, part);
        }

        result -= &self.b;
        result
    }
}

impl<Rq: PolyRing> SparseQuadraticFunction<Rq> {
    /// Evaluate the sparse quadratic function on the witness.
    fn evaluate(&self, witness: &LabradorWitness<Rq>) -> Rq {
        let parts = &witness.parts;

        let mut result = Rq::zero();

        // Quadratic terms: Σ (i,j,coeff) · ⟨sᵢ, sⱼ⟩
        for &(i, j, ref coeff) in &self.ij_a {
            debug_assert!(
                i <= j,
                "sparse ij_a indices must satisfy i <= j, got ({}, {})",
                i,
                j
            );
            let dot = Ring::dot_product(&parts[i], &parts[j]);
            result += Ring::mul_ref(coeff, &dot);
        }

        // Sparse linear terms: Σ (part_idx, entry_idx, coeff) · s_part[entry]
        for &(part_idx, entry_idx, ref coeff) in &self.phi {
            debug_assert!(
                part_idx < parts.len(),
                "sparse phi part_idx {} out of range (num_parts={})",
                part_idx,
                parts.len()
            );
            debug_assert!(
                entry_idx < parts[part_idx].len(),
                "sparse phi entry_idx {} out of range for part {} (rank={})",
                entry_idx,
                part_idx,
                parts[part_idx].len()
            );
            result += Ring::mul_ref(coeff, &parts[part_idx][entry_idx]);
        }

        result -= &self.b;
        result
    }
}

// --- LabradorStatement ---

impl<Rq> Default for LabradorStatement<Rq> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Rq> LabradorStatement<Rq> {
    pub fn new() -> Self {
        Self {
            f: Vec::new(),
            f_prime: Vec::new(),
        }
    }

    pub fn num_f(&self) -> usize {
        self.f.len()
    }

    pub fn num_f_prime(&self) -> usize {
        self.f_prime.len()
    }
}

// --- LabradorWitness ---

impl<Rq: PolyRing> LabradorWitness<Rq> {
    pub fn new(parts: Vec<Vec<Rq>>) -> Self {
        Self { parts }
    }

    pub fn num_parts(&self) -> usize {
        self.parts.len()
    }

    pub fn rank(&self) -> usize {
        if self.parts.is_empty() {
            return 0;
        }
        self.parts[0].len()
    }

    /// Compute Σᵢ ‖sᵢ‖₂² as f64.
    ///
    /// Calls `squared_l2_norm` per-poly (each poly's `coeffs()` is a contiguous
    /// slice) and sums the results — zero heap allocation.
    pub fn l2_norm_squared(&self) -> f64
    where
        Rq: PolyRing<Coeff: IntegerRing<Uint = u64>>,
    {
        let mut sum: f64 = 0.0;
        for part in &self.parts {
            for poly in part.iter() {
                sum +=
                    crate::main_protocol::squared_l2_norm(poly.coeffs()).unwrap_or(f64::INFINITY);
            }
        }
        sum
    }

    /// Exact u128 squared L2 norm of all witness coefficients.
    ///
    /// Same as `l2_norm_squared` but accumulates in u128 to preserve
    /// precision for security-critical threshold comparisons.
    pub fn l2_norm_squared_u128(&self) -> Result<u128, String>
    where
        Rq: PolyRing<Coeff: IntegerRing<Uint = u64>>,
    {
        let mut sum: u128 = 0;
        for part in &self.parts {
            for poly in part.iter() {
                let sq = crate::main_protocol::squared_l2_norm_u128(poly.coeffs())
                    .map_err(String::from)?;
                sum = sum
                    .checked_add(sq)
                    .ok_or("norm overflow: witness L2 norm exceeds u128")?;
            }
        }
        Ok(sum)
    }
}

/// Verify the principal relation.
///
/// Returns `Ok(())` iff:
/// 1. `Σᵢ ‖sᵢ‖₂² ≤ β²`
/// 2. `f(witness) = 0` for all `f ∈ F`
/// 3. `ct(f'(witness)) = 0` for all `f' ∈ F'`
pub fn verify<Rq>(
    statement: &LabradorStatement<Rq>,
    witness: &LabradorWitness<Rq>,
    beta: f64,
) -> Result<(), String>
where
    Rq: PolyRing<Coeff: IntegerRing<Uint = u64>>,
{
    if !beta.is_finite() || beta <= 0.0 {
        return Err(format!("beta must be positive and finite, got {}", beta));
    }

    let num_parts = witness.parts.len();
    if num_parts == 0 {
        return Err("witness must have at least one part".into());
    }

    // Validate consistent (r, n) shape across all parts
    let n = witness.parts[0].len();
    if n == 0 {
        return Err("witness part length n must be positive".into());
    }
    for (idx, part) in witness.parts.iter().enumerate().skip(1) {
        if part.len() != n {
            return Err(format!(
                "witness part {} has length {}, expected {} (r={}, n={})",
                idx,
                part.len(),
                n,
                num_parts,
                n
            ));
        }
    }

    // Validate function shapes against (r, n)
    for (fi, f) in statement
        .f
        .iter()
        .chain(statement.f_prime.iter())
        .enumerate()
    {
        match f {
            QuadraticFunction::Dense(d) => {
                if d.a.len() != d.ij.len() {
                    return Err(format!(
                        "function {} has {} coefficients but {} index pairs (a/ij mismatch)",
                        fi,
                        d.a.len(),
                        d.ij.len()
                    ));
                }
                for &(i, j) in &d.ij {
                    if i > j {
                        return Err(format!(
                            "function {}: quadratic index ({}, {}) violates upper-triangular invariant i <= j",
                            fi, i, j
                        ));
                    }
                    if i >= num_parts || j >= num_parts {
                        return Err(format!(
                            "function {}: quadratic index ({}, {}) out of bounds for r={}",
                            fi, i, j, num_parts
                        ));
                    }
                }
                if d.phi.len() != num_parts {
                    return Err(format!(
                        "function {} has {} phi vectors, expected {} (must match r)",
                        fi,
                        d.phi.len(),
                        num_parts
                    ));
                }
                for (pi, phi_i) in d.phi.iter().enumerate() {
                    if phi_i.len() != n {
                        return Err(format!(
                            "function {} phi[{}] has length {}, expected n={}",
                            fi,
                            pi,
                            phi_i.len(),
                            n
                        ));
                    }
                }
            }
            QuadraticFunction::Sparse(s) => {
                for &(i, j, _) in &s.ij_a {
                    if i > j {
                        return Err(format!(
                            "function {}: quadratic index ({}, {}) violates upper-triangular invariant i <= j",
                            fi, i, j
                        ));
                    }
                    if i >= num_parts || j >= num_parts {
                        return Err(format!(
                            "function {}: quadratic index ({}, {}) out of bounds for r={}",
                            fi, i, j, num_parts
                        ));
                    }
                }
                for &(part, entry, _) in &s.phi {
                    if part >= num_parts {
                        return Err(format!(
                            "function {}: phi part index {} out of bounds for r={}",
                            fi, part, num_parts
                        ));
                    }
                    if entry >= n {
                        return Err(format!(
                            "function {}: phi entry index {} out of bounds for n={}",
                            fi, entry, n
                        ));
                    }
                }
            }
        }
    }

    let norm_sq_u128 = witness.l2_norm_squared_u128()?;
    let beta_sq = beta * beta;
    if norm_sq_u128 > beta_sq as u128 {
        return Err(format!(
            "witness norm squared ({}) exceeds beta squared ({})",
            norm_sq_u128, beta_sq
        ));
    }

    // 2. Full vanishing functions
    for (k, f) in statement.f.iter().enumerate() {
        let val = f.evaluate(witness);
        if !val.is_zero() {
            return Err(format!("F[{}] does not vanish", k));
        }
    }

    // 3. Constant-term vanishing functions
    for (k, f) in statement.f_prime.iter().enumerate() {
        let ct = f.evaluate_constant_term(witness);
        if !ct.is_zero() {
            return Err(format!("F'[{}] constant term is nonzero", k));
        }
    }

    Ok(())
}

/// Apply the conjugation automorphism σ₋₁: `X → X⁻¹` in `Z_q[X]/(X^d+1)`.
///
/// This is the automorphism corresponding to k = -1 (mod 2d), i.e. k = 2d - 1.
/// Used to transform coefficient-level dot products into ring-level dot products:
/// `<a, b> = ct(⟨σ₋₁(a), b⟩)`.
///
/// Formula: σ₋₁(X⁰) = X⁰, σ₋₁(Xⁱ) = -X^(d-i) for 0 < i < d.
pub fn conjugation<Rq>(poly: &Rq) -> Rq
where
    Rq: PolyRing,
{
    let d = Rq::degree();
    let mut result = Rq::zero();
    for (i, coeff) in poly.coeffs().iter().enumerate() {
        if i == 0 {
            // X⁰ -> X⁰ (constant term unchanged)
            result.set_coeff(0, coeff.clone());
        } else {
            // Xⁱ -> -X^(d-i) in X^d+1
            // Since X^(-i) = X^(2d-i) and 2d-i >= d (for i <= d),
            // X^(2d-i) = -X^(2d-i-d) = -X^(d-i)
            let target = d - i;
            result.set_coeff(target, -coeff.clone());
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    // Small test ring: Z_17[X]/(X^8+1)
    use grid_algebra::arith::prime::PrimeField;
    use grid_algebra::poly::ring::CyclotomicPolyRing;

    type F17 = PrimeField<17>;
    type Poly8 = CyclotomicPolyRing<F17, 8>;

    fn zero_poly() -> Poly8 {
        Poly8::zero()
    }

    fn one_poly() -> Poly8 {
        Poly8::one()
    }

    fn from_u64(v: u64) -> Poly8 {
        Poly8::from_array(core::array::from_fn(|i| {
            if i == 0 {
                F17::from_u64(v)
            } else {
                F17::zero()
            }
        }))
    }

    fn make_poly(coeffs: [u64; 8]) -> Poly8 {
        Poly8::from_array(core::array::from_fn(|i| F17::from_u64(coeffs[i])))
    }

    #[test]
    fn test_quadratic_function_evaluate() {
        // Zero function on zero witness should vanish
        let func = QuadraticFunction::from_parts(Vec::new(), vec![vec![zero_poly()]], zero_poly());
        let witness = LabradorWitness::new(vec![vec![zero_poly()]]);
        assert!(
            func.evaluate(&witness).is_zero(),
            "zero function should vanish"
        );

        // f(s) = 1·<s₁,s₁> - 1 with s₁ = [1] → 1 - 1 = 0
        let quad = vec![(0, 0, one_poly())];
        let func2 = QuadraticFunction::from_parts(quad, vec![vec![zero_poly()]], one_poly());
        let witness2 = LabradorWitness::new(vec![vec![one_poly()]]);
        assert!(func2.evaluate(&witness2).is_zero(), "1 - 1 should be zero");
        assert!(func2.evaluate_constant_term(&witness2).is_zero());
    }

    #[test]
    fn test_verify() {
        #[allow(clippy::type_complexity)]
        let cases: &[(
            &str,
            LabradorStatement<Poly8>,
            LabradorWitness<Poly8>,
            f64,
            bool,
        )] = &[
            // --- Empty statement ---
            (
                "empty statement with zero witness",
                LabradorStatement::new(),
                LabradorWitness::new(vec![vec![zero_poly()]]),
                1.0,
                true,
            ),
            (
                "empty witness rejected",
                LabradorStatement::new(),
                LabradorWitness::new(vec![]),
                1.0,
                false,
            ),
            // --- Norm ---
            (
                "norm exceeds beta",
                LabradorStatement::new(),
                LabradorWitness::new(vec![vec![from_u64(10)]]),
                1.0,
                false,
            ),
            (
                "norm equals beta",
                LabradorStatement::new(),
                LabradorWitness::new(vec![vec![from_u64(3), from_u64(4)]]),
                5.0,
                true,
            ),
            // --- F vanishing ---
            (
                "vanishing F function",
                LabradorStatement {
                    f: vec![QuadraticFunction::from_parts(
                        vec![(0, 0, one_poly())],
                        vec![vec![zero_poly()]],
                        one_poly(),
                    )],
                    f_prime: vec![],
                },
                LabradorWitness::new(vec![vec![one_poly()]]),
                17.0,
                true,
            ),
            (
                "non-vanishing F function",
                LabradorStatement {
                    f: vec![QuadraticFunction::from_parts(
                        vec![(0, 0, one_poly())],
                        vec![vec![zero_poly()]],
                        from_u64(2),
                    )],
                    f_prime: vec![],
                },
                LabradorWitness::new(vec![vec![one_poly()]]),
                17.0,
                false,
            ),
            // --- F' constant term ---
            (
                "F' constant term vanishes",
                {
                    let x_poly = make_poly([0, 1, 0, 0, 0, 0, 0, 0]);
                    LabradorStatement {
                        f: vec![],
                        f_prime: vec![QuadraticFunction::from_parts(
                            vec![(0, 0, x_poly.clone())],
                            vec![vec![zero_poly()]],
                            x_poly,
                        )],
                    }
                },
                LabradorWitness::new(vec![vec![one_poly()]]),
                17.0,
                true,
            ),
            // --- Terms ---
            (
                "cross term vanishes",
                LabradorStatement {
                    f: vec![QuadraticFunction::from_parts(
                        vec![(0, 1, one_poly())],
                        vec![vec![zero_poly()], vec![zero_poly()]],
                        from_u64(6),
                    )],
                    f_prime: vec![],
                },
                LabradorWitness::new(vec![vec![from_u64(2)], vec![from_u64(3)]]),
                17.0,
                true,
            ),
            (
                "linear term vanishes",
                LabradorStatement {
                    f: vec![QuadraticFunction::from_parts(
                        Vec::new(),
                        vec![vec![from_u64(5)]],
                        from_u64(5),
                    )],
                    f_prime: vec![],
                },
                LabradorWitness::new(vec![vec![one_poly()]]),
                17.0,
                true,
            ),
            // --- Multiple functions ---
            (
                "multiple F functions vanish",
                {
                    let quad = vec![(0, 0, one_poly())];
                    let phi = vec![vec![zero_poly()]];
                    LabradorStatement {
                        f: vec![
                            QuadraticFunction::from_parts(quad.clone(), phi.clone(), one_poly()),
                            QuadraticFunction::from_parts(quad, phi, one_poly()),
                        ],
                        f_prime: vec![],
                    }
                },
                LabradorWitness::new(vec![vec![one_poly()]]),
                17.0,
                true,
            ),
            (
                "mixed F and F' both vanish",
                {
                    let f_full = QuadraticFunction::from_parts(
                        vec![(0, 0, one_poly())],
                        vec![vec![zero_poly()]],
                        one_poly(),
                    );
                    let x_poly = make_poly([0, 1, 0, 0, 0, 0, 0, 0]);
                    let f_prime = QuadraticFunction::from_parts(
                        vec![(0, 0, x_poly.clone())],
                        vec![vec![zero_poly()]],
                        x_poly,
                    );
                    LabradorStatement {
                        f: vec![f_full],
                        f_prime: vec![f_prime],
                    }
                },
                LabradorWitness::new(vec![vec![one_poly()]]),
                17.0,
                true,
            ),
        ];

        for (desc, statement, witness, beta, expect_ok) in cases {
            let result = verify(statement, witness, *beta);
            let ok = result.is_ok();
            assert_eq!(
                ok,
                *expect_ok,
                "{}: expected {}, got {:?} -> {:?}",
                desc,
                if *expect_ok { "Ok" } else { "Err" },
                result,
                result
            );
        }
    }

    #[test]
    fn test_l2_norm() {
        // [3, 4] → 9 + 16 = 25
        assert!(
            (LabradorWitness::new(vec![vec![from_u64(3), from_u64(4)]]).l2_norm_squared() - 25.0)
                .abs()
                < 1e-9
        );

        // s₁=[1], s₂=[2,3] → 1 + 4 + 9 = 14
        assert!(
            (LabradorWitness::new(vec![vec![from_u64(1)], vec![from_u64(2), from_u64(3)]])
                .l2_norm_squared()
                - 14.0)
                .abs()
                < 1e-9
        );

        // Z_17 centered: 16→-1, 15→-2 → 1 + 4 = 5
        assert!(
            (LabradorWitness::new(vec![vec![from_u64(16), from_u64(15)]]).l2_norm_squared() - 5.0)
                .abs()
                < 1e-9,
            "centered L2 norm failed"
        );
    }

    #[test]
    fn test_conjugation() {
        // σ₋₁(X⁰) = X⁰ (constant term unchanged)
        let p = make_poly([5, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(conjugation(&p).coeff(0).to_u64(), 5);

        // σ₋₁(X) = -X^7 in X^8+1
        let x = make_poly([0, 1, 0, 0, 0, 0, 0, 0]);
        let conj = conjugation(&x);
        assert_eq!(conj.coeff(7).to_u64(), F17::modulus() - 1); // -1 mod 17 = 16
        for i in 0..8 {
            if i != 7 {
                assert!(
                    conj.coeff(i).is_zero(),
                    "only coefficient 7 should be nonzero"
                );
            }
        }
    }

    #[test]
    fn test_witness_rank() {
        assert_eq!(
            LabradorWitness::new(vec![vec![one_poly(), zero_poly(), one_poly()]]).rank(),
            3
        );
        let empty: LabradorWitness<Poly8> = LabradorWitness::new(vec![]);
        assert_eq!(empty.rank(), 0);
    }

    #[test]
    fn test_verify_beta_validation() {
        let statement: LabradorStatement<Poly8> = LabradorStatement::new();
        let witness: LabradorWitness<Poly8> = LabradorWitness::new(vec![]);

        for (desc, beta) in &[
            ("NaN", f64::NAN),
            ("negative", -1.0),
            ("infinity", f64::INFINITY),
            ("zero", 0.0),
        ] {
            assert!(
                verify(&statement, &witness, *beta).is_err(),
                "{} beta rejected",
                desc
            );
        }
    }

    #[test]
    fn test_verify_shape_validation() {
        // Out-of-bounds index: ij=(5,5) but only 1 part
        let f_oob = QuadraticFunction::from_parts(
            vec![(5, 5, one_poly())],
            vec![vec![zero_poly()]],
            one_poly(),
        );
        assert!(
            verify(
                &LabradorStatement {
                    f: vec![f_oob],
                    f_prime: vec![]
                },
                &LabradorWitness::new(vec![vec![one_poly()]]),
                17.0
            )
            .is_err(),
            "out-of-bounds index rejected"
        );

        // Inverted index: i > j violates upper-triangular invariant
        let f_inverted = QuadraticFunction::Dense(DenseQuadraticFunction {
            a: vec![one_poly()],
            ij: vec![(1, 0)],
            phi: vec![vec![zero_poly()], vec![zero_poly()]],
            b: zero_poly(),
        });
        assert!(
            verify(
                &LabradorStatement {
                    f: vec![f_inverted],
                    f_prime: vec![]
                },
                &LabradorWitness::new(vec![vec![one_poly()], vec![one_poly()]]),
                17.0
            )
            .is_err(),
            "inverted index (i > j) rejected"
        );

        // Inconsistent part lengths: [2] vs [3]
        assert!(
            verify(
                &LabradorStatement::new(),
                &LabradorWitness::new(vec![
                    vec![one_poly(), one_poly()],
                    vec![one_poly(), one_poly(), one_poly()],
                ]),
                17.0
            )
            .is_err(),
            "inconsistent part lengths rejected"
        );

        // n=0 rejected
        let empty_stmt: LabradorStatement<Poly8> = LabradorStatement::new();
        let empty_part: LabradorWitness<Poly8> = LabradorWitness::new(vec![vec![]]);
        assert!(
            verify(&empty_stmt, &empty_part, 17.0).is_err(),
            "n=0 rejected"
        );

        // phi[0] length 2 but parts[0] length 1
        let f_phi_len = QuadraticFunction::from_parts(
            Vec::new(),
            vec![vec![one_poly(), one_poly()]],
            zero_poly(),
        );
        assert!(
            verify(
                &LabradorStatement {
                    f: vec![f_phi_len],
                    f_prime: vec![]
                },
                &LabradorWitness::new(vec![vec![one_poly()]]),
                17.0
            )
            .is_err(),
            "phi length mismatch rejected"
        );

        // Too few phi vectors: 0 phi but r=1
        let f_phi_few = QuadraticFunction::from_parts(Vec::new(), Vec::new(), zero_poly());
        assert!(
            verify(
                &LabradorStatement {
                    f: vec![f_phi_few],
                    f_prime: vec![]
                },
                &LabradorWitness::new(vec![vec![one_poly()]]),
                17.0
            )
            .is_err(),
            "too few phi vectors rejected"
        );

        // Too many phi vectors: 3 phi but r=2
        let f_phi_many = QuadraticFunction::from_parts(
            Vec::new(),
            vec![vec![one_poly()], vec![one_poly()], vec![one_poly()]],
            zero_poly(),
        );
        assert!(
            verify(
                &LabradorStatement {
                    f: vec![f_phi_many],
                    f_prime: vec![]
                },
                &LabradorWitness::new(vec![vec![one_poly()], vec![one_poly()]]),
                17.0
            )
            .is_err(),
            "too many phi vectors rejected"
        );

        // a/ij mismatch: constructed directly
        let f_mismatch = QuadraticFunction::Dense(DenseQuadraticFunction {
            a: vec![one_poly()],
            ij: vec![(0, 0), (0, 0)],
            phi: vec![vec![zero_poly()]],
            b: zero_poly(),
        });
        assert!(
            verify(
                &LabradorStatement {
                    f: vec![f_mismatch],
                    f_prime: vec![]
                },
                &LabradorWitness::new(vec![vec![one_poly()]]),
                17.0
            )
            .is_err(),
            "a/ij length mismatch rejected"
        );
    }
}
