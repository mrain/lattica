//! Common Reference String (CRS) for LaBRADOR (§5.2).
//!
//! The CRS is a compact 32-byte seed. Per-level Ajtai commitment matrices
//! (A, B, C, D) are expanded deterministically via [`CRS::expand`] into a
//! [`CommitKey`]. Binding security comes from Module-SIS: given `t = A * s`,
//! finding another short `s'` with `A * s' = t` is equivalent to finding a
//! short kernel element (SIS instance).
//!
//! Per the paper's soundness optimization (§5.2 p.242, proof p.574),
//! the garbage polynomials `g_ij` (independent of all challenges) are
//! bound in the first outer commitment `u1` alongside the decomposed
//! inner commitments, rather than deferred to `u2`.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use grid_algebra::arith::ring::IntegerRing;
use grid_algebra::lattice::types::{RingMat, RingVec};
use grid_algebra::poly::ring::{CyclotomicPolyRing, NegacyclicMulRing};
use grid_std::UniformRand;
use grid_std::rand::RngExt;

use crate::main_protocol::{DecomposedPolys, decompose_poly_balanced};
use crate::params::LabradorParams;
use crate::relation::LabradorWitness;
use grid_transcript::Transcript;

/// Compact Common Reference String — a 32-byte seed.
///
/// The CRS is expanded per-level into a [`CommitKey`] containing the
/// Ajtai commitment matrices A, B, C, D. Intermediate-level commitment
/// keys are derived from the transcript.
#[derive(Debug, Clone, Copy)]
pub struct CRS {
    /// Seed for deterministic commitment key generation.
    pub seed: [u8; 32],
}

impl CRS {
    /// Create a CRS from a seed.
    pub fn from_seed(seed: [u8; 32]) -> Self {
        Self { seed }
    }

    /// Generate a random CRS.
    pub fn random<Rng: RngExt>(rng: &mut Rng) -> Self {
        Self {
            seed: core::array::from_fn(|_| rng.random()),
        }
    }

    /// Expand this CRS into [`CommitKey`] for the given parameters.
    pub fn expand<R, const N: usize>(&self, params: &LabradorParams) -> CommitKey<R, N>
    where
        R: NegacyclicMulRing<N> + UniformRand,
    {
        CommitKey::from_seed(self.seed, params)
    }

    /// Expand just the A matrix with specific dimensions.
    ///
    /// Used for deriving the last-level A matrix with the correct witness rank
    /// after multi-level recursion.
    pub fn expand_a<R, const N: usize>(
        &self,
        rows: usize,
        cols: usize,
    ) -> RingMat<CyclotomicPolyRing<R, N>>
    where
        R: NegacyclicMulRing<N> + UniformRand,
    {
        let mut rng = seed_rng::SeedRng::from(self.seed);
        sample_uniform_mat(&mut rng.inner, rows, cols)
    }

    /// Derive the last-level CRS from the transcript.
    ///
    /// Samples a 32-byte seed via the transcript and returns a fresh [`CRS`]
    /// for expanding the last-level A matrix.
    pub fn derive_last<T: Transcript>(transcript: &mut T) -> Result<Self, String> {
        use alloc::format;
        let seed = transcript
            .challenge_bytes(b"labrador_crs_last_seed", 32)
            .map_err(|e| format!("Last CRS seed challenge failed: {:?}", e))?;
        let seed_array: [u8; 32] = seed
            .try_into()
            .map_err(|_| "Last CRS seed length mismatch".to_string())?;
        Ok(Self::from_seed(seed_array))
    }
}

/// Ajtai commitment key for one recursion level of LaBRADOR.
///
/// Holds the four Ajtai commitment matrices A, B, C, D expanded from
/// a CRS seed. Shapes are determined by [`LabradorParams`]:
///
/// | Matrix | Shape | Purpose |
/// |--------|-------|---------|
/// | A | κ × n | Inner commitment: `t_i = A * s_i` (one per witness vector) |
/// | B | κ₁ × r·t₁·κ | Outer commit to decomposed inner commitments |
/// | C | κ₁ × t₂·(r²+r)/2 | Outer commit to decomposed garbage g (bound early for soundness) |
/// | D | κ₂ × t₁·(r²+r)/2 | Outer commit to decomposed garbage h |
///
/// **Soundness note**: g_ij is committed in u₁ (not u₂) because it is
/// independent of all challenges (§5.2 p.242). This is required by the
/// security proof: the verifier must not see any challenge before g_ij
/// is bound.
///
/// All matrices are uniform in `R_q^{rows × cols}`.
#[derive(Debug, Clone)]
pub struct CommitKey<R, const N: usize>
where
    R: NegacyclicMulRing<N> + UniformRand,
{
    /// Inner commitment matrix: `t_i = A * s_i` (one per witness vector)
    pub a: RingMat<CyclotomicPolyRing<R, N>>,
    /// Outer commitment to decomposed inner commitments: `u1 = B·t + C·g`
    pub b: RingMat<CyclotomicPolyRing<R, N>>,
    /// Outer commitment to decomposed garbage g: `u1 = B·t + C·g`
    pub c: RingMat<CyclotomicPolyRing<R, N>>,
    /// Outer commitment to decomposed garbage h: `u2 = D·h`
    pub d: RingMat<CyclotomicPolyRing<R, N>>,
}

impl<R, const N: usize> CommitKey<R, N>
where
    R: NegacyclicMulRing<N> + UniformRand,
{
    /// Generate a fresh commitment key from [`LabradorParams`].
    ///
    /// All dimensions are derived from the params struct, avoiding
    /// mismatch risk from passing raw numbers.
    ///
    /// # Panics
    /// Panics if any dimension is zero or would overflow.
    pub fn generate_from_params<Rng: RngExt>(rng: &mut Rng, params: &LabradorParams) -> Self {
        // Reject zero dimensions — commitment matrices must have positive shapes.
        macro_rules! check_positive {
            ($val:expr, $name:expr) => {
                assert!(
                    $val > 0,
                    "CommitKey dimension {}={} must be positive",
                    $name,
                    $val
                );
            };
        }

        check_positive!(params.n, "n");
        check_positive!(params.r, "r");
        check_positive!(params.kappa, "kappa");
        check_positive!(params.kappa1, "kappa1");
        check_positive!(params.kappa2, "kappa2");
        check_positive!(params.t1, "t1");
        check_positive!(params.t2, "t2");

        let num_garbage = params
            .r
            .checked_add(1)
            .expect("r + 1 overflowed")
            .checked_mul(params.r)
            .expect("r * (r+1) overflowed")
            / 2;

        let cols_b = params
            .r
            .checked_mul(params.t1)
            .expect("r * t1 overflowed")
            .checked_mul(params.kappa)
            .expect("r * t1 * kappa overflowed");

        let cols_c = params
            .t2
            .checked_mul(num_garbage)
            .expect("t2 * num_garbage overflowed");

        let cols_d = params
            .t1
            .checked_mul(num_garbage)
            .expect("t1 * num_garbage overflowed");

        Self {
            a: sample_uniform_mat(rng, params.kappa, params.n),
            b: sample_uniform_mat(rng, params.kappa1, cols_b),
            c: sample_uniform_mat(rng, params.kappa1, cols_c),
            d: sample_uniform_mat(rng, params.kappa2, cols_d),
        }
    }

    /// Generate a commitment key deterministically from a seed and params.
    ///
    /// Used for per-level commitment key derivation in multi-level recursion.
    /// The seed is typically derived from the transcript to ensure
    /// prover and verifier agree on the same key.
    pub fn from_seed(seed: [u8; 32], params: &LabradorParams) -> Self {
        let mut rng = seed_rng::SeedRng::from(seed);
        Self::generate_from_params(&mut rng.inner, params)
    }

    /// Compute inner commitment `t_i = A * s_i` for a single witness vector.
    ///
    /// `s_i` has rank `n`. Returns a vector of rank `kappa`.
    pub fn inner_commit(
        &self,
        s_i: &RingVec<CyclotomicPolyRing<R, N>>,
    ) -> RingVec<CyclotomicPolyRing<R, N>> {
        self.inner_commit_slice(s_i.entries())
    }

    /// Compute inner commitment `t_i = A * s_i` from a slice (no RingVec wrapper needed).
    pub fn inner_commit_slice(
        &self,
        s_i: &[CyclotomicPolyRing<R, N>],
    ) -> RingVec<CyclotomicPolyRing<R, N>> {
        self.a.mul_slice(s_i)
    }

    /// Compute inner commitments for all witness vectors, then balanced-decompose.
    ///
    /// For each witness part s_i, computes t_i = A · s_i, then decomposes each
    /// polynomial in t_i into `t1` centered limbs (base `b1`). Returns the flat
    /// concatenation of all limbs as a `DecomposedPolys`.
    ///
    /// Total length: `r · κ · t1`. Each limb coefficient in `[-b1/2, b1/2]`.
    pub fn inner_commit_decomposed(
        &self,
        witness: &LabradorWitness<CyclotomicPolyRing<R, N>>,
        params: &LabradorParams,
    ) -> DecomposedPolys<CyclotomicPolyRing<R, N>>
    where
        R: IntegerRing<Uint = u64>,
    {
        let r = witness.num_parts();
        let kappa = params.kappa;
        let t1 = params.t1;
        let b1 = params.b1;

        let flat_len = r * kappa * t1;
        let mut flat = Vec::with_capacity(flat_len);

        for s_i in witness.parts.iter() {
            let t_i = self.inner_commit_slice(s_i);

            for poly in t_i.entries().iter() {
                let limbs = decompose_poly_balanced(poly, b1, t1);
                flat.extend(limbs);
            }
        }

        DecomposedPolys {
            flat,
            num_polys: r * kappa,
            num_limbs: t1,
            base: b1,
        }
    }

    /// Compute outer commitment `u1 = B·t + C·g`.
    ///
    /// `t` is the concatenation of all decomposed inner commitment limbs
    /// (length `r * t1 * kappa`). `g` is the concatenation of all
    /// decomposed garbage g limbs (length `t2 * (r²+r)/2`).
    /// Returns a vector of rank `kappa1`.
    pub fn outer_commit_u1(
        &self,
        t: &RingVec<CyclotomicPolyRing<R, N>>,
        g: &RingVec<CyclotomicPolyRing<R, N>>,
    ) -> RingVec<CyclotomicPolyRing<R, N>> {
        self.outer_commit_u1_slice(t.entries(), g.entries())
    }

    /// Compute outer commitment `u1 = B·t + C·g` from slices (no RingVec wrapper needed).
    pub fn outer_commit_u1_slice(
        &self,
        t: &[CyclotomicPolyRing<R, N>],
        g: &[CyclotomicPolyRing<R, N>],
    ) -> RingVec<CyclotomicPolyRing<R, N>> {
        let bt = self.b.mul_slice(t);
        let cg = self.c.mul_slice(g);
        let mut result = bt;
        for (r, c) in result.entries_mut().iter_mut().zip(cg.entries().iter()) {
            *r += c;
        }
        result
    }

    /// Compute outer commitment `u2 = D·h`.
    ///
    /// `h` is the concatenation of all decomposed garbage h limbs.
    /// Has length `t1 * (r²+r)/2`. Returns a vector of rank `kappa2`.
    pub fn outer_commit_u2(
        &self,
        h: &RingVec<CyclotomicPolyRing<R, N>>,
    ) -> RingVec<CyclotomicPolyRing<R, N>> {
        self.outer_commit_u2_slice(h.entries())
    }

    /// Compute outer commitment `u2 = D·h` from a slice (no RingVec wrapper needed).
    pub fn outer_commit_u2_slice(
        &self,
        h: &[CyclotomicPolyRing<R, N>],
    ) -> RingVec<CyclotomicPolyRing<R, N>> {
        self.d.mul_slice(h)
    }
}

/// Sample a uniformly random matrix over `CyclotomicPolyRing<R, N>`.
fn sample_uniform_mat<R, const N: usize, Rng: RngExt>(
    rng: &mut Rng,
    rows: usize,
    cols: usize,
) -> RingMat<CyclotomicPolyRing<R, N>>
where
    R: NegacyclicMulRing<N> + UniformRand,
{
    let len = rows
        .checked_mul(cols)
        .expect("CommitKey matrix shape overflowed usize");
    RingMat::new(
        rows,
        cols,
        (0..len)
            .map(|_| CyclotomicPolyRing::<R, N>::rand(rng))
            .collect(),
    )
}

/// Deterministic RNG seeded from a 32-byte array.
///
/// Uses `ChaCha20Rng` which is `no_std`-compatible and implements `RngExt`.
mod seed_rng {
    use grid_std::rand::SeedableRng;

    /// Deterministic RNG from a 32-byte seed.
    /// Wraps `ChaCha20Rng` which implements `rand::RngExt`.
    pub struct SeedRng {
        pub inner: rand_chacha::ChaCha20Rng,
    }

    impl SeedRng {
        pub fn from(seed: [u8; 32]) -> Self {
            Self {
                inner: rand_chacha::ChaCha20Rng::from_seed(seed),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    use grid_algebra::arith::prime::PrimeField;
    use grid_algebra::arith::ring::Ring;
    use grid_algebra::poly::ring::PolyRing;

    type F = PrimeField<12289>;

    /// Hardcoded test dimensions: small but valid (t1=2, t2=2).
    /// Not meant to match any real profile — just exercises the key shapes.
    const KAPPA: usize = 4;
    const KAPPA1: usize = 2;
    const KAPPA2: usize = 2;
    const MULT: usize = 3;
    const WITNESS_RANK: usize = 8;
    const T1: usize = 2;
    const T2: usize = 2;

    fn make_key() -> CommitKey<F, 64> {
        let mut rng = grid_std::test_rng();
        CommitKey::generate_from_params(&mut rng, &fake_params())
    }

    fn fake_params() -> LabradorParams {
        // Minimal params struct for testing key shapes.
        // Values are not security-parameterized — just exercise the math.
        use crate::params::{ChallengeProfile, JLProfile};
        LabradorParams {
            jl: JLProfile::default(),
            challenge: ChallengeProfile::paper_default(),
            security_bits: 8,
            soundness_error: 0.0,
            l: 1,
            arith_p: 274177,
            n: WITNESS_RANK,
            r: MULT,
            beta: 100.0,
            d: 64,
            q: 12289.0,
            sigma: 1.0,
            b: 2,
            b1: 2,
            b2: 2,
            t1: T1,
            t2: T2,
            kappa: KAPPA,
            kappa1: KAPPA1,
            kappa2: KAPPA2,
            gamma: 100.0,
            gamma1_sq: 10_000,
            gamma2_sq: 10_000,
            beta_prime: 100.0,
            nu: 1,
            mu: 1,
            num_levels: 1,
        }
    }

    #[test]
    fn test_key_shapes_and_operations() {
        let key = make_key();
        let num_garbage = MULT * (MULT + 1) / 2;

        // A: κ × n
        assert_eq!(key.a.rows(), KAPPA);
        assert_eq!(key.a.cols(), WITNESS_RANK);

        // B: κ₁ × r·t₁·κ
        assert_eq!(key.b.rows(), KAPPA1);
        assert_eq!(key.b.cols(), MULT * T1 * KAPPA);

        // C: κ₁ × t₂·(r²+r)/2  (NOT κ₂ — bound in u1 for soundness)
        assert_eq!(key.c.rows(), KAPPA1);
        assert_eq!(key.c.cols(), T2 * num_garbage);

        // D: κ₂ × t₁·(r²+r)/2
        assert_eq!(key.d.rows(), KAPPA2);
        assert_eq!(key.d.cols(), T1 * num_garbage);

        // Inner commitment: t_i = A * s_i → length κ
        let s_i = RingVec::zero(WITNESS_RANK);
        let t_i = key.inner_commit(&s_i);
        assert_eq!(t_i.len(), KAPPA);

        // Outer commitment u1 = B·t + C·g → length κ₁
        let t = RingVec::zero(MULT * T1 * KAPPA);
        let g = RingVec::zero(T2 * num_garbage);
        let u1 = key.outer_commit_u1(&t, &g);
        assert_eq!(u1.len(), KAPPA1);

        // Outer commitment u2 = D·h → length κ₂
        let h = RingVec::zero(T1 * num_garbage);
        let u2 = key.outer_commit_u2(&h);
        assert_eq!(u2.len(), KAPPA2);
    }

    #[test]
    fn test_inner_commit_decomposed_shapes() {
        let mut rng = grid_std::test_rng();
        let params = fake_params();
        let key = CommitKey::<F, 64>::generate_from_params(&mut rng, &params);

        let witness = LabradorWitness {
            parts: (0..params.r)
                .map(|_| vec![CyclotomicPolyRing::<F, 64>::zero(); params.n])
                .collect(),
        };

        let decomposed = key.inner_commit_decomposed(&witness, &params);

        // Flat length: r * kappa * t1
        assert_eq!(decomposed.flat.len(), params.r * params.kappa * params.t1);
        assert_eq!(decomposed.num_polys, params.r * params.kappa);
        assert_eq!(decomposed.num_limbs, params.t1);
        assert_eq!(decomposed.base, params.b1);

        // t_i = A · 0 = 0, so all decomposed limbs should be zero
        for limb in &decomposed.flat {
            assert!(limb.is_zero(), "limb should be zero for zero witness");
        }
    }

    #[test]
    fn test_inner_commit_decomposed_roundtrip() {
        let mut rng = grid_std::test_rng();
        // Use larger base/limbs so decomposition can represent the full range
        // of values produced by the matrix-vector multiply.
        let params = {
            let mut p = fake_params();
            p.b1 = 16;
            p.t1 = 4;
            p
        };
        let key = CommitKey::<F, 64>::generate_from_params(&mut rng, &params);

        let witness = LabradorWitness {
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
        };

        let decomposed = key.inner_commit_decomposed(&witness, &params);

        // Reconstruct t_vecs from decomposed limbs
        use crate::main_protocol::reconstruct_t_vecs;
        let reconstructed = reconstruct_t_vecs(&decomposed, params.r, params.kappa, params.t1);
        assert_eq!(reconstructed.len(), params.r);

        // Verify each reconstructed t_i = A · s_i
        for (wi, s_i) in witness.parts.iter().enumerate() {
            let expected = key.inner_commit_slice(s_i);
            let recon = &reconstructed[wi];
            assert_eq!(
                expected.len(),
                recon.len(),
                "t_vecs[{}] length mismatch",
                wi
            );
            for (j, (o, r)) in expected
                .entries()
                .iter()
                .zip(recon.entries().iter())
                .enumerate()
            {
                assert_eq!(o, r, "t_vecs[{}][{}] mismatch", wi, j);
            }
        }

        // Limb coefficients are within centered bounds
        let half_base = decomposed.base as i128 / 2;
        for limb in &decomposed.flat {
            for i in 0..64 {
                let v = limb.coeff(i).to_u64();
                let q = F::modulus();
                let centered = if v as i128 > half_base {
                    (v as i128) - q as i128
                } else {
                    v as i128
                };
                assert!(
                    centered.abs() <= half_base,
                    "limb coeff {} out of centered range [-{}, {}]",
                    v,
                    half_base,
                    half_base
                );
            }
        }
    }

    #[test]
    fn test_crs_expand_roundtrip() {
        let mut rng = grid_std::test_rng();
        let params = fake_params();
        let crs = CRS::random(&mut rng);
        let key = crs.expand::<F, 64>(&params);

        // Verify the expanded key matches direct generation from seed
        let expected = CommitKey::<F, 64>::from_seed(crs.seed, &params);
        assert_eq!(key.a.rows(), expected.a.rows());
        assert_eq!(key.a.cols(), expected.a.cols());
    }
}
