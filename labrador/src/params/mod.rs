use alloc::format;

mod builder;
mod challenge_profile;
mod jl_profile;

pub use builder::LabradorParamsBuilder;
pub use challenge_profile::{ChallengeProfile, ChallengeShape};
pub use jl_profile::JLProfile;

use crate::String;

/// Complete LaBRADOR protocol configuration.
///
/// Construct via [`LabradorParamsBuilder`] to derive
/// all parameters from base inputs (n, r, β, d, q, τ, κ, κ₁, κ₂).
///
/// # Derivation Chain (§5.4)
///
/// User provides: `n`, `r`, `beta`, `d`, `q`, `kappa`, `kappa1`, `kappa2`, profiles
///
/// Builder derives all remaining fields:
/// - `sigma = beta / √(r·n·d)` — coefficient standard deviation
/// - `b = ⌊√(√(12rτ))·σ⌋` — decomposition base
/// - `t1 = ⌊log₂(q) / log₂(b)⌋` — inner limb count (≥ 2)
/// - `b1 = ⌈q^(1/t1)⌉` — inner limb base
/// - `t2 = ⌊log₂(√(24nd)·σ²) / log₂(b)⌋` — garbage limb count (≥ 2)
/// - `b2 = ⌊(√(24nd)·σ²)^(1/t₂)⌋` — garbage limb base
/// - `gamma = β√τ` — challenge norm bound
/// - `gamma1_sq = b₁²·t₁/12 · r·κ·d + b₂²·t₂/12 · r(r+1)/2 · d` — combined squared norm bound
/// - `gamma2_sq = b₁²·t₁/12 · r(r+1)/2 · d` — inner squared norm bound
/// - `beta_prime = √(2/b²·γ² + γ₁² + γ₂²)` — recursed witness bound
#[derive(
    Debug, Clone, grid_serialize::CanonicalSerialize, grid_serialize::CanonicalDeserialize,
)]
pub struct LabradorParams {
    // Profiles (sub-configurations)
    /// Johnson-Lindenstrauss projection profile (§4).
    pub jl: JLProfile,

    /// Challenge space profile (§2).
    pub challenge: ChallengeProfile,

    // Security & soundness
    /// Target security level in bits.
    pub security_bits: u8,

    /// Soundness error per level: 2^(-security_bits+3).
    pub soundness_error: f64,

    /// Rounds for arithmetic R1CS reduction: ⌈security_bits / log₂(p)⌉.
    /// Derived from `arith_p` and `security_bits`.
    pub l: usize,

    /// Smallest prime factor p of the arithmetic R1CS modulus M = 2^d+1.
    /// Used to derive `l = ceil(security_bits / log2(p))` and validate soundness.
    /// This is NOT the modulus itself — it's the smallest prime dividing M.
    ///
    /// **Trusted parameter**: callers must supply the actual smallest prime factor
    /// of their R1CS modulus. Passing the full modulus M or a non-primal value will
    /// under-derive `l` and overstate soundness. Typical values: 274177 for d=64.
    pub arith_p: u64,

    // Instance geometry
    /// Witness rank — dimension of each sᵢ ∈ R_q^n.
    pub n: usize,

    /// Witness multiplicity — number of witness vectors (s₁..sᵣ).
    pub r: usize,

    /// Norm bound on honest witness (β).
    pub beta: f64,

    // Ring parameters
    /// Ring degree d (from grid-algebra; challenge shape is chosen to match d).
    pub d: usize,

    /// Modulus q (as f64 for norm comparisons).
    pub q: f64,

    // Decomposition (derived from n, r, β, τ, d, q via §5.4)
    /// Coefficient standard deviation σ = β/√(rnd).
    pub sigma: f64,

    /// Decomposition base b.
    pub b: u64,

    /// Inner limb base b₁ (uniform decomposition for inner commitments).
    pub b1: u64,

    /// Garbage limb base b₂ (Gaussian decomposition for garbage polynomials).
    pub b2: u64,

    /// Inner limb count t₁ (number of limbs per inner commitment, ≥ 2).
    pub t1: usize,

    /// Garbage limb count t₂ (number of limbs per garbage polynomial, ≥ 2).
    pub t2: usize,

    // Commitment ranks (estimated via Core-SVP/BDGL per §5.5)
    /// Inner commitment rank κ (A matrix).
    pub kappa: usize,

    /// Outer commitment rank κ₁ (u₁ = B·t + C·g, rank κ₁).
    /// B commits to decomposed inner commitments; C commits to decomposed
    /// garbage g (bound early for soundness: g_ij independent of all challenges).
    pub kappa1: usize,

    /// Outer commitment rank κ₂ (u₂ = D·h, rank κ₂).
    /// D commits to decomposed garbage h (bound after aggregation determines φᵢ).
    pub kappa2: usize,

    // Norm bounds (derived via §5.4)
    /// Challenge norm bound: γ = β√τ.
    pub gamma: f64,

    /// γ₁² as exact u128 for norm-bound comparison (avoids f64 rounding).
    pub gamma1_sq: u128,

    /// γ₂² as exact u128 for norm-bound comparison (avoids f64 rounding).
    pub gamma2_sq: u128,

    /// Recursed witness norm bound: β' = √(2/b²·γ² + γ₁² + γ₂²).
    pub beta_prime: f64,

    // Recursion (derived via §5.7)
    /// Split z into ν parts.
    pub nu: usize,

    /// Split v into μ parts.
    pub mu: usize,

    /// Total recursion levels (6–7 iterations).
    pub num_levels: usize,
}

impl LabradorParams {
    /// Validate internal consistency of derived parameters.
    ///
    /// Checks structural invariants:
    /// 1. β ≤ √(30/128) · q/125 (modulus safety, Thm 5.1)
    /// 2. Challenge shape degree == ring degree d (injectivity, LS18 Cor 1.2)
    /// 3. κ₁ = κ₂ (Thm 5.1 requirement)
    /// 4. JL profile is structurally valid (rows > 0, ordered factors, etc.)
    /// 5. Soundness l meets target security level (Thm 6.2/6.3)
    ///
    /// Does NOT verify Module-SIS rank hardness — use `validate_ranks` for that.
    pub fn validate_params(&self) -> Result<(), String> {
        // 0. Validate JL profile
        if !self.jl.validate() {
            return Err(String::from(
                "JL profile invalid — rows must be > 0, factors must be positive finite, and lower < verify < tail_upper",
            ));
        }

        let q = self.q;
        let beta = self.beta;
        let slack = self.jl.norm_slack(); // √(128/30)

        // 1. Thm 5.1: β ≤ √(30/128) · q/125
        let beta_bound = (1.0 / slack) * q / 125.0;
        if beta > beta_bound {
            return Err(format!(
                "β ({}) exceeds safety bound ({}) — need β ≤ √(30/128)·q/125",
                beta, beta_bound
            ));
        }

        // 2. Challenge space degree must equal ring degree d (LS18 Cor 1.2)
        if self.challenge.shape.degree() != self.d {
            return Err(format!(
                "challenge shape degree ({}) != ring degree d ({}) — injectivity requires equality",
                self.challenge.shape.degree(),
                self.d
            ));
        }

        // 3. Thm 5.1 requires κ₁ = κ₂
        if self.kappa1 != self.kappa2 {
            return Err(format!(
                "kappa1 ({}) != kappa2 ({}) — Thm 5.1 requires equal outer commitment ranks",
                self.kappa1, self.kappa2
            ));
        }

        // 4. §5.3 requires nu, mu >= 1 (used as divisors in split_witness)
        if self.nu < 1 {
            return Err(format!(
                "nu ({}) must be >= 1 — §5.3 requires positive z-split multiplicity",
                self.nu
            ));
        }
        if self.mu < 1 {
            return Err(format!(
                "mu ({}) must be >= 1 — §5.3 requires positive v-split multiplicity",
                self.mu
            ));
        }

        // 5. Thm 6.2/6.3: soundness l must meet target security level
        // For arithmetic R1CS: error per level is 2*p^(-l) where p is the smallest
        // prime factor of the R1CS modulus. With I levels, total error is I*2*p^(-l).
        // The builder derives l = ceil(security_bits / log2(p)) for the configured arith_p.
        // Composition budget from Lemma 3.7: total <= 2^(-bits+3).
        // This validation enforces l >= ceil((security_bits - 2) / log2(p)) as a lower bound.
        // Note: binary R1CS uses a separate l parameter (l_binary >= security_bits)
        if self.l == 0 {
            return Err("l must be >= 1 for soundness".into());
        }
        if self.arith_p <= 1 {
            return Err(format!(
                "arith_p ({}) must be > 1 — soundness prime must be a valid prime factor",
                self.arith_p
            ));
        }
        let p: f64 = self.arith_p as f64;
        let min_l_arith =
            grid_std::ceil((self.security_bits as f64 - 2.0) / grid_std::log2(p)) as usize;
        if self.l < min_l_arith {
            return Err(format!(
                "l ({}) < min_l_arith ({}) for p={} — arithmetic soundness 2*p^(-l) exceeds composition budget",
                self.l, min_l_arith, self.arith_p
            ));
        }

        Ok(())
    }

    /// Validate that commitment ranks satisfy Module-SIS hardness requirements.
    ///
    /// `min_rank_outer` and `min_rank_inner` are the minimum ranks required by
    /// Core-SVP/BDGL estimator output for the respective Module-SIS instances
    /// at the target security level.
    ///
    /// Thm 5.1 requires Module-SIS hard for:
    /// - rank κ₁=κ₂, norm 2β' (`norm_outer` from `module_sis_norms`)
    /// - rank κ, norm max(8T(b+1)β', 2(b+1)β' + 4T·slack·β) (`norm_inner`)
    ///
    /// For recursed proofs (Remark 5.2), the norm bounds are inflated by
    /// √(128/30)^(num_levels - 1) — the first level has no slack, each
    /// subsequent level adds one factor.
    pub fn validate_ranks(
        &self,
        min_rank_outer: usize,
        min_rank_inner: usize,
    ) -> Result<(), String> {
        // Thm 5.1 requires κ₁ = κ₂
        if self.kappa1 != self.kappa2 {
            return Err(format!(
                "kappa1 ({}) != kappa2 ({}) — Thm 5.1 requires equal outer commitment ranks",
                self.kappa1, self.kappa2
            ));
        }

        if self.kappa1 < min_rank_outer || self.kappa2 < min_rank_outer {
            return Err(format!(
                "outer rank too small: kappa1={}, kappa2={}, minimum={} — Module-SIS norm {} at {}-bit security",
                self.kappa1,
                self.kappa2,
                min_rank_outer,
                self.module_sis_norms().0,
                self.security_bits
            ));
        }

        if self.kappa < min_rank_inner {
            return Err(format!(
                "kappa ({}) < minimum inner rank ({}) — Module-SIS norm {} for rank {} at {}-bit security",
                self.kappa,
                min_rank_inner,
                self.module_sis_norms().1,
                min_rank_inner,
                self.security_bits
            ));
        }

        // Thm 5.1: β ≤ √(30/128) · q/125 for Ajtai binding
        #[allow(non_snake_case)]
        {
            let slack = self.jl.norm_slack(); // √(128/30)
            let max_beta = (1.0 / slack) * self.q / 125.0;
            if self.beta > max_beta {
                return Err(format!(
                    "beta ({}) exceeds √(30/128)·q/125 ({}) — Ajtai commitment may not be binding (Thm 5.1)",
                    self.beta, max_beta
                ));
            }
        }

        Ok(())
    }

    /// Module-SIS norm requirements from Thm 5.1.
    ///
    /// Returns `(norm_outer, norm_inner)` where:
    /// - `norm_outer`: Module-SIS norm for rank κ₁=κ₂
    /// - `norm_inner`: Module-SIS norm for rank κ
    ///
    /// For recursed proofs (Remark 5.2), multiply each by √(128/30)^(levels-1)
    /// via `recursed_module_sis_norms`.
    pub fn module_sis_norms(&self) -> (f64, f64) {
        let slack = self.jl.norm_slack();
        #[allow(non_snake_case)]
        let T = self.challenge.T;
        let b = self.b as f64;
        let beta = self.beta;
        let beta_prime = self.beta_prime;

        let norm_outer = 2.0 * beta_prime;
        let norm_inner_a = 8.0 * T * (b + 1.0) * beta_prime;
        let norm_inner_b = 2.0 * (b + 1.0) * beta_prime + 4.0 * T * slack * beta;
        let norm_inner = norm_inner_a.max(norm_inner_b);

        (norm_outer, norm_inner)
    }

    /// Remark 5.2: Module-SIS norms inflated for recursed levels.
    ///
    /// `total_levels` is the total number of recursion levels (including the first).
    /// The first level has no JL slack; each level beyond the first adds a
    /// √(128/30) multiplier. So inflation factor is √(128/30)^(total_levels - 1).
    pub fn recursed_module_sis_norms(&self, total_levels: usize) -> (f64, f64) {
        let (norm_outer, norm_inner) = self.module_sis_norms();
        let slack = self.jl.norm_slack();
        // First level has no slack; subsequent levels each add one factor.
        let inflation_depth = if total_levels > 0 {
            total_levels - 1
        } else {
            0
        };
        let inflation = grid_std::pow(slack, inflation_depth as f64);
        (norm_outer * inflation, norm_inner * inflation)
    }

    /// Garbage soundness: 2r / 2^128 (§5.6).
    pub fn garbage_soundness(&self) -> f64 {
        2.0 * self.r as f64 / grid_std::pow(2.0, 128.0)
    }

    /// Composition budget: I · ε₀ (Lemma 3.7).
    pub fn composition_budget(&self) -> f64 {
        self.num_levels as f64 * self.soundness_error
    }

    /// Total soundness: composition budget + garbage soundness.
    pub fn total_soundness(&self) -> f64 {
        self.composition_budget() + self.garbage_soundness()
    }

    /// Verify that β' matches the paper formula §5.3: β' = √(2/b²·γ² + γ₁² + γ₂²).
    /// Returns the relative error, or `None` if the value is unreasonable.
    pub fn verify_beta_prime(&self) -> Option<f64> {
        let b = self.b as f64;
        let expected = grid_std::sqrt(
            2.0 / (b * b) * self.gamma * self.gamma + self.gamma1_sq as f64 + self.gamma2_sq as f64,
        );
        let rel_err = (self.beta_prime - expected).abs() / expected;
        if rel_err > 1e-6 { None } else { Some(rel_err) }
    }

    /// Verify that γ₁² matches the paper formula §5.4:
    /// γ₁² = b₁²·t₁/12 · r·κ·d + b₂²·t₂/12 · r(r+1)/2 · d.
    /// Returns `true` if the value matches exactly.
    pub fn verify_gamma1_sq(&self) -> bool {
        let expected: u128 = (self.b1 as u128) * (self.b1 as u128) * (self.t1 as u128) / 12
            * (self.r as u128)
            * (self.kappa as u128)
            * (self.d as u128)
            + (self.b2 as u128) * (self.b2 as u128) * (self.t2 as u128) / 12
                * (self.r as u128)
                * (self.r as u128 + 1)
                / 2
                * (self.d as u128);
        self.gamma1_sq == expected
    }

    /// Verify that γ₂² matches the paper formula §5.4:
    /// γ₂² = b₁²·t₁/12 · r(r+1)/2 · d.
    /// Returns `true` if the value matches exactly.
    pub fn verify_gamma2_sq(&self) -> bool {
        let expected: u128 = (self.b1 as u128) * (self.b1 as u128) * (self.t1 as u128) / 12
            * (self.r as u128)
            * (self.r as u128 + 1)
            / 2
            * (self.d as u128);
        self.gamma2_sq == expected
    }
}
