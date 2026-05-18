use alloc::format;
use alloc::string::String;

use super::{ChallengeProfile, JLProfile, LabradorParams};

/// Builder for [`LabradorParams`] with automatic derivation (§5.4).
///
/// User provides base inputs; builder derives decomposition and norm parameters.
///
/// # Derivation Chain
///
/// 1. `sigma = beta / √(r·n·d)` — coefficient standard deviation
/// 2. `b = ⌊√(√(12rτ))·σ⌋` — decomposition base
/// 3. `t1 = ⌊log₂(q) / log₂(b)⌋` — inner limb count (≥ 2)
/// 4. `b1 = ⌈q^(1/t1)⌉` — inner limb base
/// 5. `t2 = ⌊log₂(√(24nd)·σ²) / log₂(b)⌋` — garbage limb count (≥ 2)
/// 6. `b2 = ⌊(√(24nd)·σ²)^(1/t₂)⌋` — garbage limb base
/// 7. `gamma = β√τ` — challenge norm bound
/// 8. `gamma2_sq = b₁²·t₁/12 · r(r+1)/2 · d` — h-garbage squared norm bound (opening for u₂ = D·h)
/// 9. `gamma1_sq = b₁²·t₁/12 · r·κ·d + b₂²·t₂/12 · r(r+1)/2 · d` — t-limbs + g-garbage squared norm bound (opening for u₁ = B·t + C·g)
/// 10. `beta_prime = √(2/b²·γ² + γ₁² + γ₂²)` — recursed witness bound
///
/// # Security Parameters
///
/// `security_bits` derives:
/// - `soundness_error = 2^(-security_bits + 3)`
/// - `l = ⌈security_bits / log₂(p)⌉` for p=274177
pub struct LabradorParamsBuilder {
    // Profiles
    jl: JLProfile,
    challenge: ChallengeProfile,

    // Security
    security_bits: u8,

    // Instance geometry
    n: usize,
    r: usize,
    beta: f64,

    // Ring parameters
    d: usize,
    q: f64,

    // Commitment ranks (Core-SVP/BDGL)
    kappa: usize,
    kappa1: usize,
    kappa2: usize,

    // Recursion
    nu: Option<usize>,
    mu: Option<usize>,
    num_levels: Option<usize>,

    // Soundness prime: smallest prime factor of R1CS modulus M = 2^d+1
    arith_p: Option<u64>,

    // Module-SIS rank validation (from Core-SVP/BDGL estimator)
    min_rank_outer: Option<usize>,
    min_rank_inner: Option<usize>,
}

impl LabradorParamsBuilder {
    /// Create a new builder with default JL and challenge profiles.
    pub fn new(n: usize, r: usize, beta: f64) -> Self {
        Self {
            jl: JLProfile::paper_default(),
            challenge: ChallengeProfile::paper_default(),
            security_bits: 128,
            n,
            r,
            beta,
            d: 0,
            q: 0.0,
            kappa: 0,
            kappa1: 0,
            kappa2: 0,
            nu: None,
            mu: None,
            num_levels: None,
            arith_p: None,
            min_rank_outer: None,
            min_rank_inner: None,
        }
    }

    // --- Profile setters ---

    /// Set JL profile (default: paper §4, 256 rows).
    pub fn jl(mut self, jl: JLProfile) -> Self {
        self.jl = jl;
        self
    }

    /// Set challenge profile (default: paper §2, d=64).
    pub fn challenge(mut self, challenge: ChallengeProfile) -> Self {
        self.challenge = challenge;
        self
    }

    /// Set security level in bits (default: 128).
    /// Derives soundness_error = 2^(-bits+3) and l = ⌈bits / log₂(p)⌉.
    pub fn security_bits(mut self, bits: u8) -> Self {
        self.security_bits = bits;
        self
    }

    // --- Ring setters ---

    /// Set ring degree d (required for derivation).
    pub fn d(mut self, d: usize) -> Self {
        self.d = d;
        self
    }

    /// Set modulus q as f64 (required for derivation).
    pub fn q(mut self, q: f64) -> Self {
        self.q = q;
        self
    }

    // --- Commitment rank setters ---

    /// Set inner commitment rank κ.
    pub fn kappa(mut self, kappa: usize) -> Self {
        self.kappa = kappa;
        self
    }

    /// Set outer commitment B rank κ₁.
    pub fn kappa1(mut self, kappa1: usize) -> Self {
        self.kappa1 = kappa1;
        self
    }

    /// Set outer commitment D rank κ₂.
    pub fn kappa2(mut self, kappa2: usize) -> Self {
        self.kappa2 = kappa2;
        self
    }

    // --- Recursion setters ---

    /// Set z-split count ν.
    pub fn nu(mut self, nu: usize) -> Self {
        self.nu = Some(nu);
        self
    }

    /// Set v-split count μ.
    pub fn mu(mut self, mu: usize) -> Self {
        self.mu = Some(mu);
        self
    }

    /// Set total recursion levels.
    pub fn num_levels(mut self, num: usize) -> Self {
        self.num_levels = Some(num);
        self
    }

    /// Set the smallest prime factor p of the arithmetic R1CS modulus M = 2^d+1.
    ///
    /// The soundness parameter `l` is derived as `ceil(security_bits / log2(p))`.
    /// This value must be > 1. Default is `p = 274177` (smallest prime factor of 2^64+1).
    ///
    /// **Trusted**: callers must supply the actual smallest prime factor of their
    /// R1CS modulus. Passing the full modulus M or a composite value will under-derive
    /// `l` and overstate soundness.
    pub fn arith_p(mut self, p: u64) -> Self {
        self.arith_p = Some(p);
        self
    }

    /// Set minimum outer commitment rank for Module-SIS hardness validation.
    ///
    /// When set, `build()` will validate that κ₁ and κ₂ meet this minimum via
    /// `LabradorParams::validate_ranks`. The value should be derived from a
    /// Core-SVP/BDGL lattice estimator for the target security level.
    pub fn min_rank_outer(mut self, rank: usize) -> Self {
        self.min_rank_outer = Some(rank);
        self
    }

    /// Set minimum inner commitment rank for Module-SIS hardness validation.
    ///
    /// When set, `build()` will validate that κ meets this minimum via
    /// `LabradorParams::validate_ranks`. The value should be derived from a
    /// Core-SVP/BDGL lattice estimator for the target security level.
    pub fn min_rank_inner(mut self, rank: usize) -> Self {
        self.min_rank_inner = Some(rank);
        self
    }

    /// Build [`LabradorParams`], deriving decomposition and norm parameters.
    ///
    /// If [`min_rank_outer`] and [`min_rank_inner`] were set, Module-SIS rank
    /// hardness is validated via [`LabradorParams::validate_ranks`]. Without them,
    /// the builder skips rank validation — the caller must invoke
    /// [`LabradorParams::validate_ranks`] manually to ensure commitment ranks meet
    /// the target security level per Thm 5.1.
    ///
    /// [`min_rank_outer`]: Self::min_rank_outer
    /// [`min_rank_inner`]: Self::min_rank_inner
    pub fn build(self) -> Result<LabradorParams, String> {
        self.validate_inputs()?;
        let ranks = (self.min_rank_outer, self.min_rank_inner);
        let derived = self.derive()?;
        derived.validate_params()?;
        if let (Some(outer), Some(inner)) = ranks {
            derived.validate_ranks(outer, inner)?;
        }
        Ok(derived)
    }

    /// Validate all required inputs are set.
    fn validate_inputs(&self) -> Result<(), String> {
        if self.n == 0 {
            return Err(String::from("witness rank n must be > 0"));
        }
        if self.r == 0 {
            return Err(String::from("witness multiplicity r must be > 0"));
        }
        if self.beta <= 0.0 {
            return Err(String::from("norm bound beta must be > 0"));
        }
        if self.d == 0 {
            return Err(String::from("ring degree d must be > 0"));
        }
        if self.q <= 0.0 {
            return Err(String::from("modulus q must be > 0"));
        }
        if self.kappa == 0 {
            return Err(String::from("inner commitment rank kappa must be > 0"));
        }
        if self.kappa1 == 0 {
            return Err(String::from("outer commitment rank kappa1 must be > 0"));
        }
        if self.kappa2 == 0 {
            return Err(String::from("outer commitment rank kappa2 must be > 0"));
        }
        if self.security_bits < 8 {
            return Err(format!(
                "security_bits ({}) too low — minimum 8 required (soundness_error would be >= 2)",
                self.security_bits
            ));
        }
        // Validate arith_p (smallest prime factor of R1CS modulus)
        if self.arith_p.is_some_and(|p| p <= 1) {
            return Err(format!(
                "arith_p ({}) must be > 1 — soundness prime must be a valid prime factor",
                self.arith_p.unwrap()
            ));
        }
        // Validate JL profile
        if !self.jl.validate() {
            return Err(String::from(
                "JL profile invalid — rows must be > 0, factors must be positive finite, and lower < verify < tail_upper",
            ));
        }
        Ok(())
    }

    /// Derive decomposition and norm parameters from base inputs (§5.4).
    fn derive(self) -> Result<LabradorParams, String> {
        let tau = self.challenge.tau();

        // 1. σ = β / √(rnd)
        let sigma = self.beta / grid_std::sqrt(self.r as f64 * self.n as f64 * self.d as f64);

        // 2. b = ⌊√(√(12rτ))·σ⌋
        let b_factor = grid_std::sqrt(grid_std::sqrt(12.0 * self.r as f64 * tau));
        let b = grid_std::floor(b_factor * sigma) as u64;
        if b < 2 {
            return Err(format!(
                "decomposition base b={} < 2 — witness too small for ring parameters",
                b
            ));
        }

        // 3. t1 = ⌊log₂(q) / log₂(b)⌋
        let log2_q = grid_std::log2(self.q);
        let log2_b = grid_std::log2(b as f64);
        let t1 = grid_std::floor(log2_q / log2_b) as usize;
        if t1 < 2 {
            return Err(format!(
                "inner limb count t1={} < 2 — base b={} too large for q={} (protocol requires decomposition into ≥ 2 parts)",
                t1, b, self.q
            ));
        }

        // 4. b1 = ⌈q^(1/t1)⌉
        let b1 = grid_std::ceil(grid_std::pow(self.q, 1.0 / t1 as f64)) as u64;

        // 5. t2 = ⌊log₂(√(24nd)·σ²) / log₂(b)⌋
        let garbage_width = grid_std::sqrt(24.0 * self.n as f64 * self.d as f64) * sigma * sigma;
        let t2 = grid_std::floor(grid_std::log2(garbage_width) / log2_b) as usize;
        if t2 < 2 {
            return Err(format!(
                "garbage limb count t2={} < 2 — protocol requires decomposition into ≥ 2 parts",
                t2
            ));
        }

        // 6. b2 = ⌊(√(24nd)·σ²)^(1/t₂)⌋
        let b2 = grid_std::floor(grid_std::pow(garbage_width, 1.0 / t2 as f64)) as u64;

        // 7. γ = β√τ
        let gamma = self.beta * grid_std::sqrt(tau);

        // Exact u128 squared gamma values for norm-bound comparison
        // γ₂² = b₁²·t₁/12 · r(r+1)/2 · d
        let r12_u128 = self.r as u128;
        let r_plus_1_u128 = (self.r + 1) as u128;
        let kappa_u128 = self.kappa as u128;
        let d_u128 = self.d as u128;
        let gamma2_sq: u128 =
            (b1 as u128) * (b1 as u128) * (t1 as u128) / 12 * r12_u128 * r_plus_1_u128 / 2 * d_u128;
        // γ₁² = b₁²·t₁/12 · r·κ·d + b₂²·t₂/12 · r(r+1)/2 · d
        let gamma1_sq: u128 =
            (b1 as u128) * (b1 as u128) * (t1 as u128) / 12 * r12_u128 * kappa_u128 * d_u128
                + (b2 as u128) * (b2 as u128) * (t2 as u128) / 12 * r12_u128 * r_plus_1_u128 / 2
                    * d_u128;

        // 10. β' = √(2/b²·γ² + γ₁² + γ₂²)
        let beta_prime = grid_std::sqrt(
            2.0 / (b as f64 * b as f64) * gamma * gamma + gamma1_sq as f64 + gamma2_sq as f64,
        );

        // Security-derived parameters
        let bits = self.security_bits as f64;
        let soundness_error = grid_std::pow(2.0, -bits + 3.0);
        let p = match self.arith_p {
            Some(p) => p,
            None if self.d == 64 => 274177,
            None => {
                return Err(format!(
                    "arith_p must be set explicitly for d={} (default 274177 is only valid for d=64)",
                    self.d
                ));
            }
        };
        let l = grid_std::ceil(bits / grid_std::log2(p as f64)) as usize;

        let nu = self.nu.unwrap_or(1);
        let mu = self.mu.unwrap_or(1);
        let num_levels = self.num_levels.unwrap_or(7);

        if nu < 1 {
            return Err("nu must be >= 1 (§5.3 positive z-split multiplicity)".into());
        }
        if mu < 1 {
            return Err("mu must be >= 1 (§5.3 positive v-split multiplicity)".into());
        }

        Ok(LabradorParams {
            jl: self.jl,
            challenge: self.challenge,
            security_bits: self.security_bits,
            soundness_error,
            l,
            arith_p: p,
            n: self.n,
            r: self.r,
            beta: self.beta,
            d: self.d,
            q: self.q,
            sigma,
            b,
            b1,
            b2,
            t1,
            t2,
            kappa: self.kappa,
            kappa1: self.kappa1,
            kappa2: self.kappa2,
            gamma,
            gamma1_sq,
            gamma2_sq,
            beta_prime,
            nu,
            mu,
            num_levels,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derivation_chain() {
        // Test with n=256 to get t1 >= 2 and t2 >= 2; d=64 matches default challenge profile
        let params = LabradorParamsBuilder::new(256, 4, 4096.0)
            .d(64)
            .q(8380417.0)
            .kappa(16)
            .kappa1(8)
            .kappa2(8)
            .build()
            .expect("builder should succeed");

        // Verify derived values are non-zero and reasonable
        assert!(params.sigma > 0.0, "sigma should be positive");
        assert!(params.b >= 2, "decomposition base b should be >= 2");
        assert!(params.t1 >= 2, "t1 should be >= 2 (paper §5.4)");
        assert!(params.b1 >= 2, "b1 should be >= 2");
        assert!(params.t2 >= 2, "t2 should be >= 2 (paper §5.4)");
        assert!(params.b2 >= 2, "b2 should be >= 2");
        assert!(params.gamma > 0.0, "gamma should be positive");
        assert!(params.gamma1_sq > 0, "gamma1_sq should be positive");
        assert!(params.gamma2_sq > 0, "gamma2_sq should be positive");
        assert!(params.beta_prime > 0.0, "beta_prime should be positive");

        // Verify gamma = β√τ
        let expected_gamma = params.beta * grid_std::sqrt(params.challenge.tau());
        assert!(
            (params.gamma - expected_gamma).abs() < 1e-6,
            "gamma mismatch: got {}, expected {}",
            params.gamma,
            expected_gamma
        );

        // Verify sigma = β/√(rnd)
        let expected_sigma =
            params.beta / grid_std::sqrt(params.r as f64 * params.n as f64 * params.d as f64);
        assert!(
            (params.sigma - expected_sigma).abs() < 1e-6,
            "sigma mismatch: got {}, expected {}",
            params.sigma,
            expected_sigma
        );

        // Verify gamma2_sq = b₁²·t₁/12 · r(r+1)/2 · d
        let expected_gamma2_sq: u128 =
            (params.b1 as u128) * (params.b1 as u128) * (params.t1 as u128) / 12
                * (params.r as u128)
                * (params.r as u128 + 1)
                / 2
                * (params.d as u128);
        assert!(
            params.gamma2_sq == expected_gamma2_sq,
            "gamma2_sq mismatch: got {}, expected {}",
            params.gamma2_sq,
            expected_gamma2_sq
        );

        // Verify gamma1_sq = b₁²·t₁/12 · r·κ·d + b₂²·t₂/12 · r(r+1)/2 · d
        let expected_gamma1_sq: u128 =
            (params.b1 as u128) * (params.b1 as u128) * (params.t1 as u128) / 12
                * (params.r as u128)
                * (params.kappa as u128)
                * (params.d as u128)
                + (params.b2 as u128) * (params.b2 as u128) * (params.t2 as u128) / 12
                    * (params.r as u128)
                    * (params.r as u128 + 1)
                    / 2
                    * (params.d as u128);
        assert!(
            params.gamma1_sq == expected_gamma1_sq,
            "gamma1_sq mismatch: got {}, expected {}",
            params.gamma1_sq,
            expected_gamma1_sq
        );
    }

    #[test]
    fn test_validation_fails_invalid_config() {
        let cases = [
            (
                "missing d",
                LabradorParamsBuilder::new(1, 4, 65536.0)
                    .kappa(1)
                    .kappa1(1)
                    .kappa2(1)
                    .build(),
            ),
            (
                "missing q",
                LabradorParamsBuilder::new(1, 4, 65536.0)
                    .d(64)
                    .kappa(1)
                    .kappa1(1)
                    .kappa2(1)
                    .build(),
            ),
            (
                "security_bits too low",
                LabradorParamsBuilder::new(256, 4, 4096.0)
                    .d(64)
                    .q(4294967291.0)
                    .kappa(16)
                    .kappa1(8)
                    .kappa2(8)
                    .security_bits(4)
                    .build(),
            ),
        ];

        for (name, result) in cases {
            assert!(result.is_err(), "expected error for case '{}'", name);
        }
    }

    #[test]
    fn test_arith_p_required_for_non64_d() {
        // d=128 without arith_p should fail
        let result = LabradorParamsBuilder::new(256, 4, 4096.0)
            .d(128)
            .q(4294967291.0)
            .kappa(16)
            .kappa1(8)
            .kappa2(8)
            .build();
        assert!(result.is_err(), "d=128 without arith_p should fail");
        let err = result.unwrap_err();
        assert!(
            err.contains("arith_p must be set explicitly"),
            "error should mention arith_p, got: {}",
            err
        );

        // d=64 without arith_p should succeed (uses default)
        let result = LabradorParamsBuilder::new(256, 4, 4096.0)
            .d(64)
            .q(4294967291.0)
            .kappa(16)
            .kappa1(8)
            .kappa2(8)
            .build();
        assert!(result.is_ok(), "d=64 without arith_p should use default");
        assert_eq!(result.unwrap().arith_p, 274177);
    }

    #[test]
    fn test_paper_profile_validation() {
        // With q ≈ 2^32 and n=256 to satisfy t1>=2, t2>=2
        let params = LabradorParamsBuilder::new(256, 4, 4096.0)
            .d(64)
            .q(4294967291.0)
            .kappa(16)
            .kappa1(8)
            .kappa2(8)
            .build()
            .expect("builder should succeed");

        params
            .validate_params()
            .expect("should pass parameter validation");
    }

    #[test]
    fn test_recursion_inflation_off_by_one() {
        let params = LabradorParamsBuilder::new(256, 4, 4096.0)
            .d(64)
            .q(4294967291.0)
            .kappa(16)
            .kappa1(8)
            .kappa2(8)
            .build()
            .expect("builder should succeed");

        let slack = params.jl.norm_slack(); // √(128/30)
        let (base_outer, base_inner) = params.module_sis_norms();

        // 1 level: no inflation (first level has no JL slack)
        let (o1, i1) = params.recursed_module_sis_norms(1);
        assert!(
            (o1 - base_outer).abs() < 1e-6,
            "1 level should have no inflation"
        );
        assert!(
            (i1 - base_inner).abs() < 1e-6,
            "1 level should have no inflation"
        );

        // 2 levels: one factor of slack
        let (o2, i2) = params.recursed_module_sis_norms(2);
        assert!(
            (o2 - base_outer * slack).abs() < 1e-6,
            "2 levels should inflate by slack^1"
        );
        assert!(
            (i2 - base_inner * slack).abs() < 1e-6,
            "2 levels should inflate by slack^1"
        );

        // 3 levels: two factors of slack
        let (o3, _i3) = params.recursed_module_sis_norms(3);
        assert!(
            (o3 - base_outer * slack * slack).abs() < 1e-6,
            "3 levels should inflate by slack^2"
        );
    }

    #[test]
    fn test_security_bits_derives_dependents() {
        let params = LabradorParamsBuilder::new(256, 4, 4096.0)
            .d(64)
            .q(4294967291.0)
            .kappa(16)
            .kappa1(8)
            .kappa2(8)
            .security_bits(128)
            .build()
            .expect("builder should succeed");

        // soundness_error should be 2^(-128+3) = 2^-125
        assert!((params.soundness_error - grid_std::pow(2.0, -125.0)).abs() < 1e-20);
        // l should be ceil(128 / log2(274177)) = ceil(128/17.99) = 8
        assert_eq!(params.l, 8);
    }

    #[test]
    fn test_rank_validation_in_build() {
        // With valid rank minimums, build succeeds
        let params = LabradorParamsBuilder::new(256, 4, 4096.0)
            .d(64)
            .q(4294967291.0)
            .kappa(16)
            .kappa1(8)
            .kappa2(8)
            .min_rank_outer(4)
            .min_rank_inner(8)
            .build()
            .expect("build should succeed with valid ranks");
        assert_eq!(params.kappa, 16);
        assert_eq!(params.kappa1, 8);

        // Outer rank too high — kappa1=8 < min_rank_outer=16
        let result = LabradorParamsBuilder::new(256, 4, 4096.0)
            .d(64)
            .q(4294967291.0)
            .kappa(16)
            .kappa1(8)
            .kappa2(8)
            .min_rank_outer(16)
            .min_rank_inner(8)
            .build();
        let err = result.unwrap_err();
        assert!(
            err.contains("outer rank too small"),
            "error should mention outer rank"
        );

        // Inner rank too high — kappa=16 < min_rank_inner=32
        let result = LabradorParamsBuilder::new(256, 4, 4096.0)
            .d(64)
            .q(4294967291.0)
            .kappa(16)
            .kappa1(8)
            .kappa2(8)
            .min_rank_outer(4)
            .min_rank_inner(32)
            .build();
        let err = result.unwrap_err();
        assert!(
            err.contains("kappa") && err.contains("minimum inner rank"),
            "error should mention kappa and inner rank"
        );

        // Without rank minimums, build succeeds (no rank validation)
        let result = LabradorParamsBuilder::new(256, 4, 4096.0)
            .d(64)
            .q(4294967291.0)
            .kappa(16)
            .kappa1(8)
            .kappa2(8)
            .build();
        assert!(result.is_ok(), "build without rank minimums should succeed");
    }
}
