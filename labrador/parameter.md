# LaBRADOR Parameters

Profile-based instantiation: `LabradorParamsBuilder` derives all parameters from base inputs (n, r, β, d, q, κ, κ₁, κ₂). `validate_params()` checks structural consistency; `validate_ranks(min_outer, min_inner)` verifies Module-SIS rank hardness.

This document is the current source of truth for both the implemented parameter surface and the
remaining parameter-generation roadmap.

---

## Universal Constants (hardcode as `const`)

Purely mathematical — independent of d, q, and security tuning.

| Constant | Value | Source |
|----------|-------|--------|
| Norm slack | √(128/30) ≈ 2.07 | §4: JL lower/upper bound ratio |

**Not here:** d (ring degree) and q (modulus) come from instantiation from algebra crate. The β condition `β ≤ √(30/128) · q/125` (Thm 5.1) depends on q and is enforced by `validate_params()` at runtime, not a compile-time constant.

---

## JL Profile (`JLProfile`)

Self-contained Johnson-Lindenstrauss configuration. Defined in `params/jl_profile.rs`. Shipped default matches Lemma 4.1 (GHL21). Different profiles may use different dimensions/distributions for different tail-bound security.

The verification threshold is `||p|| ≤ √128 · β` (§5.2), NOT √337·β. The √337 upper bound (Lemma 4.1) is a tail bound guarantee that the prover's honest projection won't exceed it with overwhelming probability — it is not used in verification. The norm slack √(128/30) comes from the ratio of the verification threshold (√128) to the lower bound (√30).

```rust
pub struct JLProfile {
    pub rows:            usize,    // 256 — projection output dimension
    pub security_bits:   u8,       // 128 — tail bound gives 2^-128 failure probability
    pub lower_factor:    f64,      // √30 — Lemma 4.1 lower bound: if ||p||₂ < √30·b, witness is too large
    pub verify_factor:   f64,      // √128 — protocol acceptance threshold: verifier checks ||p|| ≤ √128·β
    pub tail_upper:      f64,      // √337 — Lemma 4.1 upper tail: ||p|| > √337·||w|| with prob < 2^-128 (not used in verification)
}
```

`LabradorParams` holds a `JLProfile` by value. Shipped default: `(256, 128-bit, √30, √128, √337)`.

---

## Challenge Profile (`ChallengeProfile`)

Self-contained challenge space configuration. Defined in `params/challenge_profile.rs`. Shape depends on d (coefficients must sum to d). Shipped default matches §2 for d=64.

```rust
pub struct ChallengeProfile {
    pub shape:           ChallengeShape, // {0: 23, ±1: 31, ±2: 10} for d=64
    pub T:               f64,     // 15 — operator norm rejection threshold
    pub space_bits:      u8,      // 128 — log2(|C|), collision resistance
}
// tau() is a method: shape.tau() → 71 (||c||²₂ squared L2 norm, derived from shape)
```

`LabradorParams` holds a `ChallengeProfile` by value. Shipped default: `(d=64 shape, τ=71, T=15, 128-bit)`. Invertible differences guaranteed by LS18 Cor 1.2 for cyclotomic rings.

---

## LabradorParams

Complete protocol configuration. Instantiation is profile-based:
- `LabradorParamsBuilder::new(n, r, beta)` — builder derives all fields from base inputs
- `validate_params()` — structural checks:
    - β ≤ √(30/128) · q/125 (modulus safety, Thm 5.1)
    - κ₁ = κ₂ (Thm 5.1 requirement)
    - Challenge shape degree == ring degree d (LS18 Cor 1.2)
    - JL profile is structurally valid (rows > 0, ordered factors, etc.)
    - Soundness l meets target security level (Thm 6.2/6.3)
    - nu, mu >= 1 (§5.3 requirement)
- `validate_ranks(min_outer, min_inner)` — Module-SIS rank validation:
    - κ₁ = κ₂ ≥ min_rank_outer (outer commitment, norm 2β')
    - κ ≥ min_rank_inner (inner commitment, norm from Thm 5.1)
    - **Remark 5.2**: recursed proofs inflate Module-SIS norms by √(128/30)^(levels-1)

```rust
pub struct LabradorParams {
    // Profiles (sub-configurations)
    pub jl:              JLProfile,
    pub challenge:       ChallengeProfile,

    // Security & soundness
    pub security_bits:   u8,       // 128
    pub soundness_error: f64,      // 2^-125
    pub l:               usize,    // ⌈security_bits / log2(p)⌉ — rounds for arithmetic R1CS (Figure 5)
    pub arith_p:         u64,      // smallest prime factor of R1CS modulus M = 2^d+1 (e.g., 274177 for d=64)

    // Instance geometry
    pub n:               usize,    // witness rank
    pub r:               usize,    // witness multiplicity
    pub beta:            f64,      // norm bound on honest witness

    // Ring parameters
    pub d:               usize,    // ring degree
    pub q:               f64,      // modulus (as f64 for norm comparisons)

   // Decomposition (derived from n, r, β, τ via §5.4)
    pub sigma:           f64,      // β / √(rnd) — coefficient standard deviation
    pub b:               u64,
    pub b1:              u64,
    pub b2:              u64,
    pub t1:              usize,
    pub t2:              usize,

    // Commitment ranks (estimated via Core-SVP/BDGL per §5.5, stored as profile constants)
    pub kappa:           usize,    // inner commitment A — estimated, not formula-derived
    pub kappa1:          usize,    // outer commitment B — estimated, not formula-derived
    pub kappa2:          usize,    // outer commitment D (u₂ = D·h) — estimated, not formula-derived

    // Norm bounds (derived via §5.4)
    pub gamma:           f64,
    pub gamma1_sq:       u128,    // γ₁² as exact u128 (avoids f64 rounding)
    pub gamma2_sq:       u128,    // γ₂² as exact u128 (avoids f64 rounding)
    pub beta_prime:      f64,

    // Recursion (derived via §5.7)
    pub nu:              usize,    // split z into ν parts
    pub mu:              usize,    // split v into μ parts
    pub num_levels:      usize,    // 6–7 iterations
}

// Derived — fn methods, not stored fields
impl LabradorParams {
    /// 2r / 2^128 — last-level diagonal verification soundness (§5.6)
    pub fn garbage_soundness(&self) -> f64 {
        2.0 * self.r as f64 / 2.0_f64.powi(128)
    }

    /// I · ε₀ — additive soundness over recursion levels (Lemma 3.7)
    pub fn composition_budget(&self) -> f64 {
        self.num_levels as f64 * self.soundness_error
    }
}
```

### Parameter Derivation Table

Values that vary per parameter profile or input instance. Relation geometry (n, r, levels, proof size) comes from paper Tables 1/2/3. Commitment ranks (κ, κ₁, κ₂) come from offline Core-SVP/BDGL estimator artifacts. All other derived values computed at setup time from §5.4/§5.5 formulas.

| Parameter | Nature | Source |
|-----------|--------|--------|
| n | Input | Witness rank (from R1CS instance or profile) |
| r | Input | Witness multiplicity (chosen at profile design) |
| β | Input | Norm bound on honest witness |
| σ | Derived | §5.4: `σ = β / √(rnd)` — coefficient standard deviation (requires d from ring) |
| b | Derived | §5.4: `b = ⌊√(√(12rτ))·σ⌋` from n, r, β, τ, d |
| b₁ | Derived | §5.4: `b₁ = ⌈q^(1/t₁)⌉` for uniform decomposition (requires q from ring) |
| b₂ | Derived | §5.4: from Gaussian garbage width (requires d, σ) |
| t₁ | Derived | §5.4: `t₁ = ⌊log q / log b⌋` (requires q from ring) |
| t₂ | Derived | §5.4: `t₂ = ⌊log(√(24nd)·σ²) / log b⌋` (requires d from ring) |
| κ | Estimated (Core-SVP/BDGL) | §5.5: Module-SIS rank for inner commitment A — estimated offline, stored in profile, validated post-hoc |
| κ₁ | Estimated (Core-SVP/BDGL) | §5.5: Module-SIS rank for outer commitment B — estimated offline, stored in profile, validated post-hoc |
| κ₂ | Estimated (Core-SVP/BDGL) | §5.5: Module-SIS rank for outer commitment D (u₂ = D·h) — estimated offline, stored in profile, validated post-hoc |
| γ | Derived | §5.4: `γ = β√τ` |
| γ₁ | Derived | §5.4: from b₁, t₁, b₂, t₂, r, κ, d |
| γ₂ | Derived | §5.4: from b₁, t₁, r, d |
| β' | Derived | §5.4: `β' = √(2/b²·γ² + γ₁² + γ₂²)` |
| ν | Derived | §5.7: split z into ν parts so n/ν ≈ m/μ |
| μ | Derived | §5.7: split v into μ parts |
| num_levels | Derived | §5.7: 6–7 iterations, from recursion strategy |

Strategy: Commitment ranks (κ, κ₁, κ₂) are estimated offline via Core-SVP/BDGL and stored directly in `LabradorParams`. Tables 1/2/3 provide proof sizes and relation parameters but NOT commitment ranks — those require separate estimator computation. `validate_ranks(min_outer, min_inner)` checks that the configured ranks satisfy the Module-SIS requirements of Thm 5.1 at the requested `security_bits`.

---

## Current Boundary

The current implementation has a real LaBRADOR parameter object and builder, but it is still a
profile-driven interface. It does not yet generate a production profile from an arbitrary statement
family.

Implemented today:

- `LabradorParamsBuilder` takes base geometry (`n`, `r`, `beta`), ring data (`d`, `q`),
  commitment ranks (`kappa`, `kappa1`, `kappa2`), challenge/JL profiles, recursion splits, and the
  arithmetic soundness prime `arith_p`.
- The builder derives decomposition bases, limb counts, norm bounds, `beta_prime`, soundness
  metadata, and default recursion metadata.
- `validate_params()` checks structural and paper-level constraints that can be checked locally.
- `validate_ranks(min_outer, min_inner)` checks configured commitment ranks against external
  Core-SVP/BDGL estimator thresholds.

Not implemented yet:

- statement-driven parameter search
- automatic witness partition selection
- automatic proof-ring/profile selection
- in-crate Core-SVP/BDGL estimator integration
- machine-readable derivation provenance for generated production profiles
- a production zero-knowledge parameter story tied to a supported statement family

The main design lesson from LaZer-style parameter generation remains useful: lattice-proof
parameters should be generated from the concrete relation and witness bounds, not treated as a
single universal "128-bit" tuple.

---

## Future Statement-Driven Generation

A future generator should treat `LabradorParams` as the emitted concrete profile, not as the input
request. The request should capture the statement family and its bounds; the generator should select
compatible proof parameters and record how they were derived.

### Statement Profile

The request should describe the relation being proved:

- relation kind, such as binary R1CS, arithmetic R1CS mod `2^d+1`, mixed R1CS, or later CCS
- application modulus / ring family
- application ring degree when applicable
- number of public inputs and witness variables
- witness partition layout, or enough metadata for the generator to choose one
- per-block witness norm bounds

### Proof Backend Profile

The proof backend should be chosen from a vetted menu rather than arbitrary caller-supplied
arithmetic:

- proof modulus family
- proof ring degree
- supported profile identifier
- challenge profile and JL profile compatibility
- support expectations

Applications may live over an application ring such as `Z_p[X]/(X^d+1)` while the proof uses a
separate proof-optimized `R_q`. The current reductions already make this separation visible in
places, but there is no general profile-selection layer yet.

### Commitment And Security Profile

The generator should own the security-critical choices that are currently external inputs:

- commitment rank candidates (`kappa`, `kappa1`, `kappa2`)
- Module-SIS estimator output and assumptions used to justify those ranks
- arithmetic soundness prime `arith_p` for R1CS mod `2^d+1`
- target security level and composition budget
- threat-model note, such as classical vs quantum target if the estimator distinguishes them
- derivation metadata: generator version, selected backend profile, estimator artifacts, and input
  request hash

`security_bits` alone is not a security proof. It only drives local soundness formulas in the
current builder. A production profile must also justify rank hardness, norm slack, statement
compatibility, and reduction-specific soundness.

### Protocol Internal Profile

The generator should derive or justify:

- recursion splits `nu` and `mu`
- `num_levels`
- arithmetic aggregation count `l`
- witness rank and multiplicity after reductions
- proof-size vs prover-time tradeoffs if multiple profiles satisfy the same security target

Named helpers such as a future `paper_128_r1cs(...)` should only exist after these inputs and
derivation artifacts are represented explicitly.

---

## Parameter Generation TODOs

- Add a request type for statement-driven parameter generation.
- Separate application-ring metadata from proof-ring metadata.
- Add witness partition and per-block norm-bound metadata.
- Define a vetted menu of proof backend profiles.
- Add deterministic generation from request to `LabradorParams`.
- Record derivation metadata alongside generated profiles.
- Integrate or consume Core-SVP/BDGL estimator artifacts in a reproducible format.
- Add tests that identical requests produce identical generated profiles.
- Add negative tests for unsupported statement/profile combinations.
- Add named production-target helpers only after the generated profile path can justify them.
