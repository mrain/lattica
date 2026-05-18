# grid-labrador

LaBRADOR — a lattice-based proof system built on Module-SIS hardness. Implements the protocol from the LaBRADOR paper (§2–§6), with recursive folding via amortized commitments, Johnson-Lindenstrauss norm compression, and a last-level optimization that eliminates outer commitments.

## Overview

Given a quadratic constraint system (principal relation `R`), LaBRADOR proves knowledge of a short witness `(s₁..sᵣ)` with `||sᵢ|| ≤ β` satisfying dot-product constraints:

```
f_k(s₁..sᵣ) = Σ aᵢⱼ⁽ᵏ⁾⟨sᵢ,sⱼ⟩ + Σ ⟨φᵢ⁽ᵏ⁾,sᵢ⟩ - b⁽ᵏ⁾ = 0
```

The protocol recurses: each level compresses the witness via amortization and JL projection, producing a smaller target relation for the next level. The last level skips outer commitments entirely, sending `z` directly.

## Protocol Flow

```
CRS: A ∈ R_q^(κ×n), B ∈ R_q^(κ₁×m₁), C ∈ R_q^(κ₁×m₂), D ∈ R_q^(κ₂×m₃)

Per main recursion level (public messages only — z and decomposed openings are
private, retained by the prover for target derivation, and proven by the next level):
  1. Commit:   tᵢ = A·sᵢ, gᵢⱼ = ⟨sᵢ,sⱼ⟩, send u₁ = B·t + C·g
  2. Project:  verifier samples Π ← JLDist, prover sends p = Πs, check ||p|| ≤ √128·β
  3. Aggregate: compress F' and JL → b'', then aggregate all → single (φ, a, b)
  4. Commit2:  hᵢⱼ = (⟨φᵢ,sⱼ⟩+⟨φⱼ,sᵢ⟩)/2, send u₂ = D·h
  5. Amortize: verifier sends c₁..cᵣ ← C, prover computes z = Σ cᵢ·sᵢ (private)
  6. Verify:   u₁=B·t+C·g, A·z=Σcᵢtᵢ ∧ ||z||≤γ, garbage equations, u₂=D·h
  → Target relation ((G, {}, β'), (sᵢ')) for next level

Last level: skip u₁, u₂; reduced garbage (2r-1 + 2ν+1 terms); z sent directly (not decomposed)
```

## Module Layout

```
labrador/src/
├── challenges.rs           # §2: Fixed-weight challenge polynomials, operator norm rejection
├── crs.rs                  # CRS matrices A, B, C, D + CommitKey setup
├── error.rs                # LabradorError
├── jl.rs                   # §4: Johnson-Lindenstrauss projection (seed-based, JLDist)
├── relation.rs             # §5.1: Principal relation R (quadratic dot-product constraints)
├── traits.rs               # LabradorProofRing trait bundle
│
├── main_protocol/          # §5.2: One recursive step
│   ├── mod.rs              # Shared helpers: decomposition, garbage_index, transcript utils
│   ├── aggregation.rs      # Two-step: F'+JL → ℓ batches, then all → single
│   ├── amortization.rs     # z = Σ cᵢ·sᵢ, challenge sampling
│   ├── garbage.rs          # gᵢⱼ = ⟨sᵢ,sⱼ⟩, hᵢⱼ = (⟨φᵢ,sⱼ⟩+⟨φⱼ,sᵢ⟩)/2
│   ├── step_prover.rs      # prove_step, inner+outer commitments
│   ├── step_verifier.rs    # verify_step, target derivation
│   └── verify_equations.rs # Equations (1)–(4): u1, Az, ⟨z,z⟩, u2 verification
│
├── recursion/              # §5.3: Recursion and decomposition
│   ├── mod.rs              # Recursion helpers: decompose_z, split_witness, target derivation
│   ├── decompose.rs        # z = z⁰ + b·z¹, v = t||g||h bundling
│   ├── split.rs            # Split into r' = 2ν+μ vectors, zero-pad to n'
│   └── target_relation.rs  # New family G, K' = κ+κ₁+κ₂+3 constraints
│
├── last_level/             # §5.6: Last-level optimization
│   ├── mod.rs              # Last-level proof types, challenge sampling helpers
│   ├── prover.rs           # prove_last_level, z sent directly
│   ├── reduced_garbage.rs  # h_{2i-1}, h_{2i}, g_0, g_{2i-1}, g_{2i}
│   └── verifier.rs         # Garbage checks, no u₁/u₂, JL + z norm checks
│
├── reduction/              # §6: R1CS → R reductions
│   ├── mod.rs              # Shared reduction utilities
│   ├── binary_r1cs.rs      # Binary R1CS (Figure 4), Hadamard product, σ₋₁ automorphism
│   ├── r1cs_mod_rns.rs     # R1CS mod 2^d+1 (Figure 5), NAF encoding, φ: X↦2 morphism
│   └── mixed_r1cs.rs       # Binary + arithmetic → single R instance
│
├── params/                 # §5.4 + §5.7: Parameters and profiles
│   ├── mod.rs              # LabradorParams struct, validate_params, validate_ranks
│   ├── builder.rs          # LabradorParamsBuilder, full derivation chain
│   ├── jl_profile.rs       # JLProfile (rows=256, √30/√128/√337 factors)
│   └── challenge_profile.rs # ChallengeProfile (shape {23/31/10}, T=15, tau=71)
│
├── proof.rs                # LabradorProof, LastLevelProof
├── prover.rs               # Public API: prove(crs, params, statement, witness, num_main_levels, transcript) → Proof
├── verifier.rs             # Public API: verify(crs, statement, proof, params, transcript) → Result
└── lib.rs                  # Crate root, public re-exports
```

## Key Concepts

### Challenge Space (§2)

Polynomials in `R_q` with fixed coefficient weight distribution. For `d=64`: 23 zeros, 31 ±1s, 10 ±2s. Two constraints ensure soundness: coefficients sum to `d` (injectivity via LS18 Cor 1.2), and operator norm `||c||_op ≤ T=15` (rejection sampling). `τ = ||c||²₂ = 71` propagates into downstream norm bounds.

### Johnson-Lindenstrauss Projection (§4)

`Π ∈ {-1,0,1}^(256×nd)` with i.i.d. entries `Pr[0]=1/2, Pr[±1]=1/4`. Lemma 4.1 guarantees with probability `> 1 - 2⁻¹²⁸`:
- `||Πw|| ≥ √30 · ||w||` (lower bound)
- `||Πw|| ≤ √337 · ||w||` (upper tail)

Verifier checks `||p|| ≤ √128·β`. The norm slack `√(128/30) ≈ 2.07` inflates Module-SIS bounds at recursed levels (Remark 5.2).

### Principal Relation R (§5.1)

Two constraint families: `F` (fully vanishing — constant term and all coefficients zero) and `F'` (zero constant term only). The distinction matters for aggregation: `F'` constraints are compressed first with JL projection, then all constraints are aggregated together.

### Recursion (§5.3)

Output `(z, t, g, h)` from one step becomes the witness for the next level. Decomposition: `z = z⁰ + b·z¹` (binary-norm vectors). Bundle: `v = t||g||h`. Split `z` into `ν` parts, `v` into `μ` parts, zero-pad to `n'`. New multiplicity `r' = 2ν+μ`. Consolidated norm:

```
β' = √(2/b²·γ² + γ₁² + γ₂²)
```

Target relation `G` has `K' = κ+κ₁+κ₂+3` constraints with symmetric tridiagonal `aᵢⱼ` structure (nonzero only for `i,j ≤ 2ν`).

### Last-Level Optimization (§5.6)

At the final recursion level, skip `u₁`, `u₂` entirely. Interactive rounds send reduced garbage polynomials directly: `2r-1` h-terms and `2ν+1` g-terms. Witness `z` is sent directly (not decomposed). Multiplicity `r = ν+μ` (not `r' = 2ν+μ` from regular recursion).

### R1CS → R Reduction (§6)

Binary R1CS uses `σ₋₁` automorphism (`X↦X⁻¹`) to express binary constraints via `a∘σ₋₁(a) = a`. Arithmetic R1CS uses NAF encoding and `φ: X↦2` morphism. Mixed reduction combines both into a single relation instance.

## Parameter Selection (§5.4)

See [parameter.md](parameter.md) for the full implemented parameter surface and the remaining
statement-driven parameter-generation roadmap.

Given instance geometry `(n, r, β)`, ring `(d, q)`, and challenge profile `τ`:

| Derived | Formula |
|---------|---------|
| `σ` | `β / √(rnd)` |
| `b` | `⌊√(√(12rτ))·σ⌋` |
| `t₁` | `⌊log q / log b⌋` |
| `b₁` | `⌈q^(1/t₁)⌉` |
| `t₂` | `⌊log(√(24nd)·σ²) / log b⌋` |
| `b₂` | `⌊(√(24nd)·σ²)^(1/t₂)⌋` |
| `γ` | `β√τ` |
| `β'` | `√(2/b²·γ² + γ₁² + γ₂²)` |

`LabradorParamsBuilder` derives all values from base inputs. `validate_params()` checks structural consistency; `validate_ranks()` verifies Module-SIS rank hardness.

## Workspace Dependencies

| Concept | Location |
|---------|----------|
| R1CS instance, witness | `grid-relations::r1cs` |
| Ajtai commitment scheme | `grid-commit::ajtai` |
| Ring `R_q`, polynomial | `grid-algebra` |
| Transcript (Fiat-Shamir) | `grid-transcript` |
| Serialization traits | `grid-serialize` |

`grid-labrador` implements only what is unique to the LaBRADOR protocol.
