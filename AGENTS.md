# AGENTS.md

## Project Overview

**Gridland** is a layered Rust workspace for lattice-based cryptography. It targets commitments and proof systems, and is designed for extensibility to future applications (FHE, signatures, etc.).

- **Current status and roadmap:** See [docs/overview.md](docs/overview.md) for the workspace status snapshot, shipped scope, and open work.
- **Deeper status and follow-up docs:** Use [docs/large_modulus_support.md](docs/large_modulus_support.md), [labrador/parameter.md](labrador/parameter.md), and [docs/techdebts.md](docs/techdebts.md) when the narrower subsystem status or remaining follow-up work matters.

## Status Snapshot

| Area | Status | Notes |
|---|---|---|
| Foundation & Algebra | Complete | Shared algebra layer shipped, including the first large-prime and fixed-profile large-RNS milestone |
| Commitments | Complete | Ajtai, BDLOP, gadget commitments, plus explicit NTT-native Ajtai / BDLOP runtime paths |
| Relations / Transcript | Complete | R1CS / CCS, norm-tracked witnesses, SHAKE and Poseidon2 transcript surfaces |
| LaBRADOR | Complete for current scope | Generic prover/verifier, examples, and tests |

## Current Shipped Highlights

- Large-modulus support now includes fixed-limb Montgomery large-prime backends for `Bn254Fr`, `Bn254Fq`, `BLS12-381Fr`, and `BLS12-381Fq`, plus a shipped fixed-profile 3-limb large-RNS backend. Wider prepared/NTT paths, dynamic FHE-style `RnsContext`, and transcript/profile widening remain follow-up work.
- `grid-labrador` ships a generic proof core. Statement-driven parameter generation and zero-knowledge hardening are still tracked as follow-up work.

## Workspace Structure

| Crate | Purpose |
|---|---|
| `grid-std` | `no_std` compat, shared re-exports (`rand`, `rayon`), test helpers |
| `grid-serialize` | Canonical serialization/deserialization traits |
| `grid-algebra` | Modular arithmetic (`arith/`), polynomial rings (`poly/`), lattice types & sampling (`lattice/`), large-prime and fixed-profile large-RNS backends |
| `grid-commit` | Commitment schemes (Ajtai, BDLOP, gadget-based) plus explicit NTT-native Ajtai / BDLOP runtime helpers |
| `grid-relations` | Constraint systems (R1CS, CCS) with norm-tracked witnesses and widened large-norm metadata companions |
| `grid-transcript` | Shared Fiat-Shamir transcript (Poseidon, SHAKE) |
| `grid-labrador` | Generic LaBRADOR proof system |

## Build & Test

```bash
# Build everything
cargo build --workspace

# Run all tests
cargo test --workspace

# Run tests for a specific crate
cargo test -p grid-algebra

# Run benchmarks
cargo bench --workspace

# Check formatting
cargo fmt --all -- --check

# Run clippy
cargo clippy --workspace --tests --benches -- -D warnings
```

Benchmark note:

- The workspace bench profile is pinned with `codegen-units = 1` in the root `Cargo.toml`. Use the checked-in bench profile when comparing benchmark runs or refreshing performance reference numbers.

## Code Conventions

- **Rust edition:** 2024
- **Formatting:** `rustfmt` defaults. Run `cargo fmt` before committing.
- **Linting:** All code must pass `cargo clippy -- -D warnings`.
- **`no_std` support:** All crates should support `no_std` via `grid-std`. Use `#![no_std]` and feature-gate `std`-dependent code behind a `std` feature.
- **Naming:** Follow Rust API guidelines. Crate names use `grid-` prefix; internal module names are lowercase.
- **Traits over concrete types:** Define traits in leaf crates, implementations in separate crates or modules.
- **Generics:** Algebraic types are generic over modulus/ring types. Use trait bounds, not concrete types.
- **Tests:** Every public function should have unit tests. Place tests in `#[cfg(test)] mod tests` blocks within the same file.
- **Documentation:** All public items must have `///` doc comments. Crate roots must have `//!` module-level docs.
- **Tech debt tracking:** When making progress or reviewing code, record any non-blocking issues, performance concerns, or deferred improvements in [`docs/techdebts.md`](docs/techdebts.md).

## Key Domain Concepts

- **`Z_q`**: Integers modulo `q`. `q` can be prime, power-of-two, or composite (RNS). Prime-power `Z_{p^k}` is future work.
- **Large modulus backends**: The workspace now ships fixed-limb Montgomery large-prime fields and one fixed-profile large-RNS backend. Dynamic modulus-chain support for FHE-style workflows is still future work.
- **`R_q`**: Polynomial ring `Z_q[X]/(X^n+1)`. The fundamental algebraic object.
- **NTT**: Number Theoretic Transform — FFT over finite fields. `arith/ntt.rs` defines the `NTTRing` trait (modulus supports NTT); `poly/ntt.rs` implements the algorithm.
- **SIS/LWE**: Hardness assumptions. Short Integer Solution / Learning With Errors.
- **Norm bounds**: Lattice witnesses must remain "short." Norm tracking is enforced in `grid-relations`.
- **CCS**: Customizable Constraint System — generalizes R1CS, AIR, Plonkish.
- **Ajtai/BDLOP**: Lattice-native commitment schemes based on SIS.
