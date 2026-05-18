# SIMD Acceleration Plan

This document tracks the current shipped SIMD state for `grid-algebra` and the remaining follow-up work. Commitment benchmark tables live in [bench.md](../bench.md). Non-blocking follow-ups live in [techdebts.md](techdebts.md).

## Current Status

- Internal SIMD dispatch is landed with `GRID_SIMD=scalar|auto|avx2|neon`.
- `std` x86_64 ships runtime AVX2 detection under `GRID_SIMD=auto` for the slice kernels that
  still clear the current benchmark gates. Word-sized prime NTT is now back on the shipped AVX2
  path under `auto` for qualified moduli, including the direct `Rq23Np8` transform workloads that
  previously forced a scalar gate.
- `std` aarch64 now enables the NEON backend under `GRID_SIMD=auto` when runtime NEON detection succeeds.
- `Ring` and `NTTRing` now provide the hook layer that eligible slice, polynomial, NTT, and lattice-container operations use to reach SIMD fast paths while preserving scalar fallback.
- `PrimeField<Q, L>` and `Z2K<K>` layout assumptions are hardened for internal SIMD use without changing arithmetic semantics or canonical serialization.
- `PrimeField<Q, L>` SIMD dispatch is now routed through an internal Montgomery-prime helper with formula-based qualification bands instead of per-modulus branch ladders in `prime.rs`. The default limb is `u64`, and the shipped primitive-limb backends include `u8`, `u16`, `u32`, and `u64`.
- The current x86_64 AVX2 surface includes:
  - `Z2K<K>` add/sub for all `K`
  - `Z2K<K <= 32>` multiply and scalar-multiply
  - prime-field add/sub for `PrimeField<Q, L>` when the selected limb backend is AVX2-qualified
  - prime-field multiply / scalar-multiply for AVX2-qualified `PrimeField<Q, L>` primitive-limb backends
  - prime-field butterfly NTT for AVX2-qualified `PrimeField<Q, L>` primitive-limb backends under both `auto` and explicit `GRID_SIMD=avx2`
  - `CyclotomicPolyRing`, `RingVec`, and `RingMat` fast paths that reuse those kernels
- The current aarch64 NEON surface includes:
  - `Z2K<K>` add/sub for all `K`
  - `Z2K<K <= 32>` multiply and scalar-multiply
  - prime-field add/sub for NEON-qualified `PrimeField<Q, L>` primitive-limb backends
  - prime-field multiply / scalar-multiply and butterfly NTT for narrow `u8`, `u16`, and `u32` limb backends
  - no `u64` prime-field Montgomery multiply or NTT dispatch; those paths degraded scalar on ARM and remain disabled
- `LargePrimeField<P, LIMBS>` does not currently use SIMD. An x86_64 AVX2 prototype for 4-limb
  slice add/sub was explored and benchmarked, but it regressed the scalar baseline on the
  documented host and was removed from dispatch rather than left as an opt-in path.
- Representative algebra coverage now includes narrow-limb `PrimeField` profiles, a modulus just above `2^32`, and a modulus above the AVX2 add/sub band so the qualification boundaries are exercised directly in tests.
- The biggest current x86_64 wins come from the primitive-limb Montgomery AVX2 multiply paths and the `poly_mul_ntt` cleanup that batches twist/untwist and pointwise multiplication.
- The first AVX2 transform attempt that relied on scratch buffers regressed and was removed; the current winning transform design is in-place butterfly processing with precomputed stage twiddles.
- The earlier x86_64 scalar gate is no longer the current behavior. After the direct AVX2
  butterfly path and surrounding `Rq23Np8` polynomial stack were requalified on the documented
  host, `auto` now ships that transform path again for the qualified word-sized prime profiles.

## Validation Status

Standard repo validation target:

```bash
cargo test --workspace
cargo clippy --workspace --tests --benches -- -D warnings
cargo fmt --all -- --check
```

Current validation status:

- Local x86_64 on `rustc 1.94.0`: the const-eval UI tests now check compile-fail invariants directly instead of snapshotting full stderr, so they are no longer tied to rustc's exact panic-frame formatting.
- Pi `aarch64`: formatting, clippy, and full workspace tests pass, including the shared const-eval UI fixture set.
- SIMD-specific validation also relies on randomized scalar-vs-SIMD differential tests, target-gated unsupported-arch coverage, and tail-handling tests for non-lane-multiple slice lengths.
- aarch64 NEON correctness coverage exists for the new narrow-limb backends, but the current
  benchmark snapshot has not yet validated their expected performance gains.

## x86_64 Large-Prime Findings

Benchmark host:

- Architecture: `x86_64`
- CPU: `AMD Ryzen 7 9800X3D 8-Core Processor`
- Toolchain: `rustc 1.94.0`

Current scalar baseline command:

```bash
cargo bench -p grid-algebra --bench large_prime -- field_slice
```

Exploratory AVX2 command used for the removed prototype:

```bash
GRID_SIMD=avx2 cargo bench -p grid-algebra --bench large_prime -- field_slice
```

Median timings from the last exploratory x86_64 run:

| Benchmark | `auto` / scalar baseline | `GRID_SIMD=avx2` | Delta |
|---|---:|---:|---:|
| `field_slice_add/bn254_fr` | `1.6444 µs` | `2.3861 µs` | `+45.1%` |
| `field_slice_sub/bn254_fr` | `1.6461 µs` | `1.8578 µs` | `+12.9%` |
| `field_slice_add/bn254_fq` | `1.5547 µs` | `2.3342 µs` | `+50.1%` |
| `field_slice_sub/bn254_fq` | `1.6699 µs` | `1.8897 µs` | `+13.2%` |
| `field_slice_add/bls12_381_fr` | `1.6519 µs` | `2.3433 µs` | `+41.9%` |
| `field_slice_sub/bls12_381_fr` | `1.5834 µs` | `1.7766 µs` | `+12.2%` |
| `field_slice_add/bls12_381_fq` | `2.3265 µs` | `2.2925 µs` | `-1.5%` |
| `field_slice_sub/bls12_381_fq` | `2.4680 µs` | `2.4593 µs` | `-0.4%` |

Interpretation:

- The exploratory AVX2 path was correct, but it is not shipped and is not used by the current
  codebase.
- The repo intentionally keeps large-prime arithmetic scalar on x86_64 today because the
  transpose-heavy array-of-structs strategy regressed the 4-limb slice benchmarks enough that even
  an explicit override was not worth keeping live.
- In the current tree, setting `GRID_SIMD=avx2` no longer changes large-prime slice dispatch;
  the environment variable still affects the shipped word-sized SIMD kernels, but large-prime
  arithmetic remains scalar.
- The current slowdown is not mysterious: for 4-limb add/sub, the scalar path already compiles to a
  very compact carry-chain on x86_64, while the AVX2 prototype has to emulate per-lane
  carry/borrow with extra compares and logical ops and also pay three `4x4` limb transposes
  (load-pack, modulus correction, store-unpack) for every four field elements. A single slice pass
  does not give that reshaping work enough arithmetic to amortize it.
- The 6-limb `BLS12-381Fq` profile also did not show a compelling reason to keep a separate AVX2
  path alive; the tiny timing deltas there were just run-to-run noise.

## x86_64 Large-RNS Snapshot

Current scalar baseline command:

```bash
cargo bench -p grid-algebra --bench rns_large -- fixed_profile_add/rns3_v0
cargo bench -p grid-algebra --bench rns_large -- fixed_profile_slice_add/rns3_v0
```

Current comparison command with the SIMD override still set:

```bash
GRID_SIMD=avx2 cargo bench -p grid-algebra --bench rns_large -- fixed_profile_slice_add/rns3_v0
```

Median timings after replacing scalar `% modulus` addition with a single compare-and-subtract:

| Benchmark | Current scalar path | `GRID_SIMD=avx2` | Delta |
|---|---:|---:|---:|
| `fixed_profile_add/rns3_v0` | `4.4885 µs` | n/a | n/a |
| `fixed_profile_slice_add/rns3_v0` | `3.9625 µs` | `3.9696 µs` | `+0.2%` |

Interpretation:

- The meaningful improvement came from fixing scalar large-RNS addition itself: canonical residues
  satisfy `a + b < 2q`, so the old `% modulus` reduction was unnecessary. Replacing it with a
  single compare-and-subtract cut the scalar add benchmarks roughly in half.
- After that scalar cleanup, the x86_64 AVX2 block-packing experiment for slice addition no longer
  showed a real win, so it was removed from dispatch instead of being kept as a marginal fast path.
- In the current tree, `GRID_SIMD=avx2` no longer changes large-RNS arithmetic; the environment
  variable still affects the shipped word-sized SIMD kernels, but large-RNS stays scalar.

## Not Done

- `aarch64` NEON `u64` prime Montgomery multiply and NTT remain disabled because they degraded scalar on the documented ARM host.
- `aarch64` NEON narrow-limb prime and power-of-two paths still need refreshed benchmark validation under `GRID_SIMD=auto`.
- `aarch64` NEON transform-heavy workloads still need new measurements for the enabled narrow-limb paths before performance claims should be made.
- Benchmark reporting is still a manually refreshed snapshot process rather than an automated check.
- `LargePrimeField<P, LIMBS>` still needs either a layout-aware packed batch strategy or stronger
  per-call amortization before x86_64 SIMD should be reconsidered.
