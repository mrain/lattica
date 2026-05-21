# Large Modulus Support Status

Status: first milestone shipped. This file is the source of truth for the current large-prime and
large-RNS surface in the repo.

This document covers:

- prime fields whose modulus does not fit in a single `u64`
- RNS bases whose combined range exceeds `u128`
- the current downstream support boundary for those backends

It is intentionally separate from [commit_backend_support.md](commit_backend_support.md). That
document tracks future basis-aware commitment support, especially `CompositeRing`.

## Current Scope

The current large-modulus surface is built around two explicit backend families:

- fixed-profile large-prime fields
- fixed-profile large-RNS scalars plus dynamic-basis interop utilities

The shipped design keeps the existing single-word-modulus fast path intact:

- `PrimeField<Q, L>`, `Z2K<K>`, Goldilocks, and `Rq23Np8` remain the canonical optimized
  path for moduli that fit in primitive machine limbs
- large-modulus support is exposed through `IntegerRing::Canonical` and explicit backend types,
  while the small-modulus fast path keeps `Canonical = u64`
- dynamic FHE-style modulus-chain state is not stored inside each scalar element

## Shipped Algebra Surface

The current `grid-algebra` large-modulus surface includes:

- `BigUint<const N: usize>` fixed-limb substrate helpers in
  [algebra/src/arith/bigint.rs](../algebra/src/arith/bigint.rs)
- `LargePrimeProfile` and `LargeRnsProfile` in
  [algebra/src/arith/large_modulus.rs](../algebra/src/arith/large_modulus.rs)
- `LargePrimeField<P, LIMBS>` and the shipped profile aliases in
  [algebra/src/arith/large_prime.rs](../algebra/src/arith/large_prime.rs)
- `LargeRns<P, LIMBS>` and the shipped `Rns3V0` alias in
  [algebra/src/arith/large_rns.rs](../algebra/src/arith/large_rns.rs)
- dynamic-basis interop helpers in [algebra/src/arith/rns.rs](../algebra/src/arith/rns.rs) and
  [algebra/src/arith/composite.rs](../algebra/src/arith/composite.rs)

The current helper layer includes:

- `LargeNormValue`, `CanonicalNormEmbedding`, `LargeNormedRing`, `LargeNormStats`, and
  `LargeNormBound` in [algebra/src/lattice/params.rs](../algebra/src/lattice/params.rs)
- large-modulus toy samplers in
  [algebra/src/lattice/sampling/toy.rs](../algebra/src/lattice/sampling/toy.rs)
- large-modulus gadget decomposition helpers in
  [algebra/src/poly/decomposition.rs](../algebra/src/poly/decomposition.rs)

## Shipped Profiles

### Large-Prime Profiles

The shipped large-prime profile set is:

- `Bn254Fr`
- `Bn254Fq`
- `BLS12-381Fr`
- `BLS12-381Fq`

Current properties of that backend family:

- fixed-limb
- Montgomery-form internal arithmetic
- fixed-width canonical serialization
- no per-operation heap allocation in hot field arithmetic

### Large-RNS Profile

The shipped fixed-profile large-RNS target is `Rns3V0`.

Its component moduli are:

- `17592216453121`
- `17592247910401`
- `17592258396161`

Its product is `5444568820299413499809579413277774970881`, which is strictly above `2^128`.

Current properties of that backend family:

- fixed-profile pointer-free scalar storage
- `[u64; 3]` residue layout
- checked interop with the dynamic `RnsBasis` / `CompositeRing` utility path
- exact canonical export above `u128`
- shared two-adicity `20`

## Current Downstream Support

### `grid-algebra`

Current state:

- shipped scalar large-prime backends
- shipped scalar `NTTRing` support for fixed-profile large-prime fields through profile-supplied
  two-adic roots
- shipped fixed-profile scalar large-RNS backend
- shipped scalar `NTTRing` support for fixed-profile large-RNS rings through profile-supplied
  two-adic roots and inverse scales
- shipped large norms, large samplers, large gadget decomposition, and dynamic-basis interop

Current limits:

- `Field::pow` still exposes a `u64` exponent surface
- the current SIMD / prepared NTT fast path is still word-sized-only
- `Bn254Fq` and `BLS12-381Fq` only support size-`2` scalar NTT because their two-adicity is `1`
- the shipped `Rns3V0` profile supports scalar NTT up to `2^20`, which is enough for the current
  degree-`256` negacyclic polynomial path but still not a tuned prepared/SIMD profile

### `grid-commit`

Current state:

- Ajtai and BDLOP direct coefficient paths work over the shipped large-prime and fixed-profile
  large-RNS backends
- commitment parameter families are generic over their bound object
- direct runtime bound checks use `VectorNormBound` instead of hard-coding `NormBound`

Current limits:

- prepared / NTT commitment acceleration remains limited to the current word-sized cyclotomic path
- `CompositeRing` commitments remain unsupported
- gadget commitments remain coefficient-domain and are not part of the shipped large-modulus
  downstream target set

`CompositeRing` follow-up work is tracked separately in
[commit_backend_support.md](commit_backend_support.md).

### `grid-relations`

Current state:

- widened witness norm metadata is shipped
- widened witness bounds are shipped
- R1CS / CCS witness and instance containers can carry large-prime and large-RNS norm metadata

### `grid-transcript`

Current state:

- no large-modulus-specific transcript profile is shipped

Current limits:

- transcript scalar reduction remains word-sized
- additional transcript-field profiles remain separate follow-up work

### `grid-labrador`

Current state:

- the shipped LaBRADOR proof path remains on the single-word-modulus `PrimeField<Q, L>` surface

Current limits:

- no large-modulus proof profile is part of the shipped surface

## Current Constraints

The current codebase still has explicit word-sized or profile-bounded assumptions in these areas:

- `PrimeField<Q, L>` still uses `const Q: u64`; it supports primitive storage limbs (`u8`,
  `u16`, `u32`, and `u64`) but is not a runtime-modulus or multi-limb large-prime field in
  [algebra/src/arith/prime.rs](../algebra/src/arith/prime.rs)
- `IntegerRing` still exposes `from_u64`, `to_u64`, and `modulus() -> u64` in
  [algebra/src/arith/ring.rs](../algebra/src/arith/ring.rs)
- `Field::pow` still exposes a `u64` exponent surface, and large-prime `NTTRing` support is
  currently limited to profile-supplied two-adic roots rather than a general wide-modulus root
  search in [algebra/src/arith/ring.rs](../algebra/src/arith/ring.rs),
  [algebra/src/arith/ntt.rs](../algebra/src/arith/ntt.rs), and
  [algebra/src/arith/large_prime.rs](../algebra/src/arith/large_prime.rs)
- fixed-profile large-RNS `NTTRing` support is currently profile-driven and bounded by the chosen
  component moduli's shared two-adicity in
  [algebra/src/arith/ntt.rs](../algebra/src/arith/ntt.rs) and
  [algebra/src/arith/large_rns.rs](../algebra/src/arith/large_rns.rs)
- `CompositeRing::to_u128` remains only a convenience surface for small values in
  [algebra/src/arith/composite.rs](../algebra/src/arith/composite.rs)
- transcript scalar challenges and some proof-profile metadata remain word-sized in
  [transcript/src/traits.rs](../transcript/src/traits.rs)

These are known boundaries, not accidental gaps.

## Benchmarks

The canonical algebra benchmark commands for this surface are:

```bash
cargo bench -p grid-algebra --bench ops
cargo bench -p grid-algebra --bench large_prime
cargo bench -p grid-algebra --bench rns_large
```

The workspace now pins `[profile.bench] codegen-units = 1` in [Cargo.toml](../Cargo.toml) so the
benchmark configuration is stable for the recorded fast-path regression guard.

## Explicit Follow-Ups

These are not part of the shipped first milestone:

- prime-power rings such as `Z_(p^k)`
- runtime-modulus generic prime fields
- large-prime SIMD
- large-prime NTT acceleration beyond the current scalar path
- additional large-RNS profiles beyond the current `Rns3V0` basis
- residue-accelerated NTT / base extension
- transcript-field expansion for large-modulus consumers
- large-modulus proof profiles in `grid-labrador`
- basis-aware `CompositeRing` commitments
- dynamic `RnsContext` / modulus-chain support for FHE

## Large-Modulus NTT Improvement Plan

The current large-prime and large-RNS `NTTRing` implementations are correct scalar landings, but
they are not yet tuned transform backends. The next work should treat the two backend families
separately.

### Large-Prime NTT Plan

Target backend:

- `LargePrimeField<P, LIMBS>` in
  [algebra/src/arith/large_prime.rs](../algebra/src/arith/large_prime.rs)

Current shape:

- generic scalar radix-2 butterflies from [algebra/src/arith/ntt.rs](../algebra/src/arith/ntt.rs)
- profile-supplied two-adic roots
- Montgomery-form field arithmetic inside each butterfly

Recommended sequence:

- add explicit `forward`, `inverse`, and `poly_mul_ntt` benchmarks for `Bn254Fr` and
  `Bls12_381Fr`
- cache plain `NttPlan` stage data the same way twisted plans are cached today
- add a large-prime-specific scalar NTT override that works on raw Montgomery limbs instead of the
  clone-heavy generic `Ring` butterfly path
- precompute stage twiddles in Montgomery form per profile/size
- only revisit SIMD after the scalar specialized path is benchmarked

Success condition:

- measurable speedups on the shipped scalar large-prime profiles without changing the current
  public NTT API

### Large-RNS NTT Plan

Target backend:

- `LargeRns<P, LIMBS>` in [algebra/src/arith/large_rns.rs](../algebra/src/arith/large_rns.rs)

Current shape:

- generic scalar radix-2 butterflies over `[u64; LIMBS]` residue tuples
- profile-supplied shared two-adic roots and inverse scales
- mathematically equivalent to componentwise NTT, but not laid out like one

Recommended sequence:

- benchmark the current scalar tuple-level `forward`, `inverse`, and `poly_mul_ntt` path for
  `Rns3V0`
- do not over-invest in the tuple-scalar butterflies; the main win is to stop treating each
  residue tuple as one opaque NTT scalar
- add an `Rns3V0`-specific path that splits coefficients into per-modulus streams, runs the
  existing word-sized prime NTTs per component, and packs the result back
- add a prepared twisted path on top of that componentwise transform
- only revisit SIMD after the per-component path is working and benchmarked

Success condition:

- `Rns3V0` polynomial multiplication and transform workloads move onto the existing fast
  word-sized prime infrastructure rather than the current tuple-scalar fallback

### Implementation Order

Recommended order for the next NTT work:

1. cache plain `NttPlan` instances
2. specialize scalar large-prime NTT
3. add `Rns3V0` per-component NTT
4. extend prepared twisted-domain paths for the improved large-modulus transforms
5. only then reconsider SIMD

Dynamic modulus-chain follow-up is tracked in [rns_context.md](rns_context.md).

Performance-oriented follow-ups and narrower downstream limitations are tracked in
[techdebts.md](techdebts.md).
