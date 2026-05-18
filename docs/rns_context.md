# Dynamic RNS Context Follow-Up

Status: planned follow-up. This note is separate from
[large_modulus_support.md](large_modulus_support.md), which tracks the shipped first milestone for
fixed-profile scalar large-RNS support.

## Why This Exists

The first shipped large-RNS backend in `grid-algebra` is intentionally scalar, fixed-profile,
and pointer-free:

- `LargeRns<P, LIMBS>` stores only fixed residues
- the first shipped profile is `Rns3V0`
- dynamic-basis `RnsBasis` / `CompositeRing` remain interop and validation utilities

That is the right shape for compact scalar arithmetic, but it is not the full long-term answer for
FHE-oriented modulus chains.

Future FHE work needs dynamic context owned once above scalar storage, not an `Arc<RnsBasis>` or
equivalent pointer duplicated inside every coefficient.

## Target Scope

The future `RnsContext` layer should own dynamic chain state such as:

- the active basis / modulus chain
- cached CRT / Garner metadata
- NTT tables and twiddle metadata where applicable
- basis-extension and basis-switch metadata
- level-management state for rescaling / modulus dropping

That context should then be referenced once by higher-level containers such as:

- ring wrappers
- polynomial buffers
- ciphertexts
- evaluation-key / switching-key structures

## Non-Goals

This follow-up should not:

- replace the shipped fixed-profile scalar `LargeRns<P, LIMBS>` backend
- force every scalar element to carry dynamic context
- retroactively force `CompositeRing` into the plain `R: Ring` model
- widen prepared / NTT / proof-system paths before there is a concrete downstream need

## Recommended Shape

An approximate future shape is:

```rust
pub struct RnsContext {
    basis: alloc::sync::Arc<RnsBasis>,
    // future cached metadata:
    // crt, garner, basis-switch, ntt, level state
}

pub struct ContextualRnsPoly<'a> {
    ctx: &'a RnsContext,
    residues: alloc::vec::Vec<u64>,
}
```

The exact API can change, but the ownership rule should stay stable:

- scalar elements remain compact
- context is owned once at the ring / polynomial / ciphertext layer
- basis validation stays explicit

## Suggested Adoption Order

1. Keep fixed-profile scalar large-RNS as the arithmetic baseline.
2. Add dynamic context only when an FHE-style consumer actually needs modulus-chain features.
3. Land basis-switch / level-management support inside the context layer, not the scalar type.
4. Treat commitment and proof-system adoption over dynamic RNS as separate downstream projects.

## Relationship To Other Docs

- [large_modulus_support.md](large_modulus_support.md) tracks the shipped scalar large-RNS first
  milestone and the remaining word-sized-only surfaces.
- [commit_backend_support.md](commit_backend_support.md) tracks future basis-aware commitment work,
  especially `CompositeRing`.
- [techdebts.md](techdebts.md) tracks non-blocking follow-up items once concrete consumers appear.
