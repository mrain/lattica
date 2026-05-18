# Commitment Backend Support Status

Status: partial support shipped. This file records the current backend boundary in
`grid-commit` and the remaining unsupported commitment backend target: `CompositeRing`.

## Current Supported Backend Families

Today the concrete commitment implementations support these backend families:

- `PrimeField<Q, L>`
- `Z2K<K>`
- `LargePrimeField<P, LIMBS>`
- `LargeRns<P, LIMBS>`
- `CyclotomicPolyRing<R, N>` over the currently supported word-sized coefficient rings

In practice that means:

- Ajtai and BDLOP direct coefficient paths work over the shipped large-prime and fixed-profile
  large-RNS backends
- prepared / NTT commitment acceleration remains limited to the current word-sized cyclotomic path
- gadget commitments remain coefficient-domain

The implementation surface is still carried by internal traits such as:

- `CommitmentSampleRing` in [commit/src/sampling.rs](../commit/src/sampling.rs)
- `PreparedLinearOps` in [commit/src/linear.rs](../commit/src/linear.rs)
- `GadgetRing` in [commit/src/gadget/mod.rs](../commit/src/gadget/mod.rs)

The public scheme types still look broader than the actual implementation boundary because they
read as if any `R: Ring` backend should work.

## Current Unsupported Target

`CompositeRing` is still unsupported in `grid-commit`.

Why:

- it is basis-aware and runtime-parameterized
- it stores an `Arc<RnsBasis>` internally in
  [algebra/src/arith/composite.rs](../algebra/src/arith/composite.rs)
- `zero()` and `one()` require basis context
- deserialization is basis-aware rather than plain `CanonicalDeserialize`

That clashes with the current commitment stack, which assumes basis-free `R: Ring` values through:

- `Ring` identities in [algebra/src/arith/ring.rs](../algebra/src/arith/ring.rs)
- `RingVec<R>` and `RingMat<R>`
- commitment sampling, norm checking, setup, and validation helpers

So the missing support is not “one impl away.” The current commitment API still has no place to
carry basis context through setup, containers, sampling, serialization, and verification.

## Current Expansion Target

The next meaningful backend expansion target remains:

- Ajtai over `CompositeRing`
- BDLOP over `CompositeRing`

That future expansion still does not imply:

- gadget commitments over `CompositeRing`
- cyclotomic polynomials over `CompositeRing`
- NTT/prepared acceleration for composite backends
- arbitrary `BigUint` commitment backends

## Required Design Direction

Any future `CompositeRing` support should preserve these rules:

- do not force `CompositeRing: Ring` just to fit the old basis-free model
- carry basis context explicitly in the scheme state or backend context
- keep commitment/opening serialization basis-aware
- reject basis mismatch across setup, commit, verify, and decoding

## Related Docs

- [large_modulus_support.md](large_modulus_support.md) tracks the shipped large-prime and
  fixed-profile large-RNS first milestone
- [rns_context.md](rns_context.md) tracks future dynamic modulus-chain follow-up
- [techdebts.md](techdebts.md) keeps the higher-level commitment backend expansion debt visible
