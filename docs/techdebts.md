# Technical Debt

This document tracks open non-blocking follow-ups only. It should not repeat the workspace status
snapshot; use [overview.md](overview.md) for status and the linked subsystem docs for support
boundaries.

## Foundation & Algebra

### 1. Large-modulus follow-up is concentrated in downstream protocol surfaces

- Location:
  - [large_modulus_support.md](large_modulus_support.md)
  - [algebra/src/arith/ring.rs](../algebra/src/arith/ring.rs)
  - [algebra/src/lattice/params.rs](../algebra/src/lattice/params.rs)
  - [algebra/src/lattice/sampling/toy.rs](../algebra/src/lattice/sampling/toy.rs)
  - [algebra/src/poly/decomposition.rs](../algebra/src/poly/decomposition.rs)
  - [algebra/src/arith/rns.rs](../algebra/src/arith/rns.rs)
  - [commit_backend_support.md](commit_backend_support.md)
  - [transcript/src/traits.rs](../transcript/src/traits.rs)
- Issue:
  - [large_modulus_support.md](large_modulus_support.md) is the source of truth for shipped
    large-prime and fixed-profile large-RNS support. The remaining debt is downstream: prepared /
    SIMD commitment fast paths, transcript challenge reduction, gadget large-modulus support,
    higher-level proof-profile wrappers, and APIs that still assume `u64` / `u128` bounds are
    universal.
- Impact:
  - Large-prime and fixed-profile large-RNS backends can move through the shipped algebra and direct
    coefficient commitment surfaces, but they are not end-to-end proof-stack profiles until the
    downstream metadata, transcript, prepared-path, and profile work is explicit.
- Proposed fix:
  - Preserve the current single-word-modulus fast path, widen downstream metadata and transcript
    surfaces deliberately, and only broaden downstream crate claims once those wider
    bound/challenge/profile paths are actually landed.

### 2. `rns_large` benchmark reference run needs a post-backend refresh

- Location:
  - [algebra/benches/rns_large.rs](../algebra/benches/rns_large.rs)
- Issue:
  - The current `rns_large` command should be treated as a fresh reference benchmark rather than as
    a direct continuation of the earliest pre-backend numbers, because the fixed-profile backend
    and the widened dynamic reconstruction path changed the measured surface materially.
- Impact:
  - Comparing current `rns_large` runs against old pre-backend numbers is misleading; the useful
    regression gate is the current command shape on the current backend.
- Proposed fix:
  - Rerun `cargo bench -p grid-algebra --bench rns_large` once the large-RNS backend shape
    stabilizes further, and use that run as the reference point for future post-backend
    comparisons.

### 3. Fixed-profile large-RNS exact norm queries still reconstruct canonical values eagerly

- Location:
  - [algebra/src/arith/large_rns.rs](../algebra/src/arith/large_rns.rs)
  - [algebra/src/lattice/params.rs](../algebra/src/lattice/params.rs)
  - [relations/src/witness/norms.rs](../relations/src/witness/norms.rs)
- Issue:
  - The shipped fixed-profile large-RNS backend now supports exact large norms through the common
    `LargeNormedRing` companion surface, but those exact norm queries currently reconstruct a full
    canonical value with Garner-style arithmetic on demand.
- Impact:
  - This keeps the first milestone simple and exact, but it makes norm-heavy large-RNS workflows
    pay more than the residue-domain hot path, especially when many exact norm checks are batched.
- Proposed fix:
  - Keep exact canonical reconstruction as the correctness baseline, then add a reviewed
    optimization pass for norm-heavy large-RNS paths once there is a concrete downstream consumer
    that needs it. Any optimization should preserve the current exact semantics.

## Commitment Layer

### 1. Commitment backend surface needs an explicit roadmap

- Location:
  - [commit_backend_support.md](commit_backend_support.md)
  - [commit/src/ajtai/mod.rs](../commit/src/ajtai/mod.rs)
  - [commit/src/bdlop/mod.rs](../commit/src/bdlop/mod.rs)
  - [commit/src/gadget/mod.rs](../commit/src/gadget/mod.rs)
- Issue:
  - The concrete commitment implementations support only the current coefficient and cyclotomic backend families, while the public API still reads as if any `R: Ring` backend should work.
- Impact:
  - This is easy to misread as “any foundational backend works,” and it leaves future `CompositeRing` work under-specified.
- Proposed fix:
  - Leave this debt item as TODO here, and use [commit_backend_support.md](commit_backend_support.md) as the source of truth for status and expansion planning. The minimum future target is Ajtai and BDLOP support over basis-aware `CompositeRing`, without forcing `CompositeRing` into the current basis-free `R: Ring` model.

### 2. Gadget commitments still do not accept `TwistedNttPoly` directly

- Location:
  - [commit/src/gadget/mod.rs](../commit/src/gadget/mod.rs)
  - [commit/src/ntt.rs](../commit/src/ntt.rs)
- Issue:
  - The new NTT-native commitment surface currently covers the linear schemes, but gadget openings
    are still fundamentally coefficient-domain because digit decomposition, canonicality, and
    recomposition are coefficient-wise operations.
- Impact:
  - Future polynomial callers that want a fully NTT-native commitment flow can use Ajtai and BDLOP
    directly, but still need a coefficient-domain bridge for gadget commitments.
- Proposed fix:
  - Revisit gadget openings as a separate design project. Either keep gadget explicitly
    coefficient-domain, or add a dedicated NTT-facing wrapper/API that makes the required inverse
    transforms and digit checks explicit.

### 3. Goldilocks Ajtai benchmark still uses a surrogate ring

- Location:
  - [commit/benches/ajtai_goldilocks.rs](../commit/benches/ajtai_goldilocks.rs)
  - [algebra/src/poly/ring.rs](../algebra/src/poly/ring.rs)
- Issue:
  - Gridland's current `CyclotomicPolyRing` only models power-of-two negacyclic rings
    `Z_q[X] / (X^N + 1)`, so the local Goldilocks Ajtai benchmark has to use a surrogate ring
    instead of a more general cyclotomic construction.
- Impact:
  - The current `ajtai_goldilocks` benchmark is useful for tracking larger-prime commitment
    performance, but it is not a strict apples-to-apples comparison against a broader cyclotomic
    Goldilocks backend.
- Proposed fix:
  - Add broader cyclotomic-ring support beyond the current power-of-two negacyclic model, then
    switch the benchmark off the surrogate ring and refresh the recorded benchmark snapshot.

### 4. Cyclotomic commitment sampling still allocates a `Vec` per sampled polynomial

- Location:
  - [commit/src/sampling.rs](../commit/src/sampling.rs)
  - [algebra/src/poly/ring.rs](../algebra/src/poly/ring.rs)
- Issue:
  - The shared `CommitmentSampleRing` impl for `CyclotomicPolyRing<C, N>` still samples
    coefficients into a temporary `Vec` before calling `try_from_coeffs`.
- Impact:
  - This matches the current coefficient-domain sampling style and is acceptable for the shipped
    word-sized paths, but it leaves avoidable allocation overhead in hot repeated sampling loops.
- Proposed fix:
  - Revisit the polynomial sampling construction once the broader commitment/runtime shape settles,
    and move to a lower-allocation or fixed-storage construction path if benchmarks show the extra
    allocation matters for the target workloads.

## LaBRADOR

### 1. LaBRADOR privacy and production-parameter hardening remain follow-up work

- Location:
  - [LaBRADOR parameter notes](../labrador/parameter.md)
  - [labrador/src/proof.rs](../labrador/src/proof.rs)
  - [labrador/src/prover.rs](../labrador/src/prover.rs)
  - [labrador/src/verifier.rs](../labrador/src/verifier.rs)
- Issue:
  - The current LaBRADOR crate has the recursive public-proof/private-witness structure in place,
    but the shipped profiles are still hand-tuned implementation profiles rather than output from a
    statement-driven parameter generator and zero-knowledge hardening review.
- Impact:
  - The crate is suitable as the current protocol implementation and benchmarking surface, but
    production use still needs parameter provenance, security-margin documentation, and privacy
    hardening/audit tied to the supported statement families.
- Proposed fix:
  - Keep the parameter-generation roadmap in [labrador/parameter.md](../labrador/parameter.md)
    current, then add statement-driven profile generation,
    machine-readable parameter provenance, and an explicit zero-knowledge/security audit checklist
    for each shipped profile.

### 2. Statement-driven LaBRADOR parameter generation remains planned

- Location:
  - [labrador/src/lib.rs](../labrador/src/lib.rs)
  - [labrador/src/params/mod.rs](../labrador/src/params/mod.rs)
  - [LaBRADOR parameter notes](../labrador/parameter.md)
- Issue:
  - `LabradorParamsBuilder` is profile-driven: callers provide relation geometry, proof ring
    metadata, recursion choices, and commitment ranks. The crate does not yet derive those choices
    from a statement/bounds request or record machine-readable derivation provenance.
- Impact:
  - Current profiles are useful for implementation, tests, and benchmarking, but production-target
    profiles still require external parameter work and estimator artifacts.
- Proposed fix:
  - Implement the statement-driven generation roadmap in
    [labrador/parameter.md](../labrador/parameter.md): request type, vetted proof-backend menu,
    deterministic generation, estimator provenance, and negative tests for unsupported profiles.

## Transcript Layer

### 1. Poseidon2 still lacks a byte-oriented compatibility backend

- Location:
  - [transcript/src/hash/poseidon2.rs](../transcript/src/hash/poseidon2.rs)
  - [transcript/src/traits.rs](../transcript/src/traits.rs)
  - [transcript/src/field.rs](../transcript/src/field.rs)
- Issue:
  - The shipped Poseidon2 landing is field-native only. That was the right boundary for the initial
    transcript landing, but callers that still live on the byte-oriented [`Transcript`] surface cannot use
    Poseidon2 without writing their own bridge.
- Impact:
  - SHAKE remains the only built-in backend for byte-native users, and any future migration that
    wants Poseidon2 without adopting the field-native surface would have to re-open transcript
    framing decisions.
- Proposed fix:
  - If a real byte-native Poseidon2 use case appears, add one explicit adapter on top of the
    shipped field-native framing rules rather than introducing a second incompatible byte-to-field
    mapping.

### 2. Poseidon2 backend coverage is still single-profile and non-circuit

- Location:
  - [transcript/src/hash/poseidon2.rs](../transcript/src/hash/poseidon2.rs)
  - [transcript/src/hash/poseidon2/profile_goldilocks.rs](../transcript/src/hash/poseidon2/profile_goldilocks.rs)
  - [poseidon2_goldilocks_profile.md](poseidon2_goldilocks_profile.md)
- Issue:
  - The current transcript backend ships exactly one checked-in profile: Goldilocks with width 12.
    There is also no recursive/circuit Poseidon2 story yet.
- Impact:
  - Future protocols that want a different transcript field, a different width, or a circuit-native
    Poseidon2 integration will need another reviewed rollout instead of plugging into a broader
    profile matrix today.
- Proposed fix:
  - Treat new transcript fields, new Poseidon2 widths, and recursive/circuit Poseidon2 support as
    explicit follow-up projects, each with reviewed constant provenance, backend tests, and a clear
    downstream consumer before broadening the surface.

### 3. Downstream protocol adoption is still pending on top of the shipped transcript surface

- Location:
  - [labrador/src/lib.rs](../labrador/src/lib.rs)
  - [transcript/src/canonical.rs](../transcript/src/canonical.rs)
- Issue:
  - The transcript crate now owns the field-native interface and canonical event model, but
    downstream proof systems still need their own reviewed encoders and adoption work.
- Impact:
  - `grid-labrador` still uses SHAKE, and downstream proof systems still need their own reviewed
    encoders and adoption work.
- Proposed fix:
  - Keep downstream adoption separate from the transcript landing: migrate LaBRADOR only if a
    concrete Poseidon2 need emerges.

## Cross-Cutting Performance Work

### SIMD acceleration

- Planning doc:
  - [simd.md](simd.md)
- Scope:
  - incremental SIMD acceleration for hot ring, polynomial, NTT, and lattice-container operations

### Large-prime SIMD is disabled after the x86_64 AVX2 experiment regressed scalar

- Location:
  - [algebra/src/arith/large_prime.rs](../algebra/src/arith/large_prime.rs)
  - [algebra/src/arith/large_prime_profiles.rs](../algebra/src/arith/large_prime_profiles.rs)
  - [algebra/src/arith/ring.rs](../algebra/src/arith/ring.rs)
  - [algebra/benches/large_prime.rs](../algebra/benches/large_prime.rs)
  - [docs/simd.md](simd.md)
- Issue:
  - An x86_64 AVX2 prototype for 4-limb `LargePrimeField<P, LIMBS>` slice add/sub was explored and
    benchmarked, but the current array-of-structs transpose strategy regressed the scalar baseline
    for `Bn254Fr`, `Bn254Fq`, and `BLS12-381Fr`. The path has therefore been removed from dispatch,
    and large-prime slice arithmetic is scalar-only again.
- Impact:
  - Batched large-prime workloads still miss the sort of SIMD uplift that the word-sized
    prime-field path already gets, but the current codebase no longer carries a live regressed
    dispatch path just for experimentation.
- Current read on root cause:
  - The bottleneck is mostly structural. The scalar 4-limb add/sub path is already close to ideal
    on x86_64 because it maps to short carry chains, while the AVX2 prototype has to emulate
    per-lane carry/borrow with extra compare-and-mask logic and pay three `4x4` limb transposes per
    batch of four elements. For a single add/sub slice pass, that reshaping overhead dominates the
    useful arithmetic.
- Proposed fix:
  - Revisit x86_64 large-prime SIMD only with a more layout-aware packed batch strategy. That
    likely means either reducing transpose overhead for the array-of-structs layout or introducing a
    temporary structure-of-arrays batch kernel that can amortize packing costs across more work
    than a single add/sub slice pass. Only after that should follow-up work consider vectorized
    mul, scaled-slice composition, dot-product overrides, or 6-limb specialization for
    `BLS12-381Fq`.

### Large-RNS SIMD is disabled after scalar add optimization closed the benchmark gap

- Location:
  - [algebra/src/arith/large_rns.rs](../algebra/src/arith/large_rns.rs)
  - [algebra/src/arith/large_rns_profiles.rs](../algebra/src/arith/large_rns_profiles.rs)
  - [algebra/src/arith/ring.rs](../algebra/src/arith/ring.rs)
  - [algebra/benches/rns_large.rs](../algebra/benches/rns_large.rs)
  - [docs/simd.md](simd.md)
- Issue:
  - The shipped `LargeRns<P, LIMBS>` scalar layout is pointer-free and compact, but it is still an
    array-of-struct representation (`[u64; 3]` per element). An x86_64 AVX2 slice-add experiment for
    `Rns3V0` briefly looked promising, but once scalar addition stopped using `% modulus` and moved
    to a single compare-and-subtract reduction, the SIMD path no longer showed a real win and was
    removed from dispatch.
- Impact:
  - Large-RNS arithmetic is faster than before because scalar addition is no longer paying an
    unnecessary `% modulus`, but there is currently no live large-RNS SIMD path. Any future SIMD
    work still needs to beat the stronger scalar baseline rather than the older inflated numbers.
- Proposed fix:
  - Keep the scalar `LargeRns` layout unchanged and only revisit x86_64 SIMD with a packed batch
    design that can beat the current single-correction scalar arithmetic. The best next candidates
    are still higher-arithmetic-intensity paths such as `pointwise_mul_assign_slice` or
    `scalar_mul_slice`, where packing costs have more work to amortize than a single add/sub pass.

### Remaining invariant-based reduction cleanups are narrower than the RNS add case

- Location:
  - [algebra/src/arith/rns.rs](../algebra/src/arith/rns.rs)
  - [algebra/src/arith/large_rns.rs](../algebra/src/arith/large_rns.rs)
  - [algebra/src/lattice/sampling/toy.rs](../algebra/src/lattice/sampling/toy.rs)
- Current status:
  - Dynamic `RnsBasis::add_assign_into` now uses the same single compare-and-subtract reduction as
    the fixed-profile large-RNS backend, so the obvious `% prime` cleanup for canonical RNS addition
    is done in both places.
- Remaining opportunities:
  - `RnsBasis::mul_assign_into` and `LargeRns::mul` still use `% prime`, but unlike addition their
    products are not bounded by `2q`, so they need a better multiplication/reduction strategy rather
    than the same one-step correction.
  - The signed toy samplers in `toy.rs` still use `% modulus` in a few encoding paths. Some callers
    only ever pass tiny magnitudes, but that is a caller invariant rather than a universal property,
    so any cleanup there should be targeted and documented instead of blanket.
  - Large-norm and serialization paths still rely on canonical reconstruction/export in several
    places. Those are good optimization targets, but they are larger design changes rather than the
    same “replace `%` with one correction” fix.

### aarch64 NEON benchmark coverage needs refresh after backend split

- Location:
  - [algebra/src/simd/aarch64](../algebra/src/simd/aarch64)
  - [algebra/src/arith/prime.rs](../algebra/src/arith/prime.rs)
  - [algebra/src/arith/z2k.rs](../algebra/src/arith/z2k.rs)
  - [simd.md](simd.md)
- Issue:
  - The aarch64 backend now enables the NEON path under `GRID_SIMD=auto` for the operations that are expected to be meaningful, including narrow-limb prime arithmetic and power-of-two slice hooks, but the current benchmark snapshot has not revalidated those enabled paths after the backend split.
- Impact:
  - Correctness coverage exists, but ARM performance claims are not yet backed by a current benchmark matrix.
- Proposed fix:
  - Rerun the `ops`, NTT, `Rq23Np8`, and container-level benchmarks on the documented Pi host plus at least one additional ARM host, then update the enabled/disabled qualification notes from measured results.

### aarch64 NEON high-band prime support is deferred past the initial larger-prime rollout

- Location:
  - [algebra/src/simd/montgomery_prime.rs](../algebra/src/simd/montgomery_prime.rs)
  - [algebra/src/simd/aarch64/u64_arith.rs](../algebra/src/simd/aarch64/u64_arith.rs)
- Issue:
  - The current larger-prime rollout targets scalar correctness plus x86_64 AVX2
    acceleration for Goldilocks, but does not include `aarch64` NEON fast paths for `Q >= 2^63`
    prime moduli.
- Impact:
  - ARM users will not get a high-band prime SIMD path in the first Goldilocks rollout, and any
    future NEON enablement needs separate correctness qualification and performance validation.
- Proposed fix:
  - Revisit high-band `aarch64` support after the first Goldilocks rollout ships. Add explicit
    qualification rules for `Q >= 2^63`, differential scalar-vs-NEON tests, and benchmark gates
    before enabling a NEON Goldilocks fast path.

### Word-sized NTT still pays explicit permutation/scaling bookends on the fast path

- Location:
  - [algebra/src/arith/ntt.rs](../algebra/src/arith/ntt.rs)
  - [algebra/src/simd/montgomery_prime.rs](../algebra/src/simd/montgomery_prime.rs)
  - [algebra/benches/ops.rs](../algebra/benches/ops.rs)
- Issue:
  - The current word-sized NTT path still performs standalone bit-reversal passes and, on the SIMD
    prime path, a separate inverse-scale sweep. A direct DIT/DIF stage-order swap was benchmarked
    during cleanup work, but it regressed the real `forward/256`, `inverse/256`, `poly_mul_ntt`,
    and `mul/rq23_np8` workloads instead of helping.
- Impact:
  - The hot `Rq23Np8` and prepared-commitment-visible polynomial paths still carry some structural
    overhead, but a simple transform-order rewrite is not a safe optimization knob.
- Proposed fix:
  - Keep the current stage order until there is a stronger redesign. Focus future work on
    prepared/in-place polynomial APIs, tighter temporary-buffer management, explicit inverse-scale
    fusion for the SIMD prime path, or new kernels/layouts that change the benchmarked cost model.

### Large-prime NTT still uses the generic scalar butterfly path

- Location:
  - [algebra/src/arith/ntt.rs](../algebra/src/arith/ntt.rs)
  - [algebra/src/arith/large_prime.rs](../algebra/src/arith/large_prime.rs)
  - [algebra/src/poly/ntt.rs](../algebra/src/poly/ntt.rs)
- Issue:
  - The shipped large-prime `NTTRing` landing is correct, but it still runs through the generic
    scalar `Ring` butterfly code. That path clones field elements at each stage, rebuilds plain
    `NttPlan` stage data per call, and does not exploit the raw Montgomery-limb structure of the
    backend.
- Impact:
  - Large-prime transforms and twisted polynomial multiplication work functionally, but leave a
    lot of performance on the table compared with what a profile-specialized scalar path should be
    able to deliver.
- Proposed fix:
  - Add plain-plan caching, then add a large-prime-specific scalar NTT override that works on raw
    Montgomery limbs and precomputed Montgomery-form stage twiddles before revisiting SIMD.

### Large-RNS NTT is still tuple-scalar instead of per-component

- Location:
  - [algebra/src/arith/large_rns.rs](../algebra/src/arith/large_rns.rs)
  - [algebra/src/arith/ntt.rs](../algebra/src/arith/ntt.rs)
  - [algebra/src/poly/ntt.rs](../algebra/src/poly/ntt.rs)
- Issue:
  - The shipped `Rns3V0` NTT path treats each `[u64; 3]` residue tuple as one generic scalar ring
    element. That is correct algebraically, but it does not take advantage of the fact that the
    transform is really three independent word-sized prime NTTs.
- Impact:
  - `Rns3V0` transforms and polynomial multiplication are locked to the generic tuple-level scalar
    fallback instead of benefiting from the existing word-sized prime kernels and future prepared
    fast paths.
- Proposed fix:
  - Keep the generic path as the correctness fallback, but add an `Rns3V0`-specific per-component
    NTT path that splits coefficients into component streams, runs the existing prime NTT per
    modulus, and packs the result back.

### Fixed `CHUNK = 64` scratch-buffer thresholds are still heuristic

- Location:
  - [algebra/src/poly/twisted_ntt.rs](../algebra/src/poly/twisted_ntt.rs)
  - [algebra/src/arith/prime.rs](../algebra/src/arith/prime.rs)
  - [algebra/src/arith/z2k.rs](../algebra/src/arith/z2k.rs)
  - [algebra/src/arith/large_prime.rs](../algebra/src/arith/large_prime.rs)
- Issue:
  - Several hot helpers switch from direct scalar accumulation to stack-backed chunked scratch
    buffers at a hard-coded `CHUNK = 64`. The recent Goldilocks Ajtai regression showed that this
    is only a heuristic: for small fixed sizes like `N = 32`, the chunked path can be much slower
    than the direct path because the temporary setup cost dominates.
- Impact:
  - Current code now protects the known small-size cases, but the threshold is still not
    benchmark-derived and may be suboptimal for other backends, element sizes, and polynomial
    degrees.
- Proposed fix:
  - Treat `64` as a provisional upper bound, not a tuned constant. Add benchmark sweeps for
    `16/32/64/128`-style thresholds on representative word-sized, twisted-NTT, and large-modulus
    workloads, then either tune per helper/backend or derive the cutoff from `size_of::<T>()` and
    measured workloads instead of hard-coding one universal value.

## Adding New Debt

If new non-blocking follow-ups are discovered, record each item with:

- location
- issue
- impact
- proposed fix
