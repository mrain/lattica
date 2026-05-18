# Goldilocks Poseidon2 Transcript Profile Provenance

This note records the first checked-in Poseidon2 transcript profile used by
`grid-transcript`.

## Profile

- transcript field: `PrimeField<GOLDILOCKS_MODULUS>`
- width: `12`
- rate: `8`
- capacity: `4`
- full rounds: `8`
- partial rounds: `22`
- S-box: `x^7`

This is the repository's first field-native transcript profile. It is not a claim that every future
protocol in the workspace must use the same field/profile.

## Sources

The checked-in constants in
[`transcript/src/hash/poseidon2/profile_goldilocks.rs`](../transcript/src/hash/poseidon2/profile_goldilocks.rs)
were copied from the following upstream sources:

- `github.com/ppd0705/poseidon_crypto` tag `v0.0.12`
  - `hash/poseidon2_goldilocks/config.go`
  - `hash/poseidon2_goldilocks/poseidon2.go`
- `github.com/Plonky3/Plonky3` commit
  `eeb4e37b20127c4daa871b2bad0df30a7c7380db`
  - `goldilocks/src/poseidon2.rs`

The external round constants and internal round constants come from
`poseidon_crypto`. The width-12 internal diffusion diagonal
`MATRIX_DIAG_12_U64` was taken from the Plonky3 Goldilocks Poseidon2 profile,
matching the note in `poseidon_crypto`'s `config.go`.

The repository does not generate these constants locally in this first landing.
They are checked in as reviewed raw `u64` words and converted to field elements
at backend construction time.

## Verification

The repository verifies this profile in two ways:

1. [`transcript/src/hash/poseidon2/permutation.rs`](../transcript/src/hash/poseidon2/permutation.rs)
   contains a zero-state permutation vector test for the checked-in profile.
2. That expected vector was cross-checked against the upstream Go reference by
   running `poseidon2.Permute` from `github.com/ppd0705/poseidon_crypto/hash/poseidon2_goldilocks`
   on an all-zero width-12 state.

The zero-state reference vector is:

```text
[7182099517097165596, 9311216678150108034, 8831900494918587432, 10774846510254277933,
 10601329242472021962, 5629867288322699978, 140799316430260029, 16680789625189310103,
 16589856342819292996, 4940126994627441183, 14089387953811494999, 8340711910841427341]
```

If any checked-in constant row is copied incorrectly, that permutation test
should fail immediately.
