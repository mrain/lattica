# Gridland

A layered Rust workspace for lattice-based cryptography.

> [!WARNING]
> This project is a work in progress. The workspace is under active development, and several
> protocol/profile hardening tracks remain follow-up work.

Gridland is being built as a layered library stack for lattice-native commitments and proof systems.
For shipped scope, open work, and subsystem status, see [docs/overview.md](docs/overview.md).

## Workspace Layout

| Crate | Purpose |
|---|---|
| `grid-std` | `no_std` compatibility layer, shared re-exports, test helpers |
| `grid-serialize` | Canonical serialization / deserialization traits |
| `grid-algebra` | Arithmetic over `Z_q`, polynomial rings, lattice vectors/matrices, sampling |
| `grid-commit` | Ajtai, BDLOP, and gadget-based commitment schemes plus shared commitment traits/helpers |
| `grid-relations` | R1CS/CCS relation containers with norm-tracked witnesses |
| `grid-transcript` | Shared Fiat-Shamir transcript traits and backends |
| [`grid-labrador`](labrador/README.md) | Generic LaBRADOR prover/verifier core |

Other top-level docs and reports live under `docs/` and [bench.md](bench.md).

## Design Goals

- Keep the crate layering explicit and easy to extend
- Keep traits and naming consistent across crates so the APIs stay predictable
- Support `no_std` from the bottom of the stack upward
- Build around lattice assumptions such as SIS, LWE, RLWE, and MLWE
- Expose correctness-first commitment and algebra building blocks before higher-level proof-system crates
- Leave room for future protocol crates without reshaping the foundation

## Building

```bash
cargo build --workspace
cargo test --workspace
cargo bench --workspace
cargo fmt --all -- --check
cargo clippy --workspace --tests --benches -- -D warnings
```

## Project Docs

- [Current status and roadmap](docs/overview.md)
- [LaBRADOR parameters](labrador/parameter.md)
- [Large modulus support status](docs/large_modulus_support.md)
- [Dynamic RNS context follow-up](docs/rns_context.md)
- [SIMD acceleration plan](docs/simd.md)
- [SIMD benchmark report and reproduction commands](bench.md)
- [Technical debt](docs/techdebts.md)

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
