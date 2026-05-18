# Benchmarks

This file tracks commitment-scheme benchmark workflow and the current commitment-focused reference numbers.

## Refresh Workflow

Regenerate this file with the scripted workflow in [scripts/](scripts/):

```bash
./scripts/refresh_commit_bench_report.sh
```

The script reruns the isolated commitment benches, captures the serialized-size snapshot, and rewrites this file from the latest Criterion outputs.

## Canonical benchmark commands

Use isolated per-bench runs as the regression gate:

```bash
cargo bench -p grid-commit --bench ajtai_large
cargo bench -p grid-commit --bench ajtai_goldilocks
cargo bench -p grid-commit --bench ajtai -- --sample-size 10
cargo bench -p grid-commit --bench bdlop -- --sample-size 10
cargo bench -p grid-commit --bench gadget -- --sample-size 10
```

Do not use a workspace-wide `cargo bench` run as the canonical number. The full-workspace run executes other test and bench binaries first and often reports a warmer, lower timing for the large Ajtai benches.

## Environment

### Current x86_64 Snapshot Host

- CPU: AMD Ryzen 7 9800X3D 8-Core Processor
- Rust toolchain at current local validation: `rustc 1.95.0 (59807616e 2026-04-14)`
- Host triple: `x86_64-unknown-linux-gnu`

## Baseline snapshot

These numbers were captured on 2026-05-17 using the isolated commands above.

### `ajtai_large`

Canonical isolated command:

```bash
cargo bench -p grid-commit --bench ajtai_large
```

| Benchmark | Avg time |
|---|---:|
| `Ajtai commit/large_1x3072` | `5.658 ms` |
| `Ajtai verify/large_1x3072` | `5.508 ms` |

Reference intervals from the same run:

- `commit`: `[5.639 ms 5.658 ms 5.679 ms]`
- `verify`: `[5.465 ms 5.508 ms 5.553 ms]`

### `ajtai_goldilocks`

Canonical isolated command:

```bash
cargo bench -p grid-commit --bench ajtai_goldilocks
```

| Benchmark | Avg time |
|---|---:|
| `Ajtai commit/goldilocks_1x32768` | `1.622 ms` |
| `Ajtai verify/goldilocks_1x32768` | `1.636 ms` |

Reference intervals from the same run:

- `commit`: `[1.611 ms 1.622 ms 1.636 ms]`
- `verify`: `[1.621 ms 1.636 ms 1.657 ms]`

Informational only:

- a warmed workspace-wide `cargo bench` run still tends to report a lower number on the same machine, but it is not the regression gate

### `ajtai`

| Benchmark | Avg time |
|---|---:|
| `commit/f17` | `122.39 ns` |
| `verify/f17` | `31.46 ns` |
| `commit/rq23_np8` | `39.14 us` |
| `verify/rq23_np8` | `27.30 us` |

Reference intervals from the same run:

- `commit/f17`: `[120.86 ns 122.39 ns 123.93 ns]`
- `verify/f17`: `[31.28 ns 31.46 ns 31.69 ns]`
- `commit/rq23_np8`: `[38.85 us 39.14 us 39.51 us]`
- `verify/rq23_np8`: `[27.05 us 27.30 us 27.51 us]`

### `bdlop`

| Benchmark | Avg time |
|---|---:|
| `commit/f17` | `144.54 ns` |
| `verify/f17` | `49.08 ns` |
| `commit/rq23_np8` | `48.53 us` |
| `verify/rq23_np8` | `37.51 us` |

Reference intervals from the same run:

- `commit/f17`: `[138.47 ns 144.54 ns 152.43 ns]`
- `verify/f17`: `[48.99 ns 49.08 ns 49.19 ns]`
- `commit/rq23_np8`: `[48.29 us 48.53 us 48.81 us]`
- `verify/rq23_np8`: `[36.47 us 37.51 us 39.23 us]`

### `gadget`

| Benchmark | Avg time |
|---|---:|
| `commit/f17` | `208.56 ns` |
| `verify/f17` | `67.63 ns` |
| `commit/rq23_np8` | `37.98 us` |
| `verify/rq23_np8` | `24.12 us` |

Reference intervals from the same run:

- `commit/f17`: `[205.75 ns 208.56 ns 212.03 ns]`
- `verify/f17`: `[67.55 ns 67.63 ns 67.70 ns]`
- `commit/rq23_np8`: `[36.21 us 37.98 us 39.90 us]`
- `verify/rq23_np8`: `[23.33 us 24.12 us 25.04 us]`

## Serialized size snapshot

This block is captured from `cargo run -q -p grid-commit --example size_snapshot` during the scripted refresh:

```text
ajtai/f17 key=40 commitment=10 opening=10
ajtai/rq23_np8 key=24608 commitment=3080 opening=3080
bdlop/f17 key=60 commitment=20 opening=10
bdlop/rq23_np8 key=36912 commitment=6160 opening=3080
gadget/f17 a_open=20 g_matrix=28 commitment=10 opening=24
gadget/rq23_np8 a_open=12304 g_matrix=36880 commitment=3080 opening=12304
```

## Scheme refresh policy

- `ajtai_large`, `ajtai_goldilocks`, and the `rq23_np8` rows in Ajtai, BDLOP, and gadget are the primary optimization targets
- `f17` rows remain useful as cheap commitment-path sanity checks, but they are not the main driver for the current optimization work
- refresh these isolated baselines after meaningful commitment-path changes

## Notes

- Commitment benchmarks should be compared using the same command shape, sample size, and host
- The scripted refresh workflow is the canonical way to update both runtime numbers and serialized-size output
