# Scripts

This directory contains repository-maintained workflows that are meant to be rerun, not copied by hand.

## Commitment benchmark refresh

Use:

```bash
./scripts/refresh_commit_bench_report.sh
```

This workflow:

- runs the isolated Phase 2 commitment benches
- reruns `cargo run -q -p grid-commit --example size_snapshot`
- rewrites [`bench.md`](../bench.md) from the latest Criterion outputs

If the benchmark artifacts are already current and only the report rendering needs to be refreshed, use:

```bash
./scripts/refresh_commit_bench_report.sh --render-only
```

Raw artifacts stay under `target/criterion/` and `target/commit-bench-workflow/`.
