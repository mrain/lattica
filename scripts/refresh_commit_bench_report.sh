#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT_DIR/target}"
CRITERION_DIR="$TARGET_DIR/criterion"
ARTIFACT_DIR="$TARGET_DIR/commit-bench-workflow"
BENCH_MD="$ROOT_DIR/bench.md"
SIZE_SNAPSHOT_OUT="$ARTIFACT_DIR/size_snapshot.txt"
TMP_BENCH_MD="$(mktemp)"
MODE="refresh"

trap 'rm -f "$TMP_BENCH_MD"' EXIT

usage() {
    cat <<'EOF'
Usage:
  ./scripts/refresh_commit_bench_report.sh
  ./scripts/refresh_commit_bench_report.sh --render-only

Default mode reruns the commitment benchmarks, refreshes the size snapshot,
and rewrites bench.md. The render-only mode skips command execution and only
rewrites bench.md from the latest Criterion outputs and saved size snapshot.
EOF
}

require_tool() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "missing required tool: $1" >&2
        exit 1
    fi
}

log() {
    printf '[commit-bench] %s\n' "$*" >&2
}

run() {
    log "running: $*"
    (cd "$ROOT_DIR" && "$@")
}

criterion_file() {
    printf '%s\n' "$CRITERION_DIR/$1/new/estimates.json"
}

nested_criterion_file() {
    printf '%s\n' "$CRITERION_DIR/$1/$2/new/estimates.json"
}

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "missing criterion output: $1" >&2
        exit 1
    fi
}

json_number() {
    jq -r "$2" "$1"
}

mean_point() {
    json_number "$1" '.mean.point_estimate'
}

mean_lower() {
    json_number "$1" '.mean.confidence_interval.lower_bound'
}

mean_upper() {
    json_number "$1" '.mean.confidence_interval.upper_bound'
}

format_ns() {
    awk -v ns="$1" 'BEGIN {
        if (ns >= 1000000) {
            printf "%.3f ms", ns / 1000000;
        } else if (ns >= 1000) {
            printf "%.2f us", ns / 1000;
        } else {
            printf "%.2f ns", ns;
        }
    }'
}

format_interval() {
    printf '[%s %s %s]' \
        "$(format_ns "$1")" \
        "$(format_ns "$2")" \
        "$(format_ns "$3")"
}

cpu_model() {
    if command -v lscpu >/dev/null 2>&1; then
        local cpu=""
        local line
        while IFS= read -r line; do
            case "$line" in
                "Model name:"*)
                    cpu="${line#Model name:}"
                    cpu="${cpu#"${cpu%%[![:space:]]*}"}"
                    break
                    ;;
            esac
        done <<< "$(lscpu)"
        if [[ -n "$cpu" ]]; then
            printf '%s\n' "$cpu"
            return
        fi
    fi
    uname -m
}

host_triple() {
    local line
    while IFS= read -r line; do
        case "$line" in
            "host:"*)
                printf '%s\n' "${line#host: }"
                return
                ;;
        esac
    done <<< "$(rustc -vV)"
    echo "unknown-host"
}

git_commit() {
    git -C "$ROOT_DIR" rev-parse HEAD
}

git_commit_short() {
    git -C "$ROOT_DIR" rev-parse --short=12 HEAD
}

git_commit_date() {
    git -C "$ROOT_DIR" show -s --format='%ci' HEAD
}

render_size_snapshot_block() {
    sed -n '1,120p' "$SIZE_SNAPSHOT_OUT"
}

build_row() {
    printf '| `%s` | `%s` |\n' "$1" "$2"
}

if [[ $# -gt 1 ]]; then
    usage >&2
    exit 1
fi

if [[ $# -eq 1 ]]; then
    case "$1" in
        --render-only)
            MODE="render-only"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            exit 1
            ;;
    esac
fi

require_tool cargo
require_tool rustc
require_tool jq
require_tool awk
mkdir -p "$ARTIFACT_DIR"

if [[ "$MODE" == "refresh" ]]; then
    run cargo bench -p grid-commit --bench ajtai_large
    run cargo bench -p grid-commit --bench ajtai_goldilocks
    run cargo bench -p grid-commit --bench ajtai -- --sample-size 10
    run cargo bench -p grid-commit --bench bdlop -- --sample-size 10
    run cargo bench -p grid-commit --bench gadget -- --sample-size 10
    run cargo run -q -p grid-commit --example size_snapshot > "$SIZE_SNAPSHOT_OUT"
else
    log "render-only mode: reusing latest Criterion outputs and $SIZE_SNAPSHOT_OUT"
fi

AJTAI_LARGE_COMMIT_JSON="$(nested_criterion_file ajtai_large ajtai_commit_large_1x3072)"
AJTAI_LARGE_VERIFY_JSON="$(nested_criterion_file ajtai_large ajtai_verify_large_1x3072)"
AJTAI_GOLDILOCKS_COMMIT_JSON="$(nested_criterion_file ajtai_goldilocks ajtai_commit_goldilocks_1x32768)"
AJTAI_GOLDILOCKS_VERIFY_JSON="$(nested_criterion_file ajtai_goldilocks ajtai_verify_goldilocks_1x32768)"
AJTAI_COMMIT_F17_JSON="$(criterion_file ajtai_commit_f17)"
AJTAI_VERIFY_F17_JSON="$(criterion_file ajtai_verify_f17)"
AJTAI_COMMIT_RQ23_NP8_JSON="$(criterion_file ajtai_commit_rq23_np8)"
AJTAI_VERIFY_RQ23_NP8_JSON="$(criterion_file ajtai_verify_rq23_np8)"
BDLOP_COMMIT_F17_JSON="$(criterion_file bdlop_commit_f17)"
BDLOP_VERIFY_F17_JSON="$(criterion_file bdlop_verify_f17)"
BDLOP_COMMIT_RQ23_NP8_JSON="$(criterion_file bdlop_commit_rq23_np8)"
BDLOP_VERIFY_RQ23_NP8_JSON="$(criterion_file bdlop_verify_rq23_np8)"
GADGET_COMMIT_F17_JSON="$(criterion_file gadget_commit_f17)"
GADGET_VERIFY_F17_JSON="$(criterion_file gadget_verify_f17)"
GADGET_COMMIT_RQ23_NP8_JSON="$(criterion_file gadget_commit_rq23_np8)"
GADGET_VERIFY_RQ23_NP8_JSON="$(criterion_file gadget_verify_rq23_np8)"

for path in \
    "$AJTAI_LARGE_COMMIT_JSON" \
    "$AJTAI_LARGE_VERIFY_JSON" \
    "$AJTAI_GOLDILOCKS_COMMIT_JSON" \
    "$AJTAI_GOLDILOCKS_VERIFY_JSON" \
    "$AJTAI_COMMIT_F17_JSON" \
    "$AJTAI_VERIFY_F17_JSON" \
    "$AJTAI_COMMIT_RQ23_NP8_JSON" \
    "$AJTAI_VERIFY_RQ23_NP8_JSON" \
    "$BDLOP_COMMIT_F17_JSON" \
    "$BDLOP_VERIFY_F17_JSON" \
    "$BDLOP_COMMIT_RQ23_NP8_JSON" \
    "$BDLOP_VERIFY_RQ23_NP8_JSON" \
    "$GADGET_COMMIT_F17_JSON" \
    "$GADGET_VERIFY_F17_JSON" \
    "$GADGET_COMMIT_RQ23_NP8_JSON" \
    "$GADGET_VERIFY_RQ23_NP8_JSON" \
    "$SIZE_SNAPSHOT_OUT"; do
    require_file "$path"
done

AJTAI_LARGE_COMMIT_MEAN="$(mean_point "$AJTAI_LARGE_COMMIT_JSON")"
AJTAI_LARGE_COMMIT_LOW="$(mean_lower "$AJTAI_LARGE_COMMIT_JSON")"
AJTAI_LARGE_COMMIT_HIGH="$(mean_upper "$AJTAI_LARGE_COMMIT_JSON")"
AJTAI_LARGE_VERIFY_MEAN="$(mean_point "$AJTAI_LARGE_VERIFY_JSON")"
AJTAI_LARGE_VERIFY_LOW="$(mean_lower "$AJTAI_LARGE_VERIFY_JSON")"
AJTAI_LARGE_VERIFY_HIGH="$(mean_upper "$AJTAI_LARGE_VERIFY_JSON")"
AJTAI_GOLDILOCKS_COMMIT_MEAN="$(mean_point "$AJTAI_GOLDILOCKS_COMMIT_JSON")"
AJTAI_GOLDILOCKS_COMMIT_LOW="$(mean_lower "$AJTAI_GOLDILOCKS_COMMIT_JSON")"
AJTAI_GOLDILOCKS_COMMIT_HIGH="$(mean_upper "$AJTAI_GOLDILOCKS_COMMIT_JSON")"
AJTAI_GOLDILOCKS_VERIFY_MEAN="$(mean_point "$AJTAI_GOLDILOCKS_VERIFY_JSON")"
AJTAI_GOLDILOCKS_VERIFY_LOW="$(mean_lower "$AJTAI_GOLDILOCKS_VERIFY_JSON")"
AJTAI_GOLDILOCKS_VERIFY_HIGH="$(mean_upper "$AJTAI_GOLDILOCKS_VERIFY_JSON")"

AJTAI_COMMIT_F17_MEAN="$(mean_point "$AJTAI_COMMIT_F17_JSON")"
AJTAI_VERIFY_F17_MEAN="$(mean_point "$AJTAI_VERIFY_F17_JSON")"
AJTAI_COMMIT_RQ23_NP8_MEAN="$(mean_point "$AJTAI_COMMIT_RQ23_NP8_JSON")"
AJTAI_VERIFY_RQ23_NP8_MEAN="$(mean_point "$AJTAI_VERIFY_RQ23_NP8_JSON")"

BDLOP_COMMIT_F17_MEAN="$(mean_point "$BDLOP_COMMIT_F17_JSON")"
BDLOP_VERIFY_F17_MEAN="$(mean_point "$BDLOP_VERIFY_F17_JSON")"
BDLOP_COMMIT_RQ23_NP8_MEAN="$(mean_point "$BDLOP_COMMIT_RQ23_NP8_JSON")"
BDLOP_VERIFY_RQ23_NP8_MEAN="$(mean_point "$BDLOP_VERIFY_RQ23_NP8_JSON")"

GADGET_COMMIT_F17_MEAN="$(mean_point "$GADGET_COMMIT_F17_JSON")"
GADGET_VERIFY_F17_MEAN="$(mean_point "$GADGET_VERIFY_F17_JSON")"
GADGET_COMMIT_RQ23_NP8_MEAN="$(mean_point "$GADGET_COMMIT_RQ23_NP8_JSON")"
GADGET_VERIFY_RQ23_NP8_MEAN="$(mean_point "$GADGET_VERIFY_RQ23_NP8_JSON")"

AJTAI_COMMIT_F17_LOW="$(mean_lower "$AJTAI_COMMIT_F17_JSON")"
AJTAI_COMMIT_F17_HIGH="$(mean_upper "$AJTAI_COMMIT_F17_JSON")"
AJTAI_VERIFY_F17_LOW="$(mean_lower "$AJTAI_VERIFY_F17_JSON")"
AJTAI_VERIFY_F17_HIGH="$(mean_upper "$AJTAI_VERIFY_F17_JSON")"
AJTAI_COMMIT_RQ23_NP8_LOW="$(mean_lower "$AJTAI_COMMIT_RQ23_NP8_JSON")"
AJTAI_COMMIT_RQ23_NP8_HIGH="$(mean_upper "$AJTAI_COMMIT_RQ23_NP8_JSON")"
AJTAI_VERIFY_RQ23_NP8_LOW="$(mean_lower "$AJTAI_VERIFY_RQ23_NP8_JSON")"
AJTAI_VERIFY_RQ23_NP8_HIGH="$(mean_upper "$AJTAI_VERIFY_RQ23_NP8_JSON")"

BDLOP_COMMIT_F17_LOW="$(mean_lower "$BDLOP_COMMIT_F17_JSON")"
BDLOP_COMMIT_F17_HIGH="$(mean_upper "$BDLOP_COMMIT_F17_JSON")"
BDLOP_VERIFY_F17_LOW="$(mean_lower "$BDLOP_VERIFY_F17_JSON")"
BDLOP_VERIFY_F17_HIGH="$(mean_upper "$BDLOP_VERIFY_F17_JSON")"
BDLOP_COMMIT_RQ23_NP8_LOW="$(mean_lower "$BDLOP_COMMIT_RQ23_NP8_JSON")"
BDLOP_COMMIT_RQ23_NP8_HIGH="$(mean_upper "$BDLOP_COMMIT_RQ23_NP8_JSON")"
BDLOP_VERIFY_RQ23_NP8_LOW="$(mean_lower "$BDLOP_VERIFY_RQ23_NP8_JSON")"
BDLOP_VERIFY_RQ23_NP8_HIGH="$(mean_upper "$BDLOP_VERIFY_RQ23_NP8_JSON")"

GADGET_COMMIT_F17_LOW="$(mean_lower "$GADGET_COMMIT_F17_JSON")"
GADGET_COMMIT_F17_HIGH="$(mean_upper "$GADGET_COMMIT_F17_JSON")"
GADGET_VERIFY_F17_LOW="$(mean_lower "$GADGET_VERIFY_F17_JSON")"
GADGET_VERIFY_F17_HIGH="$(mean_upper "$GADGET_VERIFY_F17_JSON")"
GADGET_COMMIT_RQ23_NP8_LOW="$(mean_lower "$GADGET_COMMIT_RQ23_NP8_JSON")"
GADGET_COMMIT_RQ23_NP8_HIGH="$(mean_upper "$GADGET_COMMIT_RQ23_NP8_JSON")"
GADGET_VERIFY_RQ23_NP8_LOW="$(mean_lower "$GADGET_VERIFY_RQ23_NP8_JSON")"
GADGET_VERIFY_RQ23_NP8_HIGH="$(mean_upper "$GADGET_VERIFY_RQ23_NP8_JSON")"

SNAPSHOT_DATE="$(date +%F)"
CPU_MODEL="$(cpu_model)"
RUSTC_VERSION="$(rustc -V)"
HOST_TRIPLE="$(host_triple)"
GIT_COMMIT="$(git_commit)"
GIT_COMMIT_SHORT="$(git_commit_short)"
GIT_COMMIT_DATE="$(git_commit_date)"
SIZE_SNAPSHOT_BLOCK="$(render_size_snapshot_block)"

cat > "$TMP_BENCH_MD" <<EOF
# Benchmarks

This file tracks commitment-scheme benchmark workflow and the current commitment-focused reference numbers.

## Refresh Workflow

Regenerate this file with the scripted workflow in [scripts/](scripts/):

\`\`\`bash
./scripts/refresh_commit_bench_report.sh
\`\`\`

The script reruns the isolated commitment benches, captures the serialized-size snapshot, and rewrites this file from the latest Criterion outputs.

## Canonical benchmark commands

Use isolated per-bench runs as the regression gate:

\`\`\`bash
cargo bench -p grid-commit --bench ajtai_large
cargo bench -p grid-commit --bench ajtai_goldilocks
cargo bench -p grid-commit --bench ajtai -- --sample-size 10
cargo bench -p grid-commit --bench bdlop -- --sample-size 10
cargo bench -p grid-commit --bench gadget -- --sample-size 10
\`\`\`

Do not use a workspace-wide \`cargo bench\` run as the canonical number. The full-workspace run executes other test and bench binaries first and often reports a warmer, lower timing for the large Ajtai benches.

## Environment

### Current x86_64 Snapshot Host

- CPU: $CPU_MODEL
- Rust toolchain at current local validation: \`$RUSTC_VERSION\`
- Host triple: \`$HOST_TRIPLE\`
- Git commit: \`$GIT_COMMIT_SHORT\` (\`$GIT_COMMIT\`)
- Git commit date: \`$GIT_COMMIT_DATE\`

## Baseline snapshot

These numbers were captured on $SNAPSHOT_DATE using the isolated commands above.

### \`ajtai_large\`

Canonical isolated command:

\`\`\`bash
cargo bench -p grid-commit --bench ajtai_large
\`\`\`

| Benchmark | Avg time |
|---|---:|
$(build_row "Ajtai commit/large_1x3072" "$(format_ns "$AJTAI_LARGE_COMMIT_MEAN")")
$(build_row "Ajtai verify/large_1x3072" "$(format_ns "$AJTAI_LARGE_VERIFY_MEAN")")

Reference intervals from the same run:

- \`commit\`: \`$(format_interval "$AJTAI_LARGE_COMMIT_LOW" "$AJTAI_LARGE_COMMIT_MEAN" "$AJTAI_LARGE_COMMIT_HIGH")\`
- \`verify\`: \`$(format_interval "$AJTAI_LARGE_VERIFY_LOW" "$AJTAI_LARGE_VERIFY_MEAN" "$AJTAI_LARGE_VERIFY_HIGH")\`

### \`ajtai_goldilocks\`

Canonical isolated command:

\`\`\`bash
cargo bench -p grid-commit --bench ajtai_goldilocks
\`\`\`

| Benchmark | Avg time |
|---|---:|
$(build_row "Ajtai commit/goldilocks_1x32768" "$(format_ns "$AJTAI_GOLDILOCKS_COMMIT_MEAN")")
$(build_row "Ajtai verify/goldilocks_1x32768" "$(format_ns "$AJTAI_GOLDILOCKS_VERIFY_MEAN")")

Reference intervals from the same run:

- \`commit\`: \`$(format_interval "$AJTAI_GOLDILOCKS_COMMIT_LOW" "$AJTAI_GOLDILOCKS_COMMIT_MEAN" "$AJTAI_GOLDILOCKS_COMMIT_HIGH")\`
- \`verify\`: \`$(format_interval "$AJTAI_GOLDILOCKS_VERIFY_LOW" "$AJTAI_GOLDILOCKS_VERIFY_MEAN" "$AJTAI_GOLDILOCKS_VERIFY_HIGH")\`

Informational only:

- a warmed workspace-wide \`cargo bench\` run still tends to report a lower number on the same machine, but it is not the regression gate

### \`ajtai\`

| Benchmark | Avg time |
|---|---:|
$(build_row "commit/f17" "$(format_ns "$AJTAI_COMMIT_F17_MEAN")")
$(build_row "verify/f17" "$(format_ns "$AJTAI_VERIFY_F17_MEAN")")
$(build_row "commit/rq23_np8" "$(format_ns "$AJTAI_COMMIT_RQ23_NP8_MEAN")")
$(build_row "verify/rq23_np8" "$(format_ns "$AJTAI_VERIFY_RQ23_NP8_MEAN")")

Reference intervals from the same run:

- \`commit/f17\`: \`$(format_interval "$AJTAI_COMMIT_F17_LOW" "$AJTAI_COMMIT_F17_MEAN" "$AJTAI_COMMIT_F17_HIGH")\`
- \`verify/f17\`: \`$(format_interval "$AJTAI_VERIFY_F17_LOW" "$AJTAI_VERIFY_F17_MEAN" "$AJTAI_VERIFY_F17_HIGH")\`
- \`commit/rq23_np8\`: \`$(format_interval "$AJTAI_COMMIT_RQ23_NP8_LOW" "$AJTAI_COMMIT_RQ23_NP8_MEAN" "$AJTAI_COMMIT_RQ23_NP8_HIGH")\`
- \`verify/rq23_np8\`: \`$(format_interval "$AJTAI_VERIFY_RQ23_NP8_LOW" "$AJTAI_VERIFY_RQ23_NP8_MEAN" "$AJTAI_VERIFY_RQ23_NP8_HIGH")\`

### \`bdlop\`

| Benchmark | Avg time |
|---|---:|
$(build_row "commit/f17" "$(format_ns "$BDLOP_COMMIT_F17_MEAN")")
$(build_row "verify/f17" "$(format_ns "$BDLOP_VERIFY_F17_MEAN")")
$(build_row "commit/rq23_np8" "$(format_ns "$BDLOP_COMMIT_RQ23_NP8_MEAN")")
$(build_row "verify/rq23_np8" "$(format_ns "$BDLOP_VERIFY_RQ23_NP8_MEAN")")

Reference intervals from the same run:

- \`commit/f17\`: \`$(format_interval "$BDLOP_COMMIT_F17_LOW" "$BDLOP_COMMIT_F17_MEAN" "$BDLOP_COMMIT_F17_HIGH")\`
- \`verify/f17\`: \`$(format_interval "$BDLOP_VERIFY_F17_LOW" "$BDLOP_VERIFY_F17_MEAN" "$BDLOP_VERIFY_F17_HIGH")\`
- \`commit/rq23_np8\`: \`$(format_interval "$BDLOP_COMMIT_RQ23_NP8_LOW" "$BDLOP_COMMIT_RQ23_NP8_MEAN" "$BDLOP_COMMIT_RQ23_NP8_HIGH")\`
- \`verify/rq23_np8\`: \`$(format_interval "$BDLOP_VERIFY_RQ23_NP8_LOW" "$BDLOP_VERIFY_RQ23_NP8_MEAN" "$BDLOP_VERIFY_RQ23_NP8_HIGH")\`

### \`gadget\`

| Benchmark | Avg time |
|---|---:|
$(build_row "commit/f17" "$(format_ns "$GADGET_COMMIT_F17_MEAN")")
$(build_row "verify/f17" "$(format_ns "$GADGET_VERIFY_F17_MEAN")")
$(build_row "commit/rq23_np8" "$(format_ns "$GADGET_COMMIT_RQ23_NP8_MEAN")")
$(build_row "verify/rq23_np8" "$(format_ns "$GADGET_VERIFY_RQ23_NP8_MEAN")")

Reference intervals from the same run:

- \`commit/f17\`: \`$(format_interval "$GADGET_COMMIT_F17_LOW" "$GADGET_COMMIT_F17_MEAN" "$GADGET_COMMIT_F17_HIGH")\`
- \`verify/f17\`: \`$(format_interval "$GADGET_VERIFY_F17_LOW" "$GADGET_VERIFY_F17_MEAN" "$GADGET_VERIFY_F17_HIGH")\`
- \`commit/rq23_np8\`: \`$(format_interval "$GADGET_COMMIT_RQ23_NP8_LOW" "$GADGET_COMMIT_RQ23_NP8_MEAN" "$GADGET_COMMIT_RQ23_NP8_HIGH")\`
- \`verify/rq23_np8\`: \`$(format_interval "$GADGET_VERIFY_RQ23_NP8_LOW" "$GADGET_VERIFY_RQ23_NP8_MEAN" "$GADGET_VERIFY_RQ23_NP8_HIGH")\`

## Serialized size snapshot

This block is captured from \`cargo run -q -p grid-commit --example size_snapshot\` during the scripted refresh:

\`\`\`text
$SIZE_SNAPSHOT_BLOCK
\`\`\`

## Scheme refresh policy

- \`ajtai_large\`, \`ajtai_goldilocks\`, and the \`rq23_np8\` rows in Ajtai, BDLOP, and gadget are the primary optimization targets
- \`f17\` rows remain useful as cheap commitment-path sanity checks, but they are not the main driver for the current optimization work
- refresh these isolated baselines after meaningful commitment-path changes

## Notes

- Commitment benchmarks should be compared using the same command shape, sample size, and host
- The scripted refresh workflow is the canonical way to update both runtime numbers and serialized-size output
EOF

mv "$TMP_BENCH_MD" "$BENCH_MD"
log "updated $BENCH_MD"
