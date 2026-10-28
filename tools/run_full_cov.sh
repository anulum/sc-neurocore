#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Full pytest sweep in memory-reclaiming file batches (closes #58 OOM)
#
# The full repo has 10k+ tests across thousands of modules. Running them in
# a single `pytest --cov` invocation can OOM
# because pytest's plugin caches + numpy fixtures + coverage tracker
# accumulate across the entire run.
#
# This script chunks every pytest-discoverable test module, exiting each
# pytest process between batches so the OS reclaims memory. Coverage
# is appended via `--cov-append` so the final `.coverage` file is
# the union of all batches.
#
# Usage:
#   bash tools/run_full_cov.sh
#   bash tools/run_full_cov.sh --html       # also write htmlcov/
#   bash tools/run_full_cov.sh --no-cov     # compatibility matrix leg
#   bash tools/run_full_cov.sh --batch-size=384
#   bash tools/run_full_cov.sh --cov-fail-under=100
#   bash tools/run_full_cov.sh --skip-slow  # exclude known-heavy dirs
#   bash tools/run_full_cov.sh --shard=1/2  # run batches 1, 3, 5, ... only
#   bash tools/run_full_cov.sh --combine-from=DIR --expect-shards=2 --cov-fail-under=100
#
# Sharding: a shard runs every Nth batch of the same global batch list and
# leaves its `.coverage` data for the union; with coverage on it writes no
# report, because a partial union cannot pass the exact gate. The union run
# (--combine-from) runs no tests: it requires exactly the expected number of
# `.coverage.*` shard files in DIR, combines them, and applies the gate.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="bridge:src"
COV_DATA="$REPO_ROOT/.coverage"
JUNIT_DIR="$REPO_ROOT/test-results"
DEFAULT_BATCH_SIZE=384
DEFAULT_COV_FAIL_UNDER=100

# Known-heavy / opt-out directories. The user can flip --skip-slow to
# bypass them while iterating.
SKIP_SLOW=("tests/test_safety" "tests/test_nas" "tests/test_studio")

# Gate the union: optional external data (native platform legs) joins first.
report_union() {
    if [[ -n "${SCN_EXTRA_COVERAGE_DIR:-}" ]]; then
        python -m coverage combine --append --keep "$SCN_EXTRA_COVERAGE_DIR"
    fi
    python -m coverage report --skip-empty --show-missing --precision=2 --fail-under="$cov_fail_under"
    python -m coverage xml -o coverage.xml
    if $write_html; then
        python -m coverage html -d htmlcov
    fi
}

skip_slow=false
coverage=true
write_html=false
batch_size="$DEFAULT_BATCH_SIZE"
cov_fail_under="$DEFAULT_COV_FAIL_UNDER"
shard=""
combine_from=""
expect_shards=""
for arg in "$@"; do
    case "$arg" in
        --skip-slow) skip_slow=true ;;
        --html)      write_html=true ;;
        --no-cov)    coverage=false ;;
        --batch-size=*) batch_size="${arg#*=}" ;;
        --cov-fail-under=*) cov_fail_under="${arg#*=}" ;;
        --shard=*)   shard="${arg#*=}" ;;
        --combine-from=*) combine_from="${arg#*=}" ;;
        --expect-shards=*) expect_shards="${arg#*=}" ;;
        *)           echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

if ! [[ "$batch_size" =~ ^[1-9][0-9]*$ ]]; then
    echo "batch size must be a positive integer: $batch_size" >&2
    exit 2
fi
if ! [[ "$cov_fail_under" =~ ^([0-9]|[1-9][0-9]|100)$ ]]; then
    echo "coverage threshold must be an integer from 0 through 100: $cov_fail_under" >&2
    exit 2
fi
shard_index=1
shard_count=1
if [[ -n "$shard" ]]; then
    if ! [[ "$shard" =~ ^([1-9][0-9]*)/([1-9][0-9]*)$ ]] \
        || ((BASH_REMATCH[1] > BASH_REMATCH[2])); then
        echo "shard must be K/N with 1 <= K <= N: $shard" >&2
        exit 2
    fi
    shard_index=${BASH_REMATCH[1]}
    shard_count=${BASH_REMATCH[2]}
fi

# Union of shard coverage: no tests run here, so a missing shard cannot pass.
if [[ -n "$combine_from" ]]; then
    if [[ -n "$shard" ]] || ! $coverage; then
        echo "--combine-from excludes --shard and --no-cov" >&2
        exit 2
    fi
    if ! [[ "$expect_shards" =~ ^[1-9][0-9]*$ ]]; then
        echo "--combine-from needs --expect-shards=N with N >= 1" >&2
        exit 2
    fi
    mapfile -d '' -t shard_files < <(
        find "$combine_from" -maxdepth 1 -type f -name '.coverage.*' -print0 2>/dev/null | sort -z
    )
    if ((${#shard_files[@]} != expect_shards)); then
        echo "expected $expect_shards shard coverage files in $combine_from, found ${#shard_files[@]}" >&2
        exit 1
    fi
    rm -f "$COV_DATA" "$COV_DATA".*
    python -m coverage combine --keep "${shard_files[@]}"
    report_union
    exit 0
elif [[ -n "$expect_shards" ]]; then
    echo "--expect-shards needs --combine-from" >&2
    exit 2
fi

if $coverage && [[ -z "$combine_from" ]]; then
    rm -f "$COV_DATA" "$COV_DATA".*
fi
mkdir -p "$JUNIT_DIR"

# Match pytest's default Python module patterns across the complete tests tree.
# NUL delimiters preserve any future whitespace-bearing path without shell splitting.
mapfile -d '' -t discovered < <(
    find tests -type f \( -name 'test_*.py' -o -name '*_test.py' \) -print0 | sort -z
)

test_files=()
for path in "${discovered[@]}"; do
    # External formal proofs are independent of the Python interpreter. Run
    # them once in the coverage-bearing primary leg, not in every compatibility
    # leg where five concurrent solver copies only duplicate the same evidence.
    if ! $coverage && [[ "$path" == "tests/test_catalogue_formal_smoke.py" ]]; then
        echo "[primary-only] $path"
        continue
    fi
    if $skip_slow; then
        skip=false
        for prefix in "${SKIP_SLOW[@]}"; do
            [[ "$path" == "$prefix/"* ]] && skip=true && break
        done
        $skip && { echo "[skip-slow] $path"; continue; }
    fi
    test_files+=("$path")
done

if ((${#test_files[@]} == 0)); then
    echo "no pytest modules discovered" >&2
    exit 2
fi

ran=0
total=${#test_files[@]}
batch_count=$(((total + batch_size - 1) / batch_size))
for ((offset = 0; offset < total; offset += batch_size)); do
    batch=("${test_files[@]:offset:batch_size}")
    batch_number=$((offset / batch_size + 1))
    # Shards take every Nth batch of the one global list, so the union is exact.
    if (((batch_number - 1) % shard_count != shard_index - 1)); then
        continue
    fi
    junit_path=$(printf '%s/test-results-batch-%03d.xml' "$JUNIT_DIR" "$batch_number")
    echo
    echo "=========================================================="
    echo "  batch $batch_number/$batch_count (${#batch[@]} modules)"
    echo "=========================================================="

    pytest_args=(
        -m pytest "${batch[@]}" -q --tb=short
        --junitxml="$junit_path"
    )
    if $coverage; then
        # The canonical 100% threshold applies to the union below. Enforcing
        # it inside each partial batch rejects valid intermediate coverage.
        pytest_args+=(--cov=sc_neurocore --cov-append --cov-report= --cov-fail-under=0)
    fi

    if python "${pytest_args[@]}"; then
        ran=$((ran + ${#batch[@]}))
    else
        echo "batch $batch_number failed" >&2
        exit 1
    fi
done

echo
echo "=========================================================="
echo "  FULL-SUITE SUMMARY"
echo "=========================================================="
echo "  Modules run: $ran"
if ((shard_count > 1)); then
    echo "  Batches run: shard $shard_index/$shard_count of $batch_count"
else
    echo "  Batches run: $batch_count"
fi

if $coverage; then
    if ((shard_count > 1)); then
        echo "  Shard $shard_index/$shard_count: coverage data left in $COV_DATA for the union"
    else
        report_union
    fi
fi
