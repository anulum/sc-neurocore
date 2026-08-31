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

skip_slow=false
coverage=true
write_html=false
batch_size="$DEFAULT_BATCH_SIZE"
cov_fail_under="$DEFAULT_COV_FAIL_UNDER"
for arg in "$@"; do
    case "$arg" in
        --skip-slow) skip_slow=true ;;
        --html)      write_html=true ;;
        --no-cov)    coverage=false ;;
        --batch-size=*) batch_size="${arg#*=}" ;;
        --cov-fail-under=*) cov_fail_under="${arg#*=}" ;;
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

if $coverage; then
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
        pytest_args+=(--cov=sc_neurocore --cov-append --cov-report=)
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
echo "  Batches run: $batch_count"

if $coverage; then
    python -m coverage report --skip-empty --precision=2 --fail-under="$cov_fail_under"
    python -m coverage xml -o coverage.xml
    if $write_html; then
        python -m coverage html -d htmlcov
    fi
fi
