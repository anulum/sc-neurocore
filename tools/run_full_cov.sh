#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Full pytest --cov sweep in batched process-per-dir mode (closes #58 OOM)
#
# The full repo has 10k+ tests across 80+ packages. Running them in
# a single `pytest --cov` invocation OOMs around the 27 % mark
# because pytest's plugin caches + numpy fixtures + coverage tracker
# accumulate across the entire run.
#
# This script chunks by top-level test directory, exiting each
# pytest process between batches so the OS reclaims memory. Coverage
# is appended via `--cov-append` so the final `.coverage` file is
# the union of all batches.
#
# Usage:
#   bash tools/run_full_cov.sh
#   bash tools/run_full_cov.sh --html       # also write htmlcov/
#   bash tools/run_full_cov.sh --skip-slow  # exclude known-heavy dirs

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="bridge:src"
COV_DATA="$REPO_ROOT/.coverage"
rm -f "$COV_DATA" "$COV_DATA".*

# Known-heavy / opt-out directories. The user can flip --skip-slow to
# bypass them while iterating.
SKIP_SLOW=("tests/test_safety" "tests/test_nas" "tests/test_studio")

skip_slow=false
extra_args=()
for arg in "$@"; do
    case "$arg" in
        --skip-slow) skip_slow=true ;;
        --html)      extra_args+=(--cov-report=html) ;;
        *)           echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

# Collect every top-level test directory.
mapfile -t DIRS < <(find tests -maxdepth 1 -type d -name "test_*" | sort)

ran=0
failed_dirs=()
for d in "${DIRS[@]}"; do
    if $skip_slow; then
        skip=false
        for s in "${SKIP_SLOW[@]}"; do
            [[ "$d" == "$s" ]] && skip=true && break
        done
        $skip && { echo "[skip-slow] $d"; continue; }
    fi
    echo
    echo "=========================================================="
    echo "  $d"
    echo "=========================================================="
    if python3 -m pytest "$d" \
        --cov=src/sc_neurocore --cov-append \
        --no-cov-on-fail --no-header -q --tb=line \
        --ignore-glob='*test_studio_full_synthesis*' \
        2>&1 | tail -8; then
        ran=$((ran + 1))
    else
        failed_dirs+=("$d")
    fi
done

echo
echo "=========================================================="
echo "  FULL-SUITE SUMMARY"
echo "=========================================================="
echo "  Directories run:    $ran"
echo "  Directories failed: ${#failed_dirs[@]}"
for f in "${failed_dirs[@]}"; do echo "    - $f"; done
echo
python3 -m coverage report --skip-empty --precision=2 "${extra_args[@]}" 2>&1 | tail -25
