#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Exercise original Lapicque capture and consumer contracts in CI

set -euo pipefail

lapicque_checkout=""
lapicque_output=""
lapicque_python="python"
lapicque_unsupported=""
while (($#)); do
    case "$1" in
        --checkout|--output-dir|--python|--unsupported-python)
            if (($# < 2)); then
                printf 'Missing value for %s\n' "$1" >&2
                exit 2
            fi
            case "$1" in
                --checkout) lapicque_checkout=$2 ;;
                --output-dir) lapicque_output=$2 ;;
                --python) lapicque_python=$2 ;;
                --unsupported-python) lapicque_unsupported=$2 ;;
            esac
            shift 2
            ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
if [[ -z "$lapicque_checkout" || -z "$lapicque_output" || -z "$lapicque_unsupported" ]]; then
    printf 'Required: --checkout PATH --output-dir NEW_PATH --unsupported-python EXECUTABLE\n' >&2
    exit 2
fi

lapicque_tools=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
lapicque_checkout=$(cd -- "$lapicque_checkout" && pwd -P)
lapicque_output=$("$lapicque_python" -B -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$lapicque_output")
"$lapicque_python" -B -c 'import sys; assert sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 12) and sys.platform == "linux"'
"$lapicque_unsupported" -B -c 'import sys; assert sys.implementation.name == "cpython" and sys.version_info[:2] != (3, 12)'
mkdir -- "$lapicque_output" "$lapicque_output/tmp"
cd -- "$lapicque_checkout"

unset COVERAGE_PROCESS_START COVERAGE_PROCESS_CONFIG COVERAGE_FILE COVERAGE_RCFILE
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$lapicque_output/tmp"
export LAPICQUE_ORIGINAL_CHECKOUT="$lapicque_checkout"
export LAPICQUE_UNSUPPORTED_PYTHON="$lapicque_unsupported"

"$lapicque_python" -B -m pytest -q -p no:cacheprovider -c "$lapicque_checkout/pyproject.toml" \
    --rootdir="$lapicque_checkout" "$lapicque_tools/tests/test_capture_lapicque_samples.py" \
    --basetemp="$lapicque_output/normal-tests" --junitxml="$lapicque_output/normal-junit.xml"

lapicque_cpu=$("$lapicque_python" -B -c 'import os; print(min(os.sched_getaffinity(0)))')
taskset -c "$lapicque_cpu" "$lapicque_python" -B "$lapicque_tools/capture_lapicque_samples.py" \
    --checkout "$lapicque_checkout" --output-dir "$lapicque_output/consumer-seed"

export COVERAGE_RCFILE="$lapicque_tools/lapicque_capture.coveragerc"
export COVERAGE_FILE="$lapicque_output/.coverage-capture"
"$lapicque_python" -B -m coverage run --rcfile="$COVERAGE_RCFILE" -m pytest \
    -q -p no:cacheprovider -c "$lapicque_checkout/pyproject.toml" --rootdir="$lapicque_checkout" \
    "$lapicque_tools/tests/test_capture_lapicque_diagnostics.py" \
    "$lapicque_tools/tests/test_capture_lapicque_monitor_ownership.py" \
    "$lapicque_tools/tests/test_capture_lapicque_integrity.py" \
    "$lapicque_tools/tests/test_lapicque_return_records.py" \
    "$lapicque_tools/tests/test_capture_lapicque_admission.py" \
    --basetemp="$lapicque_output/diagnostic-tests" --junitxml="$lapicque_output/diagnostic-junit.xml"
"$lapicque_python" -B -m coverage combine --keep --rcfile="$COVERAGE_RCFILE"
"$lapicque_python" -B -m coverage json --rcfile="$COVERAGE_RCFILE" -o "$lapicque_output/capture-coverage.json"

export COVERAGE_RCFILE="$lapicque_tools/lapicque_consumer.coveragerc"
export COVERAGE_FILE="$lapicque_output/.coverage-consumer"
"$lapicque_python" -B -m coverage run --rcfile="$COVERAGE_RCFILE" -m pytest \
    -q -p no:cacheprovider -c "$lapicque_checkout/pyproject.toml" --rootdir="$lapicque_checkout" \
    "$lapicque_tools/tests/test_validate_lapicque_samples.py" \
    --lapicque-checkout="$lapicque_checkout" \
    --lapicque-samples="$lapicque_output/consumer-seed/original-seven-duration-samples.json" \
    --lapicque-report="$lapicque_output/consumer-seed/original-harness-report.json" \
    --basetemp="$lapicque_output/consumer-tests" --junitxml="$lapicque_output/consumer-junit.xml"
"$lapicque_python" -B -m coverage combine --keep --rcfile="$COVERAGE_RCFILE"
"$lapicque_python" -B -m coverage json --rcfile="$COVERAGE_RCFILE" -o "$lapicque_output/consumer-coverage.json"

"$lapicque_python" -B - "$lapicque_output" <<'PY'
from pathlib import Path
import json
import sys
import xml.etree.ElementTree as ET

root = Path(sys.argv[1])
for filename, expected in (("normal-junit.xml", 9), ("diagnostic-junit.xml", 57), ("consumer-junit.xml", 44)):
    cases = list(ET.parse(root / filename).getroot().iter("testcase"))
    assert len(cases) == expected, (filename, len(cases), expected)
    assert not any(case.find(tag) is not None for case in cases for tag in ("failure", "error", "skipped"))
for filename in ("capture-coverage.json", "consumer-coverage.json"):
    totals = json.loads((root / filename).read_text())["totals"]
    assert totals["percent_covered"] == 100 and totals["excluded_lines"] == 0, filename
print("Lapicque capture and consumer contracts passed; external performance qualification is separate.")
PY
