#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pre-push preflight gate — mirrors CI checks locally."""

import argparse
import subprocess
import sys

GATES = [
    ("black", ["python", "-m", "black", "--check", "src/", "tests/"]),
    ("bandit", ["python", "-m", "bandit", "-r", "src/sc_neurocore/", "-c", "pyproject.toml", "-q"]),
    ("spdx-guard", [
        "bash", "-c",
        'MISSING=$(find src tests engine/src engine/tests engine/benches hdl bridge '
        '-type f \\( -name "*.py" -o -name "*.rs" -o -name "*.v" \\) '
        '! -path "*/__pycache__/*" '
        '-exec grep -rL "SPDX-License-Identifier" {} +) || true; '
        '[ -z "$MISSING" ] || { echo "Missing SPDX headers:"; echo "$MISSING"; exit 1; }'
    ]),
    ("pytest", [
        "python", "-m", "pytest", "tests/", "-v",
        "--cov=sc_neurocore", "--cov-report=term", "--cov-fail-under=98",
    ]),
]


def run_gate(name: str, cmd: list[str]) -> bool:
    print(f"\n{'='*60}")
    print(f"  GATE: {name}")
    print(f"{'='*60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  FAIL: {name}")
        return False
    print(f"  PASS: {name}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="SC-NeuroCore preflight checks")
    parser.add_argument("--no-tests", action="store_true", help="Skip pytest (fast lint-only mode)")
    parser.add_argument("--coverage", action="store_true", help="Same as default (tests always include coverage)")
    args = parser.parse_args()

    gates = GATES
    if args.no_tests:
        gates = [(n, c) for n, c in gates if n != "pytest"]

    passed, failed = [], []
    for name, cmd in gates:
        if run_gate(name, cmd):
            passed.append(name)
        else:
            failed.append(name)

    print(f"\n{'='*60}")
    print(f"  PREFLIGHT: {len(passed)} passed, {len(failed)} failed")
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
    print(f"{'='*60}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
