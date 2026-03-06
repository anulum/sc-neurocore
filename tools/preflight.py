#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pre-push preflight gate — mirrors CI checks locally."""

import argparse
import pathlib
import subprocess
import sys

SPDX_DIRS = ["src", "tests", "engine/src", "engine/tests", "engine/benches", "hdl", "bridge"]
SPDX_EXTS = {".py", ".rs", ".v"}
SPDX_MARKER = "SPDX-License-Identifier"

GATES = [
    ("black", ["python", "-m", "black", "--check", "src/", "tests/"]),
    ("ruff", ["python", "-m", "ruff", "check", "src/", "tests/"]),
    ("bandit", ["python", "-m", "bandit", "-r", "src/sc_neurocore/", "-c", "pyproject.toml", "-q"]),
    ("spdx-guard", None),
    (
        "pytest",
        [
            "python",
            "-m",
            "pytest",
            "tests/",
            "-v",
            "--cov=sc_neurocore",
            "--cov-report=term",
            "--cov-fail-under=98",
        ],
    ),
]


def check_spdx() -> bool:
    missing = []
    for d in SPDX_DIRS:
        root = pathlib.Path(d)
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.suffix not in SPDX_EXTS or "__pycache__" in p.parts:
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")[:2048]
            except OSError:
                continue
            if SPDX_MARKER not in text:
                missing.append(str(p))
    if missing:
        print("Missing SPDX headers:")
        for f in missing:
            print(f"  {f}")
        return False
    return True


def run_gate(name: str, cmd) -> bool:
    print(f"\n{'='*60}")
    print(f"  GATE: {name}")
    print(f"{'='*60}")
    if cmd is None:
        ok = check_spdx()
    else:
        ok = subprocess.run(cmd).returncode == 0
    print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="SC-NeuroCore preflight checks")
    parser.add_argument("--no-tests", action="store_true", help="Skip pytest (fast lint-only mode)")
    parser.add_argument(
        "--coverage", action="store_true", help="Same as default (tests always include coverage)"
    )
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
