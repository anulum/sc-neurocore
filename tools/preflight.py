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

ENGINE_DIR = pathlib.Path("engine")

# Black targets: all .py files except drivers/, matching .pre-commit-config.yaml
BLACK_DIRS = [
    "src/",
    "tests/",
    ".github/",
    "benchmarks/",
    "bridge/",
    "cosim/",
    "examples/",
    "research/",
    "scripts/",
    "tools/",
]

GATES = [
    ("cargo-fmt", ["cargo", "fmt", "--check", "--manifest-path", "engine/Cargo.toml"]),
    (
        "cargo-clippy",
        [
            "cargo",
            "clippy",
            "--all-targets",
            "--manifest-path",
            "engine/Cargo.toml",
            "--",
            "-D",
            "warnings",
        ],
    ),
    ("black", None),  # custom handler
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
            "--cov-fail-under=100",
        ],
    ),
]


def check_black() -> bool:
    """Run black --check on all directories matching pre-commit scope."""
    existing = [d for d in BLACK_DIRS if pathlib.Path(d.rstrip("/")).exists()]
    # Also check root-level .py files (conftest.py etc.)
    root_py = [str(p) for p in pathlib.Path(".").glob("*.py")]
    targets = existing + root_py
    if not targets:
        return True
    cmd = [sys.executable, "-m", "black", "--check"] + targets
    return subprocess.run(cmd).returncode == 0


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


def _has_cargo() -> bool:
    try:
        subprocess.run(["cargo", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


_CARGO_AVAILABLE: bool | None = None


def run_gate(name: str, cmd) -> bool:
    global _CARGO_AVAILABLE
    print(f"\n{'='*60}")
    print(f"  GATE: {name}")
    print(f"{'='*60}")
    if name.startswith("cargo-"):
        if _CARGO_AVAILABLE is None:
            _CARGO_AVAILABLE = _has_cargo() and ENGINE_DIR.exists()
        if not _CARGO_AVAILABLE:
            print(f"  SKIP: {name} (cargo or engine/ not found)")
            return True
    if name == "black":
        ok = check_black()
    elif name == "spdx-guard":
        ok = check_spdx()
    elif cmd is not None:
        ok = subprocess.run(cmd).returncode == 0
    else:
        ok = True
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
