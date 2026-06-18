#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Pre-push preflight gate — mirrors CI checks locally."""

import argparse
import pathlib
import subprocess
import sys
import tempfile

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

SPDX_DIRS = ["src", "tests", "engine/src", "engine/tests", "engine/benches", "hdl", "bridge"]
SPDX_EXTS = {".py", ".rs", ".v"}
SPDX_MARKER = "SPDX-License-Identifier"

ENGINE_DIR = pathlib.Path("engine")
BENCHMARK_EVIDENCE_REPORT = (
    pathlib.Path(tempfile.gettempdir()) / "sc_neurocore_benchmark_evidence_gate_report.json"
)


def _coverage_fail_under() -> int:
    pyproject = pathlib.Path("pyproject.toml")
    if not pyproject.exists():
        return 100
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return 100
    return int(data.get("tool", {}).get("coverage", {}).get("report", {}).get("fail_under", 100))


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
    ("capability-manifest", ["python", "tools/capability_manifest.py", "--check"]),
    (
        "ruff-check",
        [
            "python",
            "-m",
            "ruff",
            "check",
            "src/",
            "tests/",
            "tools/capability_manifest.py",
            "tools/benchmark_evidence_gate.py",
        ],
    ),
    (
        "ruff-format",
        [
            "python",
            "-m",
            "ruff",
            "format",
            "--check",
            "src/",
            "tests/",
            "tools/capability_manifest.py",
            "tools/benchmark_evidence_gate.py",
        ],
    ),
    (
        "benchmark-evidence",
        [
            "python",
            "tools/benchmark_evidence_gate.py",
            "--manifest",
            "benchmarks/benchmark_regression_gates.json",
            "--output",
            str(BENCHMARK_EVIDENCE_REPORT),
        ],
    ),
    ("bandit", ["python", "-m", "bandit", "-r", "src/sc_neurocore/", "-c", "pyproject.toml", "-q"]),
    ("mypy", ["python", "-m", "mypy", "--strict", "src/sc_neurocore/"]),
    (
        "docstring-policy",
        ["python", "-m", "pytest", "tests/test_public_docstring_policy.py", "-q"],
    ),
    ("spdx-guard", None),
    (
        "pytest",
        None,
    ),
]


SPDX_SKIP_PARTS = {
    "__pycache__",
    ".pixi",
    ".venv",
    "venv",
    "node_modules",
    "target",
    "build",
    "dist",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
}


def check_spdx() -> bool:
    missing = []
    for d in SPDX_DIRS:
        root = pathlib.Path(d)
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.suffix not in SPDX_EXTS:
                continue
            # Skip vendored / cached trees that are gitignored and not
            # part of the project's source set (e.g. the .pixi Python
            # environment under accel/mojo/).
            if any(part in SPDX_SKIP_PARTS for part in p.parts):
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


def run_gate(name: str, cmd: list[str] | None) -> bool:
    global _CARGO_AVAILABLE
    print(f"\n{'=' * 60}")
    print(f"  GATE: {name}")
    print(f"{'=' * 60}")
    if name.startswith("cargo-"):
        if _CARGO_AVAILABLE is None:
            _CARGO_AVAILABLE = _has_cargo() and ENGINE_DIR.exists()
        if not _CARGO_AVAILABLE:
            print(f"  SKIP: {name} (cargo or engine/ not found)")
            return True
    if name == "spdx-guard":
        ok = check_spdx()
    elif name == "pytest":
        ok = (
            subprocess.run(
                [
                    "python",
                    "-m",
                    "pytest",
                    "tests/",
                    "-v",
                    "--cov=sc_neurocore",
                    "--cov-report=term",
                    f"--cov-fail-under={_coverage_fail_under()}",
                ]
            ).returncode
            == 0
        )
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

    print(f"\n{'=' * 60}")
    print(f"  PREFLIGHT: {len(passed)} passed, {len(failed)} failed")
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
    print(f"{'=' * 60}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
