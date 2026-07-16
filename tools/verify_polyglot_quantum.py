# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Polyglot quantum cognition verification

"""Verify all four quantum cognition backends compile and run.

Checks Python, Rust, Mojo, and Julia implementations of SpinPoolMPS.
Skips backends whose toolchains are not installed.

Usage::

    python tools/verify_polyglot_quantum.py

Exit codes:
    0 — all available backends pass
    1 — at least one available backend failed
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

from sc_neurocore.accel.mojo.isa_baseline import pin_isa

_QC_DIR = Path(__file__).resolve().parent.parent / "src" / "sc_neurocore" / "quantum_cognition"

# Colours
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _check_python() -> tuple[str, bool, float]:
    """Verify Python SpinPoolMPS import and basic operation."""
    t0 = time.monotonic()
    try:
        # Add source to path
        src = str(_QC_DIR.parent.parent)
        if src not in sys.path:
            sys.path.insert(0, src)

        from sc_neurocore.quantum_cognition import SpinPoolMPS

        pool = SpinPoolMPS(n_sites=8, bond_dim=16)
        pool.apply_measurement(0, 1.0)
        pool.apply_measurement(3, 0.5)
        eff = pool.get_local_atp_efficiency(0)
        assert 0.0 <= eff <= 1.0, f"ATP efficiency out of range: {eff}"

        state = pool.get_state()
        assert "entanglement_map" in state

        dt = time.monotonic() - t0
        return "Python", True, dt
    except Exception as exc:
        dt = time.monotonic() - t0
        print(f"  Python error: {exc}")
        return "Python", False, dt


def _check_rust() -> tuple[str, bool, float]:
    """Verify all Rust quantum cognition files compile and pass tests."""
    rs_files = ["spin_pool.rs", "radical_pair.rs", "kane_mapper.rs"]
    existing = [f for f in rs_files if (_QC_DIR / f).exists()]
    if not existing:
        return "Rust", False, 0.0

    try:
        subprocess.run(["rustc", "--version"], capture_output=True, timeout=10)
    except (FileNotFoundError, subprocess.SubprocessError):
        print(f"  {_YELLOW}Rust: rustc not found (SKIPPED){_RESET}")
        return "Rust (skip)", True, 0.0

    t0 = time.monotonic()
    total_tests = 0
    try:
        with tempfile.TemporaryDirectory(prefix="scn_rust_quantum_tests_") as tmpdir:
            for rs_name in existing:
                rs_path = _QC_DIR / rs_name
                bin_name = rs_name.replace(".rs", "_test")
                out_path = Path(tmpdir) / bin_name

                result = subprocess.run(
                    ["rustc", "--test", str(rs_path), "-o", str(out_path), "-C", "opt-level=2"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if result.returncode != 0:
                    print(f"  Rust compile error ({rs_name}):\n{result.stderr[:300]}")
                    return "Rust", False, time.monotonic() - t0

                result = subprocess.run(
                    [str(out_path)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode != 0:
                    print(f"  Rust test error ({rs_name}):\n{result.stdout[:300]}")
                    return "Rust", False, time.monotonic() - t0

                # Extract test count
                for line in result.stdout.strip().split("\n"):
                    if "test result" in line:
                        print(f"  [{rs_name}] {line.strip()}")
                        # Parse "X passed"
                        import re

                        m = re.search(r"(\d+) passed", line)
                        if m:
                            total_tests += int(m.group(1))

        dt = time.monotonic() - t0
        print(f"  Rust total: {total_tests} tests across {len(existing)} files")
        return "Rust", True, dt
    except Exception as exc:
        dt = time.monotonic() - t0
        print(f"  Rust error: {exc}")
        return "Rust", False, dt


def _check_mojo() -> tuple[str, bool, float]:
    """Verify Mojo SpinPoolMPS compilation."""
    mojo_file = _QC_DIR / "spin_pool.mojo"
    if not mojo_file.exists():
        return "Mojo", False, 0.0

    try:
        subprocess.run(["mojo", "--version"], capture_output=True, timeout=10)
    except (FileNotFoundError, subprocess.SubprocessError):
        print(f"  {_YELLOW}Mojo: mojo not found (SKIPPED){_RESET}")
        return "Mojo (skip)", True, 0.0

    t0 = time.monotonic()
    try:
        result = subprocess.run(
            pin_isa(["mojo", "run", str(mojo_file)]),
            capture_output=True,
            text=True,
            timeout=120,
        )
        dt = time.monotonic() - t0
        if result.returncode != 0:
            print(f"  Mojo error:\n{result.stderr[:500]}")
            return "Mojo", False, dt
        return "Mojo", True, dt
    except Exception as exc:
        dt = time.monotonic() - t0
        print(f"  Mojo error: {exc}")
        return "Mojo", False, dt


def _check_julia() -> tuple[str, bool, float]:
    """Verify Julia SpinPoolMPS execution."""
    jl_file = _QC_DIR / "spin_pool.jl"
    if not jl_file.exists():
        return "Julia", False, 0.0

    try:
        subprocess.run(["julia", "--version"], capture_output=True, timeout=10)
    except (FileNotFoundError, subprocess.SubprocessError):
        print(f"  {_YELLOW}Julia: julia not found (SKIPPED){_RESET}")
        return "Julia (skip)", True, 0.0

    t0 = time.monotonic()
    try:
        result = subprocess.run(
            ["julia", "--project=@.", str(jl_file)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(_QC_DIR),
        )
        dt = time.monotonic() - t0
        if result.returncode != 0:
            print(f"  Julia error:\n{result.stderr[:500]}")
            return "Julia", False, dt
        return "Julia", True, dt
    except Exception as exc:
        dt = time.monotonic() - t0
        print(f"  Julia error: {exc}")
        return "Julia", False, dt


def main() -> int:
    """Run all backend checks."""
    print(f"\n{_BOLD}{_CYAN}═══ Quantum Cognition Polyglot Verification ═══{_RESET}\n")

    checks = [_check_python, _check_rust, _check_mojo, _check_julia]
    results: list[tuple[str, bool, float]] = []

    for check in checks:
        name, passed, dt = check()
        results.append((name, passed, dt))
        status = f"{_GREEN}PASS{_RESET}" if passed else f"{_RED}FAIL{_RESET}"
        print(f"  {name:15s}  {status}  ({dt * 1000:.0f} ms)")

    print(f"\n{_BOLD}{'=' * 50}{_RESET}")
    passed = sum(1 for _, p, _ in results if p)
    total = len(results)
    colour = _GREEN if passed == total else _RED
    print(f"  {colour}{passed}/{total} backends passed{_RESET}")

    failures = [name for name, p, _ in results if not p]
    if failures:
        print(f"  Failed: {', '.join(failures)}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
