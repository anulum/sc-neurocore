# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Minimal CLI for SC-NeuroCore

"""Minimal CLI for SC-NeuroCore."""

import argparse
import sys
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="sc-neurocore",
        description="SC-NeuroCore — Universal Stochastic Computing Framework",
    )
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["info", "benchmark", "preflight"],
        help="Command to run",
    )
    args = parser.parse_args()

    if args.version:
        from sc_neurocore import __version__

        print(f"sc-neurocore {__version__}")
        return 0

    if args.command == "info":
        return _cmd_info()
    if args.command == "benchmark":
        return _cmd_benchmark()
    if args.command == "preflight":
        return _cmd_preflight()

    parser.print_help()
    return 0


def _cmd_info() -> int:
    from sc_neurocore import __version__

    print(f"sc-neurocore {__version__}")
    print(f"Python {sys.version}")
    print(_format_engine_status(__version__))
    _print_optional_dependency_version("numpy", "NumPy")
    _print_optional_dependency_version("jax", "JAX")

    return 0


def _print_optional_dependency_version(module_name: str, label: str) -> None:
    try:
        module = __import__(module_name)
    except Exception:
        return
    print(f"{label}: {getattr(module, '__version__', 'unknown')}")


def _format_engine_status(expected_version: str) -> str:
    try:
        import sc_neurocore_engine as engine
    except ImportError:
        return "Rust engine: not available"

    version = getattr(engine, "__version__", "unknown")
    simd_tier = _safe_simd_tier(engine)
    if version != expected_version:
        return (
            f"Rust engine: {version} ({simd_tier}) [version mismatch: expected {expected_version}]"
        )
    return f"Rust engine: {version} ({simd_tier})"


def _safe_simd_tier(engine: Any) -> str:
    simd_tier = getattr(engine, "simd_tier", None)
    if not callable(simd_tier):
        return "unknown"
    try:
        return str(simd_tier())
    except Exception:
        return "unknown"


def _cmd_benchmark() -> int:
    import subprocess

    return subprocess.run(
        [sys.executable, "-m", "pytest", "benchmarks/benchmark_suite.py", "--benchmark-only"],
    ).returncode


def _cmd_preflight() -> int:
    import subprocess

    return subprocess.run([sys.executable, "tools/preflight.py"]).returncode


if __name__ == "__main__":
    sys.exit(main())
