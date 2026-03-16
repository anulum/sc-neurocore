# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Minimal CLI for SC-NeuroCore."""

import argparse
import sys


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

    try:
        import sc_neurocore_engine

        print(f"Rust engine: {sc_neurocore_engine.__version__} ({sc_neurocore_engine.simd_tier()})")
    except ImportError:
        print("Rust engine: not available")

    try:
        import numpy

        print(f"NumPy: {numpy.__version__}")
    except ImportError:
        pass

    try:
        import jax

        print(f"JAX: {jax.__version__}")
    except ImportError:
        pass

    return 0


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
