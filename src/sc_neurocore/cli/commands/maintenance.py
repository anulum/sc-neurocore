# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Maintenance commands

"""Delegate benchmark and preflight maintenance commands."""

from __future__ import annotations

import argparse
import subprocess
import sys


def add_maintenance_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register benchmark and preflight delegates.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction[argparse.ArgumentParser]
        Top-level command registry.
    """
    benchmark = subparsers.add_parser(
        "benchmark",
        help="Run the repository benchmark suite",
        description="Delegate to the repository pytest-benchmark suite.",
    )
    benchmark.set_defaults(handler=run_benchmark)

    preflight = subparsers.add_parser(
        "preflight",
        help="Run the repository preflight gate",
        description="Delegate to tools/preflight.py from a source checkout.",
    )
    preflight.set_defaults(handler=run_preflight)


def run_benchmark(args: argparse.Namespace) -> int:
    """Run the repository benchmark suite in a child interpreter.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``benchmark`` arguments.

    Returns
    -------
    int
        Child process exit status.
    """
    del args
    return subprocess.run(
        [sys.executable, "-m", "pytest", "benchmarks/benchmark_suite.py", "--benchmark-only"],
        check=False,
    ).returncode


def run_preflight(args: argparse.Namespace) -> int:
    """Run the repository preflight script in a child interpreter.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``preflight`` arguments.

    Returns
    -------
    int
        Child process exit status.
    """
    del args
    return subprocess.run(
        [sys.executable, "tools/preflight.py"],
        check=False,
    ).returncode
