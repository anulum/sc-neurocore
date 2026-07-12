# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Runtime information command

"""Report package, Python, engine, and optional dependency versions."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys


def add_info_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the runtime information command.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction[argparse.ArgumentParser]
        Top-level command registry.
    """
    parser = subparsers.add_parser(
        "info",
        help="Show package, Python, engine, and optional dependency versions",
        description="Inspect the installed SC-NeuroCore runtime before compiling a model.",
    )
    parser.set_defaults(handler=run_info)


def run_info(args: argparse.Namespace) -> int:
    """Print runtime information without importing optional dependencies.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``info`` arguments.

    Returns
    -------
    int
        Always zero after the status report is emitted.
    """
    del args
    from sc_neurocore import __version__

    print(f"sc-neurocore {__version__}")
    print(f"Python {sys.version}")
    print(_format_engine_status(__version__))
    _print_optional_dependency_version("numpy", "NumPy")
    _print_optional_dependency_version("jax", "JAX")
    return 0


def _print_optional_dependency_version(module_name: str, label: str) -> None:
    loaded_module = sys.modules.get(module_name)
    if loaded_module is not None:
        version = getattr(loaded_module, "__version__", None)
        if version is not None:
            print(f"{label}: {version}")
        return
    try:
        version = importlib.metadata.version(module_name)
    except importlib.metadata.PackageNotFoundError:
        return
    print(f"{label}: {version}")


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


def _safe_simd_tier(engine: object) -> str:
    simd_tier = getattr(engine, "simd_tier", None)
    if not callable(simd_tier):
        return "unknown"
    try:
        return str(simd_tier())
    except Exception:
        return "unknown"
