# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — One-command hardware path of a catalogue model

"""Run a catalogue model from its schema to synthesis, or list the models that can."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def add_hardware_path_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the ``hardware-path`` command.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction[argparse.ArgumentParser]
        Top-level command registry.
    """
    parser = subparsers.add_parser(
        "hardware-path",
        help="Take one catalogue model from its schema to synthesis in one command",
        description=(
            "Check the model's capabilities, write its RTL, co-simulate it bit-exactly "
            "against the bit-true C kernel, write its guarantee summary, then synthesise "
            "it (and place and route it for ice40 and ecp5 when nextpnr is installed). "
            "Every stage refuses by name; a refusal exits with status 1. With --list, "
            "print which catalogue models the path supports on this host and why the "
            "others are refused."
        ),
    )
    parser.add_argument(
        "models",
        nargs="*",
        metavar="MODEL",
        help="Catalogue model (one to run; any number, or none for all, with --list)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the supported set instead of running the path",
    )
    parser.add_argument("--output", "-o", default=None, help="New or empty output directory")
    parser.add_argument(
        "--q-format", default=None, help="Fixed-point format (default: the model's default)"
    )
    parser.add_argument(
        "--integrator", default=None, help="Integrator (default: the model's schema default)"
    )
    parser.add_argument(
        "--params", default=None, help="Comma-separated parameter overrides, e.g. tau=10,R=1"
    )
    parser.add_argument("--dt", type=float, default=None, help="Step override")
    parser.add_argument(
        "--current",
        type=float,
        default=None,
        help="Co-simulation input current (default: the Studio default, 10.0)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Co-simulation cycles, 1 to 2048 (default: the Studio default, 128)",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Synthesis target: ice40, ecp5, gowin or xilinx (default: the Studio default, ecp5)",
    )
    parser.add_argument("--module-name", default=None, help="Verilog module name")
    parser.set_defaults(handler=run_hardware_path_command)


def parse_params(text: str | None) -> dict[str, float]:
    """Parse ``name=value`` pairs separated by commas.

    Parameters
    ----------
    text : str or None
        The ``--params`` argument.

    Returns
    -------
    dict of str to float
        The overrides; empty for ``None`` or an empty string.

    Raises
    ------
    ValueError
        A pair has no ``=``, an empty name, or a value that is not a finite number.
    """
    result: dict[str, float] = {}
    for pair in (text or "").split(","):
        if not pair.strip():
            continue
        name, separator, value = pair.partition("=")
        if not separator or not name.strip():
            raise ValueError(f"parameter override {pair.strip()!r} is not name=value")
        try:
            number = float(value)
        except ValueError:
            raise ValueError(
                f"parameter override {name.strip()!r} has a value that is not a number"
            ) from None
        if not math.isfinite(number):
            raise ValueError(f"parameter override {name.strip()!r} is not finite")
        result[name.strip()] = number
    return result


def run_hardware_path_command(args: argparse.Namespace) -> int:
    """Run the hardware path of one model, or list the supported set.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``hardware-path`` arguments.

    Returns
    -------
    int
        Zero when the path completed or the list was printed; one on any
        refusal, on invalid arguments, or when no model or output is given.
    """
    if args.list:
        return _list_supported(args.models)
    if len(args.models) != 1 or args.output is None:
        print("Error: hardware-path needs exactly one MODEL and --output DIR (or --list)")
        return 1
    from sc_neurocore.studio.hardware_path import (
        DEFAULT_CURRENT,
        DEFAULT_N_STEPS,
        DEFAULT_TARGET,
        HardwarePathRefusal,
        run_hardware_path,
    )

    try:
        params = parse_params(args.params)
    except ValueError as error:
        print(f"Error: {error}")
        return 1
    try:
        result = run_hardware_path(
            args.models[0],
            out_dir=Path(args.output),
            q_format=args.q_format,
            integrator=args.integrator,
            params=params,
            dt=args.dt,
            current=DEFAULT_CURRENT if args.current is None else args.current,
            n_steps=DEFAULT_N_STEPS if args.steps is None else args.steps,
            target=DEFAULT_TARGET if args.target is None else args.target,
            module_name=args.module_name,
        )
    except HardwarePathRefusal as refusal:
        print(f"Error: hardware path of {args.models[0]} refused at {refusal.stage}")
        print(f"  {refusal.code}: {refusal.message}")
        if refusal.manifest_path is not None:
            print(f"  Record: {refusal.manifest_path}")
        return 1
    print(result.summary_path.read_text(encoding="utf-8"), end="")
    print(f"Manifest: {result.manifest_path}")
    return 0


def _list_supported(models: list[str]) -> int:
    """Print the supported set of ``models`` (every catalogue model when empty)."""
    from sc_neurocore.studio.hardware_path import supported_models

    rows = supported_models(models or None)
    for row in rows:
        if row["supported"]:
            combinations = ", ".join(
                f"{entry['integrator']}@{entry['q_format']}" for entry in row["combinations"]
            )
            print(f"{row['model']}: supported ({combinations})")
        else:
            refusal = row["refusal"]
            print(f"{row['model']}: refused, {refusal['operation']}: {refusal['reason']}")
    supported = sum(1 for row in rows if row["supported"])
    print(f"{supported} of {len(rows)} models supported on this host")
    return 0
