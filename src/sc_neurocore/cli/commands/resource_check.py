# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Resource and timing check command

"""Synthesise, place and route one IR graph and judge it against its FPGA device."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: Exit status for each overall verdict; an undetermined run is never success.
EXIT_CODES = {"pass": 0, "violation": 1, "undetermined": 2}


def add_resource_check_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the ``resource-check`` command.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction[argparse.ArgumentParser]
        Top-level command registry.
    """
    parser = subparsers.add_parser(
        "resource-check",
        help="Synthesise, place and route an IR graph and judge capacity, I/O, timing and the estimate",
        description=(
            "Emit one IR graph for an FPGA device, synthesise it with Yosys, place and route it "
            "out of context with nextpnr, and judge logic capacity, package I/O, timing at the "
            "given clock and, where the engine has a calibrated static estimate, that estimate. "
            "Exit 0 when every check passes, 1 on any violation, 2 when a check is undetermined."
        ),
    )
    parser.add_argument("graph", help="IR text file (sc.graph)")
    parser.add_argument("--clock-mhz", type=int, required=True, help="Target clock in MHz")
    parser.add_argument("--target", default="ice40", choices=["ice40"], help="FPGA target")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        required=True,
        help="Limit in seconds for each Yosys and nextpnr run, chosen for this host",
    )
    parser.add_argument("--output", "-o", default=None, help="Write the JSON report to this file")
    parser.set_defaults(handler=run_resource_check)


def run_resource_check(args: argparse.Namespace) -> int:
    """Run the resource and timing check of one IR graph.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``resource-check`` arguments.

    Returns
    -------
    int
        0 when every check passes, 1 on a violation, 2 when a check is
        undetermined or the graph cannot be read, parsed or verified.
    """
    from sc_neurocore_engine import parse_ir

    from sc_neurocore.studio.resource_feedback import run_resource_feedback

    source = Path(args.graph)
    try:
        graph = parse_ir(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(f"Error: cannot read IR graph {source}: {error}")
        return EXIT_CODES["undetermined"]
    problems = graph.verify()
    if problems:
        print(f"Error: {source} fails IR verification:")
        for problem in problems:
            print(f"  {problem}")
        return EXIT_CODES["undetermined"]
    if args.clock_mhz <= 0 or not args.timeout_seconds > 0:
        print("Error: --clock-mhz and --timeout-seconds must be positive")
        return EXIT_CODES["undetermined"]
    timeout = args.timeout_seconds
    report = run_resource_feedback(
        graph, clock_mhz=args.clock_mhz, target=args.target, timeout_seconds=timeout
    )
    if args.output is not None:
        Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    for check in report["checks"]:
        print(f"{check['verdict']:>12}  {check['name']}: {check['detail']}")
    if "estimate_comparison" in report:
        print(f"estimate: {report['estimate_comparison']}")
    print(f"verdict: {report['verdict']}")
    return EXIT_CODES[report["verdict"]]
