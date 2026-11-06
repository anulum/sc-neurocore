# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evidence packet export and verification commands

"""Export the hash-bound evidence packet of one dense layer IR graph, and verify one."""

from __future__ import annotations

import argparse
from pathlib import Path


def add_evidence_packet_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register ``evidence-packet export`` and ``evidence-packet verify``.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction[argparse.ArgumentParser]
        Top-level command registry.
    """
    parser = subparsers.add_parser(
        "evidence-packet",
        help="Export or verify the evidence packet of one dense layer IR graph",
        description=(
            "Write, for one dense layer IR graph, its IR and SystemVerilog, guarantee "
            "summary, parameters, co-simulation against the Python reference, formal job "
            "results, synthesis report and tool versions into one hash-bound directory, "
            "or check such a directory."
        ),
    )
    actions = parser.add_subparsers(dest="evidence_action", metavar="ACTION", required=True)
    export = actions.add_parser(
        "export",
        help="Write the evidence packet of one IR graph",
        description=(
            "The graph must be one sc.input of vec<rate,N>, constants, one "
            "sc.dense_forward and one sc.output; any other operation is refused by name."
        ),
    )
    export.add_argument("graph", help="IR text file (sc.graph)")
    export.add_argument("--output", "-o", required=True, help="New or empty packet directory")
    export.add_argument(
        "--cycles", type=int, default=256, help="Co-simulation cycles (default 256)"
    )
    export.add_argument("--target", default="ice40", help="Yosys synthesis target (default ice40)")
    export.set_defaults(handler=run_evidence_packet_export)
    verify = actions.add_parser(
        "verify",
        help="Re-hash every member of a packet",
        description=(
            "Refuse a missing, extra, altered or symbolically linked member and a "
            "manifest whose digest does not match its list. Exit status 1 on any."
        ),
    )
    verify.add_argument("directory", help="Packet directory")
    verify.set_defaults(handler=run_evidence_packet_verify)


def run_evidence_packet_export(args: argparse.Namespace) -> int:
    """Export one evidence packet.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``evidence-packet export`` arguments.

    Returns
    -------
    int
        Zero when the packet was written and the co-simulation agreed; one when the
        file cannot be read, the graph is invalid or unsupported, the output directory
        is not empty, the simulator is missing or fails, or the co-simulation differs.
    """
    from sc_neurocore.safety_cert.dense_cosim import CosimError, SimulatorUnavailableError
    from sc_neurocore.safety_cert.evidence_packet import export_evidence_packet

    source = Path(args.graph)
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        print(f"Error: cannot read IR graph {source}: {error}")
        return 1
    try:
        export = export_evidence_packet(
            text, Path(args.output), cycles=args.cycles, synthesis_target=args.target
        )
    except (ValueError, SimulatorUnavailableError, CosimError) as error:
        print(f"Error: no evidence packet for {source}: {error}")
        return 1
    difference = export.first_difference
    print(f"Evidence packet:  {export.directory}")
    print(f"Members:          {export.members}")
    print(f"Manifest SHA-256: {export.manifest_sha256}")
    print(
        "Co-simulation:    "
        + ("AGREE" if export.cosim_agrees else f"DIFFER (first at cycle {difference})")
    )
    return 0 if export.cosim_agrees else 1


def run_evidence_packet_verify(args: argparse.Namespace) -> int:
    """Verify one evidence packet.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``evidence-packet verify`` arguments.

    Returns
    -------
    int
        Zero when every member matches the manifest; one otherwise.
    """
    from sc_neurocore.safety_cert.evidence_packet import verify_evidence_packet

    report = verify_evidence_packet(Path(args.directory))
    if not report.ok:
        print(f"Error: evidence packet {report.directory} does not verify:")
        for problem in report.problems:
            print(f"  {problem}")
        return 1
    print(f"Evidence packet {report.directory} verifies: {report.members} members")
    print(f"Manifest SHA-256: {report.manifest_sha256}")
    return 0
