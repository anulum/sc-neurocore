# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Synthesis evidence command

"""Collect FPGA tool reports into optimiser evidence JSON."""

from __future__ import annotations

import argparse


def add_synthesis_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register synthesis evidence collection.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction[argparse.ArgumentParser]
        Top-level command registry.
    """
    parser = subparsers.add_parser(
        "collect-synthesis",
        help="Collect FPGA reports into an optimiser evidence record",
        description="Parse external utilisation, power, and timing reports into JSON evidence.",
    )
    parser.add_argument("--design", help="JSON compiler-design metadata")
    parser.add_argument(
        "--utilisation",
        "--utilization",
        dest="utilisation",
        help="Vivado utilisation or Quartus fitter report",
    )
    parser.add_argument("--power", help="Vivado or Quartus power report")
    parser.add_argument("--timing", help="Optional timing report")
    parser.add_argument("--accuracy-score", type=float, help="Measured accuracy or parity score")
    parser.add_argument("--latency-cycles", type=int, help="Explicit inference latency cycles")
    parser.add_argument("--clock-mhz", type=float, help="Clock used for energy calculation")
    parser.add_argument(
        "--inferences-per-run", type=int, help="Inferences represented by the reported latency"
    )
    parser.add_argument("--out", help="Output JSON evidence path")
    parser.set_defaults(handler=run_collect_synthesis)


def run_collect_synthesis(args: argparse.Namespace) -> int:
    """Collect synthesis reports into optimiser evidence JSON.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``collect-synthesis`` arguments.

    Returns
    -------
    int
        Zero on success, otherwise one for missing or invalid evidence.
    """
    from sc_neurocore.optimizer import build_payload_from_reports, write_payload

    required = (
        ("design", "--design"),
        ("utilisation", "--utilisation"),
        ("power", "--power"),
        ("accuracy_score", "--accuracy-score"),
    )
    missing = [flag for attribute, flag in required if getattr(args, attribute) is None]
    if missing:
        print(f"Error: collect-synthesis requires {', '.join(missing)}")
        return 1

    try:
        payload = build_payload_from_reports(
            design_path=args.design,
            utilisation_path=args.utilisation,
            power_path=args.power,
            timing_path=args.timing,
            accuracy_score=args.accuracy_score,
            latency_cycles=args.latency_cycles,
            clock_mhz=args.clock_mhz,
            inferences_per_run=args.inferences_per_run,
        )
        write_payload(payload, args.out)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    if args.out is not None:
        print(f"Evidence written: {args.out}")
    return 0
