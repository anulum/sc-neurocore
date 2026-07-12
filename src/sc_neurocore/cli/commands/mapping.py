# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NIR silicon mapping command

"""Generate deterministic silicon-mapping reports for NIR graphs."""

from __future__ import annotations

import argparse
from pathlib import Path


def add_mapping_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the NIR silicon-mapping command.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction[argparse.ArgumentParser]
        Top-level command registry.
    """
    parser = subparsers.add_parser(
        "map-nir",
        help="Map a NIR graph onto named neuromorphic targets",
        description="Write a deterministic silicon-mapping report for one NIR graph.",
    )
    parser.add_argument("model", nargs="?", help="NIR model file")
    parser.add_argument("--output", "-o", default="build", help="Report output directory")
    parser.add_argument("--dt", type=float, default=1.0, help="NIR simulation timestep")
    parser.add_argument("--T", type=int, default=256, help="Stochastic bitstream length")
    parser.add_argument(
        "--hardware-targets",
        default="loihi2,spinnaker2,akida",
        help="Comma-separated neuromorphic target identifiers",
    )
    parser.set_defaults(handler=run_mapping)


def run_mapping(args: argparse.Namespace) -> int:
    """Write a NIR silicon-mapping report.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``map-nir`` arguments.

    Returns
    -------
    int
        Zero on success, otherwise one for invalid input or conversion failure.
    """
    if not args.model:
        print(
            "Error: map-nir requires a NIR model file. Usage: "
            "sc-neurocore map-nir model.nir -o build/silicon"
        )
        return 1
    model_path = str(args.model)
    if Path(model_path).suffix.lower() != ".nir":
        print("Error: map-nir supports .nir files only")
        return 1

    targets = tuple(item.strip() for item in str(args.hardware_targets).split(",") if item.strip())
    if not targets:
        print("Error: --hardware-targets must name at least one target")
        return 1

    try:
        import nir as nir_lib

        from sc_neurocore.nir_bridge import from_nir
        from sc_neurocore.nir_bridge.silicon_mapping import (
            SiliconMappingConfig,
            write_silicon_mapping_report,
        )

        graph = nir_lib.read(model_path)
        network = from_nir(graph, dt=float(args.dt))
        report_path = write_silicon_mapping_report(
            str(args.output),
            network,
            SiliconMappingConfig(
                targets=targets,
                bitstream_length=int(args.T),
            ),
        )
    except (ImportError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    print("NIR silicon mapping report generated")
    print(f"  Targets:  {', '.join(targets)}")
    print(f"  Report:   {report_path}")
    return 0
