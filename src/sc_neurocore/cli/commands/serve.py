# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Streaming inference command

"""Start the NIR streaming inference server."""

from __future__ import annotations

import argparse
from pathlib import Path


def add_serve_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the streaming inference command.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction[argparse.ArgumentParser]
        Top-level command registry.
    """
    parser = subparsers.add_parser(
        "serve",
        help="Serve a NIR graph over the spike-stream protocol",
        description="Load one NIR graph and start a blocking spike-stream server.",
    )
    parser.add_argument("model", nargs="?", help="NIR model file")
    parser.add_argument("--port", type=int, default=8001, help="Server port (default: 8001)")
    parser.add_argument("--dt", type=float, default=1.0, help="NIR simulation timestep")
    parser.set_defaults(handler=run_serve)


def run_serve(args: argparse.Namespace) -> int:
    """Load a NIR graph and start its blocking spike server.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``serve`` arguments.

    Returns
    -------
    int
        Zero after a clean server shutdown, otherwise one for invalid input.
    """
    if not args.model:
        print("Error: serve requires a model file. Usage: sc-neurocore serve model.nir --port 8001")
        return 1
    model_path = str(args.model)
    extension = Path(model_path).suffix.lower()
    if extension != ".nir":
        print(f"Error: serve currently supports .nir files only, got '{extension}'")
        return 1

    import nir as nir_lib

    from sc_neurocore.nir_bridge import from_nir
    from sc_neurocore.serve import SpikeServer

    graph = nir_lib.read(model_path)
    network = from_nir(graph, dt=float(args.dt))
    print(f"Loaded NIR graph with {len(network.topo_order)} nodes")
    server = SpikeServer(network, port=int(args.port))
    server.start(blocking=True)
    return 0
