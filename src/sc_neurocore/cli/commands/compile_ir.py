# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — IR graph compilation command

"""Compile an IR graph to SystemVerilog and write its guarantee summary beside it."""

from __future__ import annotations

import argparse
from pathlib import Path

from sc_neurocore.compiler.guarantee_summary import write_guarantee_summary


def add_compile_ir_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the ``compile-ir`` command.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction[argparse.ArgumentParser]
        Top-level command registry.
    """
    parser = subparsers.add_parser(
        "compile-ir",
        help="Compile an IR graph into SystemVerilog with its guarantee summary",
        description=(
            "Verify one IR graph, emit its SystemVerilog top level, and write beside it "
            "which library module, formal job, co-simulation or stochastic estimate each "
            "operation rests on."
        ),
    )
    parser.add_argument("graph", help="IR text file (sc.graph)")
    parser.add_argument("--output", "-o", default="build", help="Artefact output directory")
    parser.add_argument(
        "--equivalence-depth",
        type=int,
        help="Check an admitted graph with bounded model checking at this frame depth",
    )
    parser.add_argument(
        "--equivalence-timeout",
        type=float,
        help="Seconds to allow for a requested equivalence check",
    )
    parser.set_defaults(handler=run_compile_ir)


def run_compile_ir(args: argparse.Namespace) -> int:
    """Emit one IR graph and its guarantee summary.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``compile-ir`` arguments.

    Returns
    -------
    int
        Zero when compilation and any requested proof pass; one otherwise.
    """
    from sc_neurocore_engine import parse_ir

    source = Path(args.graph)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as error:
        print(f"Error: cannot read IR graph {source}: {error}")
        return 1
    try:
        graph = parse_ir(text)
    except ValueError as error:
        print(f"Error: {source} is not valid IR: {error}")
        return 1
    problems = graph.verify()
    if problems:
        print(f"Error: {source} fails IR verification:")
        for problem in problems:
            print(f"  {problem}")
        return 1
    verilog = graph.emit_sv()
    proof = None
    if args.equivalence_depth is not None:
        from sc_neurocore.compiler.ir_equivalence import (
            DEFAULT_GRAPH_PROOF_TIMEOUT_SECONDS,
            check_graph_equivalence,
        )

        timeout = args.equivalence_timeout
        if timeout is None:
            timeout = DEFAULT_GRAPH_PROOF_TIMEOUT_SECONDS

        try:
            proof = check_graph_equivalence(
                graph.name,
                graph.to_text(),
                verilog,
                depth=args.equivalence_depth,
                timeout_s=timeout,
            )
        except ValueError as error:
            print(f"Error: {error}")
            return 1
    elif args.equivalence_timeout is not None:
        print("Error: --equivalence-timeout requires --equivalence-depth")
        return 1
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    sv_path = output / f"{graph.name}.sv"
    sv_path.write_text(verilog, encoding="utf-8")
    json_path, text_path = write_guarantee_summary(
        graph, sv_path, verilog=verilog, equivalence=None if proof is None else proof.to_json()
    )
    print(f"SystemVerilog:     {sv_path}")
    print(f"Guarantee summary: {json_path}")
    print(f"                   {text_path}")
    if proof is not None:
        print(f"Bounded graph equivalence {proof.verdict} at depth {proof.depth}")
        return 0 if proof.proven else 1
    return 0
