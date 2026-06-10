# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NIR / ONNX-SNN importer

"""Neuromorphic Intermediate Representation (NIR) import utilities.

Converts NIR/ONNX graph definitions from snnTorch, Norse, and Sinabs into
ODE equations for the SC-NeuroCore compiler.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NIRGraph:
    """Imported NIR/ONNX-SNN graph representation.

    Attributes
    ----------
    nodes : dict[str, dict]
        Node name → parameters.
    edges : list[tuple[str, str]]
        Directed edges (source, target).
    equations : dict[str, str]
        Extracted ODE equations per node.
    framework : str
        Source framework.
    """

    nodes: dict[str, dict]
    edges: list[tuple[str, str]]
    equations: dict[str, str]
    framework: str


def import_nir_graph(
    nir_data: dict,
    *,
    framework: str = "snnTorch",
) -> NIRGraph:
    """Import a Neuromorphic Intermediate Representation graph.

    Converts NIR node definitions into ODE equations suitable
    for the SC-NeuroCore compilation pipeline.

    Parameters
    ----------
    nir_data : dict
        NIR graph as dictionary with 'nodes' and 'edges'.
    framework : str
        Source framework name.

    Returns
    -------
    NIRGraph
        Imported graph with extracted equations.
    """
    nodes = nir_data.get("nodes", {})
    edges = nir_data.get("edges", [])
    equations: dict[str, str] = {}

    for name, params in nodes.items():
        ntype = params.get("type", "LIF")
        tau = params.get("tau", 10.0)
        if ntype in ("LIF", "lif"):
            equations[name] = f"-(v - v_rest) / {tau} + I"
        elif ntype in ("Izhikevich", "izh"):
            equations[name] = "0.04 * v * v + 5 * v + 140 - u + I"
        else:
            equations[name] = f"-(v) / {tau} + I"

    return NIRGraph(
        nodes=nodes,
        edges=[(e[0], e[1]) for e in edges],
        equations=equations,
        framework=framework,
    )
