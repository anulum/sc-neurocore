# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Preserve NIR definitions and scheduling order

"""Retain parameter representation and scheduling through reference NIR files."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import fields
import json
from typing import Any

import numpy as np

from .execution_metadata import _unique_members

GRAPH_ORDER_KEY = "sc_neurocore.graph_order"
_SCHEMA = "sc-neurocore.nir-order.v1"


def preserve_definition(original: Any, exported: Any) -> Any:
    """Restore original parameter shapes and dtypes where values are unchanged.

    Parameters
    ----------
    original : nir.NIRNode or None
        Independent primitive definition retained at import.
    exported : nir.NIRNode
        Definition reconstructed from the current executable parameters.

    Returns
    -------
    nir.NIRNode
        Current definition, with unchanged fields represented as in the source.
        Changed parameters keep their current values rather than stale originals.
    """
    if type(original) is not type(exported):
        return exported
    unchanged = True
    for field in fields(original):
        if field.name == "metadata" or (
            field.name in {"input_type", "output_type"}
            and type(original).__name__ not in {"Input", "Output"}
        ):
            continue
        old, new = getattr(original, field.name), getattr(exported, field.name)
        if isinstance(old, (np.ndarray, tuple, int, float)) and isinstance(
            new, (np.ndarray, tuple, int, float)
        ):
            a, b = np.asarray(old).reshape(-1), np.asarray(new).reshape(-1)
            equal = False
            same_geometry = field.name not in {"weight", "bias", "scale", "threshold"} or (
                np.asarray(old).shape == np.asarray(new).shape
            )
            if same_geometry and (a.size == b.size or a.size == 1 or b.size == 1):
                left, right = np.broadcast_arrays(a, b)
                equal = bool(np.array_equal(left, right, equal_nan=True))
        elif isinstance(old, dict) and isinstance(new, dict):
            equal = old.keys() == new.keys() and all(
                np.array_equal(old[key], new[key]) for key in old
            )
        else:
            equal = old is None and new is None or isinstance(old, str) and old == new
        unchanged = unchanged and equal
        if equal:
            setattr(exported, field.name, deepcopy(old))
    if unchanged:
        return deepcopy(original)
    exported.__post_init__()
    return exported


def graph_order(graph: Any) -> tuple[list[str], list[tuple[str, str]]]:
    """Read a graph's declared scheduling order, refusing stale order metadata.

    Parameters
    ----------
    graph : nir.NIRGraph
        Graph read through the ordinary NIR reader or supplied by the caller.

    Returns
    -------
    tuple
        Node names and edges in their original order. Foreign graphs retain
        their current insertion order.

    Raises
    ------
    ValueError
        Recorded order is malformed or disagrees with the graph topology.
    """
    edges = [(src, dst) for src, dst in graph.edges]
    if GRAPH_ORDER_KEY not in graph.metadata:
        return list(graph.nodes), edges
    value = graph.metadata[GRAPH_ORDER_KEY]
    try:
        if not isinstance(value, str):
            raise ValueError("order must be JSON text")
        data = json.loads(value, object_pairs_hook=_unique_members)
        if not isinstance(data, dict) or set(data) != {"schema_version", "nodes", "edges"}:
            raise ValueError("order fields differ")
        if data["schema_version"] != _SCHEMA:
            raise ValueError("unsupported order schema")
        names = data["nodes"]
        recorded = data["edges"]
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise ValueError("node order must contain names")
        if Counter(names) != Counter(graph.nodes.keys()):
            raise ValueError("node order differs from topology")
        if not isinstance(recorded, list) or not all(
            isinstance(e, list) and len(e) == 2 and all(isinstance(n, str) for n in e)
            for e in recorded
        ):
            raise ValueError("edge order must contain name pairs")
        ordered = [(e[0], e[1]) for e in recorded]
        if Counter(ordered) != Counter(edges):
            raise ValueError("edge order differs from topology")
    except (ValueError, TypeError, RecursionError) as exc:
        raise ValueError(f"Invalid NIR graph order metadata: {exc}") from exc
    return names, ordered


def write_graph_order(
    metadata: dict[str, object], nodes: list[str], edges: list[tuple[str, str]]
) -> dict[str, object]:
    """Record ordering lost when HDF5 groups enumerate node names alphabetically.

    Parameters
    ----------
    metadata : dict
        Independent metadata created for the exported graph.
    nodes : list of str
        Executable node order, excluding internal recurrent delay nodes.
    edges : list of tuple
        Original edge order, including duplicate edges where present.

    Returns
    -------
    dict
        Metadata carrying a portable JSON order contract.
    """
    metadata[GRAPH_ORDER_KEY] = json.dumps(
        {"schema_version": _SCHEMA, "nodes": nodes, "edges": edges}, separators=(",", ":")
    )
    return metadata
