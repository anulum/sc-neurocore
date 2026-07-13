# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hierarchical partitioner test support

"""Deterministic graph builders shared by focused partitioner tests."""

from __future__ import annotations

import numpy as np

from sc_neurocore.chiplet import CorrelationAwareGraph, CorrelationEdge


def build_graph(
    vertex_count: int,
    avg_degree: int = 8,
    seed: int = 42,
) -> CorrelationAwareGraph:
    """Build a deterministic sparse correlation-aware graph."""
    rng = np.random.default_rng(seed)
    edges: list[CorrelationEdge] = []
    seen: set[tuple[int, int]] = set()
    for vertex in range(vertex_count):
        sample_size = min(avg_degree, vertex_count - 1)
        for raw_neighbour in rng.choice(vertex_count, size=sample_size, replace=False):
            neighbour = int(raw_neighbour)
            key = (min(neighbour, vertex), max(neighbour, vertex))
            if neighbour == vertex or key in seen:
                continue
            seen.add(key)
            edges.append(
                CorrelationEdge(
                    u=neighbour,
                    v=vertex,
                    conn_weight=1.0,
                    scc_weight=0.1,
                )
            )
    return CorrelationAwareGraph(num_vertices=vertex_count, edges=edges)


def make_chain_graph(
    vertex_count: int,
    scc: float = 0.0,
) -> CorrelationAwareGraph:
    """Build a path graph over consecutive vertex identifiers."""
    edges = [CorrelationEdge(vertex, vertex + 1, 1.0, scc) for vertex in range(vertex_count - 1)]
    return CorrelationAwareGraph(num_vertices=vertex_count, edges=edges)


def make_biclique(
    left_count: int,
    right_count: int,
    scc: float = 0.0,
) -> CorrelationAwareGraph:
    """Build a complete bipartite graph."""
    edges = [
        CorrelationEdge(left, right, 1.0, scc)
        for left in range(left_count)
        for right in range(left_count, left_count + right_count)
    ]
    return CorrelationAwareGraph(
        num_vertices=left_count + right_count,
        edges=edges,
    )
