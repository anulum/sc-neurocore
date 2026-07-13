# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hierarchical partition metrics

"""Partition-boundary, balance, and communication-volume metrics."""

from __future__ import annotations

import numpy as np

from sc_neurocore.chiplet.hierarchical_graph import CorrelationAwareGraph


def _build_part_map(partitions: list[list[int]]) -> dict[int, int]:
    """Build a vertex-to-partition lookup."""
    return {
        vertex: partition_index
        for partition_index, partition in enumerate(partitions)
        for vertex in partition
    }


def calculate_edge_cut(
    graph: CorrelationAwareGraph,
    partitions: list[list[int]],
) -> int:
    """Count edges whose endpoints belong to different partitions."""
    part_map = _build_part_map(partitions)
    return sum(part_map.get(edge.u, -1) != part_map.get(edge.v, -1) for edge in graph.edges)


def calculate_boundary_scc(
    graph: CorrelationAwareGraph,
    partitions: list[list[int]],
) -> float:
    """Return the maximum absolute SCC across boundary edges."""
    part_map = _build_part_map(partitions)
    return max(
        (
            abs(edge.scc_weight)
            for edge in graph.edges
            if part_map.get(edge.u, -1) != part_map.get(edge.v, -1)
        ),
        default=0.0,
    )


def calculate_mean_boundary_scc(
    graph: CorrelationAwareGraph,
    partitions: list[list[int]],
) -> float:
    """Return the mean absolute SCC across boundary edges."""
    part_map = _build_part_map(partitions)
    values = [
        abs(edge.scc_weight)
        for edge in graph.edges
        if part_map.get(edge.u, -1) != part_map.get(edge.v, -1)
    ]
    return float(np.mean(values)) if values else 0.0


def calculate_total_boundary_scc(
    graph: CorrelationAwareGraph,
    partitions: list[list[int]],
) -> float:
    """Return the total absolute SCC across boundary edges."""
    part_map = _build_part_map(partitions)
    return sum(
        (
            abs(edge.scc_weight)
            for edge in graph.edges
            if part_map.get(edge.u, -1) != part_map.get(edge.v, -1)
        ),
        start=0.0,
    )


def calculate_imbalance_ratio(partitions: list[list[int]]) -> float:
    """Return maximum size divided by ideal size, minus one."""
    sizes = [len(partition) for partition in partitions]
    if not sizes:
        return 0.0
    ideal_size = sum(sizes) / len(sizes)
    if ideal_size == 0.0:
        return 0.0
    return max(sizes) / ideal_size - 1.0


def calculate_comm_volume(
    graph: CorrelationAwareGraph,
    partitions: list[list[int]],
    bytes_per_spike: int = 8,
    bitstream_length: int = 256,
) -> dict[str, int]:
    """Estimate messages and bytes transferred across partition boundaries."""
    boundary_edges = calculate_edge_cut(graph, partitions)
    return {
        "boundary_edges": boundary_edges,
        "messages": boundary_edges,
        "volume_bytes": boundary_edges * bytes_per_spike * bitstream_length,
    }


__all__ = [
    "calculate_boundary_scc",
    "calculate_comm_volume",
    "calculate_edge_cut",
    "calculate_imbalance_ratio",
    "calculate_mean_boundary_scc",
    "calculate_total_boundary_scc",
]
