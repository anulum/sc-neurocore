# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hierarchical partitioner graph models

"""Graph representations and deterministic seed allocation for partitioning."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class HierarchyLevel(Enum):
    """Physical hierarchy levels available to the MPI rank mapper."""

    RACK = "rack"
    NODE = "node"
    DIE = "die"
    TILE = "tile"


@dataclass
class CorrelationEdge:
    """An undirected edge with connection and SC-correlation weights."""

    u: int
    v: int
    conn_weight: float = 1.0
    scc_weight: float = 0.0


@dataclass
class CSRGraph:
    """Compressed sparse-row graph with constant-time adjacency slices."""

    num_vertices: int
    indptr: np.ndarray[Any, Any]
    indices: np.ndarray[Any, Any]
    conn_weights: np.ndarray[Any, Any]
    scc_weights: np.ndarray[Any, Any]
    vertex_weights: np.ndarray[Any, Any]

    @classmethod
    def from_edge_list(
        cls,
        num_vertices: int,
        edges: list[CorrelationEdge],
        vertex_weights: dict[int, float] | None = None,
    ) -> CSRGraph:
        """Build a symmetric CSR graph from an undirected edge list."""
        adjacency: dict[int, list[tuple[int, float, float]]] = {
            vertex: [] for vertex in range(num_vertices)
        }
        for edge in edges:
            adjacency[edge.u].append((edge.v, edge.conn_weight, edge.scc_weight))
            adjacency[edge.v].append((edge.u, edge.conn_weight, edge.scc_weight))

        indptr = np.zeros(num_vertices + 1, dtype=np.int64)
        all_indices: list[int] = []
        all_conn: list[float] = []
        all_scc: list[float] = []
        for vertex in range(num_vertices):
            neighbours = sorted(adjacency[vertex], key=lambda item: item[0])
            indptr[vertex + 1] = indptr[vertex] + len(neighbours)
            for neighbour, conn_weight, scc_weight in neighbours:
                all_indices.append(neighbour)
                all_conn.append(conn_weight)
                all_scc.append(scc_weight)

        weights = np.ones(num_vertices, dtype=np.float64)
        if vertex_weights:
            for vertex, weight in vertex_weights.items():
                weights[vertex] = weight

        return cls(
            num_vertices=num_vertices,
            indptr=indptr,
            indices=np.asarray(all_indices, dtype=np.int64),
            conn_weights=np.asarray(all_conn, dtype=np.float64),
            scc_weights=np.asarray(all_scc, dtype=np.float64),
            vertex_weights=weights,
        )

    def neighbors(self, vertex: int) -> np.ndarray[Any, Any]:
        """Return the adjacent-vertex slice for the vertex."""
        return self.indices[self.indptr[vertex] : self.indptr[vertex + 1]]

    def degree(self, vertex: int) -> int:
        """Return the number of neighbours of the vertex."""
        return int(self.indptr[vertex + 1] - self.indptr[vertex])

    def edge_conn(self, vertex: int) -> np.ndarray[Any, Any]:
        """Return connection weights aligned with the neighbour slice."""
        return self.conn_weights[self.indptr[vertex] : self.indptr[vertex + 1]]

    def edge_scc(self, vertex: int) -> np.ndarray[Any, Any]:
        """Return SCC weights aligned with the neighbour slice."""
        return self.scc_weights[self.indptr[vertex] : self.indptr[vertex + 1]]

    @property
    def num_edges(self) -> int:
        """Return the undirected edge count."""
        return len(self.indices) // 2


@dataclass
class CorrelationAwareGraph:
    """Adjacency graph with cached constant-time correlation-edge lookups."""

    num_vertices: int
    edges: list[CorrelationEdge] = field(default_factory=list)
    vertex_weights: dict[int, float] = field(default_factory=dict)
    _edge_cache: dict[tuple[int, int], CorrelationEdge] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def _ensure_edge_cache(self) -> dict[tuple[int, int], CorrelationEdge]:
        """Build or refresh the lookup cache after an edge-list mutation."""
        cache = self._edge_cache
        if cache is None or len(cache) != len(self.edges):
            cache = {(min(edge.u, edge.v), max(edge.u, edge.v)): edge for edge in self.edges}
            self._edge_cache = cache
        return cache

    def adjacency(self) -> dict[int, list[int]]:
        """Return a symmetric adjacency mapping."""
        adjacency: dict[int, list[int]] = {vertex: [] for vertex in range(self.num_vertices)}
        for edge in self.edges:
            adjacency[edge.u].append(edge.v)
            adjacency[edge.v].append(edge.u)
        return adjacency

    def edge_weight(self, u: int, v: int) -> float:
        """Return the connection weight for an edge, or zero if absent."""
        edge = self._ensure_edge_cache().get((min(u, v), max(u, v)))
        return edge.conn_weight if edge is not None else 0.0

    def edge_scc(self, u: int, v: int) -> float:
        """Return the SCC weight for an edge, or zero if absent."""
        edge = self._ensure_edge_cache().get((min(u, v), max(u, v)))
        return edge.scc_weight if edge is not None else 0.0

    @property
    def num_edges(self) -> int:
        """Return the undirected edge count."""
        return len(self.edges)

    def to_csr(self) -> CSRGraph:
        """Convert this graph to its symmetric CSR representation."""
        return CSRGraph.from_edge_list(
            self.num_vertices,
            self.edges,
            self.vertex_weights or None,
        )


class LFSRSeedAllocator:
    """Allocate deterministic, separated non-zero 16-bit LFSR seeds."""

    def __init__(self, base_seed: int = 0xACE1):
        """Initialise the allocator with a 16-bit base seed."""
        self.base_seed = base_seed

    def allocate(self, num_partitions: int) -> list[int]:
        """Return one deterministic seed for every requested partition."""
        spacing = max(1, 65535 // (num_partitions + 1))
        seeds: list[int] = []
        for index in range(num_partitions):
            seed = (self.base_seed + (index + 1) * spacing) & 0xFFFF
            seeds.append(seed or 1)
        return seeds

    def verify_uniqueness(self, seeds: list[int]) -> bool:
        """Return whether all supplied seeds are unique."""
        return len(seeds) == len(set(seeds))


__all__ = [
    "CSRGraph",
    "CorrelationAwareGraph",
    "CorrelationEdge",
    "HierarchyLevel",
    "LFSRSeedAllocator",
]
