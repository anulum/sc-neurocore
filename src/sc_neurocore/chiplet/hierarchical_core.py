# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hierarchical partitioner orchestration

"""Public hierarchical partitioner composed from focused algorithm modules."""

from __future__ import annotations

from typing import Any

import numpy as np

from sc_neurocore.chiplet import hierarchical_backends as backends
from sc_neurocore.chiplet.hierarchical_bisection import BisectionMixin
from sc_neurocore.chiplet.hierarchical_graph import (
    CorrelationAwareGraph,
    LFSRSeedAllocator,
)
from sc_neurocore.chiplet.hierarchical_refinement import RefinementMixin


class HierarchicalPartitioner(BisectionMixin, RefinementMixin):
    """Multi-level graph partitioner with selectable KL-refinement backend."""

    def __init__(
        self,
        num_partitions: int = 2,
        coarsen_threshold: int = 50,
        kl_iterations: int = 10,
        correlation_penalty: float = 2.0,
        seed: int = 42,
        refine_backend: str = "auto",
    ):
        """Configure deterministic bisection and refinement."""
        self.num_partitions = num_partitions
        self.coarsen_threshold = coarsen_threshold
        self.kl_iterations = kl_iterations
        self.correlation_penalty = correlation_penalty
        self.seed_allocator = LFSRSeedAllocator()
        self.rng = np.random.default_rng(seed)
        valid_backends = ("auto", "rust", "julia", "go", "mojo", "python")
        if refine_backend not in valid_backends:
            raise ValueError(
                f"refine_backend must be one of {valid_backends}, got {refine_backend!r}"
            )
        self.refine_backend = refine_backend

    def partition(
        self,
        graph: CorrelationAwareGraph,
    ) -> tuple[list[list[int]], list[int]]:
        """Partition the graph and return partitions with independent seeds."""
        vertices = list(range(graph.num_vertices))
        if self.num_partitions <= 1:
            return [vertices], self.seed_allocator.allocate(1)

        if graph.num_vertices <= self.num_partitions:
            partitions = [[vertex] for vertex in vertices]
            partitions.extend([] for _ in range(self.num_partitions - len(partitions)))
            return partitions, self.seed_allocator.allocate(len(partitions))

        adjacency = graph.adjacency()
        partitions = self._recursive_bisect(
            vertices,
            adjacency,
            graph,
            self.num_partitions,
        )
        partitions = self._dispatch_refine(partitions, adjacency, graph)
        return partitions, self.seed_allocator.allocate(len(partitions))

    def _dispatch_refine(
        self,
        partitions: list[list[int]],
        adjacency: dict[int, list[int]],
        graph: CorrelationAwareGraph,
    ) -> list[list[int]]:
        """Dispatch refinement while retaining the historical private method."""
        return backends.dispatch_refine(self, partitions, adjacency, graph)

    def _encode_csr(
        self,
        partitions: list[list[int]],
        adjacency: dict[int, list[int]],
        graph: CorrelationAwareGraph,
    ) -> backends.EncodedBuffers:
        """Encode the shared KL-refinement ABI buffers."""
        return backends.encode_csr(partitions, adjacency, graph)

    def _decode_part_map(
        self,
        part_map: np.ndarray[Any, Any],
        partition_count: int,
    ) -> list[list[int]]:
        """Decode a flat vertex-to-partition mapping."""
        return backends.decode_part_map(part_map, partition_count)

    def _refine_rust(
        self,
        partitions: list[list[int]],
        adjacency: dict[int, list[int]],
        graph: CorrelationAwareGraph,
    ) -> list[list[int]]:
        """Run the maintained Rust KL-refinement kernel."""
        return backends.refine_rust(self, partitions, adjacency, graph)

    def _refine_julia(
        self,
        partitions: list[list[int]],
        adjacency: dict[int, list[int]],
        graph: CorrelationAwareGraph,
    ) -> list[list[int]]:
        """Run the maintained Julia KL-refinement kernel."""
        return backends.refine_julia(self, partitions, adjacency, graph)

    def _refine_go(
        self,
        partitions: list[list[int]],
        adjacency: dict[int, list[int]],
        graph: CorrelationAwareGraph,
    ) -> list[list[int]]:
        """Run the maintained Go KL-refinement kernel."""
        return backends.refine_go(self, partitions, adjacency, graph)

    def _refine_mojo(
        self,
        partitions: list[list[int]],
        adjacency: dict[int, list[int]],
        graph: CorrelationAwareGraph,
    ) -> list[list[int]]:
        """Run the maintained Mojo KL-refinement kernel."""
        return backends.refine_mojo(self, partitions, adjacency, graph)


__all__ = ["HierarchicalPartitioner"]
