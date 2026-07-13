# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hierarchical partition boundaries

"""Ghost-cell construction and SCC-budgeted boundary synchronisation."""

from __future__ import annotations

from dataclasses import dataclass

from sc_neurocore.chiplet.hierarchical_graph import CorrelationAwareGraph
from sc_neurocore.chiplet.hierarchical_metrics import _build_part_map


class GhostCellManager:
    """Compute read-only halo vertices required by each partition."""

    @staticmethod
    def compute_halos(
        graph: CorrelationAwareGraph,
        partitions: list[list[int]],
    ) -> dict[int, set[int]]:
        """Return the external neighbour vertices required by each partition."""
        part_map = _build_part_map(partitions)
        adjacency = graph.adjacency()
        halos: dict[int, set[int]] = {index: set() for index in range(len(partitions))}
        for partition_index, partition in enumerate(partitions):
            for vertex in partition:
                for neighbour in adjacency.get(vertex, []):
                    if part_map.get(neighbour, partition_index) != partition_index:
                        halos[partition_index].add(neighbour)
        return halos

    @staticmethod
    def halo_sizes(
        graph: CorrelationAwareGraph,
        partitions: list[list[int]],
    ) -> dict[int, int]:
        """Return each partition's ghost-cell count."""
        return {
            partition_index: len(ghosts)
            for partition_index, ghosts in GhostCellManager.compute_halos(
                graph,
                partitions,
            ).items()
        }


@dataclass
class BoundarySyncConfig:
    """Configuration for decorrelated boundary synchronisation."""

    decorrelation_buffer_bits: int = 32
    sync_interval_timesteps: int = 1
    max_boundary_scc_budget: float = 0.1


class BoundarySyncProtocol:
    """Manage decorrelation seeds and SCC-budget violations at boundaries."""

    def __init__(self, config: BoundarySyncConfig | None = None):
        """Initialise empty buffers and violation state."""
        self.config = config or BoundarySyncConfig()
        self.boundary_buffers: dict[tuple[int, int], int] = {}
        self.violations: list[tuple[int, int, float]] = []

    def init_buffers(
        self,
        graph: CorrelationAwareGraph,
        partitions: list[list[int]],
        seeds: list[int],
    ) -> int:
        """Initialise a non-zero XOR-derived seed for every boundary edge."""
        part_map = _build_part_map(partitions)
        count = 0
        for edge in graph.edges:
            source = part_map.get(edge.u, -1)
            target = part_map.get(edge.v, -1)
            if source != target and source >= 0 and target >= 0:
                seed = (seeds[source] ^ seeds[target]) & 0xFFFF
                self.boundary_buffers[(edge.u, edge.v)] = seed or 1
                count += 1
        return count

    def check_scc_budget(
        self,
        graph: CorrelationAwareGraph,
        partitions: list[list[int]],
    ) -> list[tuple[int, int, float]]:
        """Return boundary edges whose absolute SCC exceeds the budget."""
        part_map = _build_part_map(partitions)
        budget = self.config.max_boundary_scc_budget
        self.violations = [
            (edge.u, edge.v, edge.scc_weight)
            for edge in graph.edges
            if part_map.get(edge.u, -1) != part_map.get(edge.v, -1)
            and abs(edge.scc_weight) > budget
        ]
        return self.violations

    @property
    def num_buffers(self) -> int:
        """Return the number of initialised decorrelation buffers."""
        return len(self.boundary_buffers)


__all__ = ["BoundarySyncConfig", "BoundarySyncProtocol", "GhostCellManager"]
