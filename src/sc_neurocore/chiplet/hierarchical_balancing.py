# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hierarchical partition balancing

"""Runtime load metrics, migration recommendations, and MPI rank mapping."""

from __future__ import annotations

from dataclasses import dataclass

from sc_neurocore.chiplet.hierarchical_boundary import GhostCellManager
from sc_neurocore.chiplet.hierarchical_graph import (
    CorrelationAwareGraph,
    HierarchyLevel,
)
from sc_neurocore.chiplet.hierarchical_metrics import (
    _build_part_map,
    calculate_imbalance_ratio,
)


@dataclass
class LoadMetrics:
    """Load and boundary metrics for one partition."""

    partition_id: int
    vertex_count: int
    weight_sum: float
    boundary_scc_sum: float
    ghost_count: int


@dataclass
class MigrationRecommendation:
    """A scored proposal to move one vertex between partitions."""

    vertex: int
    from_partition: int
    to_partition: int
    gain: float


class CorrelationLoadBalancer:
    """Recommend balancing moves while accounting for boundary correlation."""

    def __init__(
        self,
        imbalance_threshold: float = 0.2,
        scc_weight: float = 1.0,
    ):
        """Configure the imbalance trigger and SCC penalty."""
        self.imbalance_threshold = imbalance_threshold
        self.scc_weight = scc_weight
        self.history: list[list[MigrationRecommendation]] = []

    def compute_load_metrics(
        self,
        graph: CorrelationAwareGraph,
        partitions: list[list[int]],
    ) -> list[LoadMetrics]:
        """Compute vertex, weight, boundary-SCC, and ghost counts."""
        part_map = _build_part_map(partitions)
        halos = GhostCellManager.compute_halos(graph, partitions)
        metrics: list[LoadMetrics] = []
        for partition_index, partition in enumerate(partitions):
            weight_sum = sum(graph.vertex_weights.get(vertex, 1.0) for vertex in partition)
            boundary_scc_sum = 0.0
            for vertex in partition:
                for edge in graph.edges:
                    if edge.u == vertex or edge.v == vertex:
                        other = edge.v if edge.u == vertex else edge.u
                        if part_map.get(other, partition_index) != partition_index:
                            boundary_scc_sum += abs(edge.scc_weight)
            metrics.append(
                LoadMetrics(
                    partition_id=partition_index,
                    vertex_count=len(partition),
                    weight_sum=weight_sum,
                    boundary_scc_sum=boundary_scc_sum,
                    ghost_count=len(halos.get(partition_index, set())),
                )
            )
        return metrics

    def recommend_migrations(
        self,
        graph: CorrelationAwareGraph,
        partitions: list[list[int]],
        max_recommendations: int = 10,
    ) -> list[MigrationRecommendation]:
        """Return highest-gain moves from overloaded to underloaded partitions."""
        metrics = self.compute_load_metrics(graph, partitions)
        if calculate_imbalance_ratio(partitions) <= self.imbalance_threshold:
            return []

        sizes = [metric.vertex_count for metric in metrics]
        average_size = sum(sizes) / len(sizes) if sizes else 1.0
        overloaded = [
            metric
            for metric in metrics
            if metric.vertex_count > average_size * (1.0 + self.imbalance_threshold)
        ]
        underloaded = [
            metric
            for metric in metrics
            if metric.vertex_count < average_size * (1.0 - self.imbalance_threshold * 0.5)
        ]
        if not overloaded or not underloaded:
            return []

        adjacency = graph.adjacency()
        part_map = _build_part_map(partitions)
        recommendations: list[MigrationRecommendation] = []
        underloaded_ids = {metric.partition_id for metric in underloaded}

        for overloaded_metric in overloaded:
            for vertex in list(partitions[overloaded_metric.partition_id]):
                if len(recommendations) >= max_recommendations:
                    break
                boundary_neighbours = [
                    part_map[neighbour]
                    for neighbour in adjacency.get(vertex, [])
                    if part_map.get(neighbour, -1) != overloaded_metric.partition_id
                ]
                if not boundary_neighbours:
                    continue
                target = max(
                    set(boundary_neighbours),
                    key=boundary_neighbours.count,
                )
                if target in underloaded_ids:
                    scc_cost = sum(
                        abs(graph.edge_scc(vertex, neighbour))
                        for neighbour in adjacency.get(vertex, [])
                        if part_map.get(neighbour, -1) != overloaded_metric.partition_id
                    )
                    recommendations.append(
                        MigrationRecommendation(
                            vertex,
                            overloaded_metric.partition_id,
                            target,
                            1.0 - scc_cost * self.scc_weight,
                        )
                    )

        recommendations.sort(key=lambda recommendation: recommendation.gain, reverse=True)
        result = recommendations[:max_recommendations]
        self.history.append(result)
        return result


class RankMapper:
    """Map partitions to ranks and count inter-rank boundary edges."""

    def __init__(
        self,
        num_ranks: int,
        hierarchy: list[HierarchyLevel] | None = None,
    ):
        """Configure rank count and physical hierarchy."""
        self.num_ranks = num_ranks
        self.hierarchy = hierarchy or [HierarchyLevel.NODE]

    def assign(
        self,
        partitions: list[list[int]],
        graph: CorrelationAwareGraph | None = None,
    ) -> dict[int, int]:
        """Assign every partition to an MPI rank."""
        del graph
        if len(partitions) <= self.num_ranks:
            return {
                partition_index: partition_index % self.num_ranks
                for partition_index in range(len(partitions))
            }
        partitions_per_rank = max(1, len(partitions) // self.num_ranks)
        return {
            partition_index: min(
                partition_index // partitions_per_rank,
                self.num_ranks - 1,
            )
            for partition_index in range(len(partitions))
        }

    def cross_rank_edges(
        self,
        graph: CorrelationAwareGraph,
        partitions: list[list[int]],
    ) -> int:
        """Count partition-boundary edges that also cross rank boundaries."""
        part_map = _build_part_map(partitions)
        rank_map = self.assign(partitions, graph)
        return sum(
            part_map.get(edge.u, -1) != part_map.get(edge.v, -1)
            and rank_map.get(part_map.get(edge.u, -1), -1)
            != rank_map.get(part_map.get(edge.v, -1), -1)
            for edge in graph.edges
        )


__all__ = [
    "CorrelationLoadBalancer",
    "LoadMetrics",
    "MigrationRecommendation",
    "RankMapper",
]
