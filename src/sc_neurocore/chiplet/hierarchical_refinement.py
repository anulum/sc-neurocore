# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hierarchical partitioner refinement

"""Reference Kernighan-Lin refinement and incremental repartitioning."""

from __future__ import annotations

from sc_neurocore.chiplet.hierarchical_graph import CorrelationAwareGraph


class RefinementMixin:
    """Private reference-refinement behaviour for the public partitioner."""

    kl_iterations: int
    correlation_penalty: float

    def _refine(
        self,
        partitions: list[list[int]],
        adjacency: dict[int, list[int]],
        graph: CorrelationAwareGraph,
    ) -> list[list[int]]:
        """Refine partitions with deterministic correlation-aware local moves."""
        part_map = {
            vertex: partition_index
            for partition_index, partition in enumerate(partitions)
            for vertex in partition
        }
        graph._ensure_edge_cache()
        partition_count = len(partitions)

        for _ in range(self.kl_iterations):
            improved = False
            for source_index, partition in enumerate(partitions):
                for vertex in list(partition):
                    if len(partition) <= 1:
                        continue
                    costs = self._per_partition_cost(
                        vertex,
                        partition_count,
                        part_map,
                        adjacency,
                        graph,
                    )
                    target_index = source_index
                    best_gain = 0.0
                    for candidate_index in range(partition_count):
                        if candidate_index == source_index:
                            continue
                        gain = costs[source_index] - costs[candidate_index]
                        if gain > best_gain:
                            best_gain = gain
                            target_index = candidate_index
                    if target_index != source_index and best_gain > 0.0:
                        partition.remove(vertex)
                        partitions[target_index].append(vertex)
                        part_map[vertex] = target_index
                        improved = True
            if not improved:
                break
        return partitions

    def _per_partition_cost(
        self,
        vertex: int,
        partition_count: int,
        part_map: dict[int, int],
        adjacency: dict[int, list[int]],
        graph: CorrelationAwareGraph,
    ) -> list[float]:
        """Compute every candidate-partition cost in one neighbour scan."""
        vertex_weight = graph.vertex_weights.get(vertex, 1.0)
        weight_to = [0.0] * partition_count
        total_weight = 0.0
        for neighbour in adjacency.get(vertex, []):
            contribution = vertex_weight * (
                1.0 + abs(graph.edge_scc(vertex, neighbour)) * self.correlation_penalty
            )
            total_weight += contribution
            target = part_map.get(neighbour, -1)
            if 0 <= target < partition_count:
                weight_to[target] += contribution
        return [
            total_weight - weight_to[partition_index] for partition_index in range(partition_count)
        ]

    def _boundary_cost(
        self,
        vertex: int,
        partition_id: int,
        part_map: dict[int, int],
        adjacency: dict[int, list[int]],
        graph: CorrelationAwareGraph,
    ) -> float:
        """Return the legacy single-target placement cost."""
        vertex_weight = graph.vertex_weights.get(vertex, 1.0)
        return sum(
            vertex_weight
            * (1.0 + abs(graph.edge_scc(vertex, neighbour)) * self.correlation_penalty)
            for neighbour in adjacency.get(vertex, [])
            if part_map.get(neighbour, -1) != partition_id
        )

    def repartition_incremental(
        self,
        graph: CorrelationAwareGraph,
        partitions: list[list[int]],
        max_moves: int = 50,
    ) -> tuple[list[list[int]], int]:
        """Move the best boundary vertex repeatedly until no gain remains."""
        adjacency = graph.adjacency()
        part_map = {
            vertex: partition_index
            for partition_index, partition in enumerate(partitions)
            for vertex in partition
        }

        moves = 0
        for _ in range(max_moves):
            best_vertex = -1
            source_index = -1
            target_index = -1
            best_gain = 0.0

            for candidate_source, partition in enumerate(partitions):
                if len(partition) <= 1:
                    continue
                for vertex in partition:
                    current_cost = self._boundary_cost(
                        vertex,
                        candidate_source,
                        part_map,
                        adjacency,
                        graph,
                    )
                    if current_cost == 0.0:
                        continue
                    for candidate_target in range(len(partitions)):
                        if candidate_target == candidate_source:
                            continue
                        new_cost = self._boundary_cost(
                            vertex,
                            candidate_target,
                            part_map,
                            adjacency,
                            graph,
                        )
                        gain = current_cost - new_cost
                        if gain > best_gain:
                            best_gain = gain
                            best_vertex = vertex
                            source_index = candidate_source
                            target_index = candidate_target

            if best_vertex < 0:
                break
            partitions[source_index].remove(best_vertex)
            partitions[target_index].append(best_vertex)
            part_map[best_vertex] = target_index
            moves += 1
        return partitions, moves


__all__ = ["RefinementMixin"]
