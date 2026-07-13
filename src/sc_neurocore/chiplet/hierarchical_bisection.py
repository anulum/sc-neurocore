# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hierarchical partitioner bisection

"""Coarsening, spectral scoring, and recursive graph bisection."""

from __future__ import annotations

from sc_neurocore.chiplet.hierarchical_graph import CorrelationAwareGraph


class BisectionMixin:
    """Private multilevel-bisection behaviour for the public partitioner."""

    coarsen_threshold: int
    correlation_penalty: float

    def _recursive_bisect(
        self,
        vertices: list[int],
        adjacency: dict[int, list[int]],
        graph: CorrelationAwareGraph,
        k: int,
    ) -> list[list[int]]:
        """Recursively bisect vertices until the requested count is reached."""
        if k <= 1 or len(vertices) <= 1:
            return [vertices]

        coarsened, mapping = self._coarsen(vertices, adjacency, graph)
        left, right = self._spectral_bisect(coarsened, adjacency, graph)
        left = self._uncoarsen(left, mapping)
        right = self._uncoarsen(right, mapping)

        if k == 2:
            return [left, right]

        left_count = k // 2
        right_count = k - left_count
        return self._recursive_bisect(
            left,
            adjacency,
            graph,
            left_count,
        ) + self._recursive_bisect(right, adjacency, graph, right_count)

    def _coarsen(
        self,
        vertices: list[int],
        adjacency: dict[int, list[int]],
        graph: CorrelationAwareGraph,
    ) -> tuple[list[int], dict[int, list[int]]]:
        """Merge disjoint low-correlation edges into deterministic supernodes."""
        del adjacency
        if len(vertices) <= self.coarsen_threshold:
            return vertices, {vertex: [vertex] for vertex in vertices}

        matched: set[int] = set()
        mapping: dict[int, list[int]] = {}
        coarsened: list[int] = []
        vertex_set = set(vertices)
        sorted_edges = sorted(
            (edge for edge in graph.edges if edge.u in vertex_set and edge.v in vertex_set),
            key=lambda edge: abs(edge.scc_weight),
        )

        for edge in sorted_edges:
            if edge.u not in matched and edge.v not in matched:
                mapping[edge.u] = [edge.u, edge.v]
                coarsened.append(edge.u)
                matched.add(edge.u)
                matched.add(edge.v)

        for vertex in vertices:
            if vertex not in matched:
                mapping[vertex] = [vertex]
                coarsened.append(vertex)
        return coarsened, mapping

    def _uncoarsen(
        self,
        partition: list[int],
        mapping: dict[int, list[int]],
    ) -> list[int]:
        """Expand a coarsened partition back to original vertex identifiers."""
        result: list[int] = []
        for vertex in partition:
            result.extend(mapping.get(vertex, [vertex]))
        return result

    def _spectral_bisect(
        self,
        vertices: list[int],
        adjacency: dict[int, list[int]],
        graph: CorrelationAwareGraph,
    ) -> tuple[list[int], list[int]]:
        """Bisect by degree score adjusted for boundary correlation."""
        if len(vertices) <= 1:
            return vertices, []

        vertex_set = set(vertices)
        graph._ensure_edge_cache()
        scores: dict[int, float] = {}
        for vertex in vertices:
            neighbours = [
                neighbour for neighbour in adjacency.get(vertex, []) if neighbour in vertex_set
            ]
            scc_sum = sum(
                abs(graph.edge_scc(vertex, neighbour)) * self.correlation_penalty
                for neighbour in neighbours
            )
            scores[vertex] = len(neighbours) - scc_sum

        ordered = sorted(vertices, key=lambda vertex: scores.get(vertex, 0.0))
        midpoint = len(ordered) // 2
        return ordered[:midpoint], ordered[midpoint:]


__all__ = ["BisectionMixin"]
