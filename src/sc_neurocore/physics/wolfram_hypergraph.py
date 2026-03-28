# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Simulates the Wolfram Physics Project Hypergraph

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class WolframHypergraph:
    """
    Simulates the Wolfram Physics Project Hypergraph.
    Universe is a set of relations (Hyperedges).
    """

    edges: List[Tuple[int, ...]]
    max_node_id: int

    def evolve(self, steps: int = 1) -> None:
        """
        Applies a rewrite rule.
        Rule: {{x, y}, {y, z}} -> {{x, z}, {x, w}, {y, w}}
        (Triangle completion with new node w)
        """
        for _ in range(steps):
            new_edges = []
            matched_indices = set()

            # Naive pattern matching O(E^2)
            # Find (x, y) and (y, z)
            for i, e1 in enumerate(self.edges):
                if i in matched_indices:
                    continue
                if len(e1) != 2:
                    continue

                x, y = e1

                for j, e2 in enumerate(self.edges):
                    if i == j or j in matched_indices:
                        continue
                    if len(e2) != 2:
                        continue

                    if e2[0] == y:  # Found chain x->y->z
                        z = e2[1]

                        # Apply Rule
                        w = self.max_node_id + 1
                        self.max_node_id += 1

                        # New edges: {x,z}, {x,w}, {y,w}
                        new_edges.append((x, z))
                        new_edges.append((x, w))
                        new_edges.append((y, w))

                        matched_indices.add(i)
                        matched_indices.add(j)
                        break

            # Keep unmatched edges
            for k, e in enumerate(self.edges):
                if k not in matched_indices:
                    new_edges.append(e)

            self.edges = new_edges  # type: ignore[assignment]

    def dimension_estimate(self) -> float:
        """Estimate effective dimension via BFS neighborhood growth.

        Builds an adjacency graph from hyperedges, picks a random node,
        measures how |B(r)| grows with r. Fits d from V(r) ~ r^d.
        Returns 0.0 if the graph is too small for meaningful estimation.
        """
        if len(self.edges) < 3:
            return 0.0

        adj: dict[int, set[int]] = {}
        for edge in self.edges:
            for node in edge:
                adj.setdefault(node, set())
            for i in range(len(edge)):
                for j in range(i + 1, len(edge)):
                    adj[edge[i]].add(edge[j])
                    adj[edge[j]].add(edge[i])

        nodes = list(adj.keys())
        if len(nodes) < 4:
            return 0.0

        start = nodes[len(nodes) // 2]
        visited = {start}
        frontier = {start}
        volumes = []

        for _ in range(min(10, len(nodes))):
            next_frontier: set[int] = set()
            for n in frontier:
                for nb in adj.get(n, set()):
                    if nb not in visited:
                        visited.add(nb)
                        next_frontier.add(nb)
            if not next_frontier:
                break
            frontier = next_frontier
            volumes.append(len(visited))

        if len(volumes) < 2:
            return 0.0

        import numpy as np

        r_vals = np.arange(1, len(volumes) + 1, dtype=np.float64)
        v_vals = np.array(volumes, dtype=np.float64)
        log_r = np.log(r_vals)
        log_v = np.log(np.clip(v_vals, 1, None))
        if log_r[-1] - log_r[0] < 1e-10:  # pragma: no cover
            return 0.0
        slope = (log_v[-1] - log_v[0]) / (log_r[-1] - log_r[0])
        return float(max(slope, 0.0))
