# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Network topology analysis

"""Graph-theoretic analysis of SNN connectivity.

Compute small-world coefficient, clustering, path length, modularity,
degree distribution, centrality from weight/adjacency matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TopologyReport:
    """Network topology analysis report."""

    n_nodes: int = 0
    n_edges: int = 0
    density: float = 0.0
    clustering_coefficient: float = 0.0
    avg_path_length: float = 0.0
    small_world_sigma: float = 0.0
    degree_mean: float = 0.0
    degree_std: float = 0.0
    degree_max: int = 0
    modularity: float = 0.0
    assortativity: float = 0.0
    hub_neurons: list[int] = field(default_factory=list)

    def summary(self) -> str:
        sw = "YES" if self.small_world_sigma > 1.0 else "NO"
        return (
            f"Topology: {self.n_nodes} nodes, {self.n_edges} edges, "
            f"density={self.density:.3f}\n"
            f"  Clustering: {self.clustering_coefficient:.3f}, "
            f"Path length: {self.avg_path_length:.2f}\n"
            f"  Small-world: {sw} (sigma={self.small_world_sigma:.2f})\n"
            f"  Degree: mean={self.degree_mean:.1f}, max={self.degree_max}\n"
            f"  Hubs: {self.hub_neurons[:5]}"
        )


class TopologyAnalyzer:
    """Analyse SNN connectivity structure.

    Parameters
    ----------
    adjacency : ndarray of shape (N, N)
        Binary adjacency matrix or weight matrix (nonzero = edge).
    directed : bool
        If True, treat as directed graph.
    n_path_samples : int
        Maximum number of source nodes used by ``_avg_path_length``.
        For ``N <= n_path_samples`` the result is the true all-pairs
        average; for larger graphs it is a sample mean over the first
        ``n_path_samples`` nodes (deterministic, not randomised).
        Default 100. Set to 0 or a very large value to force full
        all-pairs (expensive at N >> 100).
    """

    def __init__(
        self,
        adjacency: np.ndarray[Any, Any],
        directed: bool = False,
        n_path_samples: int = 100,
    ):
        self.adj = (np.abs(adjacency) > 1e-10).astype(np.float64)
        np.fill_diagonal(self.adj, 0)
        self.directed = directed
        self.N = self.adj.shape[0]
        self.n_path_samples = n_path_samples

    def analyze(self) -> TopologyReport:
        """Run full topology analysis."""
        report = TopologyReport()
        report.n_nodes = self.N
        report.n_edges = int(self.adj.sum()) // (1 if self.directed else 2)
        max_edges = self.N * (self.N - 1) // (1 if self.directed else 2)
        report.density = report.n_edges / max(max_edges, 1)

        report.clustering_coefficient = self._clustering()
        report.avg_path_length = self._avg_path_length()

        degrees = self.adj.sum(axis=1).astype(int)
        report.degree_mean = float(degrees.mean())
        report.degree_std = float(degrees.std())
        report.degree_max = int(degrees.max())

        # Hubs: top-5 by degree
        report.hub_neurons = list(np.argsort(-degrees)[:5])

        # Small-world sigma: C/C_rand / (L/L_rand)
        # For random graph: C_rand ~ k/N, L_rand ~ ln(N)/ln(k)
        k = max(report.degree_mean, 1)
        C_rand = k / max(self.N, 1)
        L_rand = np.log(max(self.N, 2)) / max(np.log(max(k, 1.1)), 0.1)
        if C_rand > 0 and report.avg_path_length > 0:
            C_ratio = report.clustering_coefficient / max(C_rand, 1e-10)
            L_ratio = report.avg_path_length / max(L_rand, 1e-10)
            report.small_world_sigma = C_ratio / max(L_ratio, 1e-10)
        else:
            report.small_world_sigma = 0.0

        report.assortativity = self._assortativity(degrees)
        report.modularity = self._modularity()

        return report

    def _modularity(self, communities: list[int] | None = None) -> float:
        """Newman's modularity Q for the (undirected projection) graph.

        Q = (1 / 2m) · Σ_ij [A_ij - (k_i · k_j) / 2m] · δ(c_i, c_j)

        See Newman M. E. J. "Modularity and community structure in
        networks." *PNAS* 103(23):8577-8582 (2006).

        When ``communities`` is ``None`` (default), the partition is
        the set of connected components (each component is its own
        community). For a single-component graph Q = 0; for a graph
        with separated cliques Q approaches its theoretical maximum
        of ~1 - 1/k for k components of equal size.

        Parameters
        ----------
        communities : list[int] or None
            Optional per-node community label. Length must equal
            ``self.N``. If None, connected components are used.

        Returns
        -------
        float
            Modularity Q in roughly [-0.5, 1.0]. Higher = stronger
            community structure.
        """
        A = self.adj if not self.directed else np.maximum(self.adj, self.adj.T)

        # Validate caller-supplied partition length BEFORE the empty-graph
        # short-circuit so misuse fails fast even on edgeless inputs.
        if communities is not None and len(communities) != self.N:
            raise ValueError(f"communities length {len(communities)} != N={self.N}")

        m2 = float(A.sum())  # 2m for undirected
        if m2 < 1.0:
            return 0.0

        if communities is None:
            communities = self._connected_components(A)

        degrees = A.sum(axis=1)
        comm = np.asarray(communities, dtype=np.int64)
        # Sum (A_ij - k_i k_j / 2m) over same-community pairs
        same_community = comm[:, None] == comm[None, :]
        expected = np.outer(degrees, degrees) / m2
        q = float((A - expected)[same_community].sum() / m2)
        return q

    @staticmethod
    def _connected_components(A: np.ndarray[Any, Any]) -> list[int]:
        """Label each node with its connected-component index (BFS)."""
        N = A.shape[0]
        labels = [-1] * N
        next_label = 0
        for src in range(N):
            if labels[src] != -1:
                continue
            queue = [src]
            labels[src] = next_label
            while queue:
                node = queue.pop(0)
                for nbr in np.where(A[node] > 0)[0]:
                    if labels[nbr] == -1:
                        labels[nbr] = next_label
                        queue.append(int(nbr))
            next_label += 1
        return labels

    def _clustering(self) -> float:
        """Local clustering coefficient averaged over nodes."""
        A = self.adj if not self.directed else np.maximum(self.adj, self.adj.T)
        coeffs = []
        for i in range(self.N):
            neighbors = np.where(A[i] > 0)[0]
            k = len(neighbors)
            if k < 2:
                continue
            subgraph = A[np.ix_(neighbors, neighbors)]
            triangles = subgraph.sum() / 2
            possible = k * (k - 1) / 2
            coeffs.append(triangles / possible)
        return float(np.mean(coeffs)) if coeffs else 0.0

    def _avg_path_length(self) -> float:
        """Average shortest path length via BFS.

        For ``N <= self.n_path_samples`` this is the true all-pairs
        average. For larger graphs it samples the first
        ``self.n_path_samples`` source nodes (deterministic, not
        randomised) — caller controls the trade-off via the
        constructor argument.
        """
        A = self.adj if not self.directed else np.maximum(self.adj, self.adj.T)
        cap = self.n_path_samples if self.n_path_samples > 0 else self.N
        total = 0.0
        count = 0
        for src in range(min(self.N, cap)):
            dist = self._bfs(A, src)
            reachable = dist[dist > 0]
            if len(reachable) > 0:
                total += reachable.sum()
                count += len(reachable)
        return total / max(count, 1)

    @staticmethod
    def _bfs(A: np.ndarray[Any, Any], src: int) -> np.ndarray[Any, Any]:
        N = A.shape[0]
        dist = np.full(N, -1)
        dist[src] = 0
        queue = [src]
        while queue:
            node = queue.pop(0)
            for nbr in np.where(A[node] > 0)[0]:
                if dist[nbr] == -1:
                    dist[nbr] = dist[node] + 1
                    queue.append(nbr)
        dist[dist == -1] = 0
        return dist

    def _assortativity(self, degrees: np.ndarray[Any, Any]) -> float:
        """Degree assortativity coefficient."""
        edges = np.argwhere(self.adj > 0)
        if len(edges) < 2:
            return 0.0
        d_src = degrees[edges[:, 0]].astype(np.float64)
        d_tgt = degrees[edges[:, 1]].astype(np.float64)
        if d_src.std() < 1e-10 or d_tgt.std() < 1e-10:
            return 0.0
        return float(np.corrcoef(d_src, d_tgt)[0, 1])
