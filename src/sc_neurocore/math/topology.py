# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Topological observables for SCPN phase dynamics

"""Topological and geometric observables for coupled oscillator networks.

These are the quick-win implementations from the Holonomic Atlas
mathematical foundations audit (Round 2).

    from sc_neurocore.math.topology import (
        winding_number,
        ollivier_ricci_curvature,
        sheaf_consistency_defect,
    )
"""

from __future__ import annotations

import numpy as np


def _validate_coupling_graph(knm: np.ndarray) -> np.ndarray:
    graph = np.asarray(knm, dtype=np.float64)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("knm must be a square coupling matrix")
    if graph.shape[0] == 0:
        raise ValueError("knm must contain at least one node")
    if not np.all(np.isfinite(graph)):
        raise ValueError("knm must contain only finite values")
    if np.any(graph < 0.0):
        raise ValueError("knm must be non-negative for Ollivier-Ricci curvature")
    return graph.copy()


def _validate_node_index(name: str, index: int, n_nodes: int) -> int:
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise ValueError(f"{name} must be an integer node index")
    index = int(index)
    if index < 0 or index >= n_nodes:
        raise ValueError(f"{name} out of range for coupling graph")
    return index


def _shortest_path_distances(graph: np.ndarray) -> np.ndarray:
    adjacency = graph > 0.0
    np.fill_diagonal(adjacency, False)
    n_nodes = graph.shape[0]
    distances = np.full((n_nodes, n_nodes), np.inf, dtype=np.float64)
    for source in range(n_nodes):
        distances[source, source] = 0.0
        frontier = [source]
        while frontier:
            current = frontier.pop(0)
            next_distance = distances[source, current] + 1.0
            for target in np.flatnonzero(adjacency[current]):
                if next_distance < distances[source, target]:
                    distances[source, target] = next_distance
                    frontier.append(int(target))
    return distances


def _lazy_random_walk(graph: np.ndarray, node: int, *, idleness: float = 0.5) -> np.ndarray:
    distribution = np.zeros(graph.shape[0], dtype=np.float64)
    distribution[node] = idleness
    row = graph[node].copy()
    row[node] = 0.0
    row_sum = float(row.sum())
    if row_sum == 0.0:
        distribution[node] = 1.0
        return distribution
    distribution += (1.0 - idleness) * row / row_sum
    return distribution


def _minimum_transport_cost(source: np.ndarray, target: np.ndarray, distances: np.ndarray) -> float:
    source_nodes = np.flatnonzero(source > 0.0)
    target_nodes = np.flatnonzero(target > 0.0)
    if source_nodes.size == 0 or target_nodes.size == 0:
        return 0.0

    supply = source[source_nodes].astype(np.float64)
    demand = target[target_nodes].astype(np.float64)
    costs = distances[np.ix_(source_nodes, target_nodes)]
    if not np.all(np.isfinite(costs)):
        return float("inf")

    total_supply = supply.size
    total_demand = demand.size
    source_id = total_supply + total_demand
    sink_id = source_id + 1
    node_count = sink_id + 1
    residual = [[0.0 for _ in range(node_count)] for _ in range(node_count)]
    edge_cost = [[0.0 for _ in range(node_count)] for _ in range(node_count)]

    for idx, amount in enumerate(supply):
        residual[source_id][idx] = float(amount)
    for idx, amount in enumerate(demand):
        residual[total_supply + idx][sink_id] = float(amount)
    for s_idx in range(total_supply):
        for d_idx in range(total_demand):
            u = s_idx
            v = total_supply + d_idx
            residual[u][v] = float("inf")
            edge_cost[u][v] = float(costs[s_idx, d_idx])
            edge_cost[v][u] = -float(costs[s_idx, d_idx])

    required = float(source.sum())
    transported = 0.0
    total_cost = 0.0
    tolerance = 1e-12
    while transported + tolerance < required:
        dist = [float("inf")] * node_count
        parent = [-1] * node_count
        dist[source_id] = 0.0
        for _ in range(node_count - 1):
            updated = False
            for u in range(node_count):
                if not np.isfinite(dist[u]):
                    continue
                for v in range(node_count):
                    if residual[u][v] <= tolerance:
                        continue
                    candidate = dist[u] + edge_cost[u][v]
                    if candidate < dist[v] - tolerance:
                        dist[v] = candidate
                        parent[v] = u
                        updated = True
            if not updated:
                break
        if parent[sink_id] == -1:
            raise ValueError("transport problem is infeasible")

        increment = required - transported
        v = sink_id
        while v != source_id:
            u = parent[v]
            increment = min(increment, residual[u][v])
            v = u
        v = sink_id
        while v != source_id:
            u = parent[v]
            residual[u][v] -= increment
            residual[v][u] += increment
            total_cost += increment * edge_cost[u][v]
            v = u
        transported += increment
    return float(total_cost)


def winding_number(phases: np.ndarray) -> int:
    """Compute the winding number of a phase trajectory around S^1.

    The winding number counts how many times the phase wraps around
    the circle [0, 2*pi). It is a topological invariant — continuous
    deformations of the trajectory cannot change it.

    Parameters
    ----------
    phases : np.ndarray, shape (T,)
        Time series of phase values (radians).

    Returns
    -------
    int
        Number of complete windings (positive = counterclockwise).
    """
    diffs = np.diff(phases)
    # Unwrap: large jumps indicate wrapping
    diffs = np.where(diffs > np.pi, diffs - 2 * np.pi, diffs)
    diffs = np.where(diffs < -np.pi, diffs + 2 * np.pi, diffs)
    return int(np.round(np.sum(diffs) / (2 * np.pi)))


def ollivier_ricci_curvature(knm: np.ndarray, i: int, j: int) -> float:
    """Compute Ollivier-Ricci curvature between nodes i and j on the coupling graph.

    Ollivier (2009), "Ricci curvature of Markov chains on metric spaces."
    The curvature kappa(i,j) measures how much the neighborhoods of i and j
    overlap. Positive curvature = neighborhoods converge (community structure).
    Negative curvature = neighborhoods diverge (bottleneck).

    kappa(i,j) = 1 - W1(mu_i, mu_j) / d(i,j)
    where mu_i is the lazy random walk distribution from node i,
    and W1 is the Wasserstein-1 distance on the unweighted support graph.

    Parameters
    ----------
    knm : np.ndarray, shape (N, N)
        Coupling matrix (non-negative, not necessarily symmetric).
    i, j : int
        Node indices.

    Returns
    -------
    float
        Ollivier-Ricci curvature. Returns 0.0 for self or disconnected pairs.
    """
    graph = _validate_coupling_graph(knm)
    n_nodes = graph.shape[0]
    i = _validate_node_index("i", i, n_nodes)
    j = _validate_node_index("j", j, n_nodes)
    if i == j:
        return 0.0

    distances = _shortest_path_distances(graph)
    graph_distance = distances[i, j]
    if not np.isfinite(graph_distance) or graph_distance <= 0.0:
        return 0.0
    mu_i = _lazy_random_walk(graph, i)
    mu_j = _lazy_random_walk(graph, j)
    w1 = _minimum_transport_cost(mu_i, mu_j, distances)
    return float(1.0 - w1 / graph_distance)


def sheaf_consistency_defect(phases: np.ndarray, knm: np.ndarray) -> float:
    """Compute the sheaf consistency defect for the SCPN phase state.

    In sheaf theory, a global section exists iff the gluing conditions
    are satisfied on all overlaps. For the SCPN, the coupling matrix
    defines the overlaps, and the phase differences weighted by coupling
    measure the failure to glue.

    defect = (1/N^2) * sum_{i,j} |K_ij| * |1 - cos(theta_i - theta_j)|

    When phases are synchronized (all equal), defect = 0.
    When phases are maximally incoherent, defect approaches max(|K|).

    This is equivalent to (1 - Kuramoto_R) weighted by coupling.

    Parameters
    ----------
    phases : np.ndarray, shape (N,)
        Phase values (radians) for each layer/oscillator.
    knm : np.ndarray, shape (N, N)
        Coupling matrix.

    Returns
    -------
    float
        Sheaf consistency defect >= 0. Zero means globally coherent.
    """
    N = len(phases)
    diffs = phases[np.newaxis, :] - phases[:, np.newaxis]
    cost = np.abs(knm) * (1.0 - np.cos(diffs))
    return float(cost.sum() / (N * N))


def connection_curvature(phases: np.ndarray, knm: np.ndarray) -> np.ndarray:
    """Compute the connection curvature from PGBO phase dynamics.

    The PGBO covariant derivative u_mu = dphi_mu - alpha * A_mu
    defines a U(1) connection. The curvature F_{ij} = K_{ij} * cos(theta_i - theta_j)
    measures the obstruction to parallel transport between layers i and j.

    Parameters
    ----------
    phases : np.ndarray, shape (N,)
        Phase values.
    knm : np.ndarray, shape (N, N)
        Coupling matrix.

    Returns
    -------
    np.ndarray, shape (N, N)
        Connection curvature matrix. Diagonal is zero.
    """
    diffs = phases[np.newaxis, :] - phases[:, np.newaxis]
    return knm * np.cos(diffs)
