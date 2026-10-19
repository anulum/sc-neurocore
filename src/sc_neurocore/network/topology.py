# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Connectivity generators returning CSR arrays

"""Connectivity generators returning CSR (indptr, indices, data) tuples.

Builds adjacency matrices from a chosen topology family + parameters.
Six generators: ``random_connectivity`` (Erdős–Rényi),
``small_world`` (Watts–Strogatz), ``scale_free`` (Barabási–Albert),
``ring_topology``, ``grid_topology``, ``all_to_all``.

NOT to be confused with :mod:`sc_neurocore.topology`, which is a
different module that **measures** graph metrics (clustering,
modularity, small-world σ, hubs) on an existing adjacency matrix.
The two share the word "topology" but have disjoint roles:

- ``sc_neurocore.network.topology`` (this module) — produce graph
- ``sc_neurocore.topology`` — measure existing graph

Use this module to feed :class:`sc_neurocore.network.Projection`;
use the other to analyse the resulting connectivity.

See :doc:`docs/api/network` §6 for the generator catalogue and
:doc:`docs/api/graph_topology` for the analyser.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any
import numpy as np

CSR = tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]


def _to_csr(
    n_rows: int,
    n_cols: int,
    rows: np.ndarray[Any, Any],
    cols: np.ndarray[Any, Any],
    weights: np.ndarray[Any, Any],
) -> CSR:
    """Convert COO arrays to CSR (indptr, indices, data)."""
    if len(rows) == 0:
        return (
            np.zeros(n_rows + 1, dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
        )
    order = np.lexsort((cols, rows))
    rows, cols, weights = rows[order], cols[order], weights[order]
    indptr = np.zeros(n_rows + 1, dtype=np.int64)
    for r in rows:
        indptr[r + 1] += 1
    np.cumsum(indptr, out=indptr)
    return indptr, cols.astype(np.int64), weights.astype(np.float64)


def random_connectivity(n_src: int, n_tgt: int, p: float, weight: float, seed: int = 42) -> CSR:
    """Erdos-Renyi random connectivity."""
    rng = np.random.default_rng(seed)
    mask = rng.random((n_src, n_tgt)) < p
    rows, cols = np.nonzero(mask)
    weights = np.full(len(rows), weight, dtype=np.float64)
    return _to_csr(n_src, n_tgt, rows, cols, weights)


def small_world(n: int, k: int, p_rewire: float, weight: float, seed: int = 42) -> CSR:
    """Generate a Watts-Strogatz small-world graph as a symmetric n-by-n CSR matrix.

    A ring lattice joins every node to its ``k / 2`` nearest neighbours on each
    side; then, for each lattice offset and node in turn, the edge to the
    neighbour at that offset is moved, with probability ``p_rewire``, to a node
    chosen uniformly among those that are neither the node itself nor already
    connected to it (the simple-graph rule of the Watts-Strogatz model). Every
    edge is emitted in both directions, so the matrix holds exactly ``n * k``
    entries and no parallel or self connections.

    Parameters
    ----------
    n : int
        Number of nodes, at least 1.
    k : int
        Even lattice degree with ``0 <= k < n``.
    p_rewire : float
        Rewiring probability in ``[0, 1]``.
    weight : float
        Finite weight of every edge.
    seed : int, default=42
        Seed of the rewiring draws.

    Returns
    -------
    CSR
        ``(indptr, indices, data)`` of the symmetric adjacency matrix.

    Raises
    ------
    ValueError
        For a node count, degree, probability or weight outside the domain.
    """
    for name, value in (("n", n), ("k", k)):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer")
    n_int, k_int = int(n), int(k)
    if n_int < 1:
        raise ValueError("n must be at least 1 for small-world topology")
    if k_int < 0 or k_int % 2 or k_int >= n_int:
        raise ValueError("k must be an even degree with 0 <= k < n for small-world topology")
    if isinstance(p_rewire, bool) or not isinstance(p_rewire, Real) or not 0.0 <= p_rewire <= 1.0:
        raise ValueError("p_rewire must be a probability in [0, 1]")
    if isinstance(weight, bool) or not isinstance(weight, Real) or not np.isfinite(float(weight)):
        raise ValueError("weight must be finite")
    rng = np.random.default_rng(seed)
    neighbours: list[set[int]] = [set() for _ in range(n_int)]
    for i in range(n_int):
        for j in range(1, k_int // 2 + 1):
            other = (i + j) % n_int
            neighbours[i].add(other)
            neighbours[other].add(i)
    for j in range(1, k_int // 2 + 1):
        for i in range(n_int):
            other = (i + j) % n_int
            if other not in neighbours[i] or rng.random() >= p_rewire:
                continue
            candidates = [c for c in range(n_int) if c != i and c not in neighbours[i]]
            if not candidates:
                continue  # i is joined to every other node; nothing to rewire to
            target = candidates[int(rng.integers(0, len(candidates)))]
            neighbours[i].discard(other)
            neighbours[other].discard(i)
            neighbours[i].add(target)
            neighbours[target].add(i)
    rows = np.array([i for i in range(n_int) for _ in neighbours[i]], dtype=np.int64)
    cols = np.array([c for i in range(n_int) for c in sorted(neighbours[i])], dtype=np.int64)
    weights = np.full(len(rows), float(weight), dtype=np.float64)
    return _to_csr(n_int, n_int, rows, cols, weights)


def _validate_scale_free_parameters(n: int, m: int, weight: float) -> tuple[int, int, float]:
    """Validate Barabasi-Albert graph dimensions and edge weights."""
    if isinstance(n, bool) or not isinstance(n, Integral):
        raise ValueError("n must be an integer")
    if isinstance(m, bool) or not isinstance(m, Integral):
        raise ValueError("m must be an integer")

    n_int = int(n)
    m_int = int(m)
    if n_int < 2:
        raise ValueError("n must be at least 2 for scale-free topology")
    if m_int < 1:
        raise ValueError("m must be at least 1 for scale-free topology")
    if m_int >= n_int:
        raise ValueError("m must be smaller than n for scale-free topology")
    if isinstance(weight, bool) or not isinstance(weight, Real):
        raise ValueError("weight must be finite")

    weight_float = float(weight)
    if not np.isfinite(weight_float):
        raise ValueError("weight must be finite")
    return n_int, m_int, weight_float


def scale_free(n: int, m: int, weight: float, seed: int = 42) -> CSR:
    """Generate a Barabasi-Albert preferential-attachment graph.

    The seed graph is a star on ``m + 1`` nodes (node 0 joined to nodes
    ``1..m``), so every seed node has an edge and a degree. Each later node
    ``s`` then joins ``m`` distinct earlier nodes, drawn without replacement
    with probability proportional to their actual degree in the graph built so
    far (Barabasi and Albert, Science 286:509, 1999). The graph is simple and
    undirected: every edge is emitted in both directions, with no self or
    parallel connections, so each row sum is that node's degree.

    Parameters
    ----------
    n : int
        Number of source and target nodes. Must be at least two.
    m : int
        Number of existing nodes each new node joins. Must satisfy
        ``1 <= m < n``.
    weight : float
        Finite synaptic weight assigned to every emitted edge.
    seed : int, default=42
        Seed for deterministic preferential-attachment sampling.

    Returns
    -------
    tuple of ndarray
        CSR ``(indptr, indices, data)`` arrays for the symmetric adjacency.

    Raises
    ------
    ValueError
        If ``n``, ``m``, or ``weight`` falls outside the Barabasi-Albert domain.
    """
    n, m, weight = _validate_scale_free_parameters(n, m, weight)
    rng = np.random.default_rng(seed)
    degree = np.zeros(n, dtype=np.float64)
    row_list: list[int] = []
    col_list: list[int] = []

    def join(a: int, b: int) -> None:
        row_list.extend((a, b))
        col_list.extend((b, a))
        degree[a] += 1.0
        degree[b] += 1.0

    for leaf in range(1, m + 1):
        join(0, leaf)
    for src in range(m + 1, n):
        weights_so_far = degree[:src] / float(degree[:src].sum())
        for tgt in rng.choice(src, size=m, replace=False, p=weights_so_far):
            join(src, int(tgt))
    rows = np.array(row_list, dtype=np.int64)
    cols = np.array(col_list, dtype=np.int64)
    weights = np.full(len(rows), weight, dtype=np.float64)
    return _to_csr(n, n, rows, cols, weights)


def ring_topology(n: int, k: int, weight: float) -> CSR:
    """Build a simple ring: each node joins its ``k`` nearest neighbours on each side.

    The graph is simple and symmetric: ``2k`` distinct neighbours per node, no
    self connection and no doubled pair, which holds exactly when
    ``0 <= k <= (n - 1) // 2`` (at ``k = n / 2`` for even ``n`` the two sides
    meet at the antipode, and at ``k >= n`` they reach the node itself).

    Parameters
    ----------
    n : int
        Number of nodes, at least 1.
    k : int
        Neighbours on each side, ``0 <= k <= (n - 1) // 2``.
    weight : float
        Finite weight of every connection.

    Returns
    -------
    CSR
        ``(indptr, indices, data)`` with ``2 n k`` entries.

    Raises
    ------
    ValueError
        For a node count, neighbour count or weight outside the domain.
    """
    for name, value in (("n", n), ("k", k)):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer")
    n, k = int(n), int(k)
    if n < 1:
        raise ValueError("n must be at least 1 for ring topology")
    if not 0 <= k <= (n - 1) // 2:
        raise ValueError(
            f"k must satisfy 0 <= k <= (n - 1) // 2 = {(n - 1) // 2} for a simple ring of "
            f"{n} nodes; got {k}"
        )
    if isinstance(weight, bool) or not isinstance(weight, Real) or not np.isfinite(float(weight)):
        raise ValueError("weight must be finite")
    row_list: list[int] = []
    col_list: list[int] = []
    for i in range(n):
        for j in range(1, k + 1):
            row_list.append(i)
            col_list.append((i + j) % n)
            row_list.append(i)
            col_list.append((i - j) % n)
    rows = np.array(row_list, dtype=np.int64)
    cols = np.array(col_list, dtype=np.int64)
    weights = np.full(len(rows), float(weight), dtype=np.float64)
    return _to_csr(n, n, rows, cols, weights)


def grid_topology(rows_count: int, cols_count: int, radius: int, weight: float) -> CSR:
    """2-D lattice connectivity within a Chebyshev (square) radius.

    Each cell connects to every other cell whose row and column both differ by
    at most ``radius`` (an 8-neighbourhood at radius 1, 24 cells at radius 2), so
    the matrix is symmetric.

    Parameters
    ----------
    rows_count, cols_count : int
        Lattice shape.
    radius : int
        Chebyshev radius; 0 gives no connections.
    weight : float
        Weight of every connection.

    Returns
    -------
    CSR
        ``(indptr, indices, data)`` of the adjacency matrix.
    """
    n = rows_count * cols_count
    row_list: list[int] = []
    col_list: list[int] = []
    for r in range(rows_count):
        for c in range(cols_count):
            idx = r * cols_count + c
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows_count and 0 <= nc < cols_count:
                        row_list.append(idx)
                        col_list.append(nr * cols_count + nc)
    r_arr = np.array(row_list, dtype=np.int64)
    c_arr = np.array(col_list, dtype=np.int64)
    weights = np.full(len(r_arr), weight, dtype=np.float64)
    return _to_csr(n, n, r_arr, c_arr, weights)


def all_to_all(n_src: int, n_tgt: int, weight: float) -> CSR:
    """Full connectivity (every source to every target)."""
    rows = np.repeat(np.arange(n_src, dtype=np.int64), n_tgt)
    cols = np.tile(np.arange(n_tgt, dtype=np.int64), n_src)
    weights = np.full(len(rows), weight, dtype=np.float64)
    return _to_csr(n_src, n_tgt, rows, cols, weights)
