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
    """Watts-Strogatz small-world graph (n-by-n adjacency)."""
    rng = np.random.default_rng(seed)
    half_k = k // 2
    row_list: list[int] = []
    col_list: list[int] = []
    for i in range(n):
        for j in range(1, half_k + 1):
            tgt = (i + j) % n
            if rng.random() < p_rewire:
                tgt = int(rng.integers(0, n))
                while tgt == i:
                    tgt = int(rng.integers(0, n))
            row_list.append(i)
            col_list.append(tgt)
            row_list.append(tgt)
            col_list.append(i)
    rows = np.array(row_list, dtype=np.int64)
    cols = np.array(col_list, dtype=np.int64)
    weights = np.full(len(rows), weight, dtype=np.float64)
    return _to_csr(n, n, rows, cols, weights)


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

    Parameters
    ----------
    n : int
        Number of source and target nodes. Must be at least two.
    m : int
        Number of existing nodes sampled for each new node. Must satisfy
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
    targets = list(range(m))
    for t in targets:
        degree[t] = 1.0
    for src in range(m, n):
        probs = degree[:src].copy()
        total = float(probs.sum())
        probs /= total
        chosen = rng.choice(src, size=min(m, src), replace=False, p=probs)
        for tgt in chosen:
            row_list.append(src)
            col_list.append(int(tgt))
            row_list.append(int(tgt))
            col_list.append(src)
            degree[src] += 1
            degree[int(tgt)] += 1
    rows = np.array(row_list, dtype=np.int64)
    cols = np.array(col_list, dtype=np.int64)
    weights = np.full(len(rows), weight, dtype=np.float64)
    return _to_csr(n, n, rows, cols, weights)


def ring_topology(n: int, k: int, weight: float) -> CSR:
    """Ring topology with k nearest neighbours in each direction."""
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
    weights = np.full(len(rows), weight, dtype=np.float64)
    return _to_csr(n, n, rows, cols, weights)


def grid_topology(rows_count: int, cols_count: int, radius: int, weight: float) -> CSR:
    """2D lattice connectivity within Manhattan radius."""
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
