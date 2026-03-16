# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Connectivity generators returning CSR arrays

"""Connectivity generators returning CSR (indptr, indices, data) tuples."""

from __future__ import annotations

import numpy as np


def _to_csr(n_rows, n_cols, rows, cols, weights):
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


def random_connectivity(n_src, n_tgt, p, weight, seed=42):
    """Erdos-Renyi random connectivity."""
    rng = np.random.default_rng(seed)
    mask = rng.random((n_src, n_tgt)) < p
    rows, cols = np.nonzero(mask)
    weights = np.full(len(rows), weight, dtype=np.float64)
    return _to_csr(n_src, n_tgt, rows, cols, weights)


def small_world(n, k, p_rewire, weight, seed=42):
    """Watts-Strogatz small-world graph (n-by-n adjacency)."""
    rng = np.random.default_rng(seed)
    half_k = k // 2
    row_list, col_list = [], []
    for i in range(n):
        for j in range(1, half_k + 1):
            tgt = (i + j) % n
            if rng.random() < p_rewire:
                tgt = rng.integers(0, n)
                while tgt == i:
                    tgt = rng.integers(0, n)
            row_list.append(i)
            col_list.append(tgt)
            row_list.append(tgt)
            col_list.append(i)
    rows = np.array(row_list, dtype=np.int64)
    cols = np.array(col_list, dtype=np.int64)
    weights = np.full(len(rows), weight, dtype=np.float64)
    return _to_csr(n, n, rows, cols, weights)


def scale_free(n, m, weight, seed=42):
    """Barabasi-Albert preferential attachment (n-by-n adjacency)."""
    rng = np.random.default_rng(seed)
    degree = np.zeros(n, dtype=np.float64)
    row_list, col_list = [], []
    targets = list(range(m))
    for t in targets:
        degree[t] = 1.0
    for src in range(m, n):
        probs = degree[:src].copy()
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs[:] = 1.0 / src
        chosen = rng.choice(src, size=min(m, src), replace=False, p=probs)
        for tgt in chosen:
            row_list.append(src)
            col_list.append(tgt)
            row_list.append(tgt)
            col_list.append(src)
            degree[src] += 1
            degree[tgt] += 1
    rows = np.array(row_list, dtype=np.int64)
    cols = np.array(col_list, dtype=np.int64)
    weights = np.full(len(rows), weight, dtype=np.float64)
    return _to_csr(n, n, rows, cols, weights)


def ring_topology(n, k, weight):
    """Ring topology with k nearest neighbours in each direction."""
    row_list, col_list = [], []
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


def grid_topology(rows_count, cols_count, radius, weight):
    """2D lattice connectivity within Manhattan radius."""
    n = rows_count * cols_count
    row_list, col_list = [], []
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


def all_to_all(n_src, n_tgt, weight):
    """Full connectivity (every source to every target)."""
    rows = np.repeat(np.arange(n_src, dtype=np.int64), n_tgt)
    cols = np.tile(np.arange(n_tgt, dtype=np.int64), n_src)
    weights = np.full(len(rows), weight, dtype=np.float64)
    return _to_csr(n_src, n_tgt, rows, cols, weights)
