# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_topology_backends.py

from __future__ import annotations

"""Cross-backend parity for ``ollivier_ricci_curvature``.

The Rust, Julia, Go, and Mojo accelerators must reproduce the pure-NumPy
reference value for every node pair on a representative spread of graph
topologies (complete, ring, star, path, weighted, and random sparse) to
within float64 round-off. Backends that are not built on the host are
skipped, never silently passed."""

import numpy as np


import pytest


from sc_neurocore.math import topology


from sc_neurocore.math.topology import ollivier_ricci_curvature


ATOL = 1e-9


def _complete(n: int) -> np.ndarray:
    g = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(g, 0.0)
    return g


def _ring(n: int) -> np.ndarray:
    g = np.zeros((n, n), dtype=np.float64)
    for k in range(n):
        g[k, (k + 1) % n] = 1.0
        g[(k + 1) % n, k] = 1.0
    return g


def _star(n: int) -> np.ndarray:
    g = np.zeros((n, n), dtype=np.float64)
    for k in range(1, n):
        g[0, k] = 1.0
        g[k, 0] = 1.0
    return g


def _path(n: int) -> np.ndarray:
    g = np.zeros((n, n), dtype=np.float64)
    for k in range(n - 1):
        g[k, k + 1] = 1.0
        g[k + 1, k] = 1.0
    return g


def _weighted_random(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    g = rng.random((n, n))
    g[g < 0.6] = 0.0
    g = 0.5 * (g + g.T)  # symmetric weighted graph
    np.fill_diagonal(g, 0.0)
    return g


_GRAPHS = {
    "complete7": _complete(7),
    "ring8": _ring(8),
    "star6": _star(6),
    "path6": _path(6),
    "weighted9a": _weighted_random(9, 1),
    "weighted9b": _weighted_random(9, 2),
}


def _node_pairs(n: int) -> list[tuple[int, int]]:
    return [(0, 1), (0, n - 1), (1, n - 2), (2, 3)]


def _reference(graph: np.ndarray, i: int, j: int) -> float:
    return ollivier_ricci_curvature(graph, i, j, backend="python")


def _rust_available() -> bool:
    return topology._HAS_RUST_TOPOLOGY


def _julia_available() -> bool:
    return topology._ensure_julia_loaded()


def _go_available() -> bool:
    return topology._ensure_go_loaded()


def _mojo_available() -> bool:
    return topology._ensure_mojo_loaded()


def _parametrised_cases() -> list[tuple[str, np.ndarray, int, int]]:
    cases: list[tuple[str, np.ndarray, int, int]] = []
    for name, graph in _GRAPHS.items():
        for i, j in _node_pairs(graph.shape[0]):
            cases.append((name, graph, i, j))
    return cases


_CASES = _parametrised_cases()


_CASE_IDS = [f"{name}-{i}-{j}" for name, _g, i, j in _CASES]


__all__ = [
    "np",
    "pytest",
    "topology",
    "ollivier_ricci_curvature",
    "ATOL",
    "_complete",
    "_ring",
    "_star",
    "_path",
    "_weighted_random",
    "_GRAPHS",
    "_node_pairs",
    "_reference",
    "_rust_available",
    "_julia_available",
    "_go_available",
    "_mojo_available",
    "_parametrised_cases",
    "_CASES",
    "_CASE_IDS",
]
