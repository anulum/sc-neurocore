# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Polyglot parity tests for Ollivier-Ricci curvature

"""Cross-backend parity for ``ollivier_ricci_curvature``.

The Rust, Julia, Go, and Mojo accelerators must reproduce the pure-NumPy
reference value for every node pair on a representative spread of graph
topologies (complete, ring, star, path, weighted, and random sparse) to
within float64 round-off. Backends that are not built on the host are
skipped, never silently passed.
"""

from __future__ import annotations

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


# ───────────────────────── availability probes ─────────────────────────


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


# ───────────────────────── per-backend parity ─────────────────────────


@pytest.mark.skipif(not _rust_available(), reason="Rust topology backend unavailable")
@pytest.mark.parametrize("name,graph,i,j", _CASES, ids=_CASE_IDS)
def test_rust_parity(name: str, graph: np.ndarray, i: int, j: int) -> None:
    expected = _reference(graph, i, j)
    got = ollivier_ricci_curvature(graph, i, j, backend="rust")
    np.testing.assert_allclose(got, expected, atol=ATOL)


@pytest.mark.skipif(not _julia_available(), reason="Julia topology backend unavailable")
@pytest.mark.parametrize("name,graph,i,j", _CASES, ids=_CASE_IDS)
def test_julia_parity(name: str, graph: np.ndarray, i: int, j: int) -> None:
    expected = _reference(graph, i, j)
    got = ollivier_ricci_curvature(graph, i, j, backend="julia")
    np.testing.assert_allclose(got, expected, atol=ATOL)


@pytest.mark.skipif(not _go_available(), reason="Go topology backend unavailable")
@pytest.mark.parametrize("name,graph,i,j", _CASES, ids=_CASE_IDS)
def test_go_parity(name: str, graph: np.ndarray, i: int, j: int) -> None:
    expected = _reference(graph, i, j)
    got = ollivier_ricci_curvature(graph, i, j, backend="go")
    np.testing.assert_allclose(got, expected, atol=ATOL)


@pytest.mark.skipif(not _mojo_available(), reason="Mojo topology backend unavailable")
@pytest.mark.parametrize("name,graph,i,j", _CASES, ids=_CASE_IDS)
def test_mojo_parity(name: str, graph: np.ndarray, i: int, j: int) -> None:
    expected = _reference(graph, i, j)
    got = ollivier_ricci_curvature(graph, i, j, backend="mojo")
    np.testing.assert_allclose(got, expected, atol=ATOL)


# ───────────────────────── auto-dispatch + guards ─────────────────────────


def test_auto_matches_python_reference() -> None:
    graph = _GRAPHS["weighted9a"]
    for i, j in _node_pairs(graph.shape[0]):
        auto = ollivier_ricci_curvature(graph, i, j, backend="auto")
        ref = ollivier_ricci_curvature(graph, i, j, backend="python")
        np.testing.assert_allclose(auto, ref, atol=ATOL)


def test_invalid_backend_name_raises() -> None:
    with pytest.raises(ValueError, match="backend must be"):
        ollivier_ricci_curvature(_complete(3), 0, 1, backend="cuda")


def test_backend_param_preserves_index_validation() -> None:
    with np.testing.assert_raises_regex(ValueError, "integer"):
        ollivier_ricci_curvature(_complete(3), True, 1, backend="python")


@pytest.mark.skipif(_go_available(), reason="Go backend is built; cannot test the unavailable path")
def test_go_unavailable_raises_when_requested() -> None:
    with pytest.raises(RuntimeError, match="Go topology backend"):
        ollivier_ricci_curvature(_complete(3), 0, 1, backend="go")


@pytest.mark.skipif(
    _mojo_available(), reason="Mojo backend is built; cannot test the unavailable path"
)
def test_mojo_unavailable_raises_when_requested() -> None:
    with pytest.raises(RuntimeError, match="Mojo topology backend"):
        ollivier_ricci_curvature(_complete(3), 0, 1, backend="mojo")


def test_self_pair_zero_across_requested_backend() -> None:
    # i == j short-circuits before dispatch, so every backend returns 0.0.
    for backend in ("auto", "python"):
        assert ollivier_ricci_curvature(_complete(4), 2, 2, backend=backend) == 0.0
