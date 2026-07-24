# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (backend_parity) from former test_topology_backends.py

from __future__ import annotations

from tests.topology_backends_support import *  # noqa: F403

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


def test_auto_matches_python_reference() -> None:
    graph = _GRAPHS["weighted9a"]
    for i, j in _node_pairs(graph.shape[0]):
        auto = ollivier_ricci_curvature(graph, i, j, backend="auto")
        ref = ollivier_ricci_curvature(graph, i, j, backend="python")
        np.testing.assert_allclose(auto, ref, atol=ATOL)


def test_self_pair_zero_across_requested_backend() -> None:
    # i == j short-circuits before dispatch, so every backend returns 0.0.
    for backend in ("auto", "python"):
        assert ollivier_ricci_curvature(_complete(4), 2, 2, backend=backend) == 0.0


def test_python_curvature_zero_for_disconnected_pair() -> None:
    # Nodes in different components have infinite graph distance, so the
    # curvature is defined as 0 rather than dividing by infinity.
    graph = np.zeros((3, 3), dtype=np.float64)
    graph[0, 1] = graph[1, 0] = 1.0  # node 2 is isolated
    assert ollivier_ricci_curvature(graph, 0, 2, backend="python") == 0.0
