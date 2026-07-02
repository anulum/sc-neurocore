# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for network topology generators

"""Tests for all 6 connectivity generators returning CSR arrays."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from sc_neurocore.network.topology import (
    random_connectivity,
    small_world,
    scale_free,
    ring_topology,
    grid_topology,
    all_to_all,
)


def _csr_to_dense(
    indptr: np.ndarray[Any, Any],
    indices: np.ndarray[Any, Any],
    data: np.ndarray[Any, Any],
    n_rows: int,
    n_cols: int,
) -> np.ndarray[Any, Any]:
    """Convert a CSR tuple to a dense matrix for structural assertions."""
    mat = np.zeros((n_rows, n_cols))
    for i in range(n_rows):
        for k in range(indptr[i], indptr[i + 1]):
            mat[i, indices[k]] = data[k]
    return mat


def _validate_csr(
    indptr: np.ndarray[Any, Any],
    indices: np.ndarray[Any, Any],
    data: np.ndarray[Any, Any],
    n_rows: int,
    n_cols: int,
) -> None:
    """Check CSR structural invariants."""
    assert len(indptr) == n_rows + 1, f"indptr length {len(indptr)} != {n_rows + 1}"
    assert indptr[0] == 0, "indptr must start at 0"
    assert indptr[-1] == len(indices), "indptr[-1] must equal len(indices)"
    assert len(indices) == len(data), "indices and data length mismatch"
    for i in range(n_rows):
        assert indptr[i] <= indptr[i + 1], "indptr must be non-decreasing"
    assert np.all(indices >= 0), "negative index"
    assert np.all(indices < n_cols), f"index >= {n_cols}"


class TestRandomConnectivity:
    def test_csr_valid(self) -> None:
        indptr, indices, data = random_connectivity(50, 50, p=0.1, weight=1.0)
        _validate_csr(indptr, indices, data, 50, 50)

    def test_expected_density(self) -> None:
        n = 200
        p = 0.1
        indptr, indices, data = random_connectivity(n, n, p=p, weight=1.0, seed=42)
        n_edges = len(indices)
        expected = n * n * p
        assert abs(n_edges - expected) < 3 * np.sqrt(expected), "edge count outside 3σ"

    def test_uniform_weight(self) -> None:
        _, _, data = random_connectivity(50, 50, p=0.2, weight=0.5, seed=1)
        np.testing.assert_allclose(data, 0.5)

    def test_zero_probability(self) -> None:
        indptr, indices, data = random_connectivity(20, 20, p=0.0, weight=1.0)
        assert len(indices) == 0

    def test_full_probability(self) -> None:
        indptr, indices, data = random_connectivity(10, 10, p=1.0, weight=1.0)
        assert len(indices) == 100

    def test_deterministic_seed(self) -> None:
        a = random_connectivity(30, 30, p=0.3, weight=1.0, seed=42)
        b = random_connectivity(30, 30, p=0.3, weight=1.0, seed=42)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])

    def test_rectangular(self) -> None:
        indptr, indices, data = random_connectivity(20, 30, p=0.2, weight=1.0)
        _validate_csr(indptr, indices, data, 20, 30)


class TestSmallWorld:
    def test_csr_valid(self) -> None:
        indptr, indices, data = small_world(50, k=6, p_rewire=0.1, weight=1.0)
        _validate_csr(indptr, indices, data, 50, 50)

    def test_no_rewiring_is_ring(self) -> None:
        n, k = 20, 4
        sw = small_world(n, k=k, p_rewire=0.0, weight=1.0, seed=42)
        ring = ring_topology(n, k=k // 2, weight=1.0)
        # Same number of edges (small-world adds both directions)
        assert len(sw[1]) == len(ring[1])

    def test_rewiring_changes_structure(self) -> None:
        sw0 = small_world(50, k=6, p_rewire=0.0, weight=1.0, seed=42)
        sw1 = small_world(50, k=6, p_rewire=0.5, weight=1.0, seed=42)
        # Different index arrays (some rewired)
        assert not np.array_equal(sw0[1], sw1[1])

    def test_symmetric(self) -> None:
        n = 30
        indptr, indices, data = small_world(n, k=4, p_rewire=0.1, weight=1.0)
        mat = _csr_to_dense(indptr, indices, data, n, n)
        np.testing.assert_array_equal(mat, mat.T)


class TestScaleFree:
    def test_csr_valid(self) -> None:
        indptr, indices, data = scale_free(50, m=3, weight=1.0)
        _validate_csr(indptr, indices, data, 50, 50)

    def test_hub_exists(self) -> None:
        """Scale-free networks have high-degree hubs."""
        n = 200
        indptr, indices, data = scale_free(n, m=2, weight=1.0, seed=42)
        mat = _csr_to_dense(indptr, indices, data, n, n)
        degrees = (mat != 0).sum(axis=1)
        max_degree = degrees.max()
        mean_degree = degrees.mean()
        assert max_degree > 3 * mean_degree, "no hub found"

    def test_symmetric(self) -> None:
        n = 30
        indptr, indices, data = scale_free(n, m=2, weight=1.0, seed=42)
        mat = _csr_to_dense(indptr, indices, data, n, n)
        np.testing.assert_array_equal(mat, mat.T)

    def test_edge_count(self) -> None:
        n, m = 100, 3
        indptr, indices, data = scale_free(n, m=m, weight=1.0)
        # Undirected: each new node adds m edges (both directions) = 2*m
        # Plus initial clique of m nodes
        n_edges = len(indices)
        assert n_edges >= 2 * m * (n - m)

    @pytest.mark.parametrize(
        ("n", "m", "match"),
        [
            (cast(int, True), 1, "n must be an integer"),
            (0, 1, "n must be at least 2"),
            (5, cast(int, False), "m must be an integer"),
            (5, 0, "m must be at least 1"),
            (5, 5, "m must be smaller than n"),
        ],
    )
    def test_rejects_invalid_barabasi_albert_parameters(self, n: int, m: int, match: str) -> None:
        """Invalid Barabasi-Albert dimensions fail before probability math."""
        with pytest.raises(ValueError, match=match):
            scale_free(n, m=m, weight=1.0, seed=42)

    @pytest.mark.parametrize("weight", [cast(float, True), np.inf])
    def test_rejects_invalid_weight(self, weight: float) -> None:
        """The emitted CSR payload must never carry non-finite weights."""
        with pytest.raises(ValueError, match="weight must be finite"):
            scale_free(8, m=2, weight=weight, seed=42)


class TestRingTopology:
    def test_csr_valid(self) -> None:
        indptr, indices, data = ring_topology(50, k=3, weight=1.0)
        _validate_csr(indptr, indices, data, 50, 50)

    def test_exact_degree(self) -> None:
        n, k = 30, 4
        indptr, indices, data = ring_topology(n, k=k, weight=1.0)
        mat = _csr_to_dense(indptr, indices, data, n, n)
        degrees = (mat != 0).sum(axis=1)
        # Each node connects to k neighbours in each direction = 2k
        np.testing.assert_array_equal(degrees, 2 * k)

    def test_wrap_around(self) -> None:
        """Neuron 0 and neuron N-1 should be connected for k>=1."""
        n = 20
        indptr, indices, data = ring_topology(n, k=1, weight=1.0)
        mat = _csr_to_dense(indptr, indices, data, n, n)
        assert mat[0, n - 1] != 0
        assert mat[n - 1, 0] != 0


class TestGridTopology:
    def test_csr_valid(self) -> None:
        indptr, indices, data = grid_topology(5, 5, radius=1, weight=1.0)
        _validate_csr(indptr, indices, data, 25, 25)

    def test_corner_degree(self) -> None:
        """Corner neuron in 5x5 grid, r=1 has 3 neighbours."""
        indptr, indices, data = grid_topology(5, 5, radius=1, weight=1.0)
        mat = _csr_to_dense(indptr, indices, data, 25, 25)
        corner_deg = (mat[0] != 0).sum()
        assert corner_deg == 3, f"corner degree {corner_deg} != 3"

    def test_centre_degree(self) -> None:
        """Centre neuron in 5x5 grid, r=1 has 8 neighbours."""
        indptr, indices, data = grid_topology(5, 5, radius=1, weight=1.0)
        mat = _csr_to_dense(indptr, indices, data, 25, 25)
        centre = 2 * 5 + 2  # row 2, col 2
        centre_deg = (mat[centre] != 0).sum()
        assert centre_deg == 8, f"centre degree {centre_deg} != 8"

    def test_no_self_connections(self) -> None:
        indptr, indices, data = grid_topology(4, 4, radius=1, weight=1.0)
        mat = _csr_to_dense(indptr, indices, data, 16, 16)
        assert np.all(np.diag(mat) == 0)


class TestAllToAll:
    def test_csr_valid(self) -> None:
        indptr, indices, data = all_to_all(10, 10, weight=0.5)
        _validate_csr(indptr, indices, data, 10, 10)

    def test_full_matrix(self) -> None:
        n = 8
        indptr, indices, data = all_to_all(n, n, weight=1.0)
        assert len(indices) == n * n

    def test_rectangular(self) -> None:
        indptr, indices, data = all_to_all(5, 10, weight=0.3)
        _validate_csr(indptr, indices, data, 5, 10)
        assert len(indices) == 50

    def test_weight_value(self) -> None:
        _, _, data = all_to_all(5, 5, weight=0.42)
        np.testing.assert_allclose(data, 0.42)
