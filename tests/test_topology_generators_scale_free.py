# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestScaleFree from former test_topology_generators.py

"""Focused suite: TestScaleFree from former test_topology_generators.py."""

from __future__ import annotations

from tests.topology_generators_support import *  # noqa: F403


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
