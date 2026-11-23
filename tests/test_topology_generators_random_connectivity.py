# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRandomConnectivity from former test_topology_generators.py

"""Focused suite: TestRandomConnectivity from former test_topology_generators.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.network.topology import (
    random_connectivity,
)
from tests.topology_generators_support import (
    _validate_csr,
)


class TestRandomConnectivity:
    """The random connectivity generator."""

    def test_csr_valid(self) -> None:
        """The generated graph is a valid CSR matrix."""
        indptr, indices, data = random_connectivity(50, 50, p=0.1, weight=1.0)
        _validate_csr(indptr, indices, data, 50, 50)

    def test_expected_density(self) -> None:
        """The edge count matches the connection probability."""
        n = 200
        p = 0.1
        indptr, indices, data = random_connectivity(n, n, p=p, weight=1.0, seed=42)
        n_edges = len(indices)
        expected = n * n * p
        assert abs(n_edges - expected) < 3 * np.sqrt(expected), "edge count outside 3σ"

    def test_uniform_weight(self) -> None:
        """Every edge has the given weight."""
        _, _, data = random_connectivity(50, 50, p=0.2, weight=0.5, seed=1)
        np.testing.assert_allclose(data, 0.5)

    def test_zero_probability(self) -> None:
        """Probability zero gives no edges."""
        indptr, indices, data = random_connectivity(20, 20, p=0.0, weight=1.0)
        assert len(indices) == 0

    def test_full_probability(self) -> None:
        """Probability one connects every pair."""
        indptr, indices, data = random_connectivity(10, 10, p=1.0, weight=1.0)
        assert len(indices) == 100

    def test_deterministic_seed(self) -> None:
        """The same seed gives the same graph."""
        a = random_connectivity(30, 30, p=0.3, weight=1.0, seed=42)
        b = random_connectivity(30, 30, p=0.3, weight=1.0, seed=42)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])

    def test_rectangular(self) -> None:
        """A rectangular random graph is a valid CSR matrix."""
        indptr, indices, data = random_connectivity(20, 30, p=0.2, weight=1.0)
        _validate_csr(indptr, indices, data, 20, 30)
