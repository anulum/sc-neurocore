# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRingTopology from former test_topology_generators.py

"""Focused suite: TestRingTopology from former test_topology_generators.py."""

from __future__ import annotations

from tests.topology_generators_support import *  # noqa: F403


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
