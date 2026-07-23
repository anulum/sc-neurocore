# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSmallWorld from former test_topology_generators.py

"""Focused suite: TestSmallWorld from former test_topology_generators.py."""

from __future__ import annotations

from tests.topology_generators_support import *  # noqa: F403

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
