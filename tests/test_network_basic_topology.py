# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTopology from former test_network_basic.py

"""Focused suite: TestTopology from former test_network_basic.py."""

from __future__ import annotations

from tests.network_basic_support import *  # noqa: F403

class TestTopology:
    def test_random_connectivity(self):
        indptr, indices, data = topology.random_connectivity(5, 5, 0.5, 1.0, seed=0)
        assert indptr.shape == (6,)
        assert len(indices) == len(data)
        assert np.all(data == 1.0)

    def test_all_to_all(self):
        indptr, indices, data = topology.all_to_all(3, 4, 2.0)
        assert indptr[-1] == 12
        assert np.all(data == 2.0)

    def test_ring_topology(self):
        indptr, indices, data = topology.ring_topology(6, 1, 0.5)
        assert indptr[-1] == 12  # 6 nodes * 2 connections each

    def test_small_world(self):
        indptr, indices, data = topology.small_world(10, 4, 0.1, 1.0, seed=7)
        assert indptr.shape == (11,)
        assert len(indices) > 0

    def test_scale_free(self):
        indptr, indices, data = topology.scale_free(10, 2, 1.0, seed=7)
        assert indptr.shape == (11,)
        assert len(indices) > 0

    def test_grid_topology(self):
        indptr, indices, data = topology.grid_topology(3, 3, 1, 1.0)
        assert indptr.shape == (10,)  # 9 nodes + 1
