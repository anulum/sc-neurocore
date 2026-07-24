# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGhostCellManager from former test_hierarchical_partitioner_reporting.py

"""Focused suite: TestGhostCellManager from former test_hierarchical_partitioner_reporting.py."""

from __future__ import annotations

from hierarchical_partitioner_reporting_support import *  # noqa: F403


class TestGhostCellManager:
    def test_compute_halos(self) -> None:
        g = _make_chain_graph(6)
        parts = [[0, 1, 2], [3, 4, 5]]
        halos = GhostCellManager.compute_halos(g, parts)
        # Partition 0 needs ghost of vertex 3 (neighbor of 2)
        assert 3 in halos[0]
        # Partition 1 needs ghost of vertex 2 (neighbor of 3)
        assert 2 in halos[1]

    def test_halo_sizes(self) -> None:
        g = _make_chain_graph(6)
        parts = [[0, 1, 2], [3, 4, 5]]
        sizes = GhostCellManager.halo_sizes(g, parts)
        assert sizes[0] >= 1
        assert sizes[1] >= 1

    def test_no_halos_single_partition(self) -> None:
        g = _make_chain_graph(4)
        parts = [list(range(4))]
        sizes = GhostCellManager.halo_sizes(g, parts)
        assert sizes[0] == 0
