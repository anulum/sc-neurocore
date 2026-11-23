# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCommVolume from former test_hierarchical_partitioner_metrics.py

"""Focused suite: TestCommVolume from former test_hierarchical_partitioner_metrics.py."""

from __future__ import annotations

from sc_neurocore.chiplet import (
    calculate_comm_volume,
)
from tests.test_chiplet.hierarchical_partitioner_support import (
    make_chain_graph as _make_chain_graph,
)


class TestCommVolume:
    """Communication volume across partition boundaries."""

    def test_basic(self) -> None:
        """A two-part chain has boundary edges, bytes, and one message per boundary edge."""
        g = _make_chain_graph(6, scc=0.1)
        parts = [[0, 1, 2], [3, 4, 5]]
        cv = calculate_comm_volume(g, parts)
        assert cv["boundary_edges"] >= 1
        assert cv["volume_bytes"] > 0
        assert cv["messages"] == cv["boundary_edges"]

    def test_no_boundary(self) -> None:
        """A single partition has no boundary edges and no volume."""
        g = _make_chain_graph(4)
        parts = [list(range(4))]
        cv = calculate_comm_volume(g, parts)
        assert cv["boundary_edges"] == 0
        assert cv["volume_bytes"] == 0
