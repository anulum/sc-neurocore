# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBoundarySCCMetrics from former test_hierarchical_partitioner_metrics.py

"""Focused suite: TestBoundarySCCMetrics from former test_hierarchical_partitioner_metrics.py."""

from __future__ import annotations

from sc_neurocore.chiplet import (
    calculate_mean_boundary_scc,
    calculate_total_boundary_scc,
)
from tests.test_chiplet.hierarchical_partitioner_support import (
    make_chain_graph as _make_chain_graph,
)


class TestBoundarySCCMetrics:
    """Mean and total boundary SCC of a partitioned chain graph."""

    def test_mean_boundary_scc(self) -> None:
        """The mean boundary SCC of a two-part chain is not negative."""
        g = _make_chain_graph(6, scc=0.3)
        parts = [[0, 1, 2], [3, 4, 5]]
        mean_scc = calculate_mean_boundary_scc(g, parts)
        assert mean_scc >= 0.0

    def test_total_boundary_scc(self) -> None:
        """The total boundary SCC of a two-part chain is not negative."""
        g = _make_chain_graph(6, scc=0.4)
        parts = [[0, 1, 2], [3, 4, 5]]
        total_scc = calculate_total_boundary_scc(g, parts)
        assert total_scc >= 0.0

    def test_no_boundary(self) -> None:
        """A single partition has zero mean and zero total boundary SCC."""
        g = _make_chain_graph(4, scc=0.5)
        parts = [list(range(4))]
        assert calculate_mean_boundary_scc(g, parts) == 0.0
        assert calculate_total_boundary_scc(g, parts) == 0.0
