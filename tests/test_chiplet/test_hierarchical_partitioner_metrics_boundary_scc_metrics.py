# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBoundarySCCMetrics from former test_hierarchical_partitioner_metrics.py

"""Focused suite: TestBoundarySCCMetrics from former test_hierarchical_partitioner_metrics.py."""

from __future__ import annotations

from hierarchical_partitioner_metrics_support import *  # noqa: F403


class TestBoundarySCCMetrics:
    def test_mean_boundary_scc(self) -> None:
        g = _make_chain_graph(6, scc=0.3)
        parts = [[0, 1, 2], [3, 4, 5]]
        mean_scc = calculate_mean_boundary_scc(g, parts)
        assert mean_scc >= 0.0

    def test_total_boundary_scc(self) -> None:
        g = _make_chain_graph(6, scc=0.4)
        parts = [[0, 1, 2], [3, 4, 5]]
        total_scc = calculate_total_boundary_scc(g, parts)
        assert total_scc >= 0.0

    def test_no_boundary(self) -> None:
        g = _make_chain_graph(4, scc=0.5)
        parts = [list(range(4))]
        assert calculate_mean_boundary_scc(g, parts) == 0.0
        assert calculate_total_boundary_scc(g, parts) == 0.0
