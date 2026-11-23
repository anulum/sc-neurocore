# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBuildPartitionReport from former test_hierarchical_partitioner_reporting.py

"""Focused suite: TestBuildPartitionReport from former test_hierarchical_partitioner_reporting.py."""

from __future__ import annotations

from sc_neurocore.chiplet import (
    build_partition_report,
    HierarchicalPartitioner,
)
from tests.test_chiplet.hierarchical_partitioner_support import (
    make_chain_graph as _make_chain_graph,
)


class TestBuildPartitionReport:
    """Building a partition report from a partitioned graph."""

    def test_full_report(self) -> None:
        """A report on a bisected chain fills its counts, cut, balance, volume and seeds."""
        g = _make_chain_graph(10, scc=0.3)
        hp = HierarchicalPartitioner(num_partitions=2, seed=42)
        parts, seeds = hp.partition(g)
        report = build_partition_report(g, parts, seeds)
        assert report.num_partitions == 2
        assert report.edge_cut >= 1
        assert report.imbalance_ratio >= 0.0
        assert report.comm_volume_bytes > 0
        assert len(report.seeds) == 2

    def test_scc_budget_violations_counted(self) -> None:
        """A low SCC budget is counted as violated in the report."""
        g = _make_chain_graph(10, scc=0.5)
        parts = [list(range(5)), list(range(5, 10))]
        seeds = [0xACE1, 0xBEEF]
        report = build_partition_report(g, parts, seeds, scc_budget=0.1)
        assert report.scc_budget_violations >= 1

    def test_no_violations_when_budget_high(self) -> None:
        """A high SCC budget leaves the report with no violation."""
        g = _make_chain_graph(10, scc=0.05)
        parts = [list(range(5)), list(range(5, 10))]
        seeds = [0xACE1, 0xBEEF]
        report = build_partition_report(g, parts, seeds, scc_budget=1.0)
        assert report.scc_budget_violations == 0
