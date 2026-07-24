# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBuildPartitionReport from former test_hierarchical_partitioner_reporting.py

"""Focused suite: TestBuildPartitionReport from former test_hierarchical_partitioner_reporting.py."""

from __future__ import annotations

from hierarchical_partitioner_reporting_support import *  # noqa: F403


class TestBuildPartitionReport:
    def test_full_report(self) -> None:
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
        g = _make_chain_graph(10, scc=0.5)
        parts = [list(range(5)), list(range(5, 10))]
        seeds = [0xACE1, 0xBEEF]
        report = build_partition_report(g, parts, seeds, scc_budget=0.1)
        assert report.scc_budget_violations >= 1

    def test_no_violations_when_budget_high(self) -> None:
        g = _make_chain_graph(10, scc=0.05)
        parts = [list(range(5)), list(range(5, 10))]
        seeds = [0xACE1, 0xBEEF]
        report = build_partition_report(g, parts, seeds, scc_budget=1.0)
        assert report.scc_budget_violations == 0
