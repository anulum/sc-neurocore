# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPartitionReport from former test_hierarchical_partitioner_metrics.py

"""Focused suite: TestPartitionReport from former test_hierarchical_partitioner_metrics.py."""

from __future__ import annotations

from hierarchical_partitioner_metrics_support import *  # noqa: F403

class TestPartitionReport:
    def test_summary(self) -> None:
        r = PartitionReport(
            num_partitions=4,
            partition_sizes=[25, 25, 25, 25],
            edge_cut=12,
            max_boundary_scc=0.15,
            mean_boundary_scc=0.08,
            total_boundary_scc=0.96,
            imbalance_ratio=0.0,
            comm_volume_bytes=24576,
            comm_messages=12,
            seeds=[0xACE1, 0xBEEF, 0xCAFE, 0xDEAD],
        )
        s = r.summary()
        assert "4" in s
        assert "12" in s
        assert "Imbalance" in s
        assert "Comm" in s
