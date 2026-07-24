# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPartitionDeterministicOutput from former test_hierarchical_partitioner_graph.py

"""Focused suite: TestPartitionDeterministicOutput from former test_hierarchical_partitioner_graph.py."""

from __future__ import annotations

from hierarchical_partitioner_graph_support import *  # noqa: F403


class TestPartitionDeterministicOutput:
    """The perf fix must NOT change algorithm output — the partitioner
    is deterministic for a fixed graph + seed."""

    def test_partitions_canonical_match_baseline(self) -> None:
        # The baseline values were captured before the perf fix and
        # pinned here so any future algorithmic drift is loud.
        baseline_sizes = {50: [1, 1, 1, 47], 100: [1, 1, 1, 97], 200: [1, 1, 1, 197]}
        hp = HierarchicalPartitioner(num_partitions=4)
        for n_v, expected_sizes in baseline_sizes.items():
            g = _build_graph(n_v, avg_degree=8, seed=42)
            partitions, _seeds = hp.partition(g)
            sizes = sorted(len(p) for p in partitions)
            assert sizes == expected_sizes, (
                f"V={n_v} partition sizes drifted: got {sizes}, expected {expected_sizes}"
            )
