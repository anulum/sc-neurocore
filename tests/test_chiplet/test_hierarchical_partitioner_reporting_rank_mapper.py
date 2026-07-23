# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRankMapper from former test_hierarchical_partitioner_reporting.py

"""Focused suite: TestRankMapper from former test_hierarchical_partitioner_reporting.py."""

from __future__ import annotations

from hierarchical_partitioner_reporting_support import *  # noqa: F403

class TestRankMapper:
    def test_basic_assignment(self) -> None:
        mapper = RankMapper(num_ranks=2)
        parts = [[0, 1], [2, 3], [4, 5], [6, 7]]
        mapping = mapper.assign(parts)
        assert len(mapping) == 4
        assert all(0 <= v < 2 for v in mapping.values())

    def test_fewer_partitions_than_ranks(self) -> None:
        mapper = RankMapper(num_ranks=8)
        parts = [[0, 1], [2, 3]]
        mapping = mapper.assign(parts)
        assert mapping[0] == 0
        assert mapping[1] == 1

    def test_cross_rank_edges(self) -> None:
        g = _make_chain_graph(6)
        parts = [[0, 1, 2], [3, 4, 5]]
        mapper = RankMapper(num_ranks=2)
        cross = mapper.cross_rank_edges(g, parts)
        assert cross >= 1

    def test_hierarchy_levels(self) -> None:
        mapper = RankMapper(num_ranks=4, hierarchy=[HierarchyLevel.RACK, HierarchyLevel.NODE])
        assert mapper.hierarchy[0] == HierarchyLevel.RACK
