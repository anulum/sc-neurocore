# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLFSRSeedAllocator from former test_hierarchical_partitioner_graph.py

"""Focused suite: TestLFSRSeedAllocator from former test_hierarchical_partitioner_graph.py."""

from __future__ import annotations

from hierarchical_partitioner_graph_support import *  # noqa: F403


class TestLFSRSeedAllocator:
    def test_allocate_unique_seeds(self) -> None:
        alloc = LFSRSeedAllocator()
        seeds = alloc.allocate(8)
        assert len(seeds) == 8
        assert alloc.verify_uniqueness(seeds)

    def test_no_zero_seeds(self) -> None:
        alloc = LFSRSeedAllocator()
        seeds = alloc.allocate(100)
        assert 0 not in seeds

    def test_single_partition(self) -> None:
        alloc = LFSRSeedAllocator()
        seeds = alloc.allocate(1)
        assert len(seeds) == 1
        assert seeds[0] != 0
