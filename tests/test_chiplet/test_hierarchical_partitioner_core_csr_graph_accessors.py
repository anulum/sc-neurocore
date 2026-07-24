# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCsrGraphAccessors from former test_hierarchical_partitioner_core.py

"""Focused suite: TestCsrGraphAccessors from former test_hierarchical_partitioner_core.py."""

from __future__ import annotations

from hierarchical_partitioner_core_support import *  # noqa: F403


class TestCsrGraphAccessors:
    """`CSRGraph.edge_conn` and `LFSRSeedAllocator` zero-seed clamp
    are used by external callers but were uncovered by the existing
    suite — pin them here."""

    def test_edge_conn_returns_correct_slice(self) -> None:
        from sc_neurocore.chiplet.hierarchical_partitioner import CSRGraph

        edges = [
            CorrelationEdge(u=0, v=1, conn_weight=1.5, scc_weight=0.2),
            CorrelationEdge(u=1, v=2, conn_weight=2.5, scc_weight=0.3),
        ]
        csr = CSRGraph.from_edge_list(3, edges, None)
        # edge_conn returns a slice of conn_weights for vertex v's edges.
        conn0 = csr.edge_conn(0)
        assert len(conn0) == 1
        assert conn0[0] == 1.5

    def test_lfsr_seed_clamps_zero_to_one(self) -> None:
        from sc_neurocore.chiplet.hierarchical_partitioner import (
            LFSRSeedAllocator,
        )

        # Pick a base_seed and num_partitions that produce a 0 seed
        # for some i — the allocator must clamp it to 1.
        # base_seed=0xFFFF, spacing=65535//5 = 13107; one of the
        # combinations will hit `& 0xFFFF == 0` after addition.
        alloc = LFSRSeedAllocator(base_seed=0)
        # i=0 → 0 + 1*spacing = 13107; i=4 → 0 + 5*spacing = 65535
        # None of these are zero. Try base that wraps to zero:
        # 0xFFFF + 1*1 = 0x10000 → masked to 0
        alloc = LFSRSeedAllocator(base_seed=0xFFFF)
        seeds = alloc.allocate(num_partitions=65535)  # spacing=1
        # Some seed would be 0 without the clamp; verify none are zero.
        assert all(s != 0 for s in seeds)
