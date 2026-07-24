# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBoundarySyncProtocol from former test_hierarchical_partitioner_reporting.py

"""Focused suite: TestBoundarySyncProtocol from former test_hierarchical_partitioner_reporting.py."""

from __future__ import annotations

from hierarchical_partitioner_reporting_support import *  # noqa: F403


class TestBoundarySyncProtocol:
    def test_init_buffers(self) -> None:
        g = _make_chain_graph(6, scc=0.2)
        parts = [[0, 1, 2], [3, 4, 5]]
        seeds = [0xACE1, 0xBEEF]
        sync = BoundarySyncProtocol()
        count = sync.init_buffers(g, parts, seeds)
        assert count >= 1
        assert sync.num_buffers == count

    def test_scc_budget_no_violations(self) -> None:
        g = _make_chain_graph(6, scc=0.05)
        parts = [[0, 1, 2], [3, 4, 5]]
        sync = BoundarySyncProtocol(BoundarySyncConfig(max_boundary_scc_budget=0.1))
        violations = sync.check_scc_budget(g, parts)
        assert violations == []

    def test_scc_budget_with_violations(self) -> None:
        g = _make_chain_graph(6, scc=0.5)
        parts = [[0, 1, 2], [3, 4, 5]]
        sync = BoundarySyncProtocol(BoundarySyncConfig(max_boundary_scc_budget=0.1))
        violations = sync.check_scc_budget(g, parts)
        assert len(violations) >= 1

    def test_buffer_seed_nonzero(self) -> None:
        g = _make_chain_graph(4, scc=0.1)
        parts = [[0, 1], [2, 3]]
        seeds = [0x0001, 0x0001]  # same seed → XOR = 0 → forced to 1
        sync = BoundarySyncProtocol()
        sync.init_buffers(g, parts, seeds)
        for seed in sync.boundary_buffers.values():
            assert seed != 0
