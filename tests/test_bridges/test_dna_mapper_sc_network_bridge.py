# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSCNetworkBridge from former test_dna_mapper.py

"""Focused suite: TestSCNetworkBridge from former test_dna_mapper.py."""

from __future__ import annotations

from dna_mapper_support import *  # noqa: F403


class TestSCNetworkBridge:
    """SC network to DNA circuit bridge."""

    def test_simple_adjacency(self) -> None:
        adj = np.array(
            [
                [0, 0, 1],
                [0, 0, 1],
                [0, 0, 0],
            ],
            dtype=float,
        )
        bridge = SCNetworkBridge(seed=42)
        design = bridge.from_adjacency(adj, input_indices=[0, 1], output_indices=[2])
        assert design.total_gates >= 1

    def test_inhibitory_produces_not(self) -> None:
        adj = np.array(
            [
                [0, -1],
                [0, 0],
            ],
            dtype=float,
        )
        bridge = SCNetworkBridge(seed=42)
        design = bridge.from_adjacency(adj, input_indices=[0], output_indices=[1])
        assert any(g.gate_type == GateType.NOT for g in design.gates)

    def test_multi_fan_in(self) -> None:
        adj = np.zeros((5, 5))
        adj[0, 4] = 1
        adj[1, 4] = 1
        adj[2, 4] = 1
        bridge = SCNetworkBridge(seed=42)
        design = bridge.from_adjacency(adj, input_indices=[0, 1, 2, 3], output_indices=[4])
        assert design.total_gates >= 2  # chained AND
