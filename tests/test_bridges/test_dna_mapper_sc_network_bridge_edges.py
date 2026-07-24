# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSCNetworkBridgeEdges from former test_dna_mapper.py

"""Focused suite: TestSCNetworkBridgeEdges from former test_dna_mapper.py."""

from __future__ import annotations

from dna_mapper_support import *  # noqa: F403


class TestSCNetworkBridgeEdges:
    """Adjacency-to-gate inference boundary cases."""

    def test_from_adjacency_skips_non_input_nodes_without_sources(self) -> None:
        design = SCNetworkBridge(seed=42).from_adjacency(
            np.zeros((3, 3), dtype=float),
            input_indices=[0],
            output_indices=[2],
            name="empty_graph",
        )

        assert design.total_gates == 0

    def test_from_adjacency_uses_or_for_two_source_inhibitory_mix(self) -> None:
        adjacency = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 0.0],
            ]
        )

        design = SCNetworkBridge(seed=42).from_adjacency(
            adjacency,
            input_indices=[0, 1],
            output_indices=[2],
            name="mixed_sources",
        )

        assert design.gates[0].gate_type == GateType.OR
