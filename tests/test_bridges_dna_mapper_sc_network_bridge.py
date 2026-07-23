# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSCNetworkBridge from former test_bridges_dna_mapper.py

"""Focused suite: TestSCNetworkBridge from former test_bridges_dna_mapper.py."""

from __future__ import annotations

from tests.bridges_dna_mapper_support import *  # noqa: F403

class TestSCNetworkBridge:
    def test_from_adjacency(self) -> None:
        adj = np.array([[0, 1, 0.5], [0, 0, 1], [0, 0, 0]], dtype=float)
        bridge = SCNetworkBridge(seed=42)
        design = bridge.from_adjacency(adj, input_indices=[0], output_indices=[2])
        assert isinstance(design, DNACircuitDesign)
