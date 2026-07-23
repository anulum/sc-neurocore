# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSCToPhotonic from former test_photonic_noc.py

"""Focused suite: TestSCToPhotonic from former test_photonic_noc.py."""

from __future__ import annotations

from photonic_noc_support import *  # noqa: F403

class TestSCToPhotonic:
    """Top-level photonic compiler tests."""

    def test_compile_basic(self, simple_adjacency: np.ndarray) -> None:
        design = SCToPhotonic().compile(simple_adjacency, name="test")
        assert design.n_nodes == 4
        assert len(design.waveguides) > 0
        assert len(design.mzi_gates) > 0
        assert len(design.wdm_channels) == 4

    def test_with_labels(self, simple_adjacency: np.ndarray) -> None:
        design = SCToPhotonic().compile(simple_adjacency, node_labels=["A", "B", "C", "D"])
        assert design.wdm_channels[0].signal_name == "A"

    def test_area_positive(self, simple_adjacency: np.ndarray) -> None:
        design = SCToPhotonic().compile(simple_adjacency)
        assert design.total_area_um2 > 0

    def test_with_custom_gates(self, simple_adjacency: np.ndarray) -> None:
        gates = [{"type": "MUL", "inputs": [0, 1], "output": 2}]
        design = SCToPhotonic().compile(simple_adjacency, gate_specs=gates)
        assert len(design.mzi_gates) == 1
