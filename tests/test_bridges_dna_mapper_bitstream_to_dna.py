# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBitstreamToDNA from former test_bridges_dna_mapper.py

"""Focused suite: TestBitstreamToDNA from former test_bridges_dna_mapper.py."""

from __future__ import annotations

from sc_neurocore.bridges.dna_mapper import (
    BitstreamToDNA,
    DNACircuitDesign,
)


class TestBitstreamToDNA:
    """Smoke tests of compiling to DNA."""

    def test_compile_network(self) -> None:
        """Compilation returns a DNA circuit design."""
        bridge = BitstreamToDNA(seed=42)
        gates = [
            {"type": "AND", "inputs": ["a", "b"], "output": "c"},
            {"type": "OR", "inputs": ["c", "d"], "output": "e"},
        ]
        design = bridge.compile_network(gates, input_names=["a", "b", "d"], output_names=["e"])
        assert isinstance(design, DNACircuitDesign)

    def test_validate(self) -> None:
        """The result is a dict; its contents are not checked."""
        bridge = BitstreamToDNA(seed=42)
        gates = [{"type": "NOT", "inputs": ["a"], "output": "b"}]
        design = bridge.compile_network(gates, input_names=["a"], output_names=["b"])
        result = bridge.validate(design)
        assert isinstance(result, dict)
