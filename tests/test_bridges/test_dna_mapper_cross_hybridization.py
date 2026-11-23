# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCrossHybridization from former test_dna_mapper.py

"""Focused suite: TestCrossHybridization from former test_dna_mapper.py."""

from __future__ import annotations

from sc_neurocore.bridges.dna_mapper import (
    BitstreamToDNA,
    CrossHybridizationChecker,
    DNACircuitDesign,
)


class TestCrossHybridization:
    """Cross-hybridization detection."""

    def test_returns_list(self, simple_and_circuit: DNACircuitDesign) -> None:
        """The checker returns a list; its flags are not checked."""
        checker = CrossHybridizationChecker(max_complementary_run=8)
        flags = checker.check(simple_and_circuit)
        assert isinstance(flags, list)

    def test_flag_structure(self) -> None:
        """Each flag names both strands and the complementary run."""
        checker = CrossHybridizationChecker(max_complementary_run=3)
        c = BitstreamToDNA(seed=42)
        design = c.compile_network(
            gates=[{"type": "AND", "inputs": ["A", "B"], "output": "C"}],
            input_names=["A", "B"],
            output_names=["C"],
        )
        flags = checker.check(design)
        for flag in flags:
            assert "strand_a" in flag
            assert "strand_b" in flag
            assert "complementary_run" in flag
            assert "severity" in flag

    def test_longest_common_substring(self) -> None:
        """The longest common substring has length 4."""
        result = CrossHybridizationChecker._longest_common_substring("ABCDEF", "XBCDEY")
        assert result == 4
