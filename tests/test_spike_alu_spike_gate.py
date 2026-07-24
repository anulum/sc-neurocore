# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeGate from former test_spike_alu.py

"""Focused suite: TestSpikeGate from former test_spike_alu.py."""

from __future__ import annotations

from tests.spike_alu_support import *  # noqa: F403


class TestSpikeGate:
    @pytest.mark.parametrize("a,b,expected", [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)])
    def test_and_truth_table(self, a, b, expected):
        assert SpikeGate("AND")(a, b) == expected

    @pytest.mark.parametrize("a,b,expected", [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)])
    def test_or_truth_table(self, a, b, expected):
        assert SpikeGate("OR")(a, b) == expected

    @pytest.mark.parametrize("a,expected", [(0, 1), (1, 0)])
    def test_not_truth_table(self, a, expected):
        assert SpikeGate("NOT")(a) == expected

    @pytest.mark.parametrize("a,b,expected", [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0)])
    def test_nand_truth_table(self, a, b, expected):
        assert SpikeGate("NAND")(a, b) == expected

    @pytest.mark.parametrize("a,b,expected", [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)])
    def test_xor_truth_table(self, a, b, expected):
        assert SpikeGate("XOR")(a, b) == expected

    def test_lif_config_exists(self):
        for gate_type in ["AND", "OR", "NOT", "NAND", "XOR"]:
            gate = SpikeGate(gate_type)
            config = gate.lif_config
            assert isinstance(config, dict)

    def test_de_morgan_and(self):
        """NOT(A AND B) = NOT(A) OR NOT(B)."""
        nand = SpikeGate("NAND")
        not_g = SpikeGate("NOT")
        or_g = SpikeGate("OR")
        for a in [0, 1]:
            for b in [0, 1]:
                assert nand(a, b) == or_g(not_g(a), not_g(b))
