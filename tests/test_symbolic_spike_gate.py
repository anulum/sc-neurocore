# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeGate from former test_symbolic.py

"""Focused suite: TestSpikeGate from former test_symbolic.py."""

from __future__ import annotations

from tests.symbolic_support import *  # noqa: F403

class TestSpikeGate:
    @pytest.mark.parametrize(
        "gate,inputs,expected",
        [
            ("AND", (1, 1), 1),
            ("AND", (1, 0), 0),
            ("AND", (0, 1), 0),
            ("AND", (0, 0), 0),
            ("OR", (1, 1), 1),
            ("OR", (1, 0), 1),
            ("OR", (0, 1), 1),
            ("OR", (0, 0), 0),
            ("NOT", (1,), 0),
            ("NOT", (0,), 1),
            ("NAND", (1, 1), 0),
            ("NAND", (1, 0), 1),
            ("NAND", (0, 0), 1),
            ("XOR", (1, 1), 0),
            ("XOR", (1, 0), 1),
            ("XOR", (0, 1), 1),
            ("XOR", (0, 0), 0),
        ],
    )
    def test_truth_tables(self, gate, inputs, expected):
        g = SpikeGate(gate)
        assert g(*inputs) == expected

    def test_xor_three_inputs(self):
        g = SpikeGate("XOR")
        assert g(1, 1, 1) == 1
        assert g(1, 1, 0) == 0
        assert g(1, 0, 0) == 1

    def test_lif_config_keys(self):
        for gate_type in ("AND", "OR", "NOT", "NAND", "XOR"):
            config = SpikeGate(gate_type).lif_config
            assert isinstance(config, dict)
            assert len(config) > 0

    def test_and_gate_lif_threshold(self):
        assert SpikeGate("AND").lif_config["threshold"] == 2

    def test_or_gate_lif_threshold(self):
        assert SpikeGate("OR").lif_config["threshold"] == 1
