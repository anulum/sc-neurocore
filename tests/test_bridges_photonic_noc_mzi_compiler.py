# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMZICompiler from former test_bridges_photonic_noc.py

"""Focused suite: TestMZICompiler from former test_bridges_photonic_noc.py."""

from __future__ import annotations

from tests.bridges_photonic_noc_support import *  # noqa: F403


class TestMZICompiler:
    def test_compile_gate(self):
        compiler = MZICompiler()
        gate = compiler.compile_gate(gate_type="cross", input_ports=[0, 1], output_port=0)
        assert isinstance(gate, MZIGate)

    def test_compile_network(self):
        compiler = MZICompiler()
        gates = [
            {"type": "cross", "inputs": [0, 1], "output": 0},
            {"type": "bar", "inputs": [1, 2], "output": 1},
        ]
        result = compiler.compile_network(gates)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_phase_shift_range(self):
        compiler = MZICompiler()
        gate = compiler.compile_gate(gate_type="bar", input_ports=[0, 1], output_port=1)
        assert 0 <= gate.phase_shift_rad <= 2 * math.pi
