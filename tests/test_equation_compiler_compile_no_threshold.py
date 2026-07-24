# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCompileNoThreshold from former test_equation_compiler.py

"""Focused suite: TestCompileNoThreshold from former test_equation_compiler.py."""

from __future__ import annotations

from tests.equation_compiler_support import *  # noqa: F403


class TestCompileNoThreshold:
    def test_integrator_no_spike(self):
        neuron = EquationNeuron(
            equations={"v": "I"},
            state={"v": 0.0},
            dt=1.0,
        )
        verilog = compile_to_verilog(neuron, module_name="integrator")
        assert "module integrator" in verilog
        assert "spike_out <= 1'b0" in verilog
        # No threshold branch
        assert "spike_out <= 1'b1" not in verilog
