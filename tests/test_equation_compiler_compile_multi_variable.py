# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCompileMultiVariable from former test_equation_compiler.py

"""Focused suite: TestCompileMultiVariable from former test_equation_compiler.py."""

from __future__ import annotations

from tests.equation_compiler_support import *  # noqa: F403


class TestCompileMultiVariable:
    def test_fitzhugh_nagumo(self):
        neuron = EquationNeuron(
            equations={
                "v": "v - v**3 / 3 - w + I",
                "w": "epsilon * (v + a - b * w)",
            },
            parameters={"epsilon": 0.08, "a": 0.7, "b": 0.8},
            state={"v": -1.0, "w": -0.5},
            threshold="v > 1.0",
            reset={"v": "-1.0"},
            dt=0.1,
        )
        verilog = compile_to_verilog(neuron, module_name="fhn_neuron")
        assert "module fhn_neuron" in verilog
        assert "v_reg" in verilog
        assert "w_reg" in verilog
        assert "v_out" in verilog
        assert "w_out" in verilog
        assert "v_next" in verilog
        assert "w_next" in verilog

    def test_izhikevich(self):
        neuron = EquationNeuron(
            equations={
                "v": "0.04 * v**2 + 5 * v + 140 - u + I",
                "u": "a * (b * v - u)",
            },
            parameters={"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0},
            state={"v": -65.0, "u": -14.0},
            threshold="v > 30",
            reset={"v": "c", "u": "u + d"},
            dt=1.0,
        )
        verilog = compile_to_verilog(neuron, module_name="izh_neuron")
        assert "module izh_neuron" in verilog
        assert "v_reg" in verilog
        assert "u_reg" in verilog
        # v**2 should generate a multiply intermediate
        assert "_mul" in verilog
