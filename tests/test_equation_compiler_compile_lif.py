# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCompileLIF from former test_equation_compiler.py

"""Focused suite: TestCompileLIF from former test_equation_compiler.py."""

from __future__ import annotations

from tests.equation_compiler_support import *  # noqa: F403


class TestCompileLIF:
    def test_basic_lif_generates_verilog(self):
        neuron = from_equations(
            "dv/dt = -(v - E_L)/tau_m + I/C",
            threshold="v > -50",
            reset="v = -65",
            params=dict(E_L=-65, tau_m=10, C=1),
            init=dict(v=-65),
        )
        verilog = compile_to_verilog(neuron, module_name="test_lif")
        assert "module test_lif" in verilog
        assert "endmodule" in verilog
        assert "clk" in verilog
        assert "rst_n" in verilog
        assert "I_t" in verilog
        assert "spike_out" in verilog
        assert "v_out" in verilog
        assert "v_reg" in verilog

    def test_lif_has_threshold_logic(self):
        neuron = from_equations(
            "dv/dt = -(v - E_L)/tau_m + I/C",
            threshold="v > -50",
            reset="v = -65",
            params=dict(E_L=-65, tau_m=10, C=1),
            init=dict(v=-65),
        )
        verilog = compile_to_verilog(neuron)
        assert "spike_out <= 1'b1" in verilog
        assert "spike_out <= 1'b0" in verilog

    def test_lif_has_parameters(self):
        neuron = from_equations(
            "dv/dt = -(v - E_L)/tau_m + I/C",
            threshold="v > -50",
            reset="v = -65",
            params=dict(E_L=-65, tau_m=10, C=1),
            init=dict(v=-65),
        )
        verilog = compile_to_verilog(neuron)
        assert "P_E_L" in verilog
        assert "P_TAU_M" in verilog
        assert "P_C" in verilog

    def test_strict_units_lif_compiles_with_named_threshold_and_reset_constants(self):
        neuron = from_equations(
            "dv/dt = (-(v - E_L) + R * I) / tau_m",
            threshold="v > v_threshold",
            reset="v = v_reset",
            params={
                "E_L": -65.0 * UNIT_REGISTRY.millivolt,
                "R": 100e6 * UNIT_REGISTRY.ohm,
                "tau_m": 10.0 * UNIT_REGISTRY.millisecond,
            },
            init={"v": -65.0 * UNIT_REGISTRY.millivolt},
            constants={
                "v_threshold": -50.0 * UNIT_REGISTRY.millivolt,
                "v_reset": -65.0 * UNIT_REGISTRY.millivolt,
            },
            dt=1.0 * UNIT_REGISTRY.millisecond,
            units="strict",
            input_unit=1.0 * UNIT_REGISTRY.nanoampere,
        )
        verilog = compile_to_verilog(neuron, module_name="strict_lif", fraction=16)
        assert "module strict_lif" in verilog
        assert "P_V_THRESHOLD" in verilog
        assert "P_V_RESET" in verilog
        assert "if ((v_next > P_V_THRESHOLD))" in verilog
        assert "v_reg <= P_V_RESET;" in verilog
