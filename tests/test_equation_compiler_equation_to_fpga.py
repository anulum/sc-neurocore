# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEquationToFPGA from former test_equation_compiler.py

"""Focused suite: TestEquationToFPGA from former test_equation_compiler.py."""

from __future__ import annotations

from tests.equation_compiler_support import *  # noqa: F403


class TestEquationToFPGA:
    def test_one_liner(self):
        neuron, verilog = equation_to_fpga(
            "dv/dt = -(v - E_L)/tau_m + I/C",
            threshold="v > -50",
            reset="v = -65",
            params=dict(E_L=-65, tau_m=10, C=1),
            init=dict(v=-65),
            module_name="oneliner_lif",
        )
        assert isinstance(neuron, EquationNeuron)
        assert "module oneliner_lif" in verilog
        # Python neuron still works
        spikes = sum(neuron.step(I=30.0) for _ in range(200))
        assert spikes > 0

    def test_multi_eq_one_liner(self):
        neuron, verilog = equation_to_fpga(
            "dv/dt = -(v - v_rest) / tau + I",
            "dw/dt = (v - w) / tau_w",
            params={"v_rest": 0.0, "tau": 10.0, "tau_w": 100.0},
            init={"v": 0.0, "w": 0.0},
            module_name="two_var",
        )
        assert "v_reg" in verilog
        assert "w_reg" in verilog

    def test_strict_units_one_liner_export_path(self):
        neuron, verilog = equation_to_fpga(
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
            module_name="strict_oneliner_lif",
            fraction=16,
            units="strict",
            input_unit=1.0 * UNIT_REGISTRY.nanoampere,
        )
        assert "module strict_oneliner_lif" in verilog
        assert "P_V_THRESHOLD" in verilog
        assert str(neuron.get_state()["v"].units) == "millivolt"
