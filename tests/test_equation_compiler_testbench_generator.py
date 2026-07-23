# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTestbenchGenerator from former test_equation_compiler.py

"""Focused suite: TestTestbenchGenerator from former test_equation_compiler.py."""

from __future__ import annotations

from tests.equation_compiler_support import *  # noqa: F403

class TestTestbenchGenerator:
    def test_generates_testbench(self):
        from sc_neurocore.compiler.equation_compiler import generate_testbench

        neuron = from_equations(
            "dv/dt = -(v - E_L)/tau_m + I/C",
            threshold="v > -50",
            reset="v = -65",
            params=dict(E_L=-65, tau_m=10, C=1),
            init=dict(v=-65),
        )
        tb = generate_testbench(neuron, module_name="test_lif", n_steps=100, input_current=2.0)
        assert "module tb_test_lif" in tb
        assert "endmodule" in tb
        assert "$dumpfile" in tb
        assert "$dumpvars" in tb
        assert "spike_count" in tb
        assert "100" in tb
        assert "v_out" in tb
        assert "uut" in tb

    def test_testbench_multi_variable(self):
        from sc_neurocore.compiler.equation_compiler import generate_testbench

        neuron = EquationNeuron(
            equations={"v": "I - w", "w": "0.01 * v"},
            state={"v": 0.0, "w": 0.0},
            dt=0.1,
        )
        tb = generate_testbench(neuron, module_name="two_var_tb")
        assert "v_out" in tb
        assert "w_out" in tb

    def test_cycles_per_step_scales_run_length(self):
        """A latency-aware testbench runs n_steps × cycles_per_step clocks for a pipelined DUT."""
        from sc_neurocore.compiler.equation_compiler import generate_testbench

        neuron = from_equations(
            "dv/dt = -(v - E_L)/tau_m + I/C",
            threshold="v > -50",
            reset="v = -65",
            params=dict(E_L=-65, tau_m=10, C=1),
            init=dict(v=-65),
        )
        tb1 = generate_testbench(neuron, module_name="lif_p1", n_steps=100, cycles_per_step=1)
        tb3 = generate_testbench(neuron, module_name="lif_p3", n_steps=100, cycles_per_step=3)
        assert "repeat (100)" in tb1
        assert "repeat (300)" in tb3
        assert "spikes in 300 cycles" in tb3

    def test_cycles_per_step_must_be_positive(self):
        """cycles_per_step below 1 is rejected rather than emitting a zero-length run."""
        import pytest

        from sc_neurocore.compiler.equation_compiler import generate_testbench

        neuron = EquationNeuron(equations={"v": "I"}, state={"v": 0.0}, dt=0.1)
        with pytest.raises(ValueError, match="cycles_per_step must be >= 1"):
            generate_testbench(neuron, module_name="bad", cycles_per_step=0)
