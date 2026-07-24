# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNeuronKernelC from former test_bit_true_kernel.py

"""Focused suite: TestNeuronKernelC from former test_bit_true_kernel.py."""

from __future__ import annotations

from tests.bit_true_kernel_support import *  # noqa: F403


class TestNeuronKernelC:
    def test_reset_and_step(self) -> None:
        code = generate_bittrue_kernel_from_neuron(_lif(), "sc_lif")
        assert "sc_lif_reset(sc_lif_state_t *s)" in code
        assert "int sc_lif_step(sc_lif_state_t *s, int16_t I_t)" in code

    def test_bit_identical_claim_present(self) -> None:
        code = generate_bittrue_kernel_from_neuron(_lif(), "sc_lif")
        assert "Bit-identical to compile_to_verilog" in code

    def test_threshold_and_spike_sequencing(self) -> None:
        code = generate_bittrue_kernel_from_neuron(_lif(), "sc_lif")
        # On a spike, state and output expose the same post-reset value.
        assert "s->v = _rst_v;" in code
        assert "s->v_out = _rst_v;" in code
        assert "int _spk" in code and "return _spk;" in code

    def test_reset_rule_lowered(self) -> None:
        code = generate_bittrue_kernel_from_neuron(_lif(), "sc_lif")
        assert "_rst_v = sat(" in code

    def test_no_threshold_branch(self) -> None:
        neuron = from_equations("dv/dt = -v + I", init=dict(v=0.0), dt=1.0)
        code = generate_bittrue_kernel_from_neuron(neuron, "sc_leak")
        assert "return 0;" in code and "spike_out = 0;" in code
        assert "_spk" not in code

    def test_multi_var_with_two_resets(self) -> None:
        code = generate_bittrue_kernel_from_neuron(
            _adaptive_reset_neuron(), "sc_izh", data_width=32, fraction=16
        )
        assert "_rst_v = sat(" in code and "_rst_u = sat(" in code
        reset_u = next(line for line in code.splitlines() if "_rst_u = sat(" in line)
        assert "_next_u" in reset_u
        assert "s->u = _rst_u;" in code and "s->u_out = _rst_u;" in code
        assert "int32_t v;" in code and "int32_t u;" in code

    def test_map_commit_and_previous_state_threshold(self) -> None:
        """Map kernels commit f(state) directly and keep aliases on the register."""
        code = generate_bittrue_kernel_from_neuron(
            _wrapped_phase_neuron(), "sc_phase", data_width=32, fraction=16
        )
        next_line = next(line for line in code.splitlines() if "_next_theta =" in line)
        spike_line = next(line for line in code.splitlines() if "int _spk =" in line)
        assert "fxmod(" in next_line
        assert "+ fxmul" not in next_line
        assert "s->theta" in spike_line
        assert "_next_theta" not in spike_line
