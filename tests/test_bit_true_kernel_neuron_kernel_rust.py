# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNeuronKernelRust from former test_bit_true_kernel.py

"""Focused suite: TestNeuronKernelRust from former test_bit_true_kernel.py."""

from __future__ import annotations

from tests.bit_true_kernel_support import *  # noqa: F403

class TestNeuronKernelRust:
    def test_reset_and_step(self) -> None:
        code = generate_bittrue_kernel_from_neuron(_lif(), "sc_lif", language="rust")
        assert "pub fn reset(&mut self)" in code
        assert "pub fn step(&mut self, I_t: i16) -> i32" in code

    def test_threshold_sequencing(self) -> None:
        code = generate_bittrue_kernel_from_neuron(_lif(), "sc_lif", language="rust")
        assert "self.v = _rst_v;" in code
        assert "self.v_out = _rst_v;" in code
        assert "if _spk != 0" in code

    def test_state_dependent_reset_reads_candidate(self) -> None:
        code = generate_bittrue_kernel_from_neuron(
            _adaptive_reset_neuron(),
            "sc_izh",
            data_width=32,
            fraction=16,
            language="rust",
        )
        reset_u = next(line for line in code.splitlines() if "_rst_u:" in line)
        assert "_next_u" in reset_u
        assert "self.u = _rst_u;" in code and "self.u_out = _rst_u;" in code

    def test_no_threshold_branch(self) -> None:
        neuron = from_equations("dv/dt = -v + I", init=dict(v=0.0), dt=1.0)
        code = generate_bittrue_kernel_from_neuron(neuron, "sc_leak", language="rust")
        assert "return 0;" in code

    def test_map_commit_and_previous_state_threshold(self) -> None:
        code = generate_bittrue_kernel_from_neuron(
            _wrapped_phase_neuron(),
            "sc_phase",
            data_width=32,
            fraction=16,
            language="rust",
        )
        next_line = next(line for line in code.splitlines() if "_next_theta:" in line)
        spike_line = next(line for line in code.splitlines() if "let _spk:" in line)
        assert "fxmod(" in next_line
        assert "+ (fxmul" not in next_line
        assert "self.theta" in spike_line
        assert "_next_theta" not in spike_line
