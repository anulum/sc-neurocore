# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTranscendentalStatements from former test_bit_true_kernel.py

"""Focused suite: TestTranscendentalStatements from former test_bit_true_kernel.py."""

from __future__ import annotations

from tests.bit_true_kernel_support import *  # noqa: F403


class TestTranscendentalStatements:
    """Cover the LUT-statement paths (deriv, threshold and table declarations)."""

    def test_simple_rust_lut_and_input(self) -> None:
        code = generate_bittrue_kernel("sc_th", {"v": "tanh(v) + I"}, language="rust")
        assert "const _tanh_lut0: [i16;" in code  # rust table declaration
        assert ", I_t: i16" in code  # input argument
        assert "let _tanh_lut0_arg" in code  # LUT statement inside step

    def _transcendental_neuron(self) -> EquationNeuron:
        return from_equations(
            "dv/dt = 0.1*(exp(v) - v) + I",
            threshold="tanh(v) > 0.5",
            reset="v = 0",
            init=dict(v=0.0),
            dt=0.5,
        )

    def test_neuron_c_deriv_and_threshold_statements(self) -> None:
        code = generate_bittrue_kernel_from_neuron(self._transcendental_neuron(), "sc_tr")
        assert "_exp_lut0_arg" in code  # derivative LUT statement (line 534)
        assert "_tanh_lut" in code  # threshold LUT statement (line 557)

    def test_neuron_rust_deriv_and_threshold_statements(self) -> None:
        code = generate_bittrue_kernel_from_neuron(
            self._transcendental_neuron(), "sc_tr", language="rust"
        )
        assert "let _exp_lut0_arg" in code  # derivative LUT statement (line 609)
        assert "_tanh_lut" in code  # threshold LUT statement (line 634)
