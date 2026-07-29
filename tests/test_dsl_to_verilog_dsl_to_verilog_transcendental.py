# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDSLToVerilogTranscendental from former test_dsl_to_verilog.py

"""Focused suite: TestDSLToVerilogTranscendental from former test_dsl_to_verilog.py."""

from __future__ import annotations

from tests.dsl_to_verilog_support import *  # noqa: F403


class TestDSLToVerilogTranscendental:
    """Test models with transcendental functions (exp, tanh via LUT)."""

    @pytest.mark.parametrize("model_name", _TRANSCENDENTAL_MODELS)
    def test_transcendental_model_compiles(self, model_name: str) -> None:
        """Models with exp/tanh should compile using LUT approximations."""
        neuron = UniversalNeuron.from_schema(model_name)
        verilog = neuron.to_verilog()
        assert "module sc_" in verilog
        assert "endmodule" in verilog
