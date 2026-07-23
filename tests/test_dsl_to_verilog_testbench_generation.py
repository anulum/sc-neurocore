# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTestbenchGeneration from former test_dsl_to_verilog.py

"""Focused suite: TestTestbenchGeneration from former test_dsl_to_verilog.py."""

from __future__ import annotations

from tests.dsl_to_verilog_support import *  # noqa: F403

class TestTestbenchGeneration:
    """Test automatic testbench generation."""

    @pytest.mark.parametrize("model_name", _SIMPLE_MODELS)
    def test_testbench_generates(self, model_name: str) -> None:
        neuron = UniversalNeuron.from_schema(model_name)
        eq_neuron = neuron.to_equation_neuron()
        module_name = f"sc_{model_name}"
        tb = generate_testbench(eq_neuron, module_name=module_name)

        assert f"module tb_{module_name}" in tb
        assert "$dumpfile" in tb
        assert "spike_count" in tb
        assert "$finish" in tb

    @pytest.mark.parametrize("model_name", _SIMPLE_MODELS)
    def test_testbench_has_state_monitors(self, model_name: str) -> None:
        neuron = UniversalNeuron.from_schema(model_name)
        eq_neuron = neuron.to_equation_neuron()
        tb = generate_testbench(eq_neuron, module_name=f"sc_{model_name}")

        for var in neuron.list_state_variables():
            assert f"{var}_out" in tb
