# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDSLToVerilogCompilation from former test_dsl_to_verilog.py

"""Focused suite: TestDSLToVerilogCompilation from former test_dsl_to_verilog.py."""

from __future__ import annotations

from tests.dsl_to_verilog_support import *  # noqa: F403


class TestDSLToVerilogCompilation:
    """Test that schema-loaded models compile to valid Verilog."""

    @pytest.mark.parametrize("model_name", _SIMPLE_MODELS)
    def test_simple_model_compiles(self, model_name: str) -> None:
        """Simple (polynomial) models should compile without errors."""
        neuron = UniversalNeuron.from_schema(model_name)
        verilog = neuron.to_verilog()

        assert "module sc_" in verilog
        assert "endmodule" in verilog
        assert "clk" in verilog
        assert "rst_n" in verilog
        assert "spike_out" in verilog

    @pytest.mark.parametrize("model_name", _SIMPLE_MODELS)
    def test_simple_model_has_state_outputs(self, model_name: str) -> None:
        """Each state variable should have a corresponding output port."""
        neuron = UniversalNeuron.from_schema(model_name)
        verilog = neuron.to_verilog()

        for var in neuron.list_state_variables():
            assert f"{var}_out" in verilog, f"Missing output port for {var}"

    @pytest.mark.parametrize("model_name", _SIMPLE_MODELS)
    def test_simple_model_has_parameters(self, model_name: str) -> None:
        """Parameters should appear as Verilog parameters."""
        neuron = UniversalNeuron.from_schema(model_name)
        verilog = neuron.to_verilog()

        assert "parameter" in verilog

    def test_lif_module_name(self) -> None:
        """LIF should generate a clean module name."""
        neuron = UniversalNeuron.from_schema("lif")
        verilog = neuron.to_verilog()
        assert "module sc_lif" in verilog

    def test_izhikevich_module_name(self) -> None:
        neuron = UniversalNeuron.from_schema("izhikevich")
        verilog = neuron.to_verilog()
        assert "module sc_izhikevich" in verilog

    def test_custom_module_name(self) -> None:
        neuron = UniversalNeuron.from_schema("lif")
        verilog = neuron.to_verilog(module_name="my_custom_lif")
        assert "module my_custom_lif" in verilog

    def test_lif_has_reset_logic(self) -> None:
        neuron = UniversalNeuron.from_schema("lif")
        verilog = neuron.to_verilog()
        # Should have threshold comparison and reset assignment
        assert "spike_out <= 1'b1" in verilog
