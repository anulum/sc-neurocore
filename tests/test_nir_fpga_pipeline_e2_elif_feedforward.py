# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestE2ELIFFeedforward from former test_nir_fpga_pipeline.py

"""Focused suite: TestE2ELIFFeedforward from former test_nir_fpga_pipeline.py."""

from __future__ import annotations

from tests.nir_fpga_pipeline_support import *  # noqa: F403

class TestE2ELIFFeedforward:
    """Full pipeline: Input→Affine→LIF→Affine→LIF→Output → Verilog."""

    def test_pipeline_produces_valid_artefacts(self):
        graph = _build_lif_feedforward()
        result = _full_pipeline(graph, module_name="lif_ff")

        # Neuron modules: only 1 type (LIF), compiled once
        assert "lif" in result.neuron_modules
        assert len(result.neuron_modules) == 1

        # Top module exists and has correct module name
        assert "module lif_ff" in result.top_module
        assert "endmodule" in result.top_module

        # Weight ROM exists and contains entries
        assert "module sc_nir_weight_rom" in result.weight_rom
        assert "endmodule" in result.weight_rom

    def test_neuron_module_contains_ode(self):
        graph = _build_lif_feedforward()
        result = _full_pipeline(graph)

        lif_v = result.neuron_modules["lif"]
        # Must contain state register (v_reg)
        assert "v_reg" in lif_v
        # Must contain spike output
        assert "spike_out" in lif_v
        # Must contain rst_n (reset)
        assert "rst_n" in lif_v
        # Must contain clk
        assert "clk" in lif_v

    def test_weight_rom_has_correct_entries(self):
        graph = _build_lif_feedforward(n_in=4, n_hidden=8, n_out=2)
        result = _full_pipeline(graph)

        # Total weights: 4×8 + 8×2 = 32 + 16 = 48
        assert result.total_synapses == 48
        # ROM should have 48 case entries
        case_entries = re.findall(r"\d+'d\d+:", result.weight_rom)
        # +1 for default
        assert len(case_entries) >= 48

    def test_top_module_instantiates_populations(self):
        graph = _build_lif_feedforward()
        result = _full_pipeline(graph, module_name="ff_net")

        # Must instantiate neuron modules
        assert "sc_nir_lif" in result.top_module
        # Must instantiate one RTL neuron per biological/NIR neuron
        assert "p0_n0_inst" in result.top_module
        assert "p1_n1_inst" in result.top_module

    def test_top_module_preserves_input_vector_and_spike_width(self):
        graph = _build_lif_feedforward(n_in=4, n_hidden=8, n_out=2)
        result = _full_pipeline(graph, module_name="ff_net")

        assert "input  wire signed [63:0] I_ext_flat" in result.top_module
        assert "output wire [9:0] spike_bus" in result.top_module

    def test_resource_counts(self):
        graph = _build_lif_feedforward(n_in=4, n_hidden=8, n_out=2)
        result = _full_pipeline(graph)

        assert result.total_neurons == 10
        assert result.total_synapses == 48
        assert result.q_format == "Q8.8"
        assert result.interconnect == "direct"  # 10 ≤ 64
