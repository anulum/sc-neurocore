# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestE2EMultiType from former test_nir_fpga_pipeline.py

"""Focused suite: TestE2EMultiType from former test_nir_fpga_pipeline.py."""

from __future__ import annotations

from tests.nir_fpga_pipeline_support import *  # noqa: F403

class TestE2EMultiType:
    """Network mixing IF and LIF neurons → two distinct Verilog modules."""

    def test_mixed_if_lif(self):
        graph = _build_mixed_type_network()
        result = _full_pipeline(graph, module_name="mixed_net")

        # Must generate two distinct neuron types
        assert "if" in result.neuron_modules
        assert "lif" in result.neuron_modules
        assert len(result.neuron_modules) == 2

        # Each module should have different ODE structures
        if_v = result.neuron_modules["if"]
        lif_v = result.neuron_modules["lif"]

        # Both must be valid Verilog
        assert "module sc_nir_if" in if_v
        assert "module sc_nir_lif" in lif_v
        assert "endmodule" in if_v
        assert "endmodule" in lif_v

        # Top module must reference both types
        assert "sc_nir_if" in result.top_module
        assert "sc_nir_lif" in result.top_module

    def test_mixed_type_resource_counts(self):
        graph = _build_mixed_type_network(n_in=4)
        result = _full_pipeline(graph)

        # IF layer: 6 neurons, LIF layer: 3 neurons
        assert result.total_neurons == 9
        # Weights: 4×6 + 6×3 = 24 + 18 = 42
        assert result.total_synapses == 42
