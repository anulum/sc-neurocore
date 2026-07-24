# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestE2EHighPrecision from former test_nir_fpga_pipeline.py

"""Focused suite: TestE2EHighPrecision from former test_nir_fpga_pipeline.py."""

from __future__ import annotations

from tests.nir_fpga_pipeline_support import *  # noqa: F403


class TestE2EHighPrecision:
    """Full pipeline at Q16.16 (32-bit) precision."""

    def test_q16_16_wire_widths(self):
        graph = _build_lif_feedforward()
        result = _full_pipeline(graph, data_width=32, fraction=16, module_name="hd_net")

        assert result.q_format == "Q16.16"

        # Neuron module must use 32-bit wires
        lif_v = result.neuron_modules["lif"]
        assert "[31:0]" in lif_v

        # Top module must use 32-bit data
        assert "localparam integer DATA_WIDTH = 32;" in result.top_module
        assert "input  wire signed [127:0] I_ext_flat" in result.top_module

    def test_q16_16_weight_precision(self):
        graph = _build_lif_feedforward(n_in=2, n_hidden=3, n_out=1)
        result = _full_pipeline(graph, data_width=32, fraction=16)

        # Weight ROM should use 32-bit words
        assert "[31:0]" in result.weight_rom
