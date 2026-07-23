# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestOutputConsistency from former test_pipeline_stages.py

"""Focused suite: TestOutputConsistency from former test_pipeline_stages.py."""

from __future__ import annotations

from tests.pipeline_stages_support import *  # noqa: F403

class TestOutputConsistency:
    """Verify pipelined vs non-pipelined Verilog is structurally consistent."""

    def test_both_compile_successfully(self, lif_neuron):
        """Both pipelined and non-pipelined must produce valid Verilog."""
        v0 = compile_to_verilog(lif_neuron, pipeline_stages=0)
        v1 = compile_to_verilog(lif_neuron, pipeline_stages=1)
        assert "module" in v0
        assert "module" in v1
        assert "endmodule" in v0
        assert "endmodule" in v1

    def test_module_structure_preserved(self, lif_neuron):
        """Core structure (ports, always block) must be preserved."""
        v = compile_to_verilog(lif_neuron, pipeline_stages=1)
        assert "input wire clk" in v
        assert "input wire rst_n" in v
        assert "output reg spike_out" in v
        assert "always @(posedge clk or negedge rst_n)" in v

    def test_state_vars_preserved(self, izhikevich_neuron):
        """All state variables must appear in pipelined output."""
        v = compile_to_verilog(izhikevich_neuron, pipeline_stages=1)
        assert "v_reg" in v
        assert "u_reg" in v
        assert "v_out" in v
        assert "u_out" in v

    def test_q1616_pipeline(self, lif_neuron):
        """Pipeline at Q16.16 should work with wider intermediates."""
        v = compile_to_verilog(
            lif_neuron,
            data_width=32,
            fraction=16,
            pipeline_stages=1,
        )
        assert "module" in v
        assert "endmodule" in v
        # 64-bit intermediate for 32-bit Q16.16
        assert "[63:0]" in v
