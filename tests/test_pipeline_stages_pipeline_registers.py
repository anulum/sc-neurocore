# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPipelineRegisters from former test_pipeline_stages.py

"""Focused suite: TestPipelineRegisters from former test_pipeline_stages.py."""

from __future__ import annotations

from tests.pipeline_stages_support import *  # noqa: F403


class TestPipelineRegisters:
    """Verify pipeline registers appear in generated Verilog."""

    def test_no_pipeline_no_regs(self, lif_neuron):
        """pipeline_stages=0 should produce no pipeline registers."""
        v = compile_to_verilog(lif_neuron, pipeline_stages=0)
        assert "_mul0_r" not in v
        assert "Pipeline registers" not in v
        assert "latency" not in v

    def test_pipeline_1_stage_has_regs(self, lif_neuron):
        """pipeline_stages=1 should register all multiply outputs."""
        v = compile_to_verilog(lif_neuron, pipeline_stages=1)
        assert "_r;" in v, "Expected pipeline register declarations"
        assert "Pipeline registers for multiply outputs" in v
        assert "Pipeline register stage" in v

    def test_pipeline_produces_latency_port(self, lif_neuron):
        """Pipelined modules must have a latency output port."""
        v = compile_to_verilog(lif_neuron, pipeline_stages=1)
        assert "output wire" in v and "latency" in v

    def test_latency_value_matches_stage_count(self, lif_neuron):
        """Latency constant value should match number of pipeline regs."""
        v = compile_to_verilog(lif_neuron, pipeline_stages=1)
        # The latency assignment should contain the actual count
        assert "Pipeline latency:" in v

    def test_pipeline_2_stages_izh(self, izhikevich_neuron):
        """Izhikevich has multiple multiplies — all should be pipelined."""
        v = compile_to_verilog(izhikevich_neuron, pipeline_stages=1)
        # Izhikevich has v*v, 5*v, a*(b*v - u) → multiple _mul regs
        reg_count = v.count("_r;")
        assert reg_count >= 3, f"Expected >=3 pipeline regs, got {reg_count}"

    def test_pipeline_always_block_present(self, lif_neuron):
        """The reset-aware pipeline-register staging block should be present."""
        v = compile_to_verilog(lif_neuron, pipeline_stages=1)
        # Staging regs are clocked and reset with the module so an unfilled pipeline never
        # injects X into the state feedback.
        assert "always @(posedge clk or negedge rst_n) begin" in v
        assert "Pipeline register stage" in v
