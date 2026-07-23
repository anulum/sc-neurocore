# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPipelineFillCounter from former test_pipeline_stages.py

"""Focused suite: TestPipelineFillCounter from former test_pipeline_stages.py."""

from __future__ import annotations

from tests.pipeline_stages_support import *  # noqa: F403

class TestPipelineFillCounter:
    """The fill-counter FSM that keeps a pipelined self-recurrent step bit-true."""

    def test_pipeline_regs_reset_to_zero(self, lif_neuron):
        """Staging registers must reset to 0 (else X propagates into the state feedback)."""
        v = compile_to_verilog(lif_neuron, pipeline_stages=1)
        staging = v.split("Pipeline register stage")[1]
        assert "if (!rst_n) begin" in staging
        assert "<= 0;" in staging

    def test_fill_counter_and_valid_present(self, lif_neuron):
        """Pipelined modules carry a fill counter and a valid strobe gating the state advance."""
        v = compile_to_verilog(lif_neuron, pipeline_stages=1)
        assert "_pl_cnt" in v
        assert "_pl_valid" in v
        assert "if (_pl_valid) begin" in v

    def test_no_fill_counter_when_not_pipelined(self, lif_neuron):
        """pipeline_stages=0 must emit no fill counter (byte-for-byte the combinational path)."""
        v = compile_to_verilog(lif_neuron, pipeline_stages=0)
        assert "_pl_cnt" not in v
        assert "_pl_valid" not in v

    def test_valid_period_matches_latency(self, izhikevich_neuron):
        """The valid strobe compares the counter against the reported latency (period = lat+1)."""
        v = compile_to_verilog(izhikevich_neuron, pipeline_stages=1)
        import re

        lat = int(re.search(r"Pipeline latency: (\d+) cycle", v).group(1))
        assert lat > 0
        assert "_pl_valid = (_pl_cnt == " in v and f"'d{lat});" in v
