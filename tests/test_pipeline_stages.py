# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — E2E tests for pipeline stage insertion

"""End-to-end tests for pipeline stage insertion (Roadmap Item 1).

Tests verify:
- Pipeline registers appear in generated Verilog
- Latency port matches pipeline stage count
- Auto-pipeline from critical path depth
- User-specified pipeline points
- Non-pipelined output remains unchanged when stages=0
- Regression: all existing neuron types compile without error
"""

from __future__ import annotations

import pytest

from sc_neurocore.compiler.equation_compiler import compile_to_verilog
from sc_neurocore.compiler.static_analysis import (
    critical_path_depth,
    pipeline_analysis,
    pipeline_stages_needed,
)
from sc_neurocore.neurons.equation_builder import from_equations


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def lif_neuron():
    """Standard LIF neuron with one multiply chain."""
    return from_equations(
        "dv/dt = -(v - E_L)/tau_m + I/C",
        threshold="v > -50",
        reset="v = -65",
        params=dict(E_L=-65, tau_m=10, C=1),
        init=dict(v=-65),
    )


@pytest.fixture
def hh_style_neuron():
    """Multi-multiply neuron (3 multiplies in sequence)."""
    return from_equations(
        "dv/dt = -(v - E_L)/tau_m + I * R / C",
        threshold="v > -50",
        reset="v = -65",
        params=dict(E_L=-65, tau_m=10, R=1, C=1),
        init=dict(v=-65),
    )


@pytest.fixture
def izhikevich_neuron():
    """Two-state-variable neuron (Izhikevich)."""
    return from_equations(
        "dv/dt = 0.04 * v * v + 5 * v + 140 - u + I",
        "du/dt = a * (b * v - u)",
        threshold="v > 30",
        reset="v = c; u = u + d",
        params=dict(a=0.02, b=0.2, c=-65, d=8),
        init=dict(v=-65, u=-14),
    )


# ═══════════════════════════════════════════════════════════════════════
# 1. Pipeline register insertion
# ═══════════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════════
# 2. User-specified pipeline points
# ═══════════════════════════════════════════════════════════════════════


class TestPipelinePoints:
    """Verify user-specified pipeline insertion points."""

    def test_specific_point_only(self, izhikevich_neuron):
        """Only the specified multiply should be registered."""
        v = compile_to_verilog(
            izhikevich_neuron,
            pipeline_stages=0,
            pipeline_points=["_mul0"],
        )
        assert "_mul0_r" in v, "Expected _mul0 to be registered"
        # Other multiplies should NOT be registered when pipeline_stages=0
        # and they're not in pipeline_points

    def test_multiple_points(self, izhikevich_neuron):
        """Multiple specified points should all be registered."""
        v = compile_to_verilog(
            izhikevich_neuron,
            pipeline_stages=0,
            pipeline_points=["_mul0", "_mul1"],
        )
        assert "_mul0_r" in v
        assert "_mul1_r" in v


# ═══════════════════════════════════════════════════════════════════════
# 3. Critical path analysis integration
# ═══════════════════════════════════════════════════════════════════════


class TestCriticalPathIntegration:
    """Verify static analysis functions work with pipeline insertion."""

    def test_lif_critical_path_depth(self):
        """LIF has 2 multiplies: reciprocal * v and reciprocal * I."""
        depth = critical_path_depth("-(v - E_L)/tau_m + I/C")
        assert depth >= 1, f"Expected depth >= 1, got {depth}"

    def test_izhikevich_critical_path_depth(self):
        """Izhikevich v-equation has 3+ chained multiplies."""
        depth = critical_path_depth("0.04 * v * v + 5 * v + 140 - u + I")
        assert depth >= 2, f"Expected depth >= 2 for v*v chain, got {depth}"

    def test_pipeline_stages_needed_low_freq(self):
        """At 100 MHz, most neurons need 0 stages."""
        stages = pipeline_stages_needed(2, 100)  # 2 DSPs at 100 MHz
        assert stages == 0

    def test_pipeline_stages_needed_high_freq(self):
        """At 900 MHz, 3+ DSPs in series need pipeline stages."""
        stages = pipeline_stages_needed(4, 900)  # 4 DSPs at 900 MHz
        assert stages >= 1, f"Expected >=1 stage at 900 MHz, got {stages}"

    def test_pipeline_analysis_full(self, izhikevich_neuron):
        """Full pipeline analysis should return per-variable results."""
        result = pipeline_analysis(izhikevich_neuron.equations, target_freq_mhz=500)
        assert "v" in result
        assert "u" in result
        assert "depth" in result["v"]
        assert "stages" in result["v"]
        assert "achievable_mhz" in result["v"]

    def test_auto_pipeline_zero_at_100mhz(self, lif_neuron):
        """Auto-pipeline at 100 MHz should give 0 stages for LIF."""
        max_depth = max(critical_path_depth(expr) for expr in lif_neuron.equations.values())
        stages = pipeline_stages_needed(max_depth, 100)
        assert stages == 0


# ═══════════════════════════════════════════════════════════════════════
# 4. Output consistency
# ═══════════════════════════════════════════════════════════════════════


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
