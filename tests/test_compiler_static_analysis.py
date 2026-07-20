# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Compiler static-analysis contracts

"""Contracts for compiler pipeline and power static analysis."""

from __future__ import annotations

import pytest


class TestPipelineAnalysis:
    """Tests for critical path depth and pipeline budget."""

    def test_no_multiply(self) -> None:
        """Pure addition has zero depth."""
        from sc_neurocore.compiler.static_analysis import critical_path_depth

        assert critical_path_depth("a + b + c") == 0

    def test_single_multiply(self) -> None:
        """Single multiply has depth 1."""
        from sc_neurocore.compiler.static_analysis import critical_path_depth

        assert critical_path_depth("a * b") == 1

    def test_chained_multiply(self) -> None:
        """Chained a * b * c has depth 2."""
        from sc_neurocore.compiler.static_analysis import critical_path_depth

        assert critical_path_depth("a * b * c") == 2

    def test_deep_chain(self) -> None:
        """a * b * c * d has depth 3."""
        from sc_neurocore.compiler.static_analysis import critical_path_depth

        assert critical_path_depth("a * b * c * d") == 3

    def test_mixed(self) -> None:
        """a * b + c * d: both branches have depth 1."""
        from sc_neurocore.compiler.static_analysis import critical_path_depth

        assert critical_path_depth("a * b + c * d") == 1

    def test_divide_counts(self) -> None:
        """Division counts as multiplicative depth."""
        from sc_neurocore.compiler.static_analysis import critical_path_depth

        assert critical_path_depth("a / b") == 1

    def test_no_pipeline_needed_slow(self) -> None:
        """No pipeline at 100 MHz with depth 1."""
        from sc_neurocore.compiler.static_analysis import pipeline_stages_needed

        assert pipeline_stages_needed(1, 100) == 0

    def test_pipeline_needed_fast(self) -> None:
        """Pipeline needed at 900 MHz with depth 4."""
        from sc_neurocore.compiler.static_analysis import pipeline_stages_needed

        stages = pipeline_stages_needed(4, 900)
        assert stages >= 1  # 4 × 3.0 ns = 12 ns > 1.11 ns period

    def test_pipeline_zero_depth(self) -> None:
        """Zero depth → zero stages."""
        from sc_neurocore.compiler.static_analysis import pipeline_stages_needed

        assert pipeline_stages_needed(0, 900) == 0

    def test_pipeline_analysis_multi(self) -> None:
        """Multi-ODE pipeline analysis."""
        from sc_neurocore.compiler.static_analysis import pipeline_analysis

        result = pipeline_analysis(
            {"v": "a * b * c + d", "w": "e + f"},
            target_freq_mhz=500,
        )
        assert result["v"]["depth"] == 2
        assert result["w"]["depth"] == 0
        assert result["w"]["stages"] == 0
        assert "achievable_mhz" in result["v"]


class TestPowerEstimation:
    """Tests for compile-time power estimation."""

    def test_basic_power(self) -> None:
        """Basic LIF-like Verilog produces non-zero power."""
        from sc_neurocore.compiler.static_analysis import estimate_power

        verilog = """
        reg signed [15:0] v_reg;
        wire signed [31:0] _mul0 = a * b;
        wire signed [15:0] _t0 = a + b - c;
        """
        p = estimate_power(verilog)
        assert p.dynamic_mw >= 0
        assert p.static_mw >= 0
        assert p.total_mw > 0
        assert p.toggle_rate >= 0

    def test_higher_freq_more_power(self) -> None:
        """Higher frequency = more dynamic power."""
        from sc_neurocore.compiler.static_analysis import estimate_power

        v = "reg signed [15:0] v_reg; wire signed [31:0] _mul0 = a * b;"
        p100 = estimate_power(v, freq_mhz=100)
        p500 = estimate_power(v, freq_mhz=500)
        assert p500.dynamic_mw > p100.dynamic_mw

    def test_energy_per_spike(self) -> None:
        """Energy per spike is computed from power and rate."""
        from sc_neurocore.compiler.static_analysis import estimate_power

        v = "reg signed [15:0] v_reg; wire signed [31:0] _mul0 = a * b;"
        p = estimate_power(v, spike_rate_hz=100.0)
        assert p.energy_per_spike_nj > 0

    def test_different_process(self) -> None:
        """Smaller process = less capacitance = less dynamic power."""
        from sc_neurocore.compiler.static_analysis import estimate_power

        v = "reg signed [15:0] v_reg; wire signed [31:0] _mul0 = a * b;"
        p28 = estimate_power(v, process_nm=28)
        p7 = estimate_power(v, process_nm=7)
        assert p7.dynamic_mw < p28.dynamic_mw

    def test_empty_verilog(self) -> None:
        """Empty Verilog produces near-zero power."""
        from sc_neurocore.compiler.static_analysis import estimate_power

        p = estimate_power("")
        assert p.total_mw == 0

    def test_vcd_activity_drives_measured_toggle_rate(self) -> None:
        """VCD switching activity overrides structural default toggles."""
        from sc_neurocore.compiler.static_analysis import estimate_power

        vcd = """
        $timescale 1ns $end
        $scope module top $end
        $var wire 4 ! data [3:0] $end
        $upscope $end
        $enddefinitions $end
        #0
        b0000 !
        #10
        b1111 !
        #20
        b1010 !
        """

        p = estimate_power("", activity_vcd=vcd, vcd_time_units_per_cycle=10)

        assert p.dynamic_mw > 0
        assert p.total_mw == p.dynamic_mw
        assert p.toggle_rate == 0.75

    def test_vcd_activity_rejects_invalid_cycle_scale(self) -> None:
        """VCD-derived activity needs a positive time-unit to cycle scale."""
        from sc_neurocore.compiler.static_analysis import estimate_power

        with pytest.raises(ValueError, match="vcd_time_units_per_cycle"):
            estimate_power("", activity_vcd="$enddefinitions $end", vcd_time_units_per_cycle=0)
