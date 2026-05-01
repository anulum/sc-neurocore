# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851

"""Tests for Wave 3: new profiles, pipeline analysis, power est, multi-target."""

import pytest


# ═══════════════════════════════════════════════════════════════════════
# A. New Profiles (Wave 1b: 7 more → 84 total)
# ═══════════════════════════════════════════════════════════════════════


class TestWave1bProfiles:
    """Verify the 7 additional profiles from §1C/1D."""

    @pytest.mark.parametrize(
        "name",
        [
            "qualcomm_nsp",
            "sambanova",
            "cambricon_mlu",
            "superconducting",
            "cim_sram",
            "analog_ai",
            "event_camera",
        ],
    )
    def test_profile_exists(self, name):
        """Profile is registered and retrievable."""
        from sc_neurocore.compiler.hardware_profiles import get_profile

        p = get_profile(name)
        assert p.name == name
        assert p.data_width > 0

    def test_total_profiles_at_least_84(self):
        """Registry should now have ≥84 profiles."""
        from sc_neurocore.compiler.hardware_profiles import list_profiles

        assert len(list_profiles()) >= 84

    def test_superconducting_is_emerging(self):
        """Superconducting is in the emerging class."""
        from sc_neurocore.compiler.hardware_profiles import get_profile

        p = get_profile("superconducting")
        assert p.platform_class == "emerging"
        assert p.overflow == "wrap"

    def test_event_camera_matches_dvs(self):
        """Event camera profile matches DVS sensor specs."""
        from sc_neurocore.compiler.hardware_profiles import get_profile

        p = get_profile("event_camera")
        assert p.vendor == "Prophesee/Sony"
        assert p.data_width == 16


# ═══════════════════════════════════════════════════════════════════════
# B. Pipeline Stage Analysis
# ═══════════════════════════════════════════════════════════════════════


class TestPipelineAnalysis:
    """Tests for critical path depth and pipeline budget."""

    def test_no_multiply(self):
        """Pure addition has zero depth."""
        from sc_neurocore.compiler.static_analysis import critical_path_depth

        assert critical_path_depth("a + b + c") == 0

    def test_single_multiply(self):
        """Single multiply has depth 1."""
        from sc_neurocore.compiler.static_analysis import critical_path_depth

        assert critical_path_depth("a * b") == 1

    def test_chained_multiply(self):
        """Chained a * b * c has depth 2."""
        from sc_neurocore.compiler.static_analysis import critical_path_depth

        assert critical_path_depth("a * b * c") == 2

    def test_deep_chain(self):
        """a * b * c * d has depth 3."""
        from sc_neurocore.compiler.static_analysis import critical_path_depth

        assert critical_path_depth("a * b * c * d") == 3

    def test_mixed(self):
        """a * b + c * d: both branches have depth 1."""
        from sc_neurocore.compiler.static_analysis import critical_path_depth

        assert critical_path_depth("a * b + c * d") == 1

    def test_divide_counts(self):
        """Division counts as multiplicative depth."""
        from sc_neurocore.compiler.static_analysis import critical_path_depth

        assert critical_path_depth("a / b") == 1

    def test_no_pipeline_needed_slow(self):
        """No pipeline at 100 MHz with depth 1."""
        from sc_neurocore.compiler.static_analysis import pipeline_stages_needed

        assert pipeline_stages_needed(1, 100) == 0

    def test_pipeline_needed_fast(self):
        """Pipeline needed at 900 MHz with depth 4."""
        from sc_neurocore.compiler.static_analysis import pipeline_stages_needed

        stages = pipeline_stages_needed(4, 900)
        assert stages >= 1  # 4 × 3.0 ns = 12 ns > 1.11 ns period

    def test_pipeline_zero_depth(self):
        """Zero depth → zero stages."""
        from sc_neurocore.compiler.static_analysis import pipeline_stages_needed

        assert pipeline_stages_needed(0, 900) == 0

    def test_pipeline_analysis_multi(self):
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


# ═══════════════════════════════════════════════════════════════════════
# C. Power Estimation
# ═══════════════════════════════════════════════════════════════════════


class TestPowerEstimation:
    """Tests for compile-time power estimation."""

    def test_basic_power(self):
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

    def test_higher_freq_more_power(self):
        """Higher frequency = more dynamic power."""
        from sc_neurocore.compiler.static_analysis import estimate_power

        v = "reg signed [15:0] v_reg; wire signed [31:0] _mul0 = a * b;"
        p100 = estimate_power(v, freq_mhz=100)
        p500 = estimate_power(v, freq_mhz=500)
        assert p500.dynamic_mw > p100.dynamic_mw

    def test_energy_per_spike(self):
        """Energy per spike is computed from power and rate."""
        from sc_neurocore.compiler.static_analysis import estimate_power

        v = "reg signed [15:0] v_reg; wire signed [31:0] _mul0 = a * b;"
        p = estimate_power(v, spike_rate_hz=100.0)
        assert p.energy_per_spike_nj > 0

    def test_different_process(self):
        """Smaller process = less capacitance = less dynamic power."""
        from sc_neurocore.compiler.static_analysis import estimate_power

        v = "reg signed [15:0] v_reg; wire signed [31:0] _mul0 = a * b;"
        p28 = estimate_power(v, process_nm=28)
        p7 = estimate_power(v, process_nm=7)
        assert p7.dynamic_mw < p28.dynamic_mw

    def test_empty_verilog(self):
        """Empty Verilog produces near-zero power."""
        from sc_neurocore.compiler.static_analysis import estimate_power

        p = estimate_power("")
        assert p.total_mw == 0


# ═══════════════════════════════════════════════════════════════════════
# D. Multi-Target Compilation
# ═══════════════════════════════════════════════════════════════════════


class TestMultiTarget:
    """Tests for multi-target --compare compilation."""

    def test_basic_multi_target(self):
        """Compile LIF to 3 targets and get results."""
        from sc_neurocore.compiler.deployment import compile_multi_target

        results = compile_multi_target(
            {"v": "-(v - v_rest) / tau + R * I"},
            ["artix7", "loihi2", "asic_16"],
        )
        assert len(results) == 3
        targets = [r.target for r in results]
        assert "artix7" in targets
        assert "loihi2" in targets
        assert "asic_16" in targets

    def test_data_widths_differ(self):
        """Different targets have different data widths."""
        from sc_neurocore.compiler.deployment import compile_multi_target

        results = compile_multi_target(
            {"v": "a * b + c"},
            ["artix7", "loihi2"],
        )
        r_map = {r.target: r for r in results}
        assert r_map["artix7"].data_width != r_map["loihi2"].data_width

    def test_guard_bits_consistent(self):
        """Guard bits should be same for all targets (expression-dependent)."""
        from sc_neurocore.compiler.deployment import compile_multi_target

        results = compile_multi_target(
            {"v": "a + b + c + d"},
            ["artix7", "ice40", "ecp5"],
        )
        guards = [r.guard_bits for r in results]
        assert len(set(guards)) == 1  # All same

    def test_format_comparison_table(self):
        """Table formatter produces markdown."""
        from sc_neurocore.compiler.deployment import (
            compile_multi_target,
            format_comparison_table,
        )

        results = compile_multi_target(
            {"v": "a * b + c"},
            ["artix7", "ice40"],
        )
        table = format_comparison_table(results)
        assert "| Target" in table
        assert "artix7" in table
        assert "ice40" in table

    def test_single_target(self):
        """Single target still works."""
        from sc_neurocore.compiler.deployment import compile_multi_target

        results = compile_multi_target(
            {"v": "a + b"},
            ["artix7"],
        )
        assert len(results) == 1
        assert results[0].target == "artix7"

    def test_dsp_allocation(self):
        """Targets with DSP blocks allocate multipliers to DSPs."""
        from sc_neurocore.compiler.deployment import compile_multi_target

        results = compile_multi_target(
            {"v": "a * b * c"},
            ["artix7"],  # has DSP48E1
        )
        assert results[0].estimated_dsps > 0
