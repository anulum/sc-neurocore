# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPowerEstimation from former test_compiler_static_analysis.py

"""Focused suite: TestPowerEstimation from former test_compiler_static_analysis.py."""

from __future__ import annotations

from tests.compiler_static_analysis_support import *  # noqa: F403

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
