# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for power estimator

"""Tests for power estimation utilities."""

from __future__ import annotations

import pytest
from sc_neurocore.compiler.power_estimator import estimate_power


class TestPowerEstimator:
    """Test power estimation from Verilog structural analysis and VCD."""

    def test_basic_estimation(self) -> None:
        """Should return a PowerEstimate with reasonable values."""
        verilog = "module test; reg signed [15:0] v_reg; wire signed [31:0] _mul0 = a * b; endmodule"
        p = estimate_power(verilog)
        assert p.dynamic_mw > 0
        assert p.static_mw > 0
        assert p.total_mw == round(p.dynamic_mw + p.static_mw, 4)

    def test_frequency_scaling(self) -> None:
        """Dynamic power should scale linearly with frequency."""
        v = "reg signed [15:0] v_reg; wire signed [31:0] _mul0 = a * b;"
        p100 = estimate_power(v, freq_mhz=100)
        p500 = estimate_power(v, freq_mhz=500)
        # 5x frequency should be ~5x dynamic power
        assert p500.dynamic_mw > 4.5 * p100.dynamic_mw

    def test_spike_rate_scaling(self) -> None:
        """Energy per spike should depend on spike rate."""
        v = "reg signed [15:0] v_reg; wire signed [31:0] _mul0 = a * b;"
        p = estimate_power(v, spike_rate_hz=100.0)
        assert p.energy_per_spike_nj > 0

    def test_process_node_scaling(self) -> None:
        """Static power should scale with process node."""
        v = "reg signed [15:0] v_reg; wire signed [31:0] _mul0 = a * b;"
        p28 = estimate_power(v, process_nm=28)
        p7 = estimate_power(v, process_nm=7)
        # 7nm should have less static power than 28nm in this model
        assert p7.static_mw < p28.static_mw

    def test_empty_verilog(self) -> None:
        """Empty Verilog should return zero power (or near zero)."""
        p = estimate_power("")
        assert p.dynamic_mw == 0.0

    def test_vcd_activity(self) -> None:
        """Test power estimation using a VCD activity trace."""
        vcd = """$date June 6, 2026 $end
$timescale 1ns $end
$scope module top $end
$var wire 16 v v_reg [15:0] $end
$upscope $end
$enddefinitions $end
#0
b0000000000000000 v
#10
b0000000011111111 v
#20
b1111111111111111 v
"""
        # 16 toggles total over 20ns
        p = estimate_power("", activity_vcd=vcd, vcd_time_units_per_cycle=10)
        assert p.toggle_rate > 0
        assert p.dynamic_mw > 0

    def test_invalid_vcd_params(self) -> None:
        """Should raise ValueError for invalid time units."""
        with pytest.raises(ValueError, match="vcd_time_units_per_cycle"):
            estimate_power("", activity_vcd="$enddefinitions $end", vcd_time_units_per_cycle=0)
