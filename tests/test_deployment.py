# SPDX-License-Identifier: AGPL-3.0-or-later
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SC-NeuroCore — Tests for deployment utilities

"""Tests for resource estimation, constraint gen, driver gen, Cocotb gen."""

from __future__ import annotations

import pytest

from sc_neurocore.compiler.deployment import (
    ResourceEstimate,
    estimate_resources,
    generate_cocotb_testbench,
    generate_constraints,
    generate_host_driver,
)

# Minimal Verilog stub for resource estimation
STUB_VERILOG = """
module sc_lif (input wire clk, input wire rst, input wire en,
               input wire signed [15:0] I_t, output wire spike_out);
    reg signed [15:0] v_reg;
    reg signed [15:0] v_rest;
    wire signed [31:0] _mul0 = v_reg * I_t;
    wire signed [31:0] _mul1 = v_rest * v_reg;
    wire signed [15:0] _t0 = (_mul0 >>> 8);
    wire signed [16:0] v_raw = v_reg + _t0 - v_rest;
    wire signed [15:0] v_next =
        (v_raw > 17'sd32767) ? 16'sd32767 :
        (v_raw < (-17'sd32768)) ? (-16'sd32768) :
        v_raw[15:0];
    assign spike_out = (v_next > 16'sd7680);
endmodule
"""

LIF_PARAMS = {"P_V_REST": 16, "P_V_THRESH": 16, "P_TAU_M": 16}


# ═══════════════════════════════════════════════════════════════════════
# Resource Estimation
# ═══════════════════════════════════════════════════════════════════════

class TestResourceEstimation:
    """Test FPGA resource estimation."""

    def test_counts_multipliers(self) -> None:
        """Should detect 2 multipliers in the stub."""
        est = estimate_resources(STUB_VERILOG)
        assert est.mul_count == 2

    def test_counts_additions(self) -> None:
        """Should detect additions and subtractions."""
        est = estimate_resources(STUB_VERILOG)
        assert est.add_count >= 2

    def test_dsps_with_dsp_blocks(self) -> None:
        """With DSP blocks, multipliers map to DSPs not LUTs."""
        est = estimate_resources(STUB_VERILOG, has_dsp=True)
        assert est.dsps == 2
        # No LUTs for multiplies
        est_no_dsp = estimate_resources(STUB_VERILOG, has_dsp=False)
        assert est_no_dsp.dsps == 0
        assert est_no_dsp.luts > est.luts

    def test_register_bits(self) -> None:
        """Should count register bits."""
        est = estimate_resources(STUB_VERILOG)
        assert est.reg_bits > 0

    def test_bram_zero_single_neuron(self) -> None:
        """Single neuron should use 0 BRAM."""
        est = estimate_resources(STUB_VERILOG)
        assert est.brams == 0


# ═══════════════════════════════════════════════════════════════════════
# Constraint File Generation
# ═══════════════════════════════════════════════════════════════════════

class TestConstraintGen:
    """Test SDC/XDC constraint generation."""

    def test_xdc_format(self) -> None:
        """XDC should contain Xilinx-style create_clock."""
        xdc = generate_constraints("sc_lif", format="xdc", target_freq_mhz=100)
        assert "create_clock" in xdc
        assert "10.000" in xdc  # 100 MHz = 10 ns
        assert "sc_lif" in xdc

    def test_sdc_format(self) -> None:
        """SDC should contain generic timing commands."""
        sdc = generate_constraints("sc_lif", format="sdc", target_freq_mhz=450)
        assert "create_clock" in sdc
        assert "2.222" in sdc  # 450 MHz ≈ 2.222 ns

    def test_io_delays(self) -> None:
        """Should include input and output delays."""
        xdc = generate_constraints("sc_lif", format="xdc")
        assert "set_input_delay" in xdc
        assert "set_output_delay" in xdc
        assert "spike_out" in xdc

    def test_false_path(self) -> None:
        """Reset should be a false path."""
        xdc = generate_constraints("sc_lif", format="xdc")
        assert "set_false_path" in xdc
        assert "rst" in xdc

    def test_custom_freq(self) -> None:
        """Custom frequency should produce correct period."""
        xdc = generate_constraints("sc_lif", format="xdc", target_freq_mhz=200)
        assert "5.000" in xdc  # 200 MHz = 5 ns


# ═══════════════════════════════════════════════════════════════════════
# Host Driver Generation
# ═══════════════════════════════════════════════════════════════════════

class TestHostDriverGen:
    """Test Python and C driver generation."""

    def test_python_driver(self) -> None:
        """Python driver should be a valid class."""
        drv = generate_host_driver("sc_lif", LIF_PARAMS, language="python")
        assert "class ScLifDriver:" in drv
        assert "def enable(self)" in drv
        assert "def set_current(self" in drv
        assert "def get_spike_count(self" in drv

    def test_python_param_setters(self) -> None:
        """Should generate setter for each parameter."""
        drv = generate_host_driver("sc_lif", LIF_PARAMS, language="python")
        assert "def set_v_rest(self" in drv
        assert "def set_v_thresh(self" in drv
        assert "def set_tau_m(self" in drv

    def test_python_register_map(self) -> None:
        """Should include register offsets."""
        drv = generate_host_driver("sc_lif", LIF_PARAMS, language="python")
        assert "REG_CTRL" in drv
        assert "0x00" in drv
        assert "REG_SPIKE_COUNT" in drv

    def test_python_q_encoding(self) -> None:
        """Should include Q-format encode/decode."""
        drv = generate_host_driver("sc_lif", LIF_PARAMS, language="python")
        assert "encode_q" in drv
        assert "decode_q" in drv

    def test_c_driver(self) -> None:
        """C driver should be a valid header."""
        drv = generate_host_driver("sc_lif", LIF_PARAMS, language="c")
        assert "#ifndef SC_LIF_DRIVER_H" in drv
        assert "#define SC_LIF_DRIVER_H" in drv
        assert "mmio_write" in drv
        assert "mmio_read" in drv

    def test_c_encode(self) -> None:
        """C driver should have Q-format encoding."""
        drv = generate_host_driver("sc_lif", LIF_PARAMS, language="c")
        assert "encode_q" in drv

    def test_c_functions(self) -> None:
        """C driver should have enable/reset/set_current."""
        drv = generate_host_driver("sc_lif", LIF_PARAMS, language="c")
        assert "sc_lif_enable" in drv
        assert "sc_lif_reset" in drv
        assert "sc_lif_set_current" in drv
        assert "sc_lif_get_spikes" in drv

    def test_custom_base_address(self) -> None:
        """Base address should be configurable."""
        drv = generate_host_driver(
            "sc_lif", LIF_PARAMS, language="python", base_address=0x8000_0000,
        )
        assert "80000000" in drv

    def test_invalid_language(self) -> None:
        """Should raise on invalid language."""
        with pytest.raises(ValueError, match="Unsupported language"):
            generate_host_driver("sc_lif", LIF_PARAMS, language="rust")  # type: ignore


# ═══════════════════════════════════════════════════════════════════════
# Cocotb Testbench Generation
# ═══════════════════════════════════════════════════════════════════════

class TestCocotbGen:
    """Test Cocotb testbench generation."""

    def test_generates_testbench(self) -> None:
        """Should produce valid Cocotb Python code."""
        tb = generate_cocotb_testbench("sc_lif")
        assert "import cocotb" in tb
        assert "@cocotb.test()" in tb

    def test_spike_test(self) -> None:
        """Should include a spike detection test."""
        tb = generate_cocotb_testbench("sc_lif")
        assert "test_sc_lif_spikes" in tb
        assert "spike_count" in tb

    def test_zero_current_test(self) -> None:
        """Should include a zero-current no-spike test."""
        tb = generate_cocotb_testbench("sc_lif")
        assert "test_sc_lif_no_spike_zero_current" in tb

    def test_reset_test(self) -> None:
        """Should include a reset test."""
        tb = generate_cocotb_testbench("sc_lif")
        assert "test_sc_lif_reset_clears_state" in tb

    def test_clock_setup(self) -> None:
        """Should set up a clock."""
        tb = generate_cocotb_testbench("sc_lif")
        assert "Clock(dut.clk" in tb

    def test_custom_params(self) -> None:
        """Custom step count and current should be reflected."""
        tb = generate_cocotb_testbench("sc_lif", n_steps=500, input_current=100.0)
        assert "500" in tb
        assert "25600" in tb  # 100.0 * 256 = 25600

    def test_custom_module_name(self) -> None:
        """Custom module name should propagate."""
        tb = generate_cocotb_testbench("sc_izh_loihi")
        assert "test_sc_izh_loihi_spikes" in tb
