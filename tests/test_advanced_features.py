# SPDX-License-Identifier: AGPL-3.0-or-later
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SC-NeuroCore — Tests for IP-XACT, VHDL, posit, CDC, TCL, bitstream

"""Tests for advanced features: IP-XACT, VHDL, posit, CDC, TCL, Makefile."""

from __future__ import annotations

import pytest

from sc_neurocore.hdl_gen.ip_xact import generate_ip_xact
from sc_neurocore.compiler.advanced_features import (
    POSIT8_0, POSIT8_1, POSIT16_1, POSIT16_2, PositConfig,
    generate_cdc_synchroniser,
    generate_oss_makefile,
    generate_tcl_project,
    posit_decode,
    posit_encode,
    verilog_to_vhdl_wrapper,
)

LIF_PARAMS = {"P_V_REST": 16, "P_V_THRESH": 16, "P_TAU_M": 16}


# ═══════════════════════════════════════════════════════════════════════
# IP-XACT Tests
# ═══════════════════════════════════════════════════════════════════════

class TestIPXACT:
    """Test IP-XACT XML generation."""

    def test_generates_xml(self) -> None:
        """Should produce valid XML."""
        xml = generate_ip_xact("sc_lif")
        assert "<?xml" in xml
        assert "spirit:component" in xml

    def test_identity(self) -> None:
        """Should contain vendor/library/name/version."""
        xml = generate_ip_xact("sc_lif", vendor="anulum.li", version="2.0")
        assert "anulum.li" in xml
        assert "sc_neurocore" in xml
        assert "sc_lif" in xml
        assert "2.0" in xml

    def test_ports(self) -> None:
        """Should list all neuron ports."""
        xml = generate_ip_xact("sc_lif", data_width=16)
        assert "clk" in xml
        assert "rst" in xml
        assert "I_t" in xml
        assert "spike_out" in xml

    def test_parameters(self) -> None:
        """Should include parameter definitions."""
        xml = generate_ip_xact("sc_lif", params=LIF_PARAMS)
        assert "P_V_REST" in xml
        assert "P_V_THRESH" in xml

    def test_axi_bus(self) -> None:
        """Should include AXI bus interface when specified."""
        xml = generate_ip_xact("sc_lif", bus="axi_lite")
        assert "aximm" in xml
        assert "S_AXI" in xml

    def test_fileset(self) -> None:
        """Should reference the Verilog source file."""
        xml = generate_ip_xact("sc_lif")
        assert "sc_lif.v" in xml
        assert "verilogSource" in xml


# ═══════════════════════════════════════════════════════════════════════
# VHDL Output Tests
# ═══════════════════════════════════════════════════════════════════════

class TestVHDLOutput:
    """Test VHDL-2008 wrapper generation."""

    def test_generates_entity(self) -> None:
        """Should produce a VHDL entity."""
        vhdl = verilog_to_vhdl_wrapper("sc_lif", data_width=16)
        assert "entity sc_lif_vhdl is" in vhdl
        assert "end entity" in vhdl

    def test_architecture(self) -> None:
        """Should produce an architecture."""
        vhdl = verilog_to_vhdl_wrapper("sc_lif")
        assert "architecture rtl of sc_lif_vhdl is" in vhdl
        assert "end architecture rtl" in vhdl

    def test_component(self) -> None:
        """Should instantiate the Verilog module as a component."""
        vhdl = verilog_to_vhdl_wrapper("sc_lif")
        assert "component sc_lif is" in vhdl
        assert "u_neuron : sc_lif" in vhdl

    def test_ports(self) -> None:
        """Should have correct ports."""
        vhdl = verilog_to_vhdl_wrapper("sc_lif", data_width=16)
        assert "clk" in vhdl
        assert "rst" in vhdl
        assert "I_t" in vhdl
        assert "spike_out" in vhdl

    def test_do254_comment(self) -> None:
        """Should reference DO-254."""
        vhdl = verilog_to_vhdl_wrapper("sc_lif")
        assert "DO-254" in vhdl

    def test_ieee_libraries(self) -> None:
        """Should use IEEE libraries."""
        vhdl = verilog_to_vhdl_wrapper("sc_lif")
        assert "ieee.std_logic_1164" in vhdl
        assert "ieee.numeric_std" in vhdl

    def test_signed_type(self) -> None:
        """Signed mode should use 'signed' type."""
        vhdl = verilog_to_vhdl_wrapper("sc_lif", signed=True)
        assert "signed(" in vhdl

    def test_unsigned_type(self) -> None:
        """Unsigned mode should use 'unsigned' type."""
        vhdl = verilog_to_vhdl_wrapper("sc_lif", signed=False)
        assert "unsigned(" in vhdl


# ═══════════════════════════════════════════════════════════════════════
# Posit Arithmetic Tests
# ═══════════════════════════════════════════════════════════════════════

class TestPositArithmetic:
    """Test posit number encoding/decoding."""

    def test_zero(self) -> None:
        """Zero should encode to 0."""
        assert posit_encode(0, POSIT8_0) == 0
        assert posit_decode(0, POSIT8_0) == 0.0

    def test_roundtrip_positive(self) -> None:
        """Positive values should roundtrip approximately."""
        for val in [1.0, 2.0, 10.0, 50.0]:
            encoded = posit_encode(val, POSIT8_0)
            decoded = posit_decode(encoded, POSIT8_0)
            assert abs(decoded - val) < val * 0.2, f"Roundtrip fail: {val} → {encoded} → {decoded}"

    def test_roundtrip_negative(self) -> None:
        """Negative values should roundtrip approximately."""
        for val in [-1.0, -5.0, -20.0]:
            encoded = posit_encode(val, POSIT8_0)
            decoded = posit_decode(encoded, POSIT8_0)
            assert abs(decoded - val) < abs(val) * 0.2

    def test_posit8_range(self) -> None:
        """Posit<8,0> max value should be 64."""
        assert POSIT8_0.max_value == 64.0

    def test_posit16_range(self) -> None:
        """Posit<16,1> should have much larger range than posit-8."""
        assert POSIT16_1.max_value > POSIT8_0.max_value

    def test_configs(self) -> None:
        """All standard configs should have valid properties."""
        for cfg in [POSIT8_0, POSIT8_1, POSIT16_1, POSIT16_2]:
            assert cfg.nbits > 0
            assert cfg.max_value > 0
            assert cfg.min_positive > 0
            assert cfg.min_positive < 1.0

    def test_useed(self) -> None:
        """Useed should be 2^(2^es)."""
        assert POSIT8_0.useed == 2    # 2^(2^0) = 2
        assert POSIT8_1.useed == 4    # 2^(2^1) = 4
        assert POSIT16_2.useed == 16  # 2^(2^2) = 16


# ═══════════════════════════════════════════════════════════════════════
# CDC Synchroniser Tests
# ═══════════════════════════════════════════════════════════════════════

class TestCDCSynchroniser:
    """Test CDC synchroniser generation."""

    def test_generates_module(self) -> None:
        """Should produce a valid Verilog module."""
        v = generate_cdc_synchroniser("spike")
        assert "module cdc_sync_spike" in v
        assert "endmodule" in v

    def test_async_reg(self) -> None:
        """Should have ASYNC_REG attribute."""
        v = generate_cdc_synchroniser("spike")
        assert "ASYNC_REG" in v

    def test_two_stages(self) -> None:
        """Default 2-stage should have sync_r0 and sync_r1."""
        v = generate_cdc_synchroniser("spike", stages=2)
        assert "sync_r0" in v
        assert "sync_r1" in v

    def test_three_stages(self) -> None:
        """3-stage for higher MTBF should have sync_r2."""
        v = generate_cdc_synchroniser("spike", stages=3)
        assert "sync_r2" in v

    def test_multi_bit(self) -> None:
        """Multi-bit signal should have vector declaration."""
        v = generate_cdc_synchroniser("data", width=8)
        assert "[7:0]" in v

    def test_custom_clocks(self) -> None:
        """Custom clock names should appear."""
        v = generate_cdc_synchroniser("spike", src_clock="clk_fast", dst_clock="clk_slow")
        assert "clk_fast" in v
        assert "clk_slow" in v

    def test_output_assignment(self) -> None:
        """Output should be from last sync register."""
        v = generate_cdc_synchroniser("spike", stages=2)
        assert "spike_out = sync_r1" in v


# ═══════════════════════════════════════════════════════════════════════
# TCL Script Tests
# ═══════════════════════════════════════════════════════════════════════

class TestTCLGen:
    """Test Vivado/Quartus TCL generation."""

    def test_vivado_project(self) -> None:
        """Should produce a Vivado project TCL."""
        tcl = generate_tcl_project("sc_lif", tool="vivado")
        assert "create_project" in tcl
        assert "synth_design" in tcl
        assert "write_bitstream" in tcl

    def test_vivado_part(self) -> None:
        """Should include FPGA part number."""
        tcl = generate_tcl_project("sc_lif", part="xc7a100t")
        assert "xc7a100t" in tcl

    def test_vivado_constraints(self) -> None:
        """Should add constraint file if provided."""
        tcl = generate_tcl_project("sc_lif", constraint_file="sc_lif.xdc")
        assert "sc_lif.xdc" in tcl
        assert "constrs_1" in tcl

    def test_vivado_reports(self) -> None:
        """Should generate utilisation and timing reports."""
        tcl = generate_tcl_project("sc_lif")
        assert "report_utilization" in tcl
        assert "report_timing_summary" in tcl

    def test_quartus_project(self) -> None:
        """Should produce a Quartus project TCL."""
        tcl = generate_tcl_project("sc_lif", tool="quartus", part="5CSEMA5F31C6")
        assert "project_new" in tcl
        assert "execute_flow" in tcl
        assert "5CSEMA5F31C6" in tcl

    def test_invalid_tool(self) -> None:
        """Should raise on invalid tool."""
        with pytest.raises(ValueError, match="Unsupported tool"):
            generate_tcl_project("sc_lif", tool="ise")  # type: ignore


# ═══════════════════════════════════════════════════════════════════════
# Bitstream Automation Tests
# ═══════════════════════════════════════════════════════════════════════

class TestBitstreamGen:
    """Test open-source Makefile generation."""

    def test_ice40_makefile(self) -> None:
        """Should produce an iCE40 Makefile."""
        mk = generate_oss_makefile("sc_lif", target="ice40")
        assert "yosys" in mk
        assert "nextpnr-ice40" in mk
        assert "icepack" in mk

    def test_ecp5_makefile(self) -> None:
        """Should produce an ECP5 Makefile."""
        mk = generate_oss_makefile("sc_lif", target="ecp5", device="um5g-85k")
        assert "yosys" in mk
        assert "nextpnr-ecp5" in mk
        assert "ecppack" in mk

    def test_custom_device(self) -> None:
        """Should include custom device name."""
        mk = generate_oss_makefile("sc_lif", target="ice40", device="lp8k")
        assert "lp8k" in mk

    def test_clean_target(self) -> None:
        """Should have a clean target."""
        mk = generate_oss_makefile("sc_lif", target="ice40")
        assert "clean:" in mk

    def test_prog_target(self) -> None:
        """Should have a prog target."""
        mk = generate_oss_makefile("sc_lif", target="ice40")
        assert "prog:" in mk

    def test_custom_sources(self) -> None:
        """Should include custom source files."""
        mk = generate_oss_makefile(
            "sc_lif", target="ice40",
            verilog_files=["sc_lif.v", "lfsr16.v"],
        )
        assert "sc_lif.v" in mk
        assert "lfsr16.v" in mk
