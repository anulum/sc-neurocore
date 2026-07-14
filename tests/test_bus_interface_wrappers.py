# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Legacy AXI4-Lite and Wishbone wrapper contracts

"""Tests for historical bus-wrapper and register-map generation."""

from __future__ import annotations

from typing import cast

import pytest

from sc_neurocore.hdl_gen.bus_interface import (
    BusProtocol,
    generate_bus_wrapper,
    generate_register_map,
)


LIF_PARAMS = {"P_V_REST": 16, "P_V_THRESH": 16, "P_TAU_M": 16}


class TestAXI4Lite:
    """Test AXI4-Lite wrapper generation."""

    def test_generates_module(self) -> None:
        """Should produce a valid Verilog module."""
        v = generate_bus_wrapper("sc_lif", LIF_PARAMS, bus="axi_lite")
        assert "module sc_lif_axi_lite" in v
        assert "endmodule" in v

    def test_has_axi_ports(self) -> None:
        """Should include all AXI4-Lite signal names."""
        v = generate_bus_wrapper("sc_lif", LIF_PARAMS, bus="axi_lite")
        for sig in [
            "S_AXI_ACLK",
            "S_AXI_ARESETN",
            "S_AXI_AWADDR",
            "S_AXI_WDATA",
            "S_AXI_RDATA",
            "S_AXI_BRESP",
        ]:
            assert sig in v, f"Missing AXI signal: {sig}"

    def test_has_interrupt(self) -> None:
        """Should export spike interrupt."""
        v = generate_bus_wrapper("sc_lif", LIF_PARAMS, bus="axi_lite")
        assert "irq_spike" in v

    def test_has_neuron_instance(self) -> None:
        """Should instantiate the inner neuron module."""
        v = generate_bus_wrapper("sc_lif", LIF_PARAMS, bus="axi_lite")
        assert "sc_lif u_neuron" in v

    def test_has_parameter_registers(self) -> None:
        """Each parameter should have a register."""
        v = generate_bus_wrapper("sc_lif", LIF_PARAMS, bus="axi_lite")
        assert "reg_p_v_rest" in v
        assert "reg_p_v_thresh" in v
        assert "reg_p_tau_m" in v

    def test_has_spike_counter(self) -> None:
        """Should include a spike counter register."""
        v = generate_bus_wrapper("sc_lif", LIF_PARAMS, bus="axi_lite")
        assert "reg_spike_count" in v

    def test_control_register(self) -> None:
        """Should have enable and reset bits in control register."""
        v = generate_bus_wrapper("sc_lif", LIF_PARAMS, bus="axi_lite")
        assert "reg_ctrl" in v
        assert "reg_ctrl[0]" in v  # enable
        assert "reg_ctrl[1]" in v  # reset


class TestWishbone:
    """Test Wishbone B4 wrapper generation."""

    def test_generates_module(self) -> None:
        """Should produce a valid Verilog module."""
        v = generate_bus_wrapper("sc_lif", LIF_PARAMS, bus="wishbone")
        assert "module sc_lif_wb" in v
        assert "endmodule" in v

    def test_has_wishbone_ports(self) -> None:
        """Should include all Wishbone signal names."""
        v = generate_bus_wrapper("sc_lif", LIF_PARAMS, bus="wishbone")
        for sig in [
            "wb_clk_i",
            "wb_rst_i",
            "wb_adr_i",
            "wb_dat_i",
            "wb_dat_o",
            "wb_we_i",
            "wb_stb_i",
            "wb_cyc_i",
            "wb_ack_o",
        ]:
            assert sig in v, f"Missing Wishbone signal: {sig}"

    def test_has_interrupt(self) -> None:
        """Should export spike interrupt."""
        v = generate_bus_wrapper("sc_lif", LIF_PARAMS, bus="wishbone")
        assert "irq_spike" in v

    def test_has_neuron_instance(self) -> None:
        """Should instantiate the inner neuron module."""
        v = generate_bus_wrapper("sc_lif", LIF_PARAMS, bus="wishbone")
        assert "sc_lif u_neuron" in v

    def test_has_parameter_registers(self) -> None:
        """Each parameter should have a register."""
        v = generate_bus_wrapper("sc_lif", LIF_PARAMS, bus="wishbone")
        assert "reg_p_v_rest" in v


class TestRegisterMap:
    """Test register map generation."""

    def test_standard_layout(self) -> None:
        """Standard registers should be at expected offsets."""
        rmap = generate_register_map(LIF_PARAMS)
        assert rmap["CTRL"] == 0
        assert rmap["I_T"] == 4
        assert rmap["SPIKE_COUNT"] == 8
        assert rmap["P_V_REST"] == 12

    def test_custom_base_address(self) -> None:
        """Base address should shift all registers."""
        rmap = generate_register_map(LIF_PARAMS, base_address=0x1000)
        assert rmap["CTRL"] == 0x1000
        assert rmap["I_T"] == 0x1004

    def test_invalid_bus(self) -> None:
        """Should raise on invalid bus protocol."""
        with pytest.raises(ValueError, match="Unsupported bus"):
            generate_bus_wrapper("sc_lif", LIF_PARAMS, bus=cast(BusProtocol, "spi"))


# ═══════════════════════════════════════════════════════════════════════
# Mixed-Precision Tests
# ═══════════════════════════════════════════════════════════════════════
