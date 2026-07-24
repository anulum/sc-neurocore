# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWishbone from former test_bus_interface_wrappers.py

"""Focused suite: TestWishbone from former test_bus_interface_wrappers.py."""

from __future__ import annotations

from tests.bus_interface_wrappers_support import *  # noqa: F403


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
