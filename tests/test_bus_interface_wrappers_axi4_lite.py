# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAXI4Lite from former test_bus_interface_wrappers.py

"""Focused suite: TestAXI4Lite from former test_bus_interface_wrappers.py."""

from __future__ import annotations

from tests.bus_interface_wrappers_support import *  # noqa: F403


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
