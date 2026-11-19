# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Focused SDC deck generation tests

"""Verify Synopsys Design Constraints generation and clock-port exclusion."""

from __future__ import annotations

from sc_neurocore.asic_flow.asic_flow import (
    DesignParams,
    PDKConfig,
    PDKType,
    SCASICOptimisationConfig,
    SDCGenerator,
)


class TestSDCGenerator:
    """Validate clock, reset, IO-delay, fanout, and driving-cell SDC commands."""

    def test_generates_sdc(self) -> None:
        """Generated SDC creates the named primary clock."""
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        sdc = SDCGenerator.generate(pdk, design)
        assert "create_clock" in sdc
        assert design.clock_name in sdc

    def test_clock_period(self) -> None:
        """Target frequency in megahertz converts to a nanosecond clock period."""
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams(target_frequency_mhz=100.0)
        sdc = SDCGenerator.generate(pdk, design)
        assert "10.000" in sdc

    def test_false_path_reset(self) -> None:
        """Asynchronous reset input is marked as a false timing path."""
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams(reset_name="rst_n")
        sdc = SDCGenerator.generate(pdk, design)
        assert "rst_n" in sdc
        assert "false_path" in sdc

    def test_sc_fanout_constraint(self) -> None:
        """Stochastic-computing maximum fanout propagates to current_design."""
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams(sc_optimisation=SCASICOptimisationConfig(max_fanout=8))
        sdc = SDCGenerator.generate(pdk, design)
        assert "set_max_fanout 8" in sdc

    def test_input_delay_and_driving_cell_exclude_clock_port(self) -> None:
        """Input-delay and driving-cell constraints exclude the clock port."""
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams(clock_name="sys_clk", target_frequency_mhz=100.0)
        sdc = SDCGenerator.generate(pdk, design)
        non_clock_expr = "[delete_from_list [all_inputs] [get_ports sys_clk]]"

        assert f"set_input_delay 2.500 -clock sys_clk {non_clock_expr}" in sdc
        assert f"set_driving_cell -lib_cell sky130_fd_sc_hd__buf_2 {non_clock_expr}" in sdc
        for line in sdc.splitlines():
            stripped = line.strip()
            if stripped.startswith(("set_input_delay", "set_driving_cell")):
                assert not stripped.endswith("[all_inputs]")
