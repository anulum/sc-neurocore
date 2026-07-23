# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCocotbGen from former test_deployment.py

"""Focused suite: TestCocotbGen from former test_deployment.py."""

from __future__ import annotations

from tests.deployment_support import *  # noqa: F403

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
