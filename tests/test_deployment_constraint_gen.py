# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestConstraintGen from former test_deployment.py

"""Focused suite: TestConstraintGen from former test_deployment.py."""

from __future__ import annotations

from tests.deployment_support import *  # noqa: F403


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
