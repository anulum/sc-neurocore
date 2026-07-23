# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestResourceEstimation from former test_deployment.py

"""Focused suite: TestResourceEstimation from former test_deployment.py."""

from __future__ import annotations

from tests.deployment_support import *  # noqa: F403

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
