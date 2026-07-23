# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHysteresis from former test_adaptive_runtime_precision.py

"""Focused suite: TestHysteresis from former test_adaptive_runtime_precision.py."""

from __future__ import annotations

from tests.adaptive_runtime_precision_support import *  # noqa: F403

class TestHysteresis:
    """Verify hysteresis threshold logic."""

    def test_thresh_up_present(self, lif_neuron):
        """THRESH_UP localparam must be present."""
        v = compile_adaptive_precision(lif_neuron)
        assert "THRESH_UP" in v

    def test_thresh_down_present(self, lif_neuron):
        """THRESH_DOWN localparam must be present."""
        v = compile_adaptive_precision(lif_neuron)
        assert "THRESH_DOWN" in v

    def test_precision_mode_register(self, lif_neuron):
        """precision_mode register must be declared."""
        v = compile_adaptive_precision(lif_neuron)
        assert "reg precision_mode" in v

    def test_precision_mode_reset_to_lp(self, lif_neuron):
        """precision_mode must reset to 0 (LP mode)."""
        v = compile_adaptive_precision(lif_neuron)
        assert "precision_mode <= 1'b0" in v

    def test_custom_hysteresis(self, lif_neuron):
        """Custom hysteresis percentages should work."""
        v = compile_adaptive_precision(
            lif_neuron,
            threshold_up_pct=0.9,
            threshold_down_pct=0.3,
        )
        assert "90%" in v
        assert "30%" in v
