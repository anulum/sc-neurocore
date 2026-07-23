# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestValidation from former test_adaptive_runtime_precision.py

"""Focused suite: TestValidation from former test_adaptive_runtime_precision.py."""

from __future__ import annotations

from tests.adaptive_runtime_precision_support import *  # noqa: F403

class TestValidation:
    """Verify that invalid configurations are rejected."""

    def test_thresholds_require_ordered_band(self, lif_neuron):
        """Swapped hysteresis thresholds must be rejected."""
        with pytest.raises(ValueError, match="threshold_down_pct"):
            compile_adaptive_precision(
                lif_neuron,
                threshold_up_pct=0.2,
                threshold_down_pct=0.8,
            )

        with pytest.raises(ValueError, match="threshold_up_pct"):
            compile_adaptive_precision(
                lif_neuron,
                threshold_up_pct=1.0,
                threshold_down_pct=0.2,
            )

    def test_thresholds_reject_nonfinite(self, lif_neuron):
        """NaN and infinities are rejected by threshold validation."""
        with pytest.raises(ValueError, match="finite"):
            compile_adaptive_precision(
                lif_neuron,
                threshold_up_pct=float("nan"),
                threshold_down_pct=0.2,
            )

        with pytest.raises(ValueError, match="finite"):
            compile_adaptive_precision(
                lif_neuron,
                threshold_up_pct=0.9,
                threshold_down_pct=float("inf"),
            )

        with pytest.raises(ValueError, match="must satisfy 0 < threshold_down_pct"):
            compile_adaptive_precision(
                lif_neuron,
                threshold_up_pct=0.6,
                threshold_down_pct=0.0,
            )

        with pytest.raises(ValueError, match="Quantised threshold codes"):
            compile_adaptive_precision(
                lif_neuron,
                threshold_up_pct=0.00001,
                threshold_down_pct=0.000001,
            )

    def test_thresholds_are_reflected_in_manifest(self, lif_neuron):
        """Manifest must retain threshold policy under compiler contract."""
        up = 0.9
        down = 0.3
        v = compile_adaptive_precision(lif_neuron, threshold_up_pct=up, threshold_down_pct=down)
        manifest = _extract_manifest(v)

        assert manifest["threshold_up_pct"] == up
        assert manifest["threshold_down_pct"] == down

    def test_lp_wider_than_hp_rejected(self, lif_neuron):
        """LP wider than HP must raise ValueError."""
        with pytest.raises(ValueError, match="strictly less"):
            compile_adaptive_precision(lif_neuron, lp_width=32, lp_frac=16, hp_width=16, hp_frac=8)

    def test_equal_widths_rejected(self, lif_neuron):
        """Equal LP and HP widths must raise ValueError."""
        with pytest.raises(ValueError, match="strictly less"):
            compile_adaptive_precision(lif_neuron, lp_width=16, lp_frac=8, hp_width=16, hp_frac=8)

    def test_zero_frac_rejected(self, lif_neuron):
        """Zero fractional bits must raise ValueError."""
        with pytest.raises(ValueError, match="fraction"):
            compile_adaptive_precision(lif_neuron, lp_width=8, lp_frac=0, hp_width=16, hp_frac=8)
