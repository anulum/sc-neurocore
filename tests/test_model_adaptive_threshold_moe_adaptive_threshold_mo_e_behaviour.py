# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAdaptiveThresholdMoEBehaviour from former test_model_adaptive_threshold_moe.py

"""Focused suite: TestAdaptiveThresholdMoEBehaviour from former test_model_adaptive_threshold_moe.py."""

from __future__ import annotations

from tests.model_adaptive_threshold_moe_support import *  # noqa: F403


class TestAdaptiveThresholdMoEBehaviour:
    def test_integer_spike_count_and_soft_reset_residual(self):
        n = AdaptiveThresholdMoENeuron(k=4.0, ema_alpha=1.0)
        spikes = n.step(2.0)
        assert spikes == 4
        assert n.v == 0.0
        assert n.v_th == 0.5

    def test_negative_drive_does_not_emit_negative_spikes(self):
        n = AdaptiveThresholdMoENeuron(k=4.0, ema_alpha=1.0)
        assert n.step(-2.0) == 0
        assert n.v == -2.0
        assert n.v_th == 0.5

    def test_collapsed_mode_preserves_membrane_state(self):
        n = AdaptiveThresholdMoENeuron(k=4.0, ema_alpha=1.0)
        n.v = 0.75
        assert n.step_collapsed(2.0) == 4
        assert n.v == 0.75

    def test_sparsity_reflects_thresholded_residual(self):
        n = AdaptiveThresholdMoENeuron(k=4.0, ema_alpha=1.0)
        n.step(0.2)
        assert n.sparsity() == 1.0
        n.v = n.v_th
        assert n.sparsity() == 0.0
