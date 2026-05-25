# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — AdaptiveThresholdMoENeuron regression tests

from __future__ import annotations

import math

import pytest

from sc_neurocore.neurons.models.adaptive_threshold_moe import AdaptiveThresholdMoENeuron


class TestAdaptiveThresholdMoEValidation:
    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, 0.0, -1.0])
    def test_rejects_non_positive_or_non_finite_k(self, value: float):
        with pytest.raises(ValueError, match="k"):
            AdaptiveThresholdMoENeuron(k=value)

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, 0.0, -0.1, 1.1])
    def test_rejects_out_of_range_ema_alpha(self, value: float):
        with pytest.raises(ValueError, match="ema_alpha"):
            AdaptiveThresholdMoENeuron(ema_alpha=value)

    @pytest.mark.parametrize("current", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite_current_before_state_mutation(self, current: float):
        n = AdaptiveThresholdMoENeuron()
        before = (n.v, n.v_th, n._mean_abs_x)
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert (n.v, n.v_th, n._mean_abs_x) == before

    @pytest.mark.parametrize("activation", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite_collapsed_activation_before_state_mutation(self, activation: float):
        n = AdaptiveThresholdMoENeuron()
        before = (n.v, n.v_th, n._mean_abs_x)
        with pytest.raises(ValueError, match="activation"):
            n.step_collapsed(activation)
        assert (n.v, n.v_th, n._mean_abs_x) == before

    @pytest.mark.parametrize(
        ("field", "message"),
        [
            ("v", "runtime membrane state"),
            ("v_th", "runtime threshold state"),
            ("_mean_abs_x", "runtime mean absolute input state"),
        ],
    )
    def test_rejects_corrupted_runtime_state_before_step(self, field: str, message: str):
        n = AdaptiveThresholdMoENeuron()
        setattr(n, field, math.nan)
        with pytest.raises(ValueError, match=message):
            n.step(1.0)
        assert math.isnan(getattr(n, field))

    def test_rejects_non_finite_adaptive_threshold_before_step_mutation(self):
        n = AdaptiveThresholdMoENeuron(k=1.0e-308)
        before = (n.v, n.v_th, n._mean_abs_x)
        with pytest.raises(ValueError, match="adaptive threshold"):
            n.step(1.0e308)
        assert (n.v, n.v_th, n._mean_abs_x) == before

    def test_rejects_non_finite_soft_reset_residual_before_mutation(self):
        n = AdaptiveThresholdMoENeuron(k=1.0, ema_alpha=1.0)
        n.v = 1.7e308
        n.v_th = 1.0
        n._mean_abs_x = 1.0
        before = (n.v, n.v_th, n._mean_abs_x)
        with pytest.raises(ValueError, match="soft reset residual"):
            n.step(1.0e308)
        assert (n.v, n.v_th, n._mean_abs_x) == before

    def test_rejects_non_finite_collapsed_threshold_before_state_mutation(self):
        n = AdaptiveThresholdMoENeuron(k=1.0e-308)
        before = (n.v, n.v_th, n._mean_abs_x)
        with pytest.raises(ValueError, match="adaptive threshold"):
            n.step_collapsed(1.0e308)
        assert (n.v, n.v_th, n._mean_abs_x) == before


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
