# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Adaptive-threshold MoE model contracts

"""Module-specific behavioural contracts for ``AdaptiveThresholdMoENeuron``."""

from __future__ import annotations

import pytest


class TestAdaptiveThresholdMoENeuron:
    @pytest.fixture()
    def neuron(self):
        from sc_neurocore.neurons.models import AdaptiveThresholdMoENeuron

        return AdaptiveThresholdMoENeuron(k=4.0)

    def test_defaults(self, neuron):
        assert neuron.k == 4.0
        assert neuron.v == 0.0
        assert neuron.v_th == 1.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"k": 0.0},
            {"k": float("nan")},
            {"ema_alpha": 0.0},
            {"ema_alpha": 1.1},
            {"ema_alpha": float("inf")},
            {"v": float("nan")},
            {"v_th": 0.0},
            {"_mean_abs_x": -0.01},
        ],
    )
    def test_rejects_non_physical_adaptive_threshold_parameters(self, kwargs):
        """Adaptive threshold dynamics require finite positive scaling and state."""
        from sc_neurocore.neurons.models import AdaptiveThresholdMoENeuron

        with pytest.raises(ValueError):
            AdaptiveThresholdMoENeuron(**kwargs)

    @pytest.mark.parametrize("current", [float("nan"), float("inf")])
    def test_rejects_non_finite_current(self, current):
        """Threshold adaptation must fail closed on non-finite current."""
        from sc_neurocore.neurons.models import AdaptiveThresholdMoENeuron

        with pytest.raises(ValueError, match="current"):
            AdaptiveThresholdMoENeuron().step(current)

    @pytest.mark.parametrize("activation", [float("nan"), float("-inf")])
    def test_rejects_non_finite_collapsed_activation(self, activation):
        """Collapsed inference must fail closed on non-finite activation."""
        from sc_neurocore.neurons.models import AdaptiveThresholdMoENeuron

        with pytest.raises(ValueError, match="activation"):
            AdaptiveThresholdMoENeuron().step_collapsed(activation)

    def test_step_returns_int(self, neuron):
        result = neuron.step(1.0)
        assert isinstance(result, int)

    def test_non_negative_spike_count(self, neuron):
        """SpikingBrain s_INT must be >= 0."""
        for _ in range(100):
            s = neuron.step(-5.0)
            assert s >= 0

    def test_integer_spike_count_gt_one(self):
        """With high input and low k, spike count can exceed 1."""
        from sc_neurocore.neurons.models import AdaptiveThresholdMoENeuron

        n = AdaptiveThresholdMoENeuron(k=10.0, ema_alpha=0.5)
        # Warm up EMA.
        for _ in range(20):
            n.step(5.0)
        # With k=10, V_th = mean(|x|)/10 = 0.5. v accumulates to 5, s=round(5/0.5)=10.
        s = n.step(5.0)
        assert s > 1, f"Expected multi-spike, got {s}"

    def test_soft_reset_preserves_residual(self):
        """After spike, v retains the sub-threshold residual."""
        from sc_neurocore.neurons.models import AdaptiveThresholdMoENeuron

        n = AdaptiveThresholdMoENeuron(k=4.0, ema_alpha=1.0)
        n.step(1.0)  # sets mean_abs_x = 1.0, v_th = 0.25
        # v = 1.0, s = round(1.0/0.25) = 4, v = 1.0 - 0.25*4 = 0.0
        assert abs(n.v) < 0.01

    def test_adaptive_threshold_tracks_input(self):
        """V_th = (1/k) * mean(|x|) tracks input magnitude."""
        from sc_neurocore.neurons.models import AdaptiveThresholdMoENeuron

        n = AdaptiveThresholdMoENeuron(k=4.0, ema_alpha=0.5)
        for _ in range(50):
            n.step(10.0)
        assert n.v_th > 1.0, "Threshold must rise with large inputs"
        n2 = AdaptiveThresholdMoENeuron(k=4.0, ema_alpha=0.5)
        for _ in range(50):
            n2.step(0.1)
        assert n2.v_th < n.v_th

    def test_sparsity_below_threshold(self, neuron):
        assert neuron.sparsity() == 1.0  # no input yet

    def test_step_collapsed(self, neuron):
        """Time-collapsed mode: s_INT = round(x / V_th)."""
        for _ in range(20):
            neuron.step_collapsed(2.0)
        s = neuron.step_collapsed(2.0)
        assert isinstance(s, int)
        assert s >= 0

    def test_reset(self, neuron):
        for _ in range(50):
            neuron.step(3.0)
        neuron.reset()
        assert neuron.v == 0.0
        assert neuron.v_th == 1.0
        assert neuron._mean_abs_x == 0.0

    def test_varying_input_produces_sparsity(self):
        """With varying inputs and k=1, not every step spikes."""
        from sc_neurocore.neurons.models import AdaptiveThresholdMoENeuron

        n = AdaptiveThresholdMoENeuron(k=1.0, ema_alpha=0.3)
        inputs = [0.0, 0.0, 5.0, 0.0, 0.0, 5.0, 0.0] * 10
        spikes = [n.step(x) for x in inputs]
        non_spiking = sum(1 for s in spikes if s == 0)
        assert non_spiking > 0, "Some steps must have zero spikes (sparsity)"
