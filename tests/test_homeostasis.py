# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Tests for sc_neurocore.homeostasis

from __future__ import annotations

import numpy as np

from sc_neurocore.homeostasis import NetworkRegulator, SleepConsolidation, StabilityMetrics


class TestNetworkRegulator:
    def test_stable(self):
        reg = NetworkRegulator(target_rate=0.1)
        rates = np.full(20, 0.1)
        thresholds = np.ones(20)
        new_th, new_lr, m = reg.regulate(rates, thresholds, 0.01)
        assert m.is_stable
        np.testing.assert_array_equal(new_th, thresholds)

    def test_too_active(self):
        reg = NetworkRegulator(target_rate=0.1, rate_tolerance=0.5)
        rates = np.full(20, 0.5)
        thresholds = np.ones(20)
        new_th, _, m = reg.regulate(rates, thresholds, 0.01)
        assert not m.is_stable
        assert (new_th > thresholds).all()

    def test_too_quiet(self):
        reg = NetworkRegulator(target_rate=0.1, rate_tolerance=0.5)
        rates = np.full(20, 0.01)
        thresholds = np.ones(20)
        new_th, _, m = reg.regulate(rates, thresholds, 0.01)
        assert (new_th < thresholds).all()

    def test_high_variance_reduces_lr(self):
        reg = NetworkRegulator(target_rate=0.1)
        rates = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        thresholds = np.ones(10)
        _, new_lr, m = reg.regulate(rates, thresholds, 0.01)
        assert new_lr < 0.01

    def test_with_weights(self):
        reg = NetworkRegulator(target_rate=0.1)
        rates = np.full(10, 0.1)
        thresholds = np.ones(10)
        weights = [np.random.randn(10, 10)]
        _, _, m = reg.regulate(rates, thresholds, 0.01, weights=weights)
        assert m.weight_norm > 0

    def test_summary(self):
        m = StabilityMetrics(mean_firing_rate=0.15, is_stable=True)
        s = m.summary()
        assert "STABLE" in s


class TestSleepConsolidation:
    def test_apply(self):
        sleep = SleepConsolidation(decay_exponent=0.5, noise_amplitude=0.001)
        weights = [np.random.randn(10, 10)]
        consolidated = sleep.apply(weights, seed=42)
        assert len(consolidated) == 1
        assert not np.array_equal(consolidated[0], weights[0])

    def test_large_weights_decay_more(self):
        sleep = SleepConsolidation(decay_exponent=1.0, noise_amplitude=0.0)
        w = np.array([[0.1, 1.0]])
        cons = sleep.apply([w], seed=42)[0]
        # Larger weight should decay proportionally more
        ratio_before = abs(w[0, 1] / w[0, 0])
        ratio_after = abs(cons[0, 1] / max(abs(cons[0, 0]), 1e-10))
        assert ratio_after < ratio_before

    def test_should_sleep(self):
        sleep = SleepConsolidation(duration_fraction=0.1)
        assert not sleep.should_sleep(0, 100)
        assert sleep.should_sleep(10, 100)
        assert sleep.should_sleep(20, 100)
        assert not sleep.should_sleep(5, 100)
