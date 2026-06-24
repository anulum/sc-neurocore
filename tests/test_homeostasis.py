# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Tests for sc_neurocore.homeostasis

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.homeostasis import NetworkRegulator, SleepConsolidation, StabilityMetrics


class TestNetworkRegulator:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("target_rate", -0.1),
            ("target_rate", float("nan")),
            ("rate_tolerance", -0.1),
            ("rate_tolerance", 1.1),
            ("threshold_step", -0.01),
            ("lr_scale_factor", 0.0),
            ("lr_scale_factor", 1.1),
            ("lr_scale_factor", float("inf")),
        ],
    )
    def test_invalid_regulator_parameters_fail_closed(self, field, value):
        kwargs = {
            "target_rate": 0.1,
            "rate_tolerance": 0.5,
            "threshold_step": 0.01,
            "lr_scale_factor": 0.95,
        }
        kwargs[field] = value
        with pytest.raises(ValueError, match=field):
            NetworkRegulator(**kwargs)

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

    @pytest.mark.parametrize(
        ("rates", "thresholds", "learning_rate"),
        [
            (np.array([0.1, float("nan")]), np.ones(2), 0.01),
            (np.array([[0.1, 0.1]]), np.ones(2), 0.01),
            (np.array([0.1, 0.1]), np.ones((2, 1)), 0.01),
            (np.array([0.1, 0.1]), np.ones(3), 0.01),
            (np.array([0.1, 0.1]), np.ones(2), -0.01),
            (np.array([0.1, 0.1]), np.ones(2), float("inf")),
        ],
    )
    def test_invalid_regulation_inputs_fail_closed(self, rates, thresholds, learning_rate):
        reg = NetworkRegulator(target_rate=0.1)
        with pytest.raises(ValueError, match="regulate"):
            reg.regulate(rates, thresholds, learning_rate)

    def test_invalid_weight_matrix_fails_closed(self):
        reg = NetworkRegulator(target_rate=0.1)
        rates = np.full(2, 0.1)
        thresholds = np.ones(2)
        with pytest.raises(ValueError, match="weights"):
            reg.regulate(rates, thresholds, 0.01, weights=[np.array([[1.0, float("nan")]])])

    def test_summary(self):
        m = StabilityMetrics(mean_firing_rate=0.15, is_stable=True)
        s = m.summary()
        assert "STABLE" in s


class TestSleepConsolidation:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("decay_exponent", -0.1),
            ("decay_exponent", float("nan")),
            ("noise_amplitude", -0.01),
            ("noise_amplitude", float("inf")),
            ("duration_fraction", 0.0),
            ("duration_fraction", 1.1),
            ("duration_fraction", float("nan")),
        ],
    )
    def test_invalid_sleep_parameters_fail_closed(self, field, value):
        kwargs = {
            "decay_exponent": 0.5,
            "noise_amplitude": 0.01,
            "duration_fraction": 0.1,
        }
        kwargs[field] = value
        with pytest.raises(ValueError, match=field):
            SleepConsolidation(**kwargs)

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

    @pytest.mark.parametrize(
        "weights",
        [
            [],
            [np.array([[1.0, float("nan")]])],
            [np.array([[1.0, float("inf")]])],
            [[1.0, 2.0]],
        ],
    )
    def test_invalid_sleep_weights_fail_closed(self, weights):
        sleep = SleepConsolidation()
        with pytest.raises(ValueError, match="weights"):
            sleep.apply(weights)

    @pytest.mark.parametrize(
        ("epoch", "total_epochs"),
        [(-1, 10), (1, 0), (1, -10), (float("nan"), 10)],
    )
    def test_invalid_sleep_schedule_fails_closed(self, epoch, total_epochs):
        sleep = SleepConsolidation()
        with pytest.raises(ValueError, match="epoch"):
            sleep.should_sleep(epoch, total_epochs)


def test_regulate_rejects_non_finite_thresholds() -> None:
    reg = NetworkRegulator(target_rate=0.1)
    rates = np.full(4, 0.1)
    thresholds = np.array([1.0, 1.0, np.nan, 1.0])
    # A correctly-shaped 1-D threshold vector with a non-finite entry is rejected.
    with pytest.raises(ValueError, match="thresholds must be finite"):
        reg.regulate(rates, thresholds, 0.01)
