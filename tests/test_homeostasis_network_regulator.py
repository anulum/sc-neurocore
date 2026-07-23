# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNetworkRegulator from former test_homeostasis.py

"""Focused suite: TestNetworkRegulator from former test_homeostasis.py."""

from __future__ import annotations

from tests.homeostasis_support import *  # noqa: F403

class TestNetworkRegulator:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("target_rate", -0.1),
            ("target_rate", float("nan")),
            ("target_rate", True),
            ("rate_tolerance", -0.1),
            ("rate_tolerance", 1.1),
            ("rate_tolerance", True),
            ("threshold_step", -0.01),
            ("threshold_step", True),
            ("lr_scale_factor", 0.0),
            ("lr_scale_factor", 1.1),
            ("lr_scale_factor", float("inf")),
            ("lr_scale_factor", True),
        ],
    )
    def test_invalid_regulator_parameters_fail_closed(self, field: str, value: float) -> None:
        kwargs: dict[str, float] = {
            "target_rate": 0.1,
            "rate_tolerance": 0.5,
            "threshold_step": 0.01,
            "lr_scale_factor": 0.95,
        }
        kwargs[field] = value
        with pytest.raises(ValueError, match=field):
            NetworkRegulator(**kwargs)

    def test_stable(self) -> None:
        reg = NetworkRegulator(target_rate=0.1)
        rates = np.full(20, 0.1)
        thresholds = np.ones(20)
        new_th, new_lr, m = reg.regulate(rates, thresholds, 0.01)
        assert m.is_stable
        np.testing.assert_array_equal(new_th, thresholds)

    def test_too_active(self) -> None:
        reg = NetworkRegulator(target_rate=0.1, rate_tolerance=0.5)
        rates = np.full(20, 0.5)
        thresholds = np.ones(20)
        new_th, _, m = reg.regulate(rates, thresholds, 0.01)
        assert not m.is_stable
        assert (new_th > thresholds).all()

    def test_too_quiet(self) -> None:
        reg = NetworkRegulator(target_rate=0.1, rate_tolerance=0.5)
        rates = np.full(20, 0.01)
        thresholds = np.ones(20)
        new_th, _, m = reg.regulate(rates, thresholds, 0.01)
        assert (new_th < thresholds).all()

    def test_high_variance_reduces_lr(self) -> None:
        reg = NetworkRegulator(target_rate=0.1)
        rates = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        thresholds = np.ones(10)
        _, new_lr, m = reg.regulate(rates, thresholds, 0.01)
        assert new_lr < 0.01

    def test_with_weights(self) -> None:
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
            (np.array([0.1, 0.1]), np.ones(2), True),
            (np.array([], dtype=float), np.array([], dtype=float), 0.01),
            (np.array([0.1, -0.1]), np.ones(2), 0.01),
            (np.array([True, False]), np.ones(2), 0.01),
            (np.array(["0.1", "0.2"], dtype=object), np.ones(2), 0.01),
        ],
    )
    def test_invalid_regulation_inputs_fail_closed(
        self,
        rates: npt.NDArray[np.float64],
        thresholds: npt.NDArray[np.float64],
        learning_rate: float,
    ) -> None:
        reg = NetworkRegulator(target_rate=0.1)
        with pytest.raises(ValueError, match="regulate"):
            reg.regulate(rates, thresholds, learning_rate)

    def test_invalid_weight_matrix_fails_closed(self) -> None:
        reg = NetworkRegulator(target_rate=0.1)
        rates = np.full(2, 0.1)
        thresholds = np.ones(2)
        with pytest.raises(ValueError, match="weights"):
            reg.regulate(rates, thresholds, 0.01, weights=[np.array([[1.0, float("nan")]])])

    def test_summary(self) -> None:
        m = StabilityMetrics(mean_firing_rate=0.15, is_stable=True)
        s = m.summary()
        assert "STABLE" in s

    def test_summary_lists_adjustments(self) -> None:
        m = StabilityMetrics(mean_firing_rate=0.25, is_stable=False, adjustments_made=["lr *0.95"])

        summary = m.summary()

        assert "UNSTABLE" in summary
        assert "Adjustments: lr *0.95" in summary
