# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAtomicity from former test_model_adaptive_threshold_if.py

"""Focused suite: TestAtomicity from former test_model_adaptive_threshold_if.py."""

from __future__ import annotations

from tests.model_adaptive_threshold_if_support import *  # noqa: F403


class TestAtomicity:
    """Rejected steps leave both dynamic states unchanged."""

    @pytest.mark.parametrize("current", [float("nan"), float("inf"), -float("inf")])
    def test_rejects_non_finite_current(self, current: float) -> None:
        n = AdaptiveThresholdIFNeuron()
        before = (n.v, n.theta)
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert (n.v, n.theta) == before

    def test_rejects_non_finite_runtime_voltage_before_update(self) -> None:
        n = AdaptiveThresholdIFNeuron(v=-60.0, theta=-45.0)
        n.v = float("nan")
        with pytest.raises(ValueError, match="state"):
            n.step(0.0)
        assert np.isnan(n.v)

    def test_rejects_non_finite_runtime_threshold_before_update(self) -> None:
        n = AdaptiveThresholdIFNeuron(v=-60.0, theta=-45.0)
        n.theta = float("nan")
        with pytest.raises(ValueError, match="state"):
            n.step(0.0)
        assert np.isnan(n.theta)

    def test_rejects_non_finite_relaxation_update_before_state_mutation(self) -> None:
        n = AdaptiveThresholdIFNeuron(v=-1.0e308, theta=-45.0)
        before = (n.v, n.theta)
        with pytest.raises(FloatingPointError, match="exact relaxation"):
            n.step(1.0e308)
        assert (n.v, n.theta) == before

    def test_rejects_non_finite_threshold_jump_before_state_mutation(self) -> None:
        n = AdaptiveThresholdIFNeuron(v=1.2e308, theta=1.0e308, delta_theta=1.0e308)
        before = (n.v, n.theta)
        with pytest.raises(FloatingPointError, match="threshold jump"):
            n.step(0.0)
        assert (n.v, n.theta) == before

    def test_rejects_invalid_runtime_configuration_before_mutation(self) -> None:
        n = AdaptiveThresholdIFNeuron()
        n.tau_m = 0.0
        before = (n.v, n.theta)
        with pytest.raises(ValueError, match="tau_m"):
            n.step(1.0)
        assert (n.v, n.theta) == before
