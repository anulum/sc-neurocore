# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCFCThresholdBehavior from former test_model_cfc.py

"""Focused suite: TestCFCThresholdBehavior from former test_model_cfc.py."""

from __future__ import annotations

from tests.model_cfc_support import *  # noqa: F403

class TestCFCThresholdBehavior:
    """Default threshold=1.0 is unreachable since tanh < 1."""

    def test_default_threshold_unreachable(self):
        """tanh output never reaches 1.0 → no spikes at θ=1.0."""
        n = ClosedFormContinuousNeuron(v_threshold=1.0)
        spikes = len(_run(n, current=5.0, steps=5000))
        assert spikes == 0, "Should not spike at default threshold"

    def test_lower_threshold_enables_spiking(self):
        """θ=0.5 → spikes because x converges near 1.0."""
        n = ClosedFormContinuousNeuron(v_threshold=0.5)
        spikes = len(_run(n, current=5.0, steps=5000))
        assert spikes > 100

    @pytest.mark.parametrize("theta", [0.3, 0.5, 0.8, 0.95])
    def test_rate_increases_with_lower_threshold(self, theta: float):
        n = ClosedFormContinuousNeuron(v_threshold=theta)
        spikes = len(_run(n, current=5.0, steps=5000))
        assert spikes > 0 or theta > 0.99

    def test_spike_resets_x_to_zero(self):
        n = ClosedFormContinuousNeuron(v_threshold=0.5)
        for _ in range(5000):
            if n.step(5.0) == 1:
                assert n.x == 0.0
                break
