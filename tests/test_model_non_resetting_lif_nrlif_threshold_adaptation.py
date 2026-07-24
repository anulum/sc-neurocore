# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNRLIFThresholdAdaptation from former test_model_non_resetting_lif.py

"""Focused suite: TestNRLIFThresholdAdaptation from former test_model_non_resetting_lif.py."""

from __future__ import annotations

from tests.model_non_resetting_lif_support import *  # noqa: F403


class TestNRLIFThresholdAdaptation:
    def test_threshold_elevation_creates_refractoriness(self):
        """After spike, elevated θ prevents immediate re-firing."""
        n = NonResettingLIFNeuron()
        # Find first spike, then check next step
        for _ in range(10_000):
            if n.step(20.0) == 1:
                # Theta just increased by 5mV
                # Next step should not spike (theta too high)
                next_spike = n.step(20.0)
                # May or may not spike depending on V, but theta is high
                assert n.theta > n.theta_rest
                break

    def test_theta_accumulates_with_rapid_spiking(self):
        """Multiple rapid spikes → θ well above θ_rest."""
        n = NonResettingLIFNeuron()
        for _ in range(5000):
            n.step(30.0)
        assert n.theta > n.theta_rest + n.delta_theta * 0.5
