# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSigmaDeltaSubtract from former test_model_sigma_delta.py

"""Focused suite: TestSigmaDeltaSubtract from former test_model_sigma_delta.py."""

from __future__ import annotations

from tests.model_sigma_delta_support import *  # noqa: F403


class TestSigmaDeltaSubtract:
    def test_positive_spike_subtracts_threshold(self):
        """On +1 spike: sigma -= threshold (not reset to 0)."""
        n = SigmaDeltaNeuron(v_threshold=1.0)
        # sigma = 0 + 1.3 = 1.3 ≥ 1.0 → spike, sigma = 1.3 - 1.0 = 0.3
        s = n.step(1.3)
        assert s == 1
        assert abs(n.sigma - 0.3) < 1e-10

    def test_negative_spike_adds_threshold(self):
        """On -1 spike: sigma += threshold."""
        n = SigmaDeltaNeuron(v_threshold=1.0)
        s = n.step(-1.3)
        assert s == -1
        assert abs(n.sigma - (-0.3)) < 1e-10

    def test_residual_carries_over(self):
        """Residual after subtraction carries into next step."""
        n = SigmaDeltaNeuron(v_threshold=1.0)
        n.step(0.7)  # sigma = 0.7, no spike
        n.step(0.7)  # sigma = 1.4 ≥ 1.0 → spike, sigma = 0.4
        assert abs(n.sigma - 0.4) < 1e-10

    def test_overflow_accumulation(self):
        """When I > threshold, sigma grows because only one threshold
        is subtracted per step (even if sigma >> threshold)."""
        n = SigmaDeltaNeuron(v_threshold=1.0)
        for _ in range(100):
            n.step(2.0)  # Each step: sigma += 2.0, then sigma -= 1.0 → net +1.0
        # After 100 steps: sigma should be large (100 * 1.0 = 100)
        assert n.sigma > 50
