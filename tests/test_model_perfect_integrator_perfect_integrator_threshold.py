# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPerfectIntegratorThreshold from former test_model_perfect_integrator.py

"""Focused suite: TestPerfectIntegratorThreshold from former test_model_perfect_integrator.py."""

from __future__ import annotations

from tests.model_perfect_integrator_support import *  # noqa: F403

class TestPerfectIntegratorThreshold:
    """Threshold, reset, and spike timing."""

    def test_exact_threshold_fires(self):
        """When V reaches exactly threshold, a spike must occur."""
        # Use I=10 so dV=1.0/step → hits threshold=1.0 exactly at step 1
        n = PerfectIntegratorNeuron(dt=0.1, c_m=1.0, v_threshold=1.0)
        s = n.step(10.0)
        assert s == 1

    def test_reset_to_v_reset(self):
        n = PerfectIntegratorNeuron()
        n.step(10.0)  # spike
        assert n.v == n.v_reset

    def test_custom_reset_potential(self):
        n = PerfectIntegratorNeuron(v_reset=-0.5, v_threshold=1.0)
        n.step(10.0)  # spike at v=1.0
        assert n.v == -0.5

    def test_superthreshold_instant_spike(self):
        """Very large current → spike on first step."""
        n = PerfectIntegratorNeuron()
        assert n.step(100.0) == 1
