# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDynamics from former test_model_gamma_renewal.py

"""Focused suite: TestDynamics from former test_model_gamma_renewal.py."""

from __future__ import annotations

from tests.model_gamma_renewal_support import *  # noqa: F403


class TestDynamics:
    def test_fires_at_test_current(self):
        n = GammaRenewalNeuron()
        spikes = _run(n, current=100.0, steps=5000)
        assert len(spikes) >= 10

    def test_rate_increases_with_current(self):
        n_low = GammaRenewalNeuron()
        n_high = GammaRenewalNeuron()
        s_low = len(_run(n_low, current=50.0, steps=5000))
        s_high = len(_run(n_high, current=500.0, steps=5000))
        assert s_high >= s_low

    def test_two_runs_differ(self):
        n1 = GammaRenewalNeuron()
        n2 = GammaRenewalNeuron()
        t1 = [n1.step(100.0) for _ in range(1000)]
        t2 = [n2.step(100.0) for _ in range(1000)]
        assert t1 != t2

    def test_gamma_hazard_uses_bounded_interval_probability(self):
        """For k=1 the gamma renewal hazard is constant, so p=1-exp(-rate*dt)."""
        n = GammaRenewalNeuron(rate_hz=100.0, shape_k=1, dt_ms=1.0)
        n._rng = FixedRng(0.098)

        assert n.step(rate_override=100.0) == 0
        assert n._time_since_spike == pytest.approx(0.001)
