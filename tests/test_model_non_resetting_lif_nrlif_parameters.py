# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNRLIFParameters from former test_model_non_resetting_lif.py

"""Focused suite: TestNRLIFParameters from former test_model_non_resetting_lif.py."""

from __future__ import annotations

from tests.model_non_resetting_lif_support import *  # noqa: F403

class TestNRLIFParameters:
    @pytest.mark.parametrize("delta_theta", [2.0, 5.0, 10.0])
    def test_delta_theta_sweep(self, delta_theta: float):
        n = NonResettingLIFNeuron(delta_theta=delta_theta)
        spikes = len(_run(n, current=20.0, steps=5000))
        assert isinstance(spikes, int)

    @pytest.mark.parametrize("tau_theta", [20.0, 50.0, 200.0])
    def test_tau_theta_sweep(self, tau_theta: float):
        n = NonResettingLIFNeuron(tau_theta=tau_theta)
        for _ in range(5000):
            n.step(20.0)
        assert np.isfinite(n.theta)

    @pytest.mark.parametrize("tau_m", [5.0, 10.0, 20.0])
    def test_tau_m_sweep(self, tau_m: float):
        n = NonResettingLIFNeuron(tau_m=tau_m)
        for _ in range(5000):
            n.step(20.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("dt", [0.05, 0.1, 0.2])
    def test_dt_stability(self, dt: float):
        n = NonResettingLIFNeuron(dt=dt)
        for _ in range(10_000):
            n.step(20.0)
        assert np.isfinite(n.v) and np.isfinite(n.theta)

    def test_larger_delta_theta_fewer_spikes(self):
        """Larger Δθ → stronger refractoriness → fewer spikes."""
        s_small = len(_run(NonResettingLIFNeuron(delta_theta=2.0), 20.0, 5000))
        s_large = len(_run(NonResettingLIFNeuron(delta_theta=15.0), 20.0, 5000))
        assert s_small >= s_large
