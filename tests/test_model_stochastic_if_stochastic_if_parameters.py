# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestStochasticIFParameters from former test_model_stochastic_if.py

"""Focused suite: TestStochasticIFParameters from former test_model_stochastic_if.py."""

from __future__ import annotations

from tests.model_stochastic_if_support import *  # noqa: F403


class TestStochasticIFParameters:
    def test_tau_m_affects_dynamics(self):
        n_fast = StochasticIFNeuron(tau_m=5.0, sigma=0.0)
        n_slow = StochasticIFNeuron(tau_m=40.0, sigma=0.0)
        s_fast = len(_run(n_fast, current=25.0, steps=10000))
        s_slow = len(_run(n_slow, current=25.0, steps=10000))
        assert s_fast > s_slow

    def test_mu_shifts_baseline(self):
        """mu adds constant offset to the input."""
        n = StochasticIFNeuron(sigma=0.0, mu=10.0)
        # Effective input = mu + I = 10 + 15 = 25, same as I=25 with mu=0
        n2 = StochasticIFNeuron(sigma=0.0, mu=0.0)
        s1 = len(_run(n, current=15.0, steps=10000))
        s2 = len(_run(n2, current=25.0, steps=10000))
        assert s1 == s2

    @pytest.mark.parametrize("dt", [0.5, 1.0, 2.0])
    def test_dt_stability(self, dt: float):
        n = StochasticIFNeuron(dt=dt)
        for _ in range(10000):
            n.step(25.0)
        assert np.isfinite(n.v)
