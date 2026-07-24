# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFHNParameters from former test_model_fitzhugh_nagumo.py

"""Focused suite: TestFHNParameters from former test_model_fitzhugh_nagumo.py."""

from __future__ import annotations

from tests.model_fitzhugh_nagumo_support import *  # noqa: F403


class TestFHNParameters:
    def test_epsilon_controls_timescale(self):
        n_fast = FitzHughNagumoNeuron(epsilon=0.2)
        n_slow = FitzHughNagumoNeuron(epsilon=0.02)
        s_fast = len(_run(n_fast, current=0.8, steps=10000))
        s_slow = len(_run(n_slow, current=0.8, steps=10000))
        assert s_fast != s_slow

    def test_a_shifts_w_nullcline(self):
        n1 = FitzHughNagumoNeuron(a=0.5)
        n2 = FitzHughNagumoNeuron(a=1.0)
        s1 = len(_run(n1, current=0.5, steps=10000))
        s2 = len(_run(n2, current=0.5, steps=10000))
        assert s1 != s2

    @pytest.mark.parametrize("dt", [0.05, 0.1, 0.2])
    def test_dt_stability(self, dt: float):
        n = FitzHughNagumoNeuron(dt=dt)
        for _ in range(10000):
            n.step(0.8)
        assert np.isfinite(n.v)

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = FitzHughNagumoNeuron()
            trace = [(n.step(0.8), n.v, n.w) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
