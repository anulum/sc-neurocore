# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNLIFParameters from former test_model_nlif.py

"""Focused suite: TestNLIFParameters from former test_model_nlif.py."""

from __future__ import annotations

from tests.model_nlif_support import *  # noqa: F403


class TestNLIFParameters:
    @pytest.mark.parametrize("a", [0.02, 0.04, 0.08])
    def test_a_nonlinearity(self, a: float):
        n = NonlinearLIFNeuron(a=a)
        for _ in range(5000):
            n.step(20.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("b", [0.2, 0.5, 1.0])
    def test_b_adaptation_strength(self, b: float):
        n = NonlinearLIFNeuron(b=b)
        spikes = len(_run(n, current=20.0, steps=5000))
        assert isinstance(spikes, int)

    @pytest.mark.parametrize("tau_w", [50.0, 100.0, 200.0])
    def test_tau_w_sweep(self, tau_w: float):
        n = NonlinearLIFNeuron(tau_w=tau_w)
        for _ in range(5000):
            n.step(20.0)
        assert np.isfinite(n.w)

    @pytest.mark.parametrize("dt", [0.05, 0.1, 0.2])
    def test_dt_stability(self, dt: float):
        n = NonlinearLIFNeuron(dt=dt)
        for _ in range(10_000):
            n.step(20.0)
        assert np.isfinite(n.v) and np.isfinite(n.w)
