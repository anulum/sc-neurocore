# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHBParameters from former test_model_huber_braun.py

"""Focused suite: TestHBParameters from former test_model_huber_braun.py."""

from __future__ import annotations

from tests.model_huber_braun_support import *  # noqa: F403

class TestHBParameters:
    @pytest.mark.parametrize("g_sd", [0.5, 1.5, 3.0])
    def test_g_sd_sweep(self, g_sd: float):
        n = HuberBraunNeuron(g_sd=g_sd)
        for _ in range(5000):
            n.step(50.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("eta", [0.0, 0.012, 0.05])
    def test_eta_noise_sweep(self, eta: float):
        n = HuberBraunNeuron(eta=eta)
        for _ in range(5000):
            n.step(50.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("dt", [0.05, 0.1, 0.2])
    def test_dt_stability(self, dt: float):
        n = HuberBraunNeuron(dt=dt)
        for _ in range(10_000):
            n.step(50.0)
        assert np.isfinite(n.v)
