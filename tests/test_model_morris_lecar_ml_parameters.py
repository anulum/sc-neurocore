# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMLParameters from former test_model_morris_lecar.py

"""Focused suite: TestMLParameters from former test_model_morris_lecar.py."""

from __future__ import annotations

from tests.model_morris_lecar_support import *  # noqa: F403


class TestMLParameters:
    @pytest.mark.parametrize("g_ca", [2.0, 4.0, 6.0])
    def test_g_ca_sweep(self, g_ca: float):
        n = MorrisLecarNeuron(g_ca=g_ca)
        for _ in range(20_000):
            n.step(100.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("g_k", [4.0, 8.0, 12.0])
    def test_g_k_sweep(self, g_k: float):
        n = MorrisLecarNeuron(g_k=g_k)
        for _ in range(20_000):
            n.step(100.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("phi", [0.04, 1.0 / 15.0, 0.1])
    def test_phi_timescale(self, phi: float):
        n = MorrisLecarNeuron(phi=phi)
        for _ in range(20_000):
            n.step(100.0)
        assert np.isfinite(n.v) and np.isfinite(n.w)

    @pytest.mark.parametrize("dt", [0.05, 0.1, 0.2])
    def test_dt_stability(self, dt: float):
        n = MorrisLecarNeuron(dt=dt)
        for _ in range(20_000):
            n.step(100.0)
        assert np.isfinite(n.v) and np.isfinite(n.w)

    def test_reversal_ordering(self):
        n = MorrisLecarNeuron()
        assert n.e_k < n.e_l < n.e_ca
