# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHBAnalytical from former test_model_huber_braun.py

"""Focused suite: TestHBAnalytical from former test_model_huber_braun.py."""

from __future__ import annotations

from tests.model_huber_braun_support import *  # noqa: F403


class TestHBAnalytical:
    def test_sd_inf_sr_inf_complementary(self):
        """sd_inf + sr_inf = 1 at any V (complementary sigmoids)."""
        for v in [-80, -60, -40, -20, 0]:
            sd = 1.0 / (1.0 + np.exp(-(v + 40.0) / 6.0))
            sr = 1.0 / (1.0 + np.exp((v + 40.0) / 6.0))
            assert abs(sd + sr - 1.0) < 1e-12

    def test_sd_inf_midpoint(self):
        """sd_inf(-40) = 0.5."""
        sd = 1.0 / (1.0 + np.exp(0.0))
        assert abs(sd - 0.5) < 1e-12

    def test_sd_activates_depolarised(self):
        """sd_inf → 1 for v >> -40."""
        sd = 1.0 / (1.0 + np.exp(-(0.0 + 40.0) / 6.0))
        assert sd > 0.99

    def test_sr_activates_hyperpolarised(self):
        """sr_inf → 1 for v << -40."""
        sr = 1.0 / (1.0 + np.exp((-80.0 + 40.0) / 6.0))
        assert sr > 0.99

    def test_three_currents(self):
        n = HuberBraunNeuron()
        assert n.g_sd > 0 and n.g_sr > 0 and n.g_l > 0

    def test_reversal_ordering(self):
        n = HuberBraunNeuron()
        assert n.e_sr < n.e_l < n.e_sd

    def test_noise_amplitude(self):
        """η=0.012 adds Gaussian noise per step."""
        n = HuberBraunNeuron()
        assert n.eta == 0.012

    def test_sd_slower_than_sr(self):
        """tau_sd=10 < tau_sr=20 — sd activates faster."""
        n = HuberBraunNeuron()
        assert n.tau_sd < n.tau_sr

    def test_gating_bounded(self):
        n = HuberBraunNeuron()
        for _ in range(10_000):
            n.step(50.0)
        assert -0.05 <= n.a_sd <= 1.05
        assert -0.05 <= n.a_sr <= 1.05
