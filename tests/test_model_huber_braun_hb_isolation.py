# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHBIsolation from former test_model_huber_braun.py

"""Focused suite: TestHBIsolation from former test_model_huber_braun.py."""

from __future__ import annotations

from tests.model_huber_braun_support import *  # noqa: F403

class TestHBIsolation:
    def test_defaults(self):
        n = HuberBraunNeuron()
        assert n.v == -50.0 and n.a_sd == 0.0 and n.a_sr == 0.0
        assert n.g_sd == 1.5 and n.g_sr == 0.4 and n.g_l == 0.1
        assert n.eta == 0.012 and n.dt == 0.1

    def test_step_returns_binary(self):
        assert HuberBraunNeuron().step(0.0) in (0, 1)

    def test_state_finite_long_run(self):
        n = HuberBraunNeuron()
        for _ in range(50_000):
            n.step(50.0)
        assert np.isfinite(n.v) and np.isfinite(n.a_sd) and np.isfinite(n.a_sr)

    def test_reset_restores_defaults(self):
        n = HuberBraunNeuron()
        for _ in range(2000):
            n.step(50.0)
        n.reset()
        assert n.v == -50.0 and n.a_sd == 0.0 and n.a_sr == 0.0

    def test_stochastic_noise_present(self):
        """η·randn() noise → voltage diverges from deterministic path."""
        n1 = HuberBraunNeuron(eta=0.0)  # no noise
        n2 = HuberBraunNeuron(eta=0.012)  # with noise
        for _ in range(500):
            n1.step(50.0)
            n2.step(50.0)
        # With noise, voltages should differ
        assert n1.v != n2.v
