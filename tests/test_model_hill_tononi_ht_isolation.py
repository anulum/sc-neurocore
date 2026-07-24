# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHTIsolation from former test_model_hill_tononi.py

"""Focused suite: TestHTIsolation from former test_model_hill_tononi.py."""

from __future__ import annotations

from tests.model_hill_tononi_support import *  # noqa: F403


class TestHTIsolation:
    def test_defaults(self):
        n = HillTononiNeuron()
        assert n.v == -65.0 and n.na_i == 5.0
        assert n.h_na == 0.6 and n.n_k == 0.3
        assert n.m_h == 0.0 and n.h_t == 0.9
        assert n.g_kna == 1.33 and n.dt == 0.05

    def test_six_state_variables(self):
        n = HillTononiNeuron()
        for attr in ["v", "h_na", "n_k", "m_h", "h_t", "na_i"]:
            assert hasattr(n, attr)

    def test_step_returns_binary(self):
        assert HillTononiNeuron().step(0.0) in (0, 1)

    def test_state_finite_long_run(self):
        n = HillTononiNeuron()
        for _ in range(50_000):
            n.step(0.0)
        for attr in ["v", "h_na", "n_k", "m_h", "h_t", "na_i"]:
            assert np.isfinite(getattr(n, attr)), f"{attr} not finite"

    def test_reset_restores_defaults(self):
        n = HillTononiNeuron()
        for _ in range(5000):
            n.step(2.0)
        n.reset()
        assert n.v == -65.0 and n.na_i == 5.0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = HillTononiNeuron()
            trace = [(n.step(0.0), n.v, n.na_i) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]
