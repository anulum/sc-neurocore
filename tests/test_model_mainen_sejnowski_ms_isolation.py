# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMSIsolation from former test_model_mainen_sejnowski.py

"""Focused suite: TestMSIsolation from former test_model_mainen_sejnowski.py."""

from __future__ import annotations

from tests.model_mainen_sejnowski_support import *  # noqa: F403

class TestMSIsolation:
    def test_defaults(self):
        n = MainenSejnowskiNeuron()
        assert n.vs == -65.0 and n.va == -65.0
        assert n.m == 0.05 and n.h == 0.6 and n.n == 0.3
        assert n.g_na == 3000.0 and n.dt == 0.005

    def test_two_compartments(self):
        n = MainenSejnowskiNeuron()
        assert hasattr(n, "vs") and hasattr(n, "va")

    def test_step_returns_binary(self):
        assert MainenSejnowskiNeuron().step(0.0) in (0, 1)

    def test_state_finite(self):
        n = MainenSejnowskiNeuron()
        for _ in range(500):
            n.step(10.0)
        for attr in ["vs", "va", "m", "h", "n"]:
            assert np.isfinite(getattr(n, attr))

    def test_reset(self):
        n = MainenSejnowskiNeuron()
        for _ in range(200):
            n.step(10.0)
        n.reset()
        assert n.vs == -65.0 and n.va == -65.0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = MainenSejnowskiNeuron()
            trace = [(n.step(10.0), n.vs) for _ in range(100)]
            traces.append(trace)
        assert traces[0] == traces[1]
