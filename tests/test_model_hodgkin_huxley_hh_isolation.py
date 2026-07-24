# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHHIsolation from former test_model_hodgkin_huxley.py

"""Focused suite: TestHHIsolation from former test_model_hodgkin_huxley.py."""

from __future__ import annotations

from tests.model_hodgkin_huxley_support import *  # noqa: F403


class TestHHIsolation:
    def test_defaults(self):
        n = HodgkinHuxleyNeuron()
        assert n.v == -65.0 and n.m == 0.05 and n.h == 0.6 and n.n == 0.32
        assert n.g_na == 120.0 and n.g_k == 36.0 and n.g_l == 0.3
        assert n.e_na == 50.0 and n.e_k == -77.0 and n.e_l == -54.4

    def test_step_returns_binary(self):
        assert HodgkinHuxleyNeuron().step(0.0) in (0, 1)

    def test_four_variables_evolve(self):
        n = HodgkinHuxleyNeuron()
        initial = (n.v, n.m, n.h, n.n)
        for _ in range(100):
            n.step(10.0)
        for name, v0, v1 in zip(["v", "m", "h", "n"], initial, (n.v, n.m, n.h, n.n)):
            assert v0 != v1, f"{name} didn't evolve"

    def test_state_finite(self):
        n = HodgkinHuxleyNeuron()
        for _ in range(5000):
            n.step(10.0)
        for var in [n.v, n.m, n.h, n.n]:
            assert np.isfinite(var)

    def test_reset(self):
        n = HodgkinHuxleyNeuron()
        for _ in range(100):
            n.step(10.0)
        n.reset()
        assert n.v == -65.0 and n.m == 0.05 and n.h == 0.6 and n.n == 0.32

    def test_100_substeps(self):
        """round(1.0/dt) = 100 sub-steps per step() call."""
        n = HodgkinHuxleyNeuron(dt=0.01)
        assert round(1.0 / n.dt) == 100
