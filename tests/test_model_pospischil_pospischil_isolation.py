# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPospischilIsolation from former test_model_pospischil.py

"""Focused suite: TestPospischilIsolation from former test_model_pospischil.py."""

from __future__ import annotations

from tests.model_pospischil_support import *  # noqa: F403

class TestPospischilIsolation:
    def test_construction_defaults(self):
        n = PospischilNeuron()
        assert n.v == -70.0
        assert n.g_m == 0.07  # RS type
        assert n.g_na == 50.0
        assert n.dt == 0.025
        assert n.v_threshold == -20.0

    def test_step_returns_binary(self):
        assert PospischilNeuron().step(0.0) in (0, 1)

    def test_five_state_variables_evolve(self):
        n = PospischilNeuron()
        initial = (n.v, n.m, n.h, n.n, n.p)
        for _ in range(500):
            n.step(5.0)
        for name, v0, v1 in zip(["v", "m", "h", "n", "p"], initial, (n.v, n.m, n.h, n.n, n.p)):
            assert v0 != v1, f"{name} didn't evolve"

    def test_state_finite_long_run(self):
        n = PospischilNeuron()
        for _ in range(50000):
            n.step(10.0)
        for var in [n.v, n.m, n.h, n.n, n.p]:
            assert np.isfinite(var)

    def test_reset_restores_initial(self):
        n = PospischilNeuron()
        for _ in range(1000):
            n.step(10.0)
        n.reset()
        assert n.v == -70.0
        assert n.p == 0.0

    def test_substep_integration(self):
        """Uses 4 sub-steps per step() call."""
        n = PospischilNeuron()
        v0 = n.v
        n.step(5.0)
        assert n.v != v0  # Integration happened
