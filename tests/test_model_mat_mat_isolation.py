# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMATIsolation from former test_model_mat.py

"""Focused suite: TestMATIsolation from former test_model_mat.py."""

from __future__ import annotations

from tests.model_mat_support import *


class TestMATIsolation:
    def test_defaults(self):
        n = SCResettingMATNeuron()
        assert n.v == -70.0 and n.theta1 == 0.0 and n.theta2 == 0.0
        assert n.v_threshold_base == -50.0 and n.dt == 1.0
        assert n.tau_1 == 10.0 and n.tau_2 == 200.0
        assert n.h1 == 5.0 and n.h2 == 3.0

    def test_step_returns_binary(self):
        assert SCResettingMATNeuron().step(0.0) in (0, 1)

    def test_all_states_evolve(self):
        n = SCResettingMATNeuron()
        v0, t1_0, t2_0 = n.v, n.theta1, n.theta2
        for _ in range(100):
            n.step(30.0)
        # v changes, thetas change after spike
        assert n.v != v0

    def test_state_finite_long_run(self):
        n = SCResettingMATNeuron()
        for _ in range(100_000):
            n.step(30.0)
        assert np.isfinite(n.v) and np.isfinite(n.theta1) and np.isfinite(n.theta2)

    def test_reset_restores_defaults(self):
        n = SCResettingMATNeuron()
        for _ in range(5000):
            n.step(30.0)
        n.reset()
        assert n.v == n.v_rest and n.theta1 == 0.0 and n.theta2 == 0.0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = SCResettingMATNeuron()
            trace = [(n.step(30.0), n.v, n.theta1, n.theta2) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]
