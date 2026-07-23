# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNRLIFIsolation from former test_model_non_resetting_lif.py

"""Focused suite: TestNRLIFIsolation from former test_model_non_resetting_lif.py."""

from __future__ import annotations

from tests.model_non_resetting_lif_support import *  # noqa: F403

class TestNRLIFIsolation:
    def test_defaults(self):
        n = NonResettingLIFNeuron()
        assert n.v == -65.0 and n.theta == -50.0
        assert n.v_rest == -65.0 and n.theta_rest == -50.0
        assert n.delta_theta == 5.0
        assert n.tau_m == 10.0 and n.tau_theta == 50.0
        assert n.r_m == 1.0 and n.dt == 0.1

    def test_step_returns_binary(self):
        assert NonResettingLIFNeuron().step(0.0) in (0, 1)

    def test_state_finite_long_run(self):
        n = NonResettingLIFNeuron()
        for _ in range(100_000):
            n.step(20.0)
        assert np.isfinite(n.v) and np.isfinite(n.theta)

    def test_reset_restores_defaults(self):
        n = NonResettingLIFNeuron()
        for _ in range(5000):
            n.step(20.0)
        n.reset()
        assert n.v == n.v_rest and n.theta == n.theta_rest

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = NonResettingLIFNeuron()
            trace = [(n.step(20.0), n.v, n.theta) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]
