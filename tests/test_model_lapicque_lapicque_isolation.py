# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLapicqueIsolation from former test_model_lapicque.py

"""Focused suite: TestLapicqueIsolation from former test_model_lapicque.py."""

from __future__ import annotations

from tests.model_lapicque_support import *  # noqa: F403


class TestLapicqueIsolation:
    def test_defaults(self):
        n = LapicqueNeuron()
        assert n.v == 0.0 and n.v_rest == 0.0
        assert n.v_threshold == 1.0 and n.v_reset == 0.0
        assert n.tau == 20.0 and n.resistance == 1.0 and n.dt == 1.0

    def test_step_returns_binary(self):
        assert LapicqueNeuron().step(0.0) in (0, 1)

    def test_state_finite_long_run(self):
        n = LapicqueNeuron()
        for _ in range(100_000):
            n.step(20.0)
        assert np.isfinite(n.v)

    def test_reset_restores_default(self):
        n = LapicqueNeuron()
        for _ in range(100):
            n.step(20.0)
        n.reset()
        assert n.v == n.v_rest

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = LapicqueNeuron()
            trace = [(n.step(20.0), n.v) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]
