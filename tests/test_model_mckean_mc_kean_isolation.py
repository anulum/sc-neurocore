# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMcKeanIsolation from former test_model_mckean.py

"""Focused suite: TestMcKeanIsolation from former test_model_mckean.py."""

from __future__ import annotations

from tests.model_mckean_support import *  # noqa: F403

class TestMcKeanIsolation:
    def test_defaults(self):
        n = McKeanNeuron()
        assert n.v == 0.0 and n.w == 0.0
        assert n.a == 0.25 and n.epsilon == 0.01 and n.gamma == 0.5
        assert n.dt == 0.1 and n.v_peak == 0.8

    def test_step_returns_binary(self):
        assert McKeanNeuron().step(0.0) in (0, 1)

    def test_both_states_evolve(self):
        n = McKeanNeuron()
        v0, w0 = n.v, n.w
        for _ in range(100):
            n.step(0.5)
        assert n.v != v0 and n.w != w0

    def test_state_finite_long_run(self):
        n = McKeanNeuron()
        for _ in range(100_000):
            n.step(0.5)
        assert np.isfinite(n.v) and np.isfinite(n.w)

    def test_reset_restores_defaults(self):
        n = McKeanNeuron()
        for _ in range(5000):
            n.step(0.5)
        n.reset()
        assert n.v == 0.0 and n.w == 0.0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = McKeanNeuron()
            trace = [(n.step(0.5), n.v, n.w) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]

    def test_runtime_non_finite_state_fails_closed_without_mutating_w(self):
        n = McKeanNeuron()
        n.v = float("nan")
        before_w = n.w

        with pytest.raises(FloatingPointError, match="v must be finite"):
            n.step(0.5)

        assert np.isnan(n.v)
        assert n.w == before_w

    def test_runtime_update_overflow_fails_closed_without_mutating_state(self):
        n = McKeanNeuron(v=1e308, w=-1.7e308)
        before = (n.v, n.w)

        with pytest.raises(FloatingPointError, match="finite"):
            n.step(1.7e308)

        assert (n.v, n.w) == before
