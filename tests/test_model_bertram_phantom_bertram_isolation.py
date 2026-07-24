# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBertramIsolation from former test_model_bertram_phantom.py

"""Focused suite: TestBertramIsolation from former test_model_bertram_phantom.py."""

from __future__ import annotations

from tests.model_bertram_phantom_support import *  # noqa: F403


class TestBertramIsolation:
    def test_defaults(self):
        n = BertramPhantomBurster()
        assert n.v == -50.0 and n.s1 == 0.1 and n.s2 == 0.1
        assert n.c_m == 5.3 and n.dt == 0.5
        assert n.v_threshold == -20.0

    def test_three_state_variables(self):
        """Model has v (fast), s1 (slow), s2 (ultra-slow)."""
        n = BertramPhantomBurster()
        assert hasattr(n, "v") and hasattr(n, "s1") and hasattr(n, "s2")

    def test_step_returns_binary(self):
        assert BertramPhantomBurster().step(0.0) in (0, 1)

    def test_all_states_evolve(self):
        n = BertramPhantomBurster()
        v0, s1_0, s2_0 = n.v, n.s1, n.s2
        for _ in range(1000):
            n.step(200.0)
        assert n.v != v0 and n.s1 != s1_0 and n.s2 != s2_0

    def test_state_finite_long_run(self):
        n = BertramPhantomBurster()
        for _ in range(100_000):
            n.step(200.0)
        assert np.isfinite(n.v) and np.isfinite(n.s1) and np.isfinite(n.s2)

    def test_reset_restores_defaults(self):
        n = BertramPhantomBurster()
        for _ in range(5000):
            n.step(200.0)
        n.reset()
        assert n.v == -50.0 and n.s1 == 0.1 and n.s2 == 0.1

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = BertramPhantomBurster()
            trace = [(n.step(200.0), n.v, n.s1, n.s2) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]
