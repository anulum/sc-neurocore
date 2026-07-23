# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTermanWangIsolation from former test_model_terman_wang.py

"""Focused suite: TestTermanWangIsolation from former test_model_terman_wang.py."""

from __future__ import annotations

from tests.model_terman_wang_support import *  # noqa: F403

class TestTermanWangIsolation:
    def test_defaults(self):
        n = TermanWangOscillator()
        assert n.v == -1.5 and n.w == -0.5
        assert n.alpha == 3.0 and n.beta == 0.2 and n.epsilon == 0.02

    def test_step_returns_binary(self):
        assert TermanWangOscillator().step(0.0) in (0, 1)

    def test_two_variables_evolve(self):
        n = TermanWangOscillator()
        v0, w0 = n.v, n.w
        for _ in range(500):
            n.step(1.0)
        assert n.v != v0 and n.w != w0

    def test_state_finite(self):
        n = TermanWangOscillator()
        for _ in range(100000):
            n.step(1.0)
        assert np.isfinite(n.v) and np.isfinite(n.w)

    def test_reset(self):
        n = TermanWangOscillator()
        for _ in range(1000):
            n.step(1.0)
        n.reset()
        assert n.v == -1.5 and n.w == -0.5
