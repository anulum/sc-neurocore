# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCNIsolation from former test_model_courage_nekorkin_map.py

"""Focused suite: TestCNIsolation from former test_model_courage_nekorkin_map.py."""

from __future__ import annotations

from tests.model_courage_nekorkin_map_support import *  # noqa: F403


class TestCNIsolation:
    def test_defaults(self):
        n = CourageNekorkinMapNeuron()
        assert n.x == 0.0 and n.y == 0.0
        assert n.m0 == 0.0864 and n.m1 == 0.65 and n.a == 0.2
        assert n.d == 0.235 and n.j == 0.2 and n.beta == 0.085 and n.eps == 0.02
        assert n.x_threshold == 0.235

    def test_default_regime_is_valid(self):
        """Defaults satisfy the published parameter region (eq. 6): Jmin<d<Jmax, 0<J<d, m0<1."""
        n = CourageNekorkinMapNeuron()
        jmin, jmax = _breakpoints()
        assert jmin < n.d < jmax
        assert 0.0 < n.j < n.d
        assert n.m0 < 1.0

    def test_step_returns_binary(self):
        assert CourageNekorkinMapNeuron().step(0.0) in (0, 1)

    def test_state_finite_long_run(self):
        n = CourageNekorkinMapNeuron()
        for _ in range(50_000):
            n.step(0.0)
        assert np.isfinite(n.x) and np.isfinite(n.y)

    def test_reset_restores_state(self):
        n = CourageNekorkinMapNeuron()
        for _ in range(1000):
            n.step(0.0)
        n.reset()
        assert n.x == 0.0 and n.y == 0.0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = CourageNekorkinMapNeuron()
            trace = [(n.step(0.0), n.x, n.y) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
