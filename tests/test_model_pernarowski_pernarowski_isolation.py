# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPernarowskiIsolation from former test_model_pernarowski.py

"""Focused suite: TestPernarowskiIsolation from former test_model_pernarowski.py."""

from __future__ import annotations

from tests.model_pernarowski_support import *  # noqa: F403


class TestPernarowskiIsolation:
    def test_construction_defaults(self):
        n = PernarowskiNeuron()
        assert n.v == -1.0
        assert n.w == 0.0
        assert n.z == 0.0
        assert n.eps1 == 0.1
        assert n.eps2 == 0.001
        assert n.v_threshold == 0.5

    def test_step_returns_binary(self):
        n = PernarowskiNeuron()
        assert n.step(0.0) in (0, 1)

    def test_three_state_variables_evolve(self):
        """All three state variables (V, w, z) should change after steps."""
        n = PernarowskiNeuron()
        v0, w0, z0 = n.v, n.w, n.z
        for _ in range(100):
            n.step(0.5)
        assert n.v != v0
        assert n.w != w0
        assert n.z != z0

    def test_state_finite_long_run(self):
        """No divergence over 50k steps."""
        n = PernarowskiNeuron()
        for _ in range(50000):
            n.step(0.5)
        assert np.isfinite(n.v)
        assert np.isfinite(n.w)
        assert np.isfinite(n.z)

    def test_reset_restores_initial(self):
        n = PernarowskiNeuron()
        for _ in range(500):
            n.step(1.0)
        n.reset()
        assert n.v == -1.0
        assert n.w == 0.0
        assert n.z == 0.0
