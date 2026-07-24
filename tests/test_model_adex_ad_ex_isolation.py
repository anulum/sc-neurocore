# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAdExIsolation from former test_model_adex.py

"""Focused suite: TestAdExIsolation from former test_model_adex.py."""

from __future__ import annotations

from tests.model_adex_support import *  # noqa: F403


class TestAdExIsolation:
    def test_construction_defaults(self):
        n = AdExNeuron()
        assert n.v == -65.0
        assert n.w == 0.0
        assert n.delta_t == 2.0
        assert n.a == 0.5
        assert n.b == 7.0
        assert n.c_m == 200.0

    def test_step_returns_binary(self):
        assert AdExNeuron().step(0.0) in (0, 1)

    def test_two_variables_evolve(self):
        n = AdExNeuron()
        v0, w0 = n.v, n.w
        for _ in range(100):
            n.step(500.0)
        assert n.v != v0 and n.w != w0

    def test_state_finite(self):
        n = AdExNeuron()
        for _ in range(50000):
            n.step(500.0)
        assert np.isfinite(n.v) and np.isfinite(n.w)

    def test_reset(self):
        n = AdExNeuron()
        for _ in range(100):
            n.step(500.0)
        n.reset()
        assert n.v == n.v_rest and n.w == 0.0

    def test_exp_clipped(self):
        """Exponential term is clipped to avoid overflow."""
        n = AdExNeuron()
        n.v = 100.0  # far above threshold → should clip exp
        n.step(0.0)
        assert np.isfinite(n.v)
