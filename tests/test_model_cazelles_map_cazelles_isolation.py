# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCazellesIsolation from former test_model_cazelles_map.py

"""Focused suite: TestCazellesIsolation from former test_model_cazelles_map.py."""

from __future__ import annotations

from tests.model_cazelles_map_support import *  # noqa: F403

class TestCazellesIsolation:
    def test_construction(self):
        n = CazellesMapNeuron()
        assert n.x == 0.1
        assert n.y == 0.0

    def test_step_returns_binary(self):
        n = CazellesMapNeuron()
        assert n.step(0.0) in (0, 1)

    def test_spikes_under_drive(self):
        n = CazellesMapNeuron()
        spikes = sum(n.step(0.2) for _ in range(5000))
        assert spikes > 100

    def test_slow_variable_modulates(self):
        """y should change from initial under sustained drive."""
        n = CazellesMapNeuron()
        y_init = n.y
        for _ in range(1000):
            n.step(0.2)
        assert n.y != y_init

    def test_x_clipped(self):
        """x should stay in [-2, 2] (np.clip in step)."""
        n = CazellesMapNeuron()
        for _ in range(10000):
            n.step(1.0)
        assert -2.0 <= n.x <= 2.0

    def test_state_finite(self):
        n = CazellesMapNeuron()
        for _ in range(10000):
            n.step(0.5)
        assert np.isfinite(n.x)
        assert np.isfinite(n.y)

    def test_reset(self):
        n = CazellesMapNeuron()
        for _ in range(100):
            n.step(0.2)
        n.reset()
        assert n.x == 0.1
        assert n.y == 0.0
