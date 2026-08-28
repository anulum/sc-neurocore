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
        assert n.alpha == 0.0

    def test_step_returns_binary(self):
        n = CazellesMapNeuron()
        assert n.step(0.0) in (0, 1)

    def test_spikes_under_drive(self):
        n = CazellesMapNeuron()
        spikes = sum(n.step(0.0) for _ in range(5000))
        assert spikes > 50

    def test_source_orbit_visits_fast_and_slow_regimes(self):
        n = CazellesMapNeuron()
        trace, events = n.simulate(600, backend="python")
        assert events == 7
        assert np.any(trace < n.x1)
        assert np.any(trace >= n.x1)

    def test_x_clipped(self):
        """The source orbit stays in the Figure-1 domain."""
        n = CazellesMapNeuron()
        for _ in range(10000):
            n.step(0.0)
        assert n.x0 <= n.x <= n.x4

    def test_state_finite(self):
        n = CazellesMapNeuron()
        for _ in range(10000):
            n.step(0.0)
        assert np.isfinite(n.x)

    def test_reset(self):
        n = CazellesMapNeuron()
        for _ in range(100):
            n.step(0.0)
        n.reset()
        assert n.x == 0.1
