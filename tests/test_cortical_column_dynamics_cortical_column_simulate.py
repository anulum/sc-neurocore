# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCorticalColumnSimulate from former test_cortical_column_dynamics.py

"""Focused suite: TestCorticalColumnSimulate from former test_cortical_column_dynamics.py."""

from __future__ import annotations

from tests.cortical_column_dynamics_support import *  # noqa: F403


class TestCorticalColumnSimulate:
    def test_simulate_returns_dict(self):
        col = CorticalColumn(scale=0.02, seed=42)
        result = col.simulate(duration_ms=5.0, dt=0.1)
        assert isinstance(result, dict)

    def test_simulate_shapes(self):
        col = CorticalColumn(scale=0.02, seed=42)
        dt = 0.1
        dur = 5.0
        n_steps = int(round(dur / dt))
        result = col.simulate(duration_ms=dur, dt=dt)
        for key, arr in result.items():
            assert arr.shape[0] == n_steps, f"{key}: {arr.shape[0]} rows != {n_steps}"

    def test_background_drive_produces_spikes(self):
        """Background Poisson input should produce at least some spikes."""
        col = CorticalColumn(scale=0.02, bg_rate=8.0, seed=42)
        result = col.simulate(duration_ms=50.0, dt=0.1)
        total_spikes = sum(arr.sum() for arr in result.values())
        assert total_spikes > 0, "no spikes at all with background drive"

    def test_no_background_minimal_activity(self):
        """With near-zero background, activity should be very low."""
        col = CorticalColumn(scale=0.02, bg_rate=0.01, seed=42)
        result = col.simulate(duration_ms=10.0, dt=0.1)
        total = sum(arr.sum() for arr in result.values())
        max_possible = sum(arr.size for arr in result.values())
        assert total < max_possible * 0.5, "too many spikes with minimal background"
