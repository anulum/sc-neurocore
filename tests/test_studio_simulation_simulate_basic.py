# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio simulation simulate basic

"""Focused suite: TestSimulateBasic from former test_studio_simulation.py."""

from __future__ import annotations

from tests.studio_simulation_support import *  # noqa: F403

class TestSimulateBasic:
    def test_lif_produces_spikes(self):
        t = TEMPLATES["lif"]
        result = simulate(
            equations=t["equations"],
            threshold=t["threshold"],
            reset=t["reset"],
            params=t["params"],
            init=t["init"],
            dt=t["dt"],
            duration=t["duration"],
            current=t["current"],
        )
        assert result["spike_count"] > 0
        assert len(result["spikes"]) == result["spike_count"]

    def test_result_has_required_keys(self):
        result = simulate(
            equations=["dv/dt = I"],
            init={"v": 0.0},
            dt=0.1,
            duration=10.0,
            current=1.0,
        )
        for key in ("time", "states", "spikes", "spike_count", "dt", "n_steps"):
            assert key in result

    def test_time_length_matches_states(self):
        result = simulate(
            equations=["dv/dt = I"],
            init={"v": 0.0},
            dt=0.1,
            duration=10.0,
            current=1.0,
        )
        n = len(result["time"])
        assert n == len(result["states"]["v"])

    def test_zero_current_no_spikes_lif(self):
        t = TEMPLATES["lif"]
        result = simulate(
            equations=t["equations"],
            threshold=t["threshold"],
            reset=t["reset"],
            params=t["params"],
            init=t["init"],
            dt=t["dt"],
            duration=50.0,
            current=0.0,
        )
        assert result["spike_count"] == 0

    def test_n_steps_matches_duration(self):
        result = simulate(
            equations=["dv/dt = I"],
            init={"v": 0.0},
            dt=0.5,
            duration=10.0,
            current=0.0,
        )
        assert result["n_steps"] == 20

