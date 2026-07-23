# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio simulation simulate edge cases

"""Focused suite: TestSimulateEdgeCases from former test_studio_simulation.py."""

from __future__ import annotations

from tests.studio_simulation_support import *  # noqa: F403

class TestSimulateEdgeCases:
    def test_max_steps_cap(self):
        result = simulate(
            equations=["dv/dt = I"],
            init={"v": 0.0},
            dt=0.001,
            duration=1_000_000.0,
            current=0.0,
        )
        assert result["n_steps"] == MAX_STEPS

    def test_downsampling_large_trace(self):
        n_steps = MAX_PLOT_POINTS * 3
        dt = 0.1
        duration = n_steps * dt
        result = simulate(
            equations=["dv/dt = I"],
            init={"v": 0.0},
            dt=dt,
            duration=duration,
            current=0.0,
        )
        assert len(result["time"]) <= MAX_PLOT_POINTS

    def test_invalid_duration_raises(self):
        with pytest.raises(ValueError, match="< 1 step"):
            simulate(
                equations=["dv/dt = I"],
                init={"v": 0.0},
                dt=1.0,
                duration=0.0,
                current=0.0,
            )

    def test_bad_equation_raises(self):
        with pytest.raises(ValueError):
            simulate(
                equations=["v = I"],
                init={"v": 0.0},
                dt=0.1,
                duration=10.0,
                current=0.0,
            )

    def test_multi_variable_states(self):
        t = TEMPLATES["izhikevich"]
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
        assert "v" in result["states"]
        assert "u" in result["states"]
        assert len(result["states"]["v"]) == len(result["states"]["u"])

