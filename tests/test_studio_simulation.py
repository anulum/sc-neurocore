# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Studio simulation engine

from __future__ import annotations

import pytest

from sc_neurocore.studio.simulation import MAX_PLOT_POINTS, MAX_STEPS, simulate
from sc_neurocore.studio.templates import TEMPLATES


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


class TestSimulateAllTemplates:
    @pytest.mark.parametrize("name", list(TEMPLATES.keys()))
    def test_template_runs_without_error(self, name):
        t = TEMPLATES[name]
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
        assert len(result["time"]) > 0
        assert all(isinstance(v, list) for v in result["states"].values())


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
