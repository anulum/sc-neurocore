# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio equation simulation contracts

"""Focused suite: TestSimulateBasic from former test_studio_simulation.py."""

from __future__ import annotations

from sc_neurocore.neurons.seed_domain import DEFAULT_LFSR16_SEED
from sc_neurocore.studio.simulation import simulate
from sc_neurocore.studio.templates import TEMPLATES


class TestSimulateBasic:
    """Basic results of the Studio equation simulator."""

    def test_lif_produces_spikes(self) -> None:
        """The LIF template spikes, and the spike list has one entry per counted spike."""
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

    def test_result_has_required_keys(self) -> None:
        """A result carries time, states, spikes, spike count, dt and step count."""
        result = simulate(
            equations=["dv/dt = I"],
            init={"v": 0.0},
            dt=0.1,
            duration=10.0,
            current=1.0,
        )
        for key in ("time", "states", "spikes", "spike_count", "dt", "n_steps"):
            assert key in result

    def test_time_length_matches_states(self) -> None:
        """Every state trace has one sample per time point."""
        result = simulate(
            equations=["dv/dt = I"],
            init={"v": 0.0},
            dt=0.1,
            duration=10.0,
            current=1.0,
        )
        n = len(result["time"])
        assert n == len(result["states"]["v"])

    def test_zero_current_no_spikes_lif(self) -> None:
        """The LIF template does not spike without input current."""
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

    def test_n_steps_matches_duration(self) -> None:
        """A 10 ms run at dt 0.5 ms takes 20 steps."""
        result = simulate(
            equations=["dv/dt = I"],
            init={"v": 0.0},
            dt=0.5,
            duration=10.0,
            current=0.0,
        )
        assert result["n_steps"] == 20

    def test_default_seed_replays_complete_stochastic_state_and_events(self) -> None:
        """Omitted and published default keys replay; distinct counter keys do not."""
        runs = [
            simulate(
                equations=["dv/dt = rand() + I", "dw/dt = v - w"],
                threshold="v > 2.5",
                reset="v = 0",
                init={"v": 0.75, "w": -0.25},
                dt=0.125,
                duration=8.0,
                current=0.5,
                seed=seed,
            )
            for seed in (None, DEFAULT_LFSR16_SEED, 44257, 1, 0)
        ]
        assert runs[0] == runs[1] == runs[2]
        assert runs[0]["spike_count"] > 0
        assert runs[0]["raw"] != runs[3]["raw"]
        assert runs[0]["raw"] != runs[4]["raw"]
