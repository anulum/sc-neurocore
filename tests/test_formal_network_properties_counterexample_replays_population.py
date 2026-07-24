# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (population) from former test_formal_network_properties_counterexample_replays.py

from __future__ import annotations

from tests.formal_network_properties_support import *  # noqa: F403

def test_population_coactivation_replay_detects_too_many_simultaneous_outputs() -> None:
    prop = NetworkPopulationCoactivationCap(
        name="population_coactivation_cap",
        max_active_outputs=1,
    )

    replay = replay_population_coactivation_counterexample(
        [[True, False, True], [False, True, False]],
        prop,
    )

    assert replay.violated
    assert replay.first_violation_cycle == 0
    assert replay.observed_active_outputs == 2
    assert replay.max_active_outputs == 1
    assert replay.cycles_checked == 1


def test_population_coactivation_replay_accepts_outputs_within_cap() -> None:
    prop = NetworkPopulationCoactivationCap(
        name="population_coactivation_cap",
        max_active_outputs=1,
    )

    replay = replay_population_coactivation_counterexample(
        [[True, False, False], [False, True, False], [False, False, True]],
        prop,
    )

    assert not replay.violated
    assert replay.first_violation_cycle is None
    assert replay.observed_active_outputs == 1
    assert replay.max_active_outputs == 1
    assert replay.cycles_checked == 3


def test_population_silence_replay_detects_spike_after_coactivation() -> None:
    prop = NetworkPopulationSilenceAfterCoactivation(
        name="population_silence_after_coactivation",
        trigger_active_outputs=2,
        silence_cycles=2,
    )

    replay = replay_population_silence_counterexample(
        [[True, True, False], [False, False, False], [False, True, False]],
        prop,
    )

    assert replay.violated
    assert replay.first_violation_cycle == 2
    assert replay.trigger_cycle == 0
    assert replay.observed_active_outputs == 1
    assert replay.remaining_silence_cycles == 1
    assert replay.trigger_active_outputs == 2
    assert replay.silence_cycles == 2
    assert replay.cycles_checked == 3


def test_population_silence_replay_accepts_silent_window() -> None:
    prop = NetworkPopulationSilenceAfterCoactivation(
        name="population_silence_after_coactivation",
        trigger_active_outputs=2,
        silence_cycles=2,
    )

    replay = replay_population_silence_counterexample(
        [[True, True, False], [False, False, False], [False, False, False], [True, False, False]],
        prop,
    )

    assert not replay.violated
    assert replay.first_violation_cycle is None
    assert replay.trigger_cycle is None
    assert replay.observed_active_outputs == 0
    assert replay.remaining_silence_cycles == 0
    assert replay.cycles_checked == 4


def test_population_inactivity_replay_detects_too_many_silent_cycles() -> None:
    prop = NetworkPopulationInactivityBound(
        name="population_inactivity_bound",
        max_silent_cycles=2,
    )

    replay = replay_population_inactivity_counterexample(
        [
            [False, False],
            [False, False],
            [True, False],
            [False, False],
            [False, False],
            [False, False],
        ],
        prop,
    )

    assert replay.violated
    assert replay.first_violation_cycle == 5
    assert replay.observed_silent_cycles == 3
    assert replay.max_silent_cycles == 2
    assert replay.cycles_checked == 6


def test_population_inactivity_replay_accepts_bounded_silent_runs() -> None:
    prop = NetworkPopulationInactivityBound(
        name="population_inactivity_bound",
        max_silent_cycles=2,
    )

    replay = replay_population_inactivity_counterexample(
        [[False, False], [True, False], [False, False], [False, False]],
        prop,
    )

    assert not replay.violated
    assert replay.first_violation_cycle is None
    assert replay.observed_silent_cycles == 2
    assert replay.max_silent_cycles == 2
    assert replay.cycles_checked == 4


def test_population_coactivation_replay_accepts_scalar_samples() -> None:
    # Scalar samples are read as a single active output, so a unit cap holds.
    prop = NetworkPopulationCoactivationCap(
        name="population_coactivation_cap",
        max_active_outputs=1,
    )

    replay = replay_population_coactivation_counterexample([True, False, True], prop)

    assert not replay.violated
    assert replay.observed_active_outputs == 1
    assert replay.cycles_checked == 3


def test_population_coactivation_replay_rejects_string_sample() -> None:
    prop = NetworkPopulationCoactivationCap(
        name="population_coactivation_cap",
        max_active_outputs=1,
    )

    with pytest.raises(ValueError, match="must contain a binary spike sample"):
        replay_population_coactivation_counterexample(cast(Any, ["ab"]), prop)
