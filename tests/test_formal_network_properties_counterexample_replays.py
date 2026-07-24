# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (counterexample_replays) from former test_formal_network_properties.py

from __future__ import annotations

from tests.formal_network_properties_support import *  # noqa: F403

def test_counterexample_replay_detects_aligned_window_rate_violation() -> None:
    prop = NetworkRateBound(
        name="output0_rate_bound",
        output_index=0,
        window_cycles=4,
        max_spikes=2,
    )

    replay = replay_rate_bound_counterexample([True, False, True, True], prop)

    assert replay.violated
    assert replay.first_violation_cycle == 3
    assert replay.window_start_cycle == 0
    assert replay.observed_spikes == 3


def test_counterexample_replay_selects_monitored_output_index() -> None:
    prop = NetworkRateBound(
        name="output1_rate_bound",
        output_index=1,
        window_cycles=4,
        max_spikes=1,
    )

    replay = replay_rate_bound_counterexample(
        [
            [True, False],
            [True, True],
            [False, False],
            [False, True],
        ],
        prop,
    )

    assert replay.violated
    assert replay.first_violation_cycle == 3
    assert replay.observed_spikes == 2


def test_counterexample_replay_resets_on_aligned_windows() -> None:
    prop = NetworkRateBound(
        name="output0_rate_bound",
        output_index=0,
        window_cycles=4,
        max_spikes=2,
    )

    replay = replay_rate_bound_counterexample([True, True, False, False, True, True], prop)

    assert not replay.violated
    assert replay.first_violation_cycle is None
    assert replay.observed_spikes == 2


def test_counterexample_replay_rejects_non_binary_trace_values() -> None:
    prop = NetworkRateBound(
        name="output0_rate_bound",
        output_index=0,
        window_cycles=4,
        max_spikes=2,
    )

    with pytest.raises(ValueError, match="binary spike"):
        replay_rate_bound_counterexample([0, 2], prop)


def test_refractory_replay_detects_spike_inside_refractory_window() -> None:
    prop = NetworkRefractoryInvariant(
        name="output0_refractory",
        output_index=0,
        refractory_cycles=3,
    )

    replay = replay_refractory_counterexample([True, False, True, False], prop)

    assert replay.violated
    assert replay.first_violation_cycle == 2
    assert replay.trigger_cycle == 0
    assert replay.remaining_refractory_cycles == 2
    assert replay.cycles_checked == 3


def test_refractory_replay_accepts_spike_after_refractory_window() -> None:
    prop = NetworkRefractoryInvariant(
        name="output0_refractory",
        output_index=0,
        refractory_cycles=2,
    )

    replay = replay_refractory_counterexample([True, False, False, True], prop)

    assert not replay.violated
    assert replay.first_violation_cycle is None
    assert replay.trigger_cycle is None
    assert replay.cycles_checked == 4


def test_refractory_replay_selects_monitored_output_index() -> None:
    prop = NetworkRefractoryInvariant(
        name="output1_refractory",
        output_index=1,
        refractory_cycles=2,
    )

    replay = replay_refractory_counterexample(
        [[True, True], [True, False], [False, True]],
        prop,
    )

    assert replay.violated
    assert replay.first_violation_cycle == 2
    assert replay.trigger_cycle == 0


def test_antagonistic_replay_detects_simultaneous_outputs() -> None:
    prop = NetworkAntagonisticOutputExclusion(
        name="motor_left_right_exclusion",
        output_a=0,
        output_b=1,
    )

    replay = replay_antagonistic_counterexample(
        [[True, False], [False, True], [True, True]],
        prop,
    )

    assert replay.violated
    assert replay.first_violation_cycle == 2
    assert replay.output_a == 0
    assert replay.output_b == 1
    assert replay.cycles_checked == 3


def test_antagonistic_replay_accepts_mutually_exclusive_outputs() -> None:
    prop = NetworkAntagonisticOutputExclusion(
        name="motor_left_right_exclusion",
        output_a=0,
        output_b=1,
    )

    replay = replay_antagonistic_counterexample(
        [[True, False], [False, True], [False, False]],
        prop,
    )

    assert not replay.violated
    assert replay.first_violation_cycle is None
    assert replay.cycles_checked == 3


def test_temporal_separation_replay_detects_bounded_window_violation() -> None:
    prop = NetworkOutputTemporalSeparation(
        name="motor_left_right_temporal_separation",
        output_a=0,
        output_b=1,
        separation_cycles=2,
    )

    replay = replay_temporal_separation_counterexample(
        [[True, False], [False, True], [False, False]],
        prop,
    )

    assert replay.violated
    assert replay.first_violation_cycle == 1
    assert replay.trigger_output == 0
    assert replay.violating_output == 1
    assert replay.remaining_separation_cycles == 2
    assert replay.cycles_checked == 2


def test_temporal_separation_replay_rejects_simultaneous_outputs() -> None:
    prop = NetworkOutputTemporalSeparation(
        name="motor_left_right_temporal_separation",
        output_a=0,
        output_b=1,
        separation_cycles=2,
    )

    replay = replay_temporal_separation_counterexample([[True, True]], prop)

    assert replay.violated
    assert replay.first_violation_cycle == 0
    assert replay.trigger_output is None
    assert replay.violating_output is None


def test_temporal_separation_replay_accepts_outputs_after_window() -> None:
    prop = NetworkOutputTemporalSeparation(
        name="motor_left_right_temporal_separation",
        output_a=0,
        output_b=1,
        separation_cycles=2,
    )

    replay = replay_temporal_separation_counterexample(
        [[True, False], [False, False], [False, False], [False, True]],
        prop,
    )

    assert not replay.violated
    assert replay.first_violation_cycle is None
    assert replay.cycles_checked == 4


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


def test_temporal_separation_replay_detects_reverse_window_violation() -> None:
    # output_b fires first, then output_a fires inside the separation window —
    # the symmetric (b-triggers-a) violation branch.
    prop = NetworkOutputTemporalSeparation(
        name="motor_left_right_temporal_separation",
        output_a=0,
        output_b=1,
        separation_cycles=2,
    )

    replay = replay_temporal_separation_counterexample(
        [[False, True], [True, False]],
        prop,
    )

    assert replay.violated
    assert replay.first_violation_cycle == 1
    assert replay.trigger_output == 1
    assert replay.violating_output == 0
    assert replay.remaining_separation_cycles == 2
    assert replay.cycles_checked == 2


def test_temporal_separation_replay_decrements_output_b_window() -> None:
    # output_b fires, then quiet cycles run its window down to zero before
    # output_a fires after the window has closed — no violation, exercising the
    # output_b countdown branch.
    prop = NetworkOutputTemporalSeparation(
        name="motor_left_right_temporal_separation",
        output_a=0,
        output_b=1,
        separation_cycles=2,
    )

    replay = replay_temporal_separation_counterexample(
        [[False, True], [False, False], [False, False], [True, False]],
        prop,
    )

    assert not replay.violated
    assert replay.first_violation_cycle is None
    assert replay.cycles_checked == 4


def test_temporal_separation_replay_rejects_scalar_sample_for_second_output() -> None:
    # A scalar sample only carries output_index 0, so monitoring output_b=1
    # raises rather than silently misreading the trace.
    prop = NetworkOutputTemporalSeparation(
        name="motor_left_right_temporal_separation",
        output_a=0,
        output_b=1,
        separation_cycles=2,
    )

    with pytest.raises(ValueError, match="only support output_index 0"):
        replay_temporal_separation_counterexample([True], prop)


def test_temporal_separation_replay_rejects_string_sample() -> None:
    prop = NetworkOutputTemporalSeparation(
        name="motor_left_right_temporal_separation",
        output_a=0,
        output_b=1,
        separation_cycles=2,
    )

    with pytest.raises(ValueError, match="must contain a binary spike sample"):
        replay_temporal_separation_counterexample(cast(Any, ["ab"]), prop)


def test_temporal_separation_replay_rejects_out_of_range_output_index() -> None:
    prop = NetworkOutputTemporalSeparation(
        name="motor_left_right_temporal_separation",
        output_a=0,
        output_b=1,
        separation_cycles=2,
    )

    with pytest.raises(ValueError, match="does not contain output_index 1"):
        replay_temporal_separation_counterexample([[True]], prop)


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


