# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (temporal_separation) from former test_formal_network_properties_counterexample_replays.py

from __future__ import annotations

from tests.formal_network_properties_support import *  # noqa: F403


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
