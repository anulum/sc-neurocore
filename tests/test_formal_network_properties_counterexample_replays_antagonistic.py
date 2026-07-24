# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (antagonistic) from former test_formal_network_properties_counterexample_replays.py

from __future__ import annotations

from tests.formal_network_properties_support import *  # noqa: F403

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
