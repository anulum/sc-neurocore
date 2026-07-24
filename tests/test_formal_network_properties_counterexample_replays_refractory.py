# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (refractory) from former test_formal_network_properties_counterexample_replays.py

from __future__ import annotations

from tests.formal_network_properties_support import *  # noqa: F403

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
