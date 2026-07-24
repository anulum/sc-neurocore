# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (rate_window) from former test_formal_network_properties_counterexample_replays.py

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
