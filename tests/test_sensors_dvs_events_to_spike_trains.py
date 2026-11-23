# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEventsToSpikeTrains from former test_sensors_dvs.py

"""Focused suite: TestEventsToSpikeTrains from former test_sensors_dvs.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.sensors import (
    events_to_spike_trains,
)
from tests.sensors_dvs_support import (
    _make_events,
)


class TestEventsToSpikeTrains:
    """Turning events into spike trains."""

    def test_basic_shape(self) -> None:
        """Spike trains have one channel per pixel and polarity."""
        events = _make_events(n=50, width=4, height=3)
        spikes = events_to_spike_trains(events, width=4, height=3, dt_us=10000.0)
        n_channels = 4 * 3 * 2  # ON + OFF
        assert spikes.shape[1] == n_channels
        assert spikes.shape[0] >= 1

    def test_binary_output(self) -> None:
        """Spike trains hold only zeros and ones."""
        events = _make_events()
        spikes = events_to_spike_trains(events, width=8, height=6, dt_us=10000.0)
        assert set(np.unique(spikes)).issubset({0, 1})

    def test_explicit_duration(self) -> None:
        """An explicit duration sets the step count."""
        events = _make_events()
        spikes = events_to_spike_trains(
            events,
            width=8,
            height=6,
            dt_us=10000.0,
            duration_us=50000.0,
        )
        assert spikes.shape[0] == 5

    def test_on_off_channels(self) -> None:
        """ON and OFF events go to separate channels."""
        dtype = np.dtype([("x", np.int32), ("y", np.int32), ("t", np.int64), ("p", np.int8)])
        events = np.zeros(2, dtype=dtype)
        events[0] = (0, 0, 0, 1)  # ON event at pixel (0,0)
        events[1] = (1, 0, 0, 0)  # OFF event at pixel (1,0)
        spikes = events_to_spike_trains(events, width=2, height=1, dt_us=10000.0)
        # ON channel for pixel 0 = index 0, OFF channel for pixel 1 = index 2+1=3
        assert spikes[0, 0] == 1  # ON pixel 0
        assert spikes[0, 3] == 1  # OFF pixel 1

    def test_empty_after_filtering(self) -> None:
        """A single early event still gives at least one step."""
        dtype = np.dtype([("x", np.int32), ("y", np.int32), ("t", np.int64), ("p", np.int8)])
        events = np.zeros(1, dtype=dtype)
        events[0] = (0, 0, 500, 1)
        spikes = events_to_spike_trains(events, width=2, height=2, dt_us=1000.0)
        assert spikes.shape[0] >= 1
