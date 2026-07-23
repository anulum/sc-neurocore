# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEventsToFrames from former test_sensors_dvs.py

"""Focused suite: TestEventsToFrames from former test_sensors_dvs.py."""

from __future__ import annotations

from tests.sensors_dvs_support import *  # noqa: F403

class TestEventsToFrames:
    def test_basic_shape(self) -> None:
        events = _make_events(n=50, width=4, height=3)
        frames = events_to_frames(events, width=4, height=3, dt_us=10000.0)
        assert frames.ndim == 4
        assert frames.shape[1] == 2  # ON and OFF channels
        assert frames.shape[2] == 3  # height
        assert frames.shape[3] == 4  # width

    def test_accumulates_counts(self) -> None:
        dtype = np.dtype([("x", np.int32), ("y", np.int32), ("t", np.int64), ("p", np.int8)])
        events = np.zeros(3, dtype=dtype)
        events[0] = (0, 0, 0, 1)
        events[1] = (0, 0, 500, 1)
        events[2] = (0, 0, 900, 1)
        frames = events_to_frames(events, width=2, height=2, dt_us=2000.0)
        # All 3 events in first frame, ON channel
        assert frames[0, 1, 0, 0] == 3.0

    def test_explicit_duration(self) -> None:
        events = _make_events()
        frames = events_to_frames(
            events,
            width=8,
            height=6,
            dt_us=25000.0,
            duration_us=100000.0,
        )
        assert frames.shape[0] == 4

    def test_float32_dtype(self) -> None:
        events = _make_events()
        frames = events_to_frames(events, width=8, height=6)
        assert frames.dtype == np.float32
