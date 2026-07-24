# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTemporalAligner from former test_sensor_fusion.py

"""Focused suite: TestTemporalAligner from former test_sensor_fusion.py."""

from __future__ import annotations

from sensor_fusion_support import *  # noqa: F403


class TestTemporalAligner:
    def test_align_overlapping(self):
        aligner = TemporalAligner(window_us=1000.0)
        s1 = EventStream(
            SensorModality.DVS,
            timestamps=np.array([100, 200, 300, 400, 500], dtype=np.float64),
            addresses=np.arange(5),
            polarities=np.ones(5, dtype=np.int8),
        )
        s2 = EventStream(
            SensorModality.TACTILE,
            timestamps=np.array([200, 300, 400, 500, 600], dtype=np.float64),
            addresses=np.arange(5),
            polarities=np.ones(5, dtype=np.int8),
        )
        aligned = aligner.align([s1, s2])
        assert len(aligned) == 2
        for a in aligned:
            assert float(a.timestamps[0]) >= 200
            assert float(a.timestamps[-1]) <= 500

    def test_slice_windows(self):
        aligner = TemporalAligner(window_us=200.0)
        s = EventStream(
            SensorModality.DVS,
            timestamps=np.array([0, 100, 200, 300, 400, 500, 600], dtype=np.float64),
            addresses=np.arange(7),
            polarities=np.ones(7, dtype=np.int8),
        )
        windows = aligner.slice_windows(s)
        assert len(windows) >= 3

    def test_empty_alignment(self):
        aligner = TemporalAligner()
        assert aligner.align([]) == []

    def test_align_non_overlapping_returns_streams_unchanged(self):
        # Streams whose active spans do not overlap give t_min >= t_max, so
        # there is no common window and the originals are returned as-is.
        aligner = TemporalAligner(window_us=1000.0)
        early = EventStream(
            SensorModality.DVS,
            timestamps=np.array([100, 200], dtype=np.float64),
            addresses=np.arange(2),
            polarities=np.ones(2, dtype=np.int8),
        )
        late = EventStream(
            SensorModality.TACTILE,
            timestamps=np.array([300, 400], dtype=np.float64),
            addresses=np.arange(2),
            polarities=np.ones(2, dtype=np.int8),
        )
        aligned = aligner.align([early, late])
        assert aligned == [early, late]

    def test_slice_windows_single_event_returns_whole_stream(self):
        # A stream with fewer than two events cannot be windowed and is passed
        # through as a single window.
        aligner = TemporalAligner(window_us=200.0)
        s = EventStream(
            SensorModality.DVS,
            timestamps=np.array([100.0], dtype=np.float64),
            addresses=np.arange(1),
            polarities=np.ones(1, dtype=np.int8),
        )
        windows = aligner.slice_windows(s)
        assert windows == [s]
