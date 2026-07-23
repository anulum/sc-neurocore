# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEventStream from former test_sensor_fusion.py

"""Focused suite: TestEventStream from former test_sensor_fusion.py."""

from __future__ import annotations

from sensor_fusion_support import *  # noqa: F403

class TestEventStream:
    def test_num_events(self):
        s = _make_stream(SensorModality.DVS, n_events=50)
        assert s.num_events == 50

    def test_duration(self):
        s = _make_stream(SensorModality.DVS, n_events=100)
        assert s.duration_us > 0

    def test_event_rate(self):
        s = _make_stream(SensorModality.DVS, n_events=100)
        assert s.event_rate > 0

    def test_to_bitstream_shape(self):
        s = _make_stream(SensorModality.DVS, n_events=50)
        bs = s.to_bitstream(length=256, num_channels=32)
        assert bs.shape == (32, 256)

    def test_empty_stream(self):
        s = EventStream(
            modality=SensorModality.DVS,
            timestamps=np.array([]),
            addresses=np.array([]),
            polarities=np.array([]),
        )
        assert s.num_events == 0
        assert s.duration_us == 0.0
        bs = s.to_bitstream(128, 16)
        assert np.sum(bs) == 0
