# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEventStream from former test_sensor_fusion.py

"""Focused suite: TestEventStream from former test_sensor_fusion.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.fusion.sensor_fusion import (
    EventStream,
    SensorModality,
)
from sensor_fusion_support import (
    _make_stream,
)


class TestEventStream:
    """Counts, timing and bitstream form of event streams."""

    def test_num_events(self) -> None:
        """A stream counts its events."""
        s = _make_stream(SensorModality.DVS, n_events=50)
        assert s.num_events == 50

    def test_duration(self) -> None:
        """A stream of events has a positive duration."""
        s = _make_stream(SensorModality.DVS, n_events=100)
        assert s.duration_us > 0

    def test_event_rate(self) -> None:
        """A stream of events has a positive event rate."""
        s = _make_stream(SensorModality.DVS, n_events=100)
        assert s.event_rate > 0

    def test_to_bitstream_shape(self) -> None:
        """A stream becomes a bitstream of the requested channels and length."""
        s = _make_stream(SensorModality.DVS, n_events=50)
        bs = s.to_bitstream(length=256, num_channels=32)
        assert bs.shape == (32, 256)

    def test_empty_stream(self) -> None:
        """An empty stream has no events, no duration and an all-zero bitstream."""
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
