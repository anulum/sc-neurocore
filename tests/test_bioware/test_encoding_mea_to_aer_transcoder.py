# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMEAToAERTranscoder from former test_encoding.py

"""Focused suite: TestMEAToAERTranscoder from former test_encoding.py."""

from __future__ import annotations

from tests.test_bioware.encoding_support import *  # noqa: F403


class TestMEAToAERTranscoder:
    def test_transcode(self) -> None:
        spikes = [
            DetectedSpike(channel=0, timestamp_s=0.001, amplitude_uv=-50),
            DetectedSpike(channel=3, timestamp_s=0.005, amplitude_uv=-40),
        ]
        tc = MEAToAERTranscoder(hw_clock_hz=1e6)
        events = tc.transcode(spikes)
        assert len(events) == 2

    def test_timestamp_conversion(self) -> None:
        spikes = [DetectedSpike(channel=0, timestamp_s=0.001, amplitude_uv=-50)]
        tc = MEAToAERTranscoder(hw_clock_hz=1e6)
        events = tc.transcode(spikes)
        assert events[0].timestamp == 1000  # 0.001s * 1MHz = 1000

    def test_channel_mapping(self) -> None:
        spikes = [DetectedSpike(channel=5, timestamp_s=0.0, amplitude_uv=-50)]
        tc = MEAToAERTranscoder(channel_map={5: 42})
        events = tc.transcode(spikes)
        assert events[0].neuron_id == 42

    def test_sorted_by_time(self) -> None:
        spikes = [
            DetectedSpike(channel=0, timestamp_s=0.005, amplitude_uv=-50),
            DetectedSpike(channel=1, timestamp_s=0.001, amplitude_uv=-30),
        ]
        tc = MEAToAERTranscoder()
        events = tc.transcode(spikes)
        assert events[0].timestamp <= events[1].timestamp
