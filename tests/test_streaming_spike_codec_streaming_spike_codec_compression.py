# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestStreamingSpikeCodecCompression from former test_streaming_spike_codec.py

"""Focused suite: TestStreamingSpikeCodecCompression from former test_streaming_spike_codec.py."""

from __future__ import annotations

from tests.streaming_spike_codec_support import *  # noqa: F403


class TestStreamingSpikeCodecCompression:
    def test_silent_frames_small(self):
        """Silent frames should be very compact."""
        silent = np.zeros((20, 64), dtype=np.int8)
        codec = StreamingSpikeCodec(window_size=20)
        frame = codec.compress_frame(silent)
        # Header(4) + skip_bitmap(8) = 12 bytes minimum
        assert len(frame) <= 20

    def test_max_frame_bytes_reported(self):
        rng = np.random.RandomState(42)
        spikes = (rng.random((100, 16)) < 0.05).astype(np.int8)
        _, result = StreamingSpikeCodec(window_size=20).compress(spikes)
        assert result.max_frame_bytes > 0
        assert result.codec_type == "streaming"
