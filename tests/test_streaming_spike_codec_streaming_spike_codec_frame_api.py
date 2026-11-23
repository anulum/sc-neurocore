# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestStreamingSpikeCodecFrameAPI from former test_streaming_spike_codec.py

"""Focused suite: TestStreamingSpikeCodecFrameAPI from former test_streaming_spike_codec.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.spike_codec.streaming_codec import (
    StreamingSpikeCodec,
)


class TestStreamingSpikeCodecFrameAPI:
    """The frame-by-frame codec interface."""

    def test_single_frame_roundtrip(self) -> None:
        """One frame round-trips losslessly."""
        rng = np.random.RandomState(42)
        window = (rng.random((20, 16)) < 0.05).astype(np.int8)
        codec = StreamingSpikeCodec(window_size=20)
        frame = codec.compress_frame(window)
        recovered = codec.decompress_frame(frame)
        np.testing.assert_array_equal(recovered, window)

    def test_frame_independence(self) -> None:
        """Each frame must be decodable without any other frame."""
        rng = np.random.RandomState(42)
        codec = StreamingSpikeCodec(window_size=10)
        for _ in range(5):
            w = (rng.random((10, 8)) < 0.1).astype(np.int8)
            f = codec.compress_frame(w)
            assert np.array_equal(codec.decompress_frame(f), w)
