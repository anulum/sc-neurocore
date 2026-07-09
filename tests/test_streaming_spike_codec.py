# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# Tests for sc_neurocore.spike_codec.streaming_codec

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.spike_codec.streaming_codec import (
    StreamingSpikeCodec,
    StreamingCompressionResult,
)


class TestStreamingSpikeCodecRoundtrip:
    def test_roundtrip_basic(self):
        rng = np.random.RandomState(42)
        spikes = (rng.random((100, 16)) < 0.05).astype(np.int8)
        codec = StreamingSpikeCodec(window_size=20)
        data, result = codec.compress(spikes)
        recovered = codec.decompress(data)
        np.testing.assert_array_equal(recovered, spikes)
        assert result.lossless

    def test_roundtrip_uneven_T(self):
        """T not divisible by window_size — last frame padded."""
        rng = np.random.RandomState(42)
        spikes = (rng.random((107, 8)) < 0.05).astype(np.int8)
        codec = StreamingSpikeCodec(window_size=20)
        data, result = codec.compress(spikes)
        recovered = codec.decompress(data)
        np.testing.assert_array_equal(recovered, spikes)
        assert result.n_frames == 6  # ceil(107/20)

    def test_roundtrip_silent(self):
        spikes = np.zeros((60, 10), dtype=np.int8)
        codec = StreamingSpikeCodec(window_size=20)
        data, result = codec.compress(spikes)
        recovered = codec.decompress(data)
        np.testing.assert_array_equal(recovered, spikes)
        assert result.mean_active_channels == 0.0

    def test_roundtrip_all_firing(self):
        spikes = np.ones((40, 8), dtype=np.int8)
        codec = StreamingSpikeCodec(window_size=10)
        data, _ = codec.compress(spikes)
        recovered = codec.decompress(data)
        np.testing.assert_array_equal(recovered, spikes)

    def test_roundtrip_large_window(self):
        """Window larger than T."""
        rng = np.random.RandomState(42)
        spikes = (rng.random((10, 4)) < 0.1).astype(np.int8)
        codec = StreamingSpikeCodec(window_size=100)
        data, result = codec.compress(spikes)
        recovered = codec.decompress(data)
        np.testing.assert_array_equal(recovered, spikes)
        assert result.n_frames == 1

    def test_roundtrip_window_size_1(self):
        """Minimum latency: one sample per frame."""
        rng = np.random.RandomState(42)
        spikes = (rng.random((50, 8)) < 0.1).astype(np.int8)
        codec = StreamingSpikeCodec(window_size=1)
        data, result = codec.compress(spikes)
        recovered = codec.decompress(data)
        np.testing.assert_array_equal(recovered, spikes)
        assert result.n_frames == 50

    def test_roundtrip_1024ch(self):
        rng = np.random.RandomState(42)
        spikes = (rng.random((40, 1024)) < 0.005).astype(np.int8)
        codec = StreamingSpikeCodec(window_size=20)
        data, result = codec.compress(spikes)
        recovered = codec.decompress(data)
        np.testing.assert_array_equal(recovered, spikes)
        assert result.n_neurons == 1024


class TestStreamingSpikeCodecFrameAPI:
    def test_single_frame_roundtrip(self):
        rng = np.random.RandomState(42)
        window = (rng.random((20, 16)) < 0.05).astype(np.int8)
        codec = StreamingSpikeCodec(window_size=20)
        frame = codec.compress_frame(window)
        recovered = codec.decompress_frame(frame)
        np.testing.assert_array_equal(recovered, window)

    def test_frame_independence(self):
        """Each frame must be decodable without any other frame."""
        rng = np.random.RandomState(42)
        codec = StreamingSpikeCodec(window_size=10)
        for _ in range(5):
            w = (rng.random((10, 8)) < 0.1).astype(np.int8)
            f = codec.compress_frame(w)
            assert np.array_equal(codec.decompress_frame(f), w)


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


class TestStreamingSpikeCodecEdgeCases:
    def test_invalid_magic_raises(self):
        codec = StreamingSpikeCodec()
        with pytest.raises(ValueError, match="Invalid header magic"):
            codec.decompress(b"XXXX" + b"\x00" * 100)

    def test_result_type(self):
        spikes = np.zeros((20, 4), dtype=np.int8)
        _, result = StreamingSpikeCodec().compress(spikes)
        assert isinstance(result, StreamingCompressionResult)
