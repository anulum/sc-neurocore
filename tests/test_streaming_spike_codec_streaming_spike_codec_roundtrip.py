# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestStreamingSpikeCodecRoundtrip from former test_streaming_spike_codec.py

"""Focused suite: TestStreamingSpikeCodecRoundtrip from former test_streaming_spike_codec.py."""

from __future__ import annotations

from tests.streaming_spike_codec_support import *  # noqa: F403

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
