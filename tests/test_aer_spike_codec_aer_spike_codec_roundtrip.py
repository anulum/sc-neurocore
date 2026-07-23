# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAERSpikeCodecRoundtrip from former test_aer_spike_codec.py

"""Focused suite: TestAERSpikeCodecRoundtrip from former test_aer_spike_codec.py."""

from __future__ import annotations

from tests.aer_spike_codec_support import *  # noqa: F403

class TestAERSpikeCodecRoundtrip:
    def test_roundtrip_sparse(self):
        rng = np.random.RandomState(42)
        spikes = (rng.random((500, 32)) < 0.02).astype(np.int8)
        codec = AERSpikeCodec()
        data, result = codec.compress(spikes)
        recovered = codec.decompress(data)
        np.testing.assert_array_equal(recovered, spikes)
        assert result.lossless

    def test_roundtrip_dense(self):
        rng = np.random.RandomState(7)
        spikes = (rng.random((100, 16)) < 0.3).astype(np.int8)
        codec = AERSpikeCodec()
        data, _ = codec.compress(spikes)
        recovered = codec.decompress(data)
        np.testing.assert_array_equal(recovered, spikes)

    def test_roundtrip_silent(self):
        spikes = np.zeros((200, 10), dtype=np.int8)
        codec = AERSpikeCodec()
        data, result = codec.compress(spikes)
        recovered = codec.decompress(data)
        np.testing.assert_array_equal(recovered, spikes)
        assert result.n_events == 0

    def test_roundtrip_all_firing(self):
        spikes = np.ones((20, 8), dtype=np.int8)
        codec = AERSpikeCodec()
        data, _ = codec.compress(spikes)
        recovered = codec.decompress(data)
        np.testing.assert_array_equal(recovered, spikes)

    def test_roundtrip_single_spike(self):
        spikes = np.zeros((100, 4), dtype=np.int8)
        spikes[42, 2] = 1
        codec = AERSpikeCodec()
        data, result = codec.compress(spikes)
        recovered = codec.decompress(data)
        np.testing.assert_array_equal(recovered, spikes)
        assert result.n_events == 1

    def test_roundtrip_1024ch(self):
        rng = np.random.RandomState(42)
        spikes = (rng.random((100, 1024)) < 0.005).astype(np.int8)
        codec = AERSpikeCodec()
        data, result = codec.compress(spikes)
        recovered = codec.decompress(data)
        np.testing.assert_array_equal(recovered, spikes)
        assert result.n_neurons == 1024

    def test_roundtrip_256ch_escape_collision(self):
        """N=256: neuron 255 = 0xFF would collide with escape marker."""
        rng = np.random.RandomState(42)
        spikes = (rng.random((200, 256)) < 0.01).astype(np.int8)
        codec = AERSpikeCodec()
        data, _ = codec.compress(spikes)
        recovered = codec.decompress(data)
        np.testing.assert_array_equal(recovered, spikes)

    def test_roundtrip_large_time_gap(self):
        """Test timestamp delta overflow (gap > 65535)."""
        spikes = np.zeros((100000, 4), dtype=np.int8)
        spikes[0, 0] = 1
        spikes[99999, 3] = 1
        codec = AERSpikeCodec(timestamp_bits=16)
        data, result = codec.compress(spikes)
        recovered = codec.decompress(data)
        np.testing.assert_array_equal(recovered, spikes)
        assert result.n_events == 2
