# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# Tests for sc_neurocore.spike_codec.aer_codec

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.spike_codec.aer_codec import AERSpikeCodec, AERCompressionResult


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


class TestAERSpikeCodecCompression:
    def test_sparse_high_compression(self):
        """Very sparse data (0.1% firing) should compress >30x."""
        rng = np.random.RandomState(42)
        spikes = (rng.random((10000, 100)) < 0.001).astype(np.int8)
        codec = AERSpikeCodec()
        _, result = codec.compress(spikes)
        # 3 bytes/event + 17 byte header → ~40x at 0.1% rate
        assert result.compression_ratio > 30.0

    def test_events_proportional_to_spikes(self):
        rng = np.random.RandomState(42)
        spikes = (rng.random((500, 20)) < 0.05).astype(np.int8)
        codec = AERSpikeCodec()
        _, result = codec.compress(spikes)
        assert result.n_events == int(np.sum(spikes))

    def test_bytes_per_event(self):
        rng = np.random.RandomState(42)
        spikes = (rng.random((200, 16)) < 0.05).astype(np.int8)
        _, result = AERSpikeCodec().compress(spikes)
        # Each event: 2 bytes timestamp delta + 1 byte neuron_id (for N<=256)
        # Plus 17 byte header amortized
        assert result.bytes_per_event > 0


class TestAERSpikeCodecEdgeCases:
    def test_invalid_magic_raises(self):
        codec = AERSpikeCodec()
        with pytest.raises(ValueError, match="Invalid header magic"):
            codec.decompress(b"XXXX" + b"\x00" * 100)

    def test_result_type(self):
        spikes = np.zeros((10, 4), dtype=np.int8)
        _, result = AERSpikeCodec().compress(spikes)
        assert isinstance(result, AERCompressionResult)
        assert result.codec_type == "aer"

    def test_single_channel(self):
        rng = np.random.RandomState(42)
        spikes = (rng.random((100, 1)) < 0.05).astype(np.int8)
        codec = AERSpikeCodec()
        data, _ = codec.compress(spikes)
        recovered = codec.decompress(data)
        np.testing.assert_array_equal(recovered, spikes)
