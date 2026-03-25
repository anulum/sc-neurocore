# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Tests for sc_neurocore.spike_codec

from __future__ import annotations
import numpy as np
from sc_neurocore.spike_codec import SpikeCodec, CompressionResult


class TestSpikeCodec:
    def test_roundtrip_lossless(self):
        rng = np.random.RandomState(42)
        spikes = (rng.random((100, 20)) < 0.05).astype(np.int8)
        codec = SpikeCodec(mode="lossless")
        data, result = codec.compress(spikes)
        reconstructed = codec.decompress(data, 100, 20)
        np.testing.assert_array_equal(reconstructed, spikes)
        assert result.lossless

    def test_compression_ratio(self):
        rng = np.random.RandomState(42)
        spikes = (rng.random((1000, 50)) < 0.01).astype(np.int8)
        codec = SpikeCodec()
        _, result = codec.compress(spikes)
        assert result.compression_ratio > 5.0

    def test_empty_spikes(self):
        spikes = np.zeros((100, 10), dtype=np.int8)
        codec = SpikeCodec()
        data, result = codec.compress(spikes)
        reconstructed = codec.decompress(data, 100, 10)
        np.testing.assert_array_equal(reconstructed, spikes)
        assert result.n_spikes == 0

    def test_dense_spikes(self):
        spikes = np.ones((10, 5), dtype=np.int8)
        codec = SpikeCodec()
        data, result = codec.compress(spikes)
        reconstructed = codec.decompress(data, 10, 5)
        np.testing.assert_array_equal(reconstructed, spikes)

    def test_lossy_mode(self):
        rng = np.random.RandomState(42)
        spikes = (rng.random((100, 10)) < 0.1).astype(np.int8)
        codec = SpikeCodec(mode="lossy", timing_precision=5)
        _, result = codec.compress(spikes)
        assert not result.lossless

    def test_summary(self):
        r = CompressionResult(1000, 100, 10.0, 50, 10, 100, True)
        assert "10.0x" in r.summary()

    def test_varint_roundtrip(self):
        for v in [0, 1, 127, 128, 16383, 16384, 1000000]:
            encoded = SpikeCodec._encode_varint(v)
            decoded, _ = SpikeCodec._decode_varint(encoded, 0)
            assert decoded == v

    def test_single_neuron(self):
        spikes = np.zeros((50, 1), dtype=np.int8)
        spikes[10, 0] = 1
        spikes[30, 0] = 1
        codec = SpikeCodec()
        data, _ = codec.compress(spikes)
        rec = codec.decompress(data, 50, 1)
        np.testing.assert_array_equal(rec, spikes)
