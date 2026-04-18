# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
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


class TestSpikeCodecHuffman:
    def test_huffman_roundtrip_sparse(self):
        rng = np.random.RandomState(42)
        spikes = (rng.random((500, 20)) < 0.01).astype(np.int8)
        codec = SpikeCodec(entropy="huffman")
        data, result = codec.compress(spikes)
        rec = codec.decompress(data, 500, 20)
        np.testing.assert_array_equal(rec, spikes)

    def test_huffman_roundtrip_dense(self):
        rng = np.random.RandomState(42)
        spikes = (rng.random((200, 16)) < 0.3).astype(np.int8)
        codec = SpikeCodec(entropy="huffman")
        data, _ = codec.compress(spikes)
        rec = codec.decompress(data, 200, 16)
        np.testing.assert_array_equal(rec, spikes)

    def test_huffman_beats_varint_on_dense(self):
        rng = np.random.RandomState(42)
        spikes = (rng.random((1000, 32)) < 0.2).astype(np.int8)
        d_var, _ = SpikeCodec(entropy="varint").compress(spikes)
        d_huf, _ = SpikeCodec(entropy="huffman").compress(spikes)
        assert len(d_huf) < len(d_var)

    def test_huffman_empty(self):
        spikes = np.zeros((100, 10), dtype=np.int8)
        codec = SpikeCodec(entropy="huffman")
        data, _ = codec.compress(spikes)
        rec = codec.decompress(data, 100, 10)
        np.testing.assert_array_equal(rec, spikes)

    def test_auto_entropy_roundtrip(self):
        rng = np.random.RandomState(42)
        for rate in [0.005, 0.05, 0.2]:
            spikes = (rng.random((500, 16)) < rate).astype(np.int8)
            codec = SpikeCodec(entropy="auto")
            data, _ = codec.compress(spikes)
            rec = codec.decompress(data, 500, 16)
            np.testing.assert_array_equal(rec, spikes)


class TestHuffmanEncoder:
    def test_encode_decode_roundtrip(self):
        from sc_neurocore.spike_codec.entropy import HuffmanEncoder

        enc = HuffmanEncoder()
        values = [1, 2, 1, 3, 1, 2, 1, 1, 4, 1]
        data = enc.encode(values)
        decoded, _ = enc.decode(data, len(values))
        assert decoded == values

    def test_empty_values(self):
        from sc_neurocore.spike_codec.entropy import HuffmanEncoder

        enc = HuffmanEncoder()
        data = enc.encode([])
        decoded, _ = enc.decode(data, 0)
        assert decoded == []

    def test_single_symbol(self):
        from sc_neurocore.spike_codec.entropy import HuffmanEncoder

        enc = HuffmanEncoder()
        values = [42, 42, 42, 42]
        data = enc.encode(values)
        decoded, _ = enc.decode(data, 4)
        assert decoded == values
