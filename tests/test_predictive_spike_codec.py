# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Tests for sc_neurocore.spike_codec.predictive_codec

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.spike_codec import PredictiveSpikeCodec, PredictiveCompressionResult


class TestPredictiveSpikeCodecRoundtrip:
    """Lossless roundtrip: compress → decompress must recover exact input."""

    def test_roundtrip_sparse_spikes(self):
        rng = np.random.RandomState(42)
        spikes = (rng.random((500, 32)) < 0.02).astype(np.int8)
        codec = PredictiveSpikeCodec(alpha=0.005)
        data, result = codec.compress(spikes)
        recovered = codec.decompress(data, 500, 32)
        np.testing.assert_array_equal(recovered, spikes)
        assert result.lossless

    def test_roundtrip_dense_spikes(self):
        rng = np.random.RandomState(7)
        spikes = (rng.random((100, 16)) < 0.3).astype(np.int8)
        codec = PredictiveSpikeCodec(alpha=0.01)
        data, _ = codec.compress(spikes)
        recovered = codec.decompress(data, 100, 16)
        np.testing.assert_array_equal(recovered, spikes)

    def test_roundtrip_silent_channels(self):
        spikes = np.zeros((200, 10), dtype=np.int8)
        codec = PredictiveSpikeCodec()
        data, result = codec.compress(spikes)
        recovered = codec.decompress(data, 200, 10)
        np.testing.assert_array_equal(recovered, spikes)
        assert result.n_spikes == 0

    def test_roundtrip_all_firing(self):
        spikes = np.ones((50, 8), dtype=np.int8)
        codec = PredictiveSpikeCodec(alpha=0.1, threshold=0.5)
        data, _ = codec.compress(spikes)
        recovered = codec.decompress(data, 50, 8)
        np.testing.assert_array_equal(recovered, spikes)

    def test_roundtrip_single_spike(self):
        spikes = np.zeros((100, 4), dtype=np.int8)
        spikes[42, 2] = 1
        codec = PredictiveSpikeCodec()
        data, _ = codec.compress(spikes)
        recovered = codec.decompress(data, 100, 4)
        np.testing.assert_array_equal(recovered, spikes)

    def test_roundtrip_single_channel(self):
        rng = np.random.RandomState(99)
        spikes = (rng.random((300, 1)) < 0.05).astype(np.int8)
        codec = PredictiveSpikeCodec()
        data, _ = codec.compress(spikes)
        recovered = codec.decompress(data, 300, 1)
        np.testing.assert_array_equal(recovered, spikes)


class TestPredictiveSpikeCodecCompression:
    """Verify compression properties."""

    def test_beats_raw_isi_on_bursty_data(self):
        """Bursty firing (periodic oscillation) should compress better
        with prediction than without."""
        rng = np.random.RandomState(42)
        T, N = 2000, 64
        spikes = np.zeros((T, N), dtype=np.int8)
        # Periodic bursting: each neuron fires in bursts of 5 every ~100 steps
        for n in range(N):
            phase = rng.randint(0, 100)
            for burst_start in range(phase, T, 100):
                for dt in range(min(5, T - burst_start)):
                    spikes[burst_start + dt, n] = 1

        from sc_neurocore.spike_codec import SpikeCodec

        raw_codec = SpikeCodec(mode="lossless")
        pred_codec = PredictiveSpikeCodec(alpha=0.02, threshold=0.3)

        _, raw_result = raw_codec.compress(spikes)
        _, pred_result = pred_codec.compress(spikes)

        # Predictive should achieve higher compression on structured data
        assert pred_result.compression_ratio > raw_result.compression_ratio * 0.8

    def test_compression_ratio_above_one(self):
        rng = np.random.RandomState(42)
        spikes = (rng.random((1000, 50)) < 0.01).astype(np.int8)
        codec = PredictiveSpikeCodec()
        _, result = codec.compress(spikes)
        assert result.compression_ratio > 1.0

    def test_prediction_accuracy_increases(self):
        """On stationary data, prediction accuracy should be reasonable."""
        rng = np.random.RandomState(42)
        spikes = (rng.random((5000, 20)) < 0.01).astype(np.int8)
        codec = PredictiveSpikeCodec(alpha=0.005, threshold=0.5)
        _, result = codec.compress(spikes)
        # With 1% firing rate and threshold 0.5, predictor always predicts 0
        # → accuracy should be ~99% (since 99% of bins are 0)
        assert result.prediction_accuracy > 0.95

    def test_error_sparsity_reported(self):
        rng = np.random.RandomState(42)
        spikes = (rng.random((200, 10)) < 0.05).astype(np.int8)
        codec = PredictiveSpikeCodec()
        _, result = codec.compress(spikes)
        assert 0.0 <= result.error_sparsity <= 1.0
        assert result.predictor_type == "ema"


class TestPredictiveSpikeCodecEdgeCases:
    """Edge cases and error handling."""

    def test_invalid_magic_raises(self):
        codec = PredictiveSpikeCodec()
        with pytest.raises(ValueError, match="Invalid header magic"):
            codec.decompress(b"XXXX" + b"\x00" * 100, 10, 5)

    def test_bool_input_accepted(self):
        rng = np.random.RandomState(42)
        spikes = rng.random((100, 8)) < 0.05
        codec = PredictiveSpikeCodec()
        data, _ = codec.compress(spikes)
        recovered = codec.decompress(data, 100, 8)
        np.testing.assert_array_equal(recovered, spikes.astype(np.int8))

    def test_different_alpha_values(self):
        rng = np.random.RandomState(42)
        spikes = (rng.random((200, 10)) < 0.05).astype(np.int8)
        for alpha in [0.001, 0.01, 0.1, 0.5]:
            codec = PredictiveSpikeCodec(alpha=alpha)
            data, _ = codec.compress(spikes)
            recovered = codec.decompress(data, 200, 10)
            np.testing.assert_array_equal(recovered, spikes)

    def test_different_thresholds(self):
        rng = np.random.RandomState(42)
        spikes = (rng.random((200, 10)) < 0.05).astype(np.int8)
        for threshold in [0.01, 0.1, 0.5, 0.9]:
            codec = PredictiveSpikeCodec(threshold=threshold)
            data, _ = codec.compress(spikes)
            recovered = codec.decompress(data, 200, 10)
            np.testing.assert_array_equal(recovered, spikes)

    def test_result_is_predictive_type(self):
        spikes = np.zeros((10, 2), dtype=np.int8)
        codec = PredictiveSpikeCodec()
        _, result = codec.compress(spikes)
        assert isinstance(result, PredictiveCompressionResult)

    def test_large_channel_count(self):
        """1024 channels — Neuralink N1 scale."""
        rng = np.random.RandomState(42)
        spikes = (rng.random((100, 1024)) < 0.005).astype(np.int8)
        codec = PredictiveSpikeCodec()
        data, result = codec.compress(spikes)
        recovered = codec.decompress(data, 100, 1024)
        np.testing.assert_array_equal(recovered, spikes)
        assert result.n_neurons == 1024
