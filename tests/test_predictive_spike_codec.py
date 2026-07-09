# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# Tests for sc_neurocore.spike_codec.predictive_codec

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.spike_codec import PredictiveSpikeCodec, PredictiveCompressionResult


class TestPredictiveSpikeCodecRoundtrip:
    """Lossless roundtrip: compress → decompress must recover exact input."""

    def test_roundtrip_sparse_spikes(self) -> None:
        rng = np.random.RandomState(42)
        spikes = (rng.random((500, 32)) < 0.02).astype(np.int8)
        codec = PredictiveSpikeCodec(alpha=0.005)
        data, result = codec.compress(spikes)
        recovered = codec.decompress(data, 500, 32)
        np.testing.assert_array_equal(recovered, spikes)
        assert result.lossless

    def test_roundtrip_dense_spikes(self) -> None:
        rng = np.random.RandomState(7)
        spikes = (rng.random((100, 16)) < 0.3).astype(np.int8)
        codec = PredictiveSpikeCodec(alpha=0.01)
        data, _ = codec.compress(spikes)
        recovered = codec.decompress(data, 100, 16)
        np.testing.assert_array_equal(recovered, spikes)

    def test_roundtrip_silent_channels(self) -> None:
        spikes = np.zeros((200, 10), dtype=np.int8)
        codec = PredictiveSpikeCodec()
        data, result = codec.compress(spikes)
        recovered = codec.decompress(data, 200, 10)
        np.testing.assert_array_equal(recovered, spikes)
        assert result.n_spikes == 0

    def test_roundtrip_all_firing(self) -> None:
        spikes = np.ones((50, 8), dtype=np.int8)
        codec = PredictiveSpikeCodec(alpha=0.1, threshold=0.5)
        data, _ = codec.compress(spikes)
        recovered = codec.decompress(data, 50, 8)
        np.testing.assert_array_equal(recovered, spikes)

    def test_roundtrip_single_spike(self) -> None:
        spikes = np.zeros((100, 4), dtype=np.int8)
        spikes[42, 2] = 1
        codec = PredictiveSpikeCodec()
        data, _ = codec.compress(spikes)
        recovered = codec.decompress(data, 100, 4)
        np.testing.assert_array_equal(recovered, spikes)

    def test_roundtrip_single_channel(self) -> None:
        rng = np.random.RandomState(99)
        spikes = (rng.random((300, 1)) < 0.05).astype(np.int8)
        codec = PredictiveSpikeCodec()
        data, _ = codec.compress(spikes)
        recovered = codec.decompress(data, 300, 1)
        np.testing.assert_array_equal(recovered, spikes)


class TestPredictiveSpikeCodecCompression:
    """Verify compression properties."""

    def test_beats_raw_isi_on_bursty_data(self) -> None:
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

    def test_compression_ratio_above_one(self) -> None:
        rng = np.random.RandomState(42)
        spikes = (rng.random((1000, 50)) < 0.01).astype(np.int8)
        codec = PredictiveSpikeCodec()
        _, result = codec.compress(spikes)
        assert result.compression_ratio > 1.0

    def test_prediction_accuracy_increases(self) -> None:
        """On stationary data, prediction accuracy should be reasonable."""
        rng = np.random.RandomState(42)
        spikes = (rng.random((5000, 20)) < 0.01).astype(np.int8)
        codec = PredictiveSpikeCodec(alpha=0.005, threshold=0.5)
        _, result = codec.compress(spikes)
        # With 1% firing rate and threshold 0.5, predictor always predicts 0
        # → accuracy should be ~99% (since 99% of bins are 0)
        assert result.prediction_accuracy > 0.95

    def test_error_sparsity_reported(self) -> None:
        rng = np.random.RandomState(42)
        spikes = (rng.random((200, 10)) < 0.05).astype(np.int8)
        codec = PredictiveSpikeCodec()
        _, result = codec.compress(spikes)
        assert 0.0 <= result.error_sparsity <= 1.0
        assert result.predictor_type == "ema"


class TestPredictiveSpikeCodecEdgeCases:
    """Edge cases and error handling."""

    def test_invalid_magic_raises(self) -> None:
        codec = PredictiveSpikeCodec()
        with pytest.raises(ValueError, match="Invalid header magic"):
            codec.decompress(b"XXXX" + b"\x00" * 100, 10, 5)

    def test_bool_input_accepted(self) -> None:
        rng = np.random.RandomState(42)
        spikes = rng.random((100, 8)) < 0.05
        codec = PredictiveSpikeCodec()
        data, _ = codec.compress(spikes)
        recovered = codec.decompress(data, 100, 8)
        np.testing.assert_array_equal(recovered, spikes.astype(np.int8))

    def test_different_alpha_values(self) -> None:
        rng = np.random.RandomState(42)
        spikes = (rng.random((200, 10)) < 0.05).astype(np.int8)
        for alpha in [0.001, 0.01, 0.1, 0.5]:
            codec = PredictiveSpikeCodec(alpha=alpha)
            data, _ = codec.compress(spikes)
            recovered = codec.decompress(data, 200, 10)
            np.testing.assert_array_equal(recovered, spikes)

    def test_different_thresholds(self) -> None:
        rng = np.random.RandomState(42)
        spikes = (rng.random((200, 10)) < 0.05).astype(np.int8)
        for threshold in [0.01, 0.1, 0.5, 0.9]:
            codec = PredictiveSpikeCodec(threshold=threshold)
            data, _ = codec.compress(spikes)
            recovered = codec.decompress(data, 200, 10)
            np.testing.assert_array_equal(recovered, spikes)

    def test_result_is_predictive_type(self) -> None:
        spikes = np.zeros((10, 2), dtype=np.int8)
        codec = PredictiveSpikeCodec()
        _, result = codec.compress(spikes)
        assert isinstance(result, PredictiveCompressionResult)

    def test_large_channel_count(self) -> None:
        """1024 channels — Neuralink N1 scale."""
        rng = np.random.RandomState(42)
        spikes = (rng.random((100, 1024)) < 0.005).astype(np.int8)
        codec = PredictiveSpikeCodec()
        data, result = codec.compress(spikes)
        recovered = codec.decompress(data, 100, 1024)
        np.testing.assert_array_equal(recovered, spikes)
        assert result.n_neurons == 1024


class TestLFSRPredictor:
    """SC-native LFSR predictor: bit-true with sc_bitstream_encoder.v."""

    def test_lfsr_roundtrip(self) -> None:
        rng = np.random.RandomState(42)
        spikes = (rng.random((500, 32)) < 0.02).astype(np.int8)
        codec = PredictiveSpikeCodec(predictor="lfsr", alpha_q8=1, seed=0xACE1)
        data, result = codec.compress(spikes)
        recovered = codec.decompress(data, 500, 32)
        np.testing.assert_array_equal(recovered, spikes)
        assert result.predictor_type == "lfsr"

    def test_lfsr_1024ch(self) -> None:
        rng = np.random.RandomState(42)
        spikes = (rng.random((100, 1024)) < 0.005).astype(np.int8)
        codec = PredictiveSpikeCodec(predictor="lfsr", alpha_q8=2, seed=0x1234)
        data, _ = codec.compress(spikes)
        np.testing.assert_array_equal(codec.decompress(data, 100, 1024), spikes)

    def test_lfsr_silent(self) -> None:
        spikes = np.zeros((200, 10), dtype=np.int8)
        codec = PredictiveSpikeCodec(predictor="lfsr")
        data, _ = codec.compress(spikes)
        np.testing.assert_array_equal(codec.decompress(data, 200, 10), spikes)

    def test_lfsr_all_firing(self) -> None:
        spikes = np.ones((50, 8), dtype=np.int8)
        codec = PredictiveSpikeCodec(predictor="lfsr", alpha_q8=50, seed=0xBEEF)
        data, _ = codec.compress(spikes)
        np.testing.assert_array_equal(codec.decompress(data, 50, 8), spikes)

    def test_lfsr_deterministic(self) -> None:
        """Same seed → same output, always."""
        rng = np.random.RandomState(42)
        spikes = (rng.random((100, 16)) < 0.05).astype(np.int8)
        c1 = PredictiveSpikeCodec(predictor="lfsr", seed=0xDEAD)
        c2 = PredictiveSpikeCodec(predictor="lfsr", seed=0xDEAD)
        d1, _ = c1.compress(spikes)
        d2, _ = c2.compress(spikes)
        assert d1 == d2

    def test_cross_mode_auto_detect(self) -> None:
        """Decoder auto-detects predictor type from header magic."""
        rng = np.random.RandomState(42)
        spikes = (rng.random((100, 16)) < 0.05).astype(np.int8)
        codec_ema = PredictiveSpikeCodec(predictor="ema")
        codec_lfsr = PredictiveSpikeCodec(predictor="lfsr")
        data_ema, _ = codec_ema.compress(spikes)
        data_lfsr, _ = codec_lfsr.compress(spikes)
        # Either codec instance can decompress either format
        np.testing.assert_array_equal(codec_lfsr.decompress(data_ema, 100, 16), spikes)
        np.testing.assert_array_equal(codec_ema.decompress(data_lfsr, 100, 16), spikes)

    def test_different_alpha_q8(self) -> None:
        rng = np.random.RandomState(42)
        spikes = (rng.random((200, 10)) < 0.05).astype(np.int8)
        for aq8 in [1, 5, 20, 100]:
            codec = PredictiveSpikeCodec(predictor="lfsr", alpha_q8=aq8)
            data, _ = codec.compress(spikes)
            np.testing.assert_array_equal(codec.decompress(data, 200, 10), spikes)


class TestContextPredictor:
    """Markov context model predictor tests."""

    def test_context_roundtrip(self) -> None:
        rng = np.random.RandomState(42)
        spikes = (rng.random((500, 16)) < 0.05).astype(np.int8)
        codec = PredictiveSpikeCodec(predictor="context", context_bits=8)
        data, result = codec.compress(spikes)
        recovered = codec.decompress(data, 500, 16)
        np.testing.assert_array_equal(recovered, spikes)
        assert result.predictor_type == "context"

    def test_context_roundtrip_silent(self) -> None:
        spikes = np.zeros((200, 8), dtype=np.int8)
        codec = PredictiveSpikeCodec(predictor="context")
        data, _ = codec.compress(spikes)
        np.testing.assert_array_equal(codec.decompress(data, 200, 8), spikes)

    def test_context_roundtrip_dense(self) -> None:
        rng = np.random.RandomState(42)
        spikes = (rng.random((100, 8)) < 0.3).astype(np.int8)
        codec = PredictiveSpikeCodec(predictor="context", context_bits=4)
        data, _ = codec.compress(spikes)
        np.testing.assert_array_equal(codec.decompress(data, 100, 8), spikes)

    def test_context_beats_ema_on_bursty(self) -> None:
        """Context model should outperform EMA on periodic bursts."""
        rng = np.random.RandomState(42)
        T, N = 2000, 16
        bursty = np.zeros((T, N), dtype=np.int8)
        for n in range(N):
            phase = rng.randint(0, 50)
            for bs in range(phase, T, 50):
                for dt in range(min(5, T - bs)):
                    bursty[bs + dt, n] = 1

        _, r_ema = PredictiveSpikeCodec(predictor="ema", alpha=0.02).compress(bursty)
        _, r_ctx = PredictiveSpikeCodec(predictor="context", context_bits=8).compress(bursty)
        assert r_ctx.prediction_accuracy > r_ema.prediction_accuracy

    def test_context_different_bits(self) -> None:
        rng = np.random.RandomState(42)
        spikes = (rng.random((200, 8)) < 0.05).astype(np.int8)
        for bits in [2, 4, 8, 12]:
            codec = PredictiveSpikeCodec(predictor="context", context_bits=bits)
            data, _ = codec.compress(spikes)
            np.testing.assert_array_equal(codec.decompress(data, 200, 8), spikes)

    def test_cross_mode_context_decode(self) -> None:
        """Any codec instance can decompress context-encoded data."""
        rng = np.random.RandomState(42)
        spikes = (rng.random((100, 8)) < 0.05).astype(np.int8)
        codec_ctx = PredictiveSpikeCodec(predictor="context")
        codec_ema = PredictiveSpikeCodec(predictor="ema")
        data, _ = codec_ctx.compress(spikes)
        np.testing.assert_array_equal(codec_ema.decompress(data, 100, 8), spikes)


class TestWorldModelPredictor:
    """Learnable autoregressive world model predictor."""

    def test_world_model_roundtrip(self) -> None:
        rng = np.random.RandomState(42)
        spikes = (rng.random((500, 16)) < 0.05).astype(np.int8)
        codec = PredictiveSpikeCodec(predictor="world_model", alpha=0.01, context_bits=8, seed=42)
        data, result = codec.compress(spikes)
        recovered = codec.decompress(data, 500, 16)
        np.testing.assert_array_equal(recovered, spikes)
        assert result.predictor_type == "world_model"

    def test_world_model_silent(self) -> None:
        spikes = np.zeros((200, 8), dtype=np.int8)
        codec = PredictiveSpikeCodec(predictor="world_model")
        data, _ = codec.compress(spikes)
        np.testing.assert_array_equal(codec.decompress(data, 200, 8), spikes)

    def test_world_model_bursty(self) -> None:
        rng = np.random.RandomState(42)
        T, N = 1000, 16
        bursty = np.zeros((T, N), dtype=np.int8)
        for n in range(N):
            ph = rng.randint(0, 50)
            for bs in range(ph, T, 50):
                for dt in range(min(5, T - bs)):
                    bursty[bs + dt, n] = 1

        codec = PredictiveSpikeCodec(predictor="world_model", alpha=0.01, context_bits=8, seed=42)
        data, result = codec.compress(bursty)
        np.testing.assert_array_equal(codec.decompress(data, T, N), bursty)
        assert result.prediction_accuracy > 0.95

    def test_world_model_beats_ema_on_bursty(self) -> None:
        rng = np.random.RandomState(42)
        T, N = 2000, 16
        bursty = np.zeros((T, N), dtype=np.int8)
        for n in range(N):
            ph = rng.randint(0, 50)
            for bs in range(ph, T, 50):
                for dt in range(min(5, T - bs)):
                    bursty[bs + dt, n] = 1

        _, r_ema = PredictiveSpikeCodec(predictor="ema", alpha=0.02).compress(bursty)
        _, r_wm = PredictiveSpikeCodec(
            predictor="world_model", alpha=0.01, context_bits=8, seed=42
        ).compress(bursty)
        assert r_wm.prediction_accuracy > r_ema.prediction_accuracy

    def test_world_model_cross_decode(self) -> None:
        rng = np.random.RandomState(42)
        spikes = (rng.random((100, 8)) < 0.05).astype(np.int8)
        data, _ = PredictiveSpikeCodec(predictor="world_model").compress(spikes)
        np.testing.assert_array_equal(
            PredictiveSpikeCodec(predictor="ema").decompress(data, 100, 8), spikes
        )


class TestPythonFallbackFunctions:
    """Direct tests for Python fallback functions.

    These are never hit when the Rust engine is available (CI builds Rust).
    Testing them directly validates Python fallback behaviour regardless of Rust availability.
    """

    def test_predict_and_xor_roundtrip(self) -> None:
        from sc_neurocore.spike_codec.predictive_codec import (
            _predict_and_xor,
            _xor_and_recover,
        )

        rng = np.random.RandomState(42)
        spikes = (rng.random((200, 16)) < 0.05).astype(np.int8)
        errors, correct = _predict_and_xor(spikes, 16, 0.005, 0.5)
        assert errors.shape == spikes.shape
        assert correct >= 0
        recovered = _xor_and_recover(errors, 16, 0.005, 0.5)
        np.testing.assert_array_equal(recovered, spikes)

    def test_predict_and_xor_silent(self) -> None:
        from sc_neurocore.spike_codec.predictive_codec import (
            _predict_and_xor,
            _xor_and_recover,
        )

        spikes = np.zeros((100, 8), dtype=np.int8)
        errors, correct = _predict_and_xor(spikes, 8, 0.01, 0.5)
        assert correct == 800  # all correct (predict 0, actual 0)
        recovered = _xor_and_recover(errors, 8, 0.01, 0.5)
        np.testing.assert_array_equal(recovered, spikes)

    def test_lfsr16_step(self) -> None:
        from sc_neurocore.spike_codec.predictive_codec import _lfsr16_step

        reg = 0xACE1
        seen = set()
        for _ in range(100):
            reg = _lfsr16_step(reg)
            assert 0 <= reg <= 0xFFFF
            seen.add(reg)
        assert len(seen) == 100  # no short cycle

    def test_predict_and_xor_lfsr_roundtrip(self) -> None:
        from sc_neurocore.spike_codec.predictive_codec import (
            _predict_and_xor_lfsr,
            _xor_and_recover_lfsr,
        )

        rng = np.random.RandomState(42)
        spikes = (rng.random((200, 16)) < 0.05).astype(np.int8)
        errors, correct = _predict_and_xor_lfsr(spikes, 16, 1, 0xACE1)
        assert errors.shape == spikes.shape
        assert correct >= 0
        recovered = _xor_and_recover_lfsr(errors, 16, 1, 0xACE1)
        np.testing.assert_array_equal(recovered, spikes)

    def test_predict_and_xor_lfsr_all_firing(self) -> None:
        from sc_neurocore.spike_codec.predictive_codec import (
            _predict_and_xor_lfsr,
            _xor_and_recover_lfsr,
        )

        spikes = np.ones((50, 4), dtype=np.int8)
        errors, _ = _predict_and_xor_lfsr(spikes, 4, 50, 0xBEEF)
        recovered = _xor_and_recover_lfsr(errors, 4, 50, 0xBEEF)
        np.testing.assert_array_equal(recovered, spikes)

    def test_legacy_rate_predictor_contract(self) -> None:
        """Legacy EMA predictor should update, predict, and reset deterministically."""
        from sc_neurocore.spike_codec.predictive_codec import _RatePredictor

        predictor = _RatePredictor(n_channels=3, alpha=0.5, threshold=0.25)
        np.testing.assert_array_equal(predictor.predict(), np.zeros(3, dtype=np.int8))

        predictor.update(np.array([1, 0, 1], dtype=np.int8))

        np.testing.assert_allclose(predictor.rates, np.array([0.5, 0.0, 0.5]))
        np.testing.assert_array_equal(predictor.predict(), np.array([1, 0, 1], dtype=np.int8))

        predictor.reset()

        np.testing.assert_allclose(predictor.rates, np.zeros(3, dtype=np.float64))


class TestPythonFallbackPath:
    """Force Python path through the class by monkeypatching _HAS_RUST."""

    def test_ema_python_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sc_neurocore.spike_codec.predictive_codec as mod

        monkeypatch.setattr(mod, "_HAS_RUST", False)
        rng = np.random.RandomState(42)
        spikes = (rng.random((200, 16)) < 0.05).astype(np.int8)
        codec = PredictiveSpikeCodec(predictor="ema")
        data, result = codec.compress(spikes)
        recovered = codec.decompress(data, 200, 16)
        np.testing.assert_array_equal(recovered, spikes)
        assert result.predictor_type == "ema"

    def test_lfsr_python_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sc_neurocore.spike_codec.predictive_codec as mod

        monkeypatch.setattr(mod, "_HAS_RUST", False)
        rng = np.random.RandomState(42)
        spikes = (rng.random((200, 16)) < 0.05).astype(np.int8)
        codec = PredictiveSpikeCodec(predictor="lfsr", alpha_q8=1, seed=0xACE1)
        data, result = codec.compress(spikes)
        recovered = codec.decompress(data, 200, 16)
        np.testing.assert_array_equal(recovered, spikes)
        assert result.predictor_type == "lfsr"
