# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestContextPredictor from former test_predictive_spike_codec.py

"""Focused suite: TestContextPredictor from former test_predictive_spike_codec.py."""

from __future__ import annotations

from tests.predictive_spike_codec_support import *  # noqa: F403

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
