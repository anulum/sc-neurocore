# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWorldModelPredictor from former test_predictive_spike_codec.py

"""Focused suite: TestWorldModelPredictor from former test_predictive_spike_codec.py."""

from __future__ import annotations

from tests.predictive_spike_codec_support import *  # noqa: F403


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
