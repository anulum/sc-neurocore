# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPredictiveSpikeCodecCompression from former test_predictive_spike_codec.py

"""Focused suite: TestPredictiveSpikeCodecCompression from former test_predictive_spike_codec.py."""

from __future__ import annotations

from tests.predictive_spike_codec_support import *  # noqa: F403

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
