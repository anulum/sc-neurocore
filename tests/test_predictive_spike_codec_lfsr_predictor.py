# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLFSRPredictor from former test_predictive_spike_codec.py

"""Focused suite: TestLFSRPredictor from former test_predictive_spike_codec.py."""

from __future__ import annotations

from tests.predictive_spike_codec_support import *  # noqa: F403

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
