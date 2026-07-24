# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPredictiveSpikeCodecEdgeCases from former test_predictive_spike_codec.py

"""Focused suite: TestPredictiveSpikeCodecEdgeCases from former test_predictive_spike_codec.py."""

from __future__ import annotations

from tests.predictive_spike_codec_support import *  # noqa: F403


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
