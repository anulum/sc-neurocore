# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPredictiveSpikeCodecRoundtrip from former test_predictive_spike_codec.py

"""Focused suite: TestPredictiveSpikeCodecRoundtrip from former test_predictive_spike_codec.py."""

from __future__ import annotations

from tests.predictive_spike_codec_support import *  # noqa: F403


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
