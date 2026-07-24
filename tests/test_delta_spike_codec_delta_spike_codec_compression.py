# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDeltaSpikeCodecCompression from former test_delta_spike_codec.py

"""Focused suite: TestDeltaSpikeCodecCompression from former test_delta_spike_codec.py."""

from __future__ import annotations

from tests.delta_spike_codec_support import *  # noqa: F403


class TestDeltaSpikeCodecCompression:
    def test_correlated_beats_uncorrelated(self):
        """Correlated data should compress better with delta codec."""
        rng = np.random.RandomState(42)
        T, N = 2000, 16

        # Correlated: shared base pattern
        base = (rng.random((T, 1)) < 0.03).astype(np.int8)
        corr_spikes = (
            np.broadcast_to(base, (T, N)) | (rng.random((T, N)) < 0.002).astype(np.int8)
        ).astype(np.int8)

        # Uncorrelated: independent
        uncorr_spikes = (rng.random((T, N)) < 0.03).astype(np.int8)

        codec = DeltaSpikeCodec(group_size=4)
        _, corr_result = codec.compress(corr_spikes)
        _, uncorr_result = codec.compress(uncorr_spikes)

        assert corr_result.compression_ratio > uncorr_result.compression_ratio

    def test_delta_sparsity_reported(self):
        rng = np.random.RandomState(42)
        spikes = (rng.random((200, 16)) < 0.05).astype(np.int8)
        codec = DeltaSpikeCodec(group_size=4)
        _, result = codec.compress(spikes)
        assert 0.0 <= result.mean_delta_sparsity <= 1.0
        assert result.codec_type == "delta"
        assert result.n_groups == 4
        assert result.group_size == 4
