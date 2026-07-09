# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# Tests for sc_neurocore.spike_codec.delta_codec

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.spike_codec.delta_codec import DeltaSpikeCodec, DeltaCompressionResult


class TestDeltaSpikeCodecRoundtrip:
    def test_roundtrip_random(self):
        rng = np.random.RandomState(42)
        spikes = (rng.random((500, 32)) < 0.05).astype(np.int8)
        codec = DeltaSpikeCodec(group_size=8)
        data, result = codec.compress(spikes)
        recovered = codec.decompress(data, 500, 32)
        np.testing.assert_array_equal(recovered, spikes)
        assert result.lossless

    def test_roundtrip_correlated(self):
        """Correlated channels: same base pattern + individual noise."""
        rng = np.random.RandomState(42)
        T, N = 1000, 16
        base = (rng.random((T, 1)) < 0.03).astype(np.int8)
        noise = (rng.random((T, N)) < 0.005).astype(np.int8)
        spikes = (np.broadcast_to(base, (T, N)) | noise).astype(np.int8)
        codec = DeltaSpikeCodec(group_size=4)
        data, _ = codec.compress(spikes)
        recovered = codec.decompress(data, T, N)
        np.testing.assert_array_equal(recovered, spikes)

    def test_roundtrip_silent(self):
        spikes = np.zeros((200, 10), dtype=np.int8)
        codec = DeltaSpikeCodec()
        data, result = codec.compress(spikes)
        recovered = codec.decompress(data, 200, 10)
        np.testing.assert_array_equal(recovered, spikes)
        assert result.n_spikes == 0

    def test_roundtrip_all_firing(self):
        spikes = np.ones((50, 8), dtype=np.int8)
        codec = DeltaSpikeCodec(group_size=4)
        data, _ = codec.compress(spikes)
        recovered = codec.decompress(data, 50, 8)
        np.testing.assert_array_equal(recovered, spikes)

    def test_roundtrip_uneven_groups(self):
        """N not divisible by group_size."""
        rng = np.random.RandomState(42)
        spikes = (rng.random((100, 13)) < 0.05).astype(np.int8)
        codec = DeltaSpikeCodec(group_size=4)
        data, result = codec.compress(spikes)
        recovered = codec.decompress(data, 100, 13)
        np.testing.assert_array_equal(recovered, spikes)
        assert result.n_groups == 4  # ceil(13/4)

    def test_roundtrip_group_size_1(self):
        """group_size=1 means no delta coding — each channel is its own reference."""
        rng = np.random.RandomState(42)
        spikes = (rng.random((100, 8)) < 0.05).astype(np.int8)
        codec = DeltaSpikeCodec(group_size=1)
        data, _ = codec.compress(spikes)
        recovered = codec.decompress(data, 100, 8)
        np.testing.assert_array_equal(recovered, spikes)

    def test_roundtrip_large_group(self):
        """group_size > N: entire array is one group."""
        rng = np.random.RandomState(42)
        spikes = (rng.random((100, 6)) < 0.05).astype(np.int8)
        codec = DeltaSpikeCodec(group_size=32)
        data, _ = codec.compress(spikes)
        recovered = codec.decompress(data, 100, 6)
        np.testing.assert_array_equal(recovered, spikes)


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


class TestDeltaSpikeCodecEdgeCases:
    def test_invalid_magic_raises(self):
        codec = DeltaSpikeCodec()
        with pytest.raises(ValueError, match="Invalid header magic"):
            codec.decompress(b"XXXX" + b"\x00" * 100, 10, 5)

    def test_result_type(self):
        spikes = np.zeros((10, 4), dtype=np.int8)
        _, result = DeltaSpikeCodec().compress(spikes)
        assert isinstance(result, DeltaCompressionResult)

    def test_single_channel(self):
        rng = np.random.RandomState(42)
        spikes = (rng.random((100, 1)) < 0.05).astype(np.int8)
        codec = DeltaSpikeCodec(group_size=4)
        data, _ = codec.compress(spikes)
        recovered = codec.decompress(data, 100, 1)
        np.testing.assert_array_equal(recovered, spikes)
