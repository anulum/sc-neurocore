# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDeltaSpikeCodecRoundtrip from former test_delta_spike_codec.py

"""Focused suite: TestDeltaSpikeCodecRoundtrip from former test_delta_spike_codec.py."""

from __future__ import annotations

from tests.delta_spike_codec_support import *  # noqa: F403


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
