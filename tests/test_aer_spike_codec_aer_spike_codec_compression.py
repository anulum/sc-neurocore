# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAERSpikeCodecCompression from former test_aer_spike_codec.py

"""Focused suite: TestAERSpikeCodecCompression from former test_aer_spike_codec.py."""

from __future__ import annotations

from tests.aer_spike_codec_support import *  # noqa: F403


class TestAERSpikeCodecCompression:
    def test_sparse_high_compression(self):
        """Very sparse data (0.1% firing) should compress >30x."""
        rng = np.random.RandomState(42)
        spikes = (rng.random((10000, 100)) < 0.001).astype(np.int8)
        codec = AERSpikeCodec()
        _, result = codec.compress(spikes)
        # 3 bytes/event + 17 byte header → ~40x at 0.1% rate
        assert result.compression_ratio > 30.0

    def test_events_proportional_to_spikes(self):
        rng = np.random.RandomState(42)
        spikes = (rng.random((500, 20)) < 0.05).astype(np.int8)
        codec = AERSpikeCodec()
        _, result = codec.compress(spikes)
        assert result.n_events == int(np.sum(spikes))

    def test_bytes_per_event(self):
        rng = np.random.RandomState(42)
        spikes = (rng.random((200, 16)) < 0.05).astype(np.int8)
        _, result = AERSpikeCodec().compress(spikes)
        # Each event: 2 bytes timestamp delta + 1 byte neuron_id (for N<=256)
        # Plus 17 byte header amortized
        assert result.bytes_per_event > 0
