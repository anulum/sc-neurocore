# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeCodecHuffman from former test_spike_codec.py

"""Focused suite: TestSpikeCodecHuffman from former test_spike_codec.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.spike_codec import (
    SpikeCodec,
)


class TestSpikeCodecHuffman:
    """Huffman and entropy coding in the spike codec."""

    def test_huffman_roundtrip_sparse(self) -> None:
        """Huffman roundtrip sparse."""
        rng = np.random.RandomState(42)
        spikes = (rng.random((500, 20)) < 0.01).astype(np.int8)
        codec = SpikeCodec(entropy="huffman")
        data, result = codec.compress(spikes)
        rec = codec.decompress(data, 500, 20)
        np.testing.assert_array_equal(rec, spikes)

    def test_huffman_roundtrip_dense(self) -> None:
        """Huffman roundtrip dense."""
        rng = np.random.RandomState(42)
        spikes = (rng.random((200, 16)) < 0.3).astype(np.int8)
        codec = SpikeCodec(entropy="huffman")
        data, _ = codec.compress(spikes)
        rec = codec.decompress(data, 200, 16)
        np.testing.assert_array_equal(rec, spikes)

    def test_huffman_beats_varint_on_dense(self) -> None:
        """Huffman beats varint on dense."""
        rng = np.random.RandomState(42)
        spikes = (rng.random((1000, 32)) < 0.2).astype(np.int8)
        d_var, _ = SpikeCodec(entropy="varint").compress(spikes)
        d_huf, _ = SpikeCodec(entropy="huffman").compress(spikes)
        assert len(d_huf) < len(d_var)

    def test_huffman_empty(self) -> None:
        """Huffman coding round-trips an empty raster."""
        spikes = np.zeros((100, 10), dtype=np.int8)
        codec = SpikeCodec(entropy="huffman")
        data, _ = codec.compress(spikes)
        rec = codec.decompress(data, 100, 10)
        np.testing.assert_array_equal(rec, spikes)

    def test_auto_entropy_roundtrip(self) -> None:
        """Auto entropy roundtrip."""
        rng = np.random.RandomState(42)
        for rate in [0.005, 0.05, 0.2]:
            spikes = (rng.random((500, 16)) < rate).astype(np.int8)
            codec = SpikeCodec(entropy="auto")
            data, _ = codec.compress(spikes)
            rec = codec.decompress(data, 500, 16)
            np.testing.assert_array_equal(rec, spikes)
