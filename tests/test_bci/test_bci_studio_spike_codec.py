# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeCodec from former test_bci_studio.py

"""Focused suite: TestSpikeCodec from former test_bci_studio.py."""

from __future__ import annotations

import numpy as np
import unittest
from sc_neurocore.bci_studio.bci_studio import (
    SpikeCodec,
)


class TestSpikeCodec(unittest.TestCase):
    """Encoding, decoding and compression of the spike codec."""

    def setUp(self) -> None:
        """Create a spike codec."""
        self.codec = SpikeCodec()

    def test_encode_decode_roundtrip(self) -> None:
        """A spike train decodes to itself."""
        spikes = np.array([1, 1, 0, 0, 0, 1, 0, 1, 1, 1], dtype=np.uint8)
        encoded = self.codec.encode(spikes)
        decoded = self.codec.decode(encoded)
        np.testing.assert_array_equal(spikes, decoded)

    def test_empty_array(self) -> None:
        """An empty spike train encodes to no bytes."""
        spikes = np.array([], dtype=np.uint8)
        encoded = self.codec.encode(spikes)
        self.assertEqual(encoded, b"")

    def test_all_zeros(self) -> None:
        """An all-zero spike train decodes to itself."""
        spikes = np.zeros(100, dtype=np.uint8)
        encoded = self.codec.encode(spikes)
        decoded = self.codec.decode(encoded)
        np.testing.assert_array_equal(spikes, decoded)

    def test_all_ones(self) -> None:
        """An all-one spike train decodes to itself."""
        spikes = np.ones(100, dtype=np.uint8)
        encoded = self.codec.encode(spikes)
        decoded = self.codec.decode(encoded)
        np.testing.assert_array_equal(spikes, decoded)

    def test_compression_ratio_sparse(self) -> None:
        """A sparse spike train compresses."""
        spikes = np.zeros(1000, dtype=np.uint8)
        spikes[::100] = 1  # very sparse
        ratio = self.codec.compression_ratio(spikes)
        self.assertGreater(ratio, 1.0)

    def test_decode_returns_empty_for_truncated_header(self) -> None:
        """A truncated header decodes to an empty train."""
        # An RLE stream shorter than the 4-byte length header carries no spikes.
        self.assertEqual(self.codec.decode(b"\x00\x00").size, 0)

    def test_compression_ratio_of_empty_array_is_unity(self) -> None:
        """An empty spike train has a compression ratio of one."""
        # An empty spike array compresses to nothing, so the ratio is defined as 1.0.
        self.assertEqual(self.codec.compression_ratio(np.array([], dtype=np.uint8)), 1.0)

    def test_compression_ratio_dense(self) -> None:
        """A dense random train has a positive compression ratio."""
        rng = np.random.default_rng(42)
        spikes = rng.integers(0, 2, size=1000, dtype=np.uint8)
        ratio = self.codec.compression_ratio(spikes)
        self.assertGreater(ratio, 0)
