# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHuffmanEncoder from former test_spike_codec.py

"""Focused suite: TestHuffmanEncoder from former test_spike_codec.py."""

from __future__ import annotations

from tests.spike_codec_support import *  # noqa: F403

class TestHuffmanEncoder:
    def test_encode_decode_roundtrip(self):
        from sc_neurocore.spike_codec.entropy import HuffmanEncoder

        enc = HuffmanEncoder()
        values = [1, 2, 1, 3, 1, 2, 1, 1, 4, 1]
        data = enc.encode(values)
        decoded, _ = enc.decode(data, len(values))
        assert decoded == values

    def test_empty_values(self):
        from sc_neurocore.spike_codec.entropy import HuffmanEncoder

        enc = HuffmanEncoder()
        data = enc.encode([])
        decoded, _ = enc.decode(data, 0)
        assert decoded == []

    def test_single_symbol(self):
        from sc_neurocore.spike_codec.entropy import HuffmanEncoder

        enc = HuffmanEncoder()
        values = [42, 42, 42, 42]
        data = enc.encode(values)
        decoded, _ = enc.decode(data, 4)
        assert decoded == values
