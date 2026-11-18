# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Canonical Huffman body tests

"""Test the public Huffman encoder and body decoder."""

from __future__ import annotations

import pytest

from sc_neurocore.spike_codec.entropy import HuffmanEncoder


class TestHuffmanEncoder:
    """Check complete bodies and exact delivery of the integer symbols."""

    def test_encode_decode_roundtrip(self) -> None:
        """A body with several code lengths must decode in original order."""
        enc = HuffmanEncoder()
        values = [1, 2, 1, 3, 1, 2, 1, 1, 4, 1]
        data = enc.encode(values)
        decoded, _ = enc.decode(data, len(values))
        assert decoded == values

    def test_empty_values(self) -> None:
        """An empty body represents exactly zero symbols."""
        enc = HuffmanEncoder()
        data = enc.encode([])
        decoded, _ = enc.decode(data, 0)
        assert decoded == []

    def test_single_symbol(self) -> None:
        """A single-symbol alphabet still carries one bit per symbol."""
        enc = HuffmanEncoder()
        values = [42, 42, 42, 42]
        data = enc.encode(values)
        decoded, _ = enc.decode(data, 4)
        assert decoded == values


def test_signed_symbols_and_exact_consumption() -> None:
    """A standalone body consumes its bits without taking the next field."""
    codec = HuffmanEncoder()
    symbols = [-(1 << 31), -1, 0, (1 << 31) - 1, -1]
    packet = codec.encode(symbols)
    assert codec.decode(packet + b"next", len(symbols)) == (symbols, len(packet))


def test_encode_refuses_unrepresentable_alphabet() -> None:
    """The signed symbol and table width bounds protect the wire format."""
    codec = HuffmanEncoder()
    with pytest.raises(ValueError, match="int32"):
        codec.encode([1 << 31])
    with pytest.raises(ValueError, match="uint16"):
        codec.encode(list(range(1 << 16)))


def test_empty_body_cannot_supply_one_symbol() -> None:
    """The empty-body marker cannot satisfy a nonempty expected stream."""
    codec = HuffmanEncoder()
    with pytest.raises(ValueError, match="symbol count"):
        codec.decode(codec.encode([]), 1)
