# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Entropy coding backends for spike codec

"""Encode and decode canonical Huffman bodies for integer streams."""

from __future__ import annotations

import heapq
import struct
from collections import Counter

_MAX_CODE_LENGTH = 63
_ENTRY = struct.Struct("!iB")


def _code_lengths(symbols: list[int]) -> dict[int, int]:
    # Counter keeps first occurrence order, which breaks equal-frequency ties.
    frequencies = Counter(symbols)
    if len(frequencies) > 0xFFFF:
        raise ValueError("Huffman table exceeds uint16")
    nodes: list[int | tuple[int, int]] = list(frequencies)
    heap = [(frequency, node) for node, frequency in enumerate(frequencies.values())]
    heapq.heapify(heap)
    while len(heap) > 1:
        left_frequency, left = heapq.heappop(heap)
        right_frequency, right = heapq.heappop(heap)
        heapq.heappush(heap, (left_frequency + right_frequency, len(nodes)))
        nodes.append((left, right))

    lengths = {}
    pending = [(heap[0][1], 0)]
    while pending:
        index, depth = pending.pop()
        node = nodes[index]
        if isinstance(node, int):
            lengths[node] = max(1, depth)
        else:
            left, right = node
            pending.extend(((left, depth + 1), (right, depth + 1)))
    return lengths


def _canonical_codes(lengths: dict[int, int]) -> dict[int, tuple[int, int]]:
    codes = {}
    code = previous = 0
    for symbol, length in sorted(lengths.items(), key=lambda entry: (entry[1], entry[0])):
        if not 1 <= length <= _MAX_CODE_LENGTH:
            raise ValueError("invalid Huffman code length")
        if code >= 1 << previous:
            raise ValueError("oversubscribed Huffman table")
        code <<= length - previous
        codes[symbol] = (code, length)
        code += 1
        previous = length
    return codes


class HuffmanEncoder:
    """Code signed 32-bit symbols with canonical codes and MSB-first bits."""

    def encode(self, values: list[int]) -> bytes:
        """Encode symbols as a length table, bit count and packed bits."""
        if not values:
            return b"\x00\x00"
        table = _canonical_codes(_code_lengths(values))
        header = bytearray(struct.pack("!H", len(table)))
        for symbol, (_, length) in table.items():
            if not -(1 << 31) <= symbol < 1 << 31:
                raise ValueError("Huffman symbol exceeds int32")
            header.extend(_ENTRY.pack(symbol, length))

        bit_count = sum(table[symbol][1] for symbol in values)
        if bit_count >= 1 << 32:
            raise ValueError("Huffman bit count exceeds uint32")
        header.extend(struct.pack("!I", bit_count))
        packed = bytearray()
        pending = width = 0
        for symbol in values:
            code, length = table[symbol]
            pending = (pending << length) | code
            width += length
            while width >= 8:
                width -= 8
                packed.append(pending >> width)
                pending &= (1 << width) - 1
        if width:
            packed.append(pending << (8 - width))
        return bytes(header + packed)

    def decode(self, data: bytes | memoryview, n_symbols: int) -> tuple[list[int], int]:
        """Decode one body and return its symbols and bytes consumed.

        Bytes after the complete body are left for the caller's enclosing
        format. A malformed table, bitstream or symbol count raises ValueError.
        """
        if len(data) < 2 or n_symbols < 0:
            raise ValueError("invalid Huffman header or symbol count")
        count = int.from_bytes(data[:2], "big")
        if count == 0 and n_symbols == 0:
            return [], 2
        if not 1 <= count <= n_symbols:
            raise ValueError("Huffman table does not match the symbol count")
        table_end = 2 + count * _ENTRY.size
        if len(data) < table_end + 4:
            raise ValueError("truncated Huffman table or bit count")
        lengths = dict(_ENTRY.iter_unpack(memoryview(data)[2:table_end]))
        if len(lengths) != count:
            raise ValueError("duplicate Huffman symbol")
        lookup = {code: symbol for symbol, code in _canonical_codes(lengths).items()}
        bit_count = int.from_bytes(data[table_end : table_end + 4], "big")
        start = table_end + 4
        end = start + (bit_count + 7) // 8
        if bit_count < n_symbols or len(data) < end:
            raise ValueError("truncated Huffman symbols")
        if bit_count % 8 and data[end - 1] & ((1 << (8 - bit_count % 8)) - 1):
            raise ValueError("nonzero Huffman padding")

        values: list[int] = []
        code = length = 0
        for bit in range(bit_count):
            code = (code << 1) | ((data[start + bit // 8] >> (7 - bit % 8)) & 1)
            length += 1
            symbol = lookup.get((code, length))
            if symbol is not None:
                if len(values) == n_symbols:
                    raise ValueError("excess Huffman symbols")
                values.append(symbol)
                code = length = 0
            elif length == _MAX_CODE_LENGTH:
                raise ValueError("invalid Huffman code")
        if len(values) != n_symbols or length:
            raise ValueError("incomplete Huffman symbol stream")
        return values, end
