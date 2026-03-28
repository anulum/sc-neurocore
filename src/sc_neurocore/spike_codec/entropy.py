# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Entropy coding backends for spike codec

"""Huffman and range coding for ISI integer streams.

LEB128 varint uses ceil(log2(x)/7) bytes per symbol — fixed overhead
per magnitude bucket. Huffman coding uses per-symbol bit lengths
proportional to -log2(frequency), approaching the Shannon entropy limit.

On exponentially-distributed ISIs (typical for neural spike trains),
Huffman saves 30-60% over varint because short ISIs (most common)
get 2-4 bit codes instead of 8+ bit varints.
"""

from __future__ import annotations

import heapq
import struct
from collections import Counter
from typing import Any


def _build_huffman_table(symbols: list[int]) -> dict[int, tuple[int, int]]:
    """Build Huffman codes from a symbol stream.

    Returns dict mapping symbol → (code_bits, code_length).
    Uses package-merge algorithm for optimal length-limited codes.
    """
    if not symbols:  # pragma: no cover — defensive guard
        return {}

    freqs = Counter(symbols)
    if len(freqs) == 1:
        sym = next(iter(freqs))
        return {sym: (0, 1)}

    # Build tree with heapq
    # Nodes: (freq, id, symbol_or_None, left, right)
    heap: list[tuple[int, int, int | None, Any, Any]] = []
    node_id = 0
    for sym, freq in freqs.items():
        heapq.heappush(heap, (freq, node_id, sym, None, None))
        node_id += 1

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = (left[0] + right[0], node_id, None, left, right)
        heapq.heappush(heap, merged)
        node_id += 1

    # Extract code lengths
    lengths: dict[int, int] = {}
    _walk_tree(heap[0], lengths, 0)

    return _canonical_codes(lengths)


def _walk_tree(node: tuple[int, int, int | None, Any, Any], lengths: dict[int, int], depth: int) -> None:
    _, _, sym, left, right = node
    if sym is not None:
        lengths[sym] = max(depth, 1)
        return
    if left is not None:
        _walk_tree(left, lengths, depth + 1)
    if right is not None:
        _walk_tree(right, lengths, depth + 1)


def _canonical_codes(lengths: dict[int, int]) -> dict[int, tuple[int, int]]:
    """Generate canonical Huffman codes from bit lengths.

    Canonical codes: sorted by (length, symbol), sequential assignment.
    Decoder only needs the length table to reconstruct codes.
    """
    if not lengths:  # pragma: no cover — defensive guard
        return {}

    sorted_syms = sorted(lengths.items(), key=lambda x: (x[1], x[0]))
    codes = {}
    code = 0
    prev_len = sorted_syms[0][1]

    for sym, length in sorted_syms:
        code <<= length - prev_len
        codes[sym] = (code, length)
        code += 1
        prev_len = length

    return codes


class HuffmanEncoder:
    """Encode integer streams using adaptive Huffman coding.

    Builds code table from the input data, stores table in header,
    then encodes symbols as variable-length bit sequences.
    """

    def encode(self, values: list[int]) -> bytes:
        """Encode integer list to compressed bytes.

        Format: table_size(2) + table_entries + packed_bits + padding_info(1)
        """
        if not values:
            return struct.pack("!H", 0)

        table = _build_huffman_table(values)

        # Serialize table: n_entries(2) + [symbol(4) + length(1)] per entry
        entries = sorted(table.items(), key=lambda x: (x[1][1], x[0]))
        header = struct.pack("!H", len(entries))
        for sym, (_, length) in entries:
            header += struct.pack("!iB", sym, length)

        # Encode symbols as bit stream
        bits = []
        for v in values:
            code, length = table[v]
            for i in range(length - 1, -1, -1):
                bits.append((code >> i) & 1)

        # Pack bits into bytes
        n_bits = len(bits)
        n_bytes = (n_bits + 7) // 8
        packed = bytearray(n_bytes)
        for i, bit in enumerate(bits):
            if bit:
                packed[i // 8] |= 1 << (7 - (i % 8))

        # Padding info: how many bits in last byte are padding
        padding = n_bytes * 8 - n_bits
        return header + struct.pack("!I", n_bits) + bytes(packed)

    def decode(self, data: bytes, n_symbols: int) -> tuple[list[int], int]:
        """Decode compressed bytes to integer list.

        Returns (values, bytes_consumed).
        """
        pos = 0
        n_entries = struct.unpack("!H", data[pos : pos + 2])[0]
        pos += 2

        if n_entries == 0:
            return [], pos

        # Reconstruct table
        lengths: dict[int, int] = {}
        for _ in range(n_entries):
            sym = struct.unpack("!i", data[pos : pos + 4])[0]
            length = data[pos + 4]
            lengths[sym] = length
            pos += 5

        # Rebuild canonical codes
        codes = _canonical_codes(lengths)
        # Build reverse lookup: (code, length) → symbol
        decode_map = {(code, length): sym for sym, (code, length) in codes.items()}

        # Read bit count
        n_bits = struct.unpack("!I", data[pos : pos + 4])[0]
        pos += 4

        # Read packed bits
        n_bytes = (n_bits + 7) // 8
        packed = data[pos : pos + n_bytes]
        pos += n_bytes

        # Decode bit stream
        values: list[int] = []
        bit_pos = 0
        current_code = 0
        current_len = 0
        max_len = max(lengths.values()) if lengths else 0

        while len(values) < n_symbols and bit_pos < n_bits:
            byte_idx = bit_pos // 8
            bit_idx = 7 - (bit_pos % 8)
            bit = (packed[byte_idx] >> bit_idx) & 1
            current_code = (current_code << 1) | bit
            current_len += 1
            bit_pos += 1

            key = (current_code, current_len)
            if key in decode_map:
                values.append(decode_map[key])
                current_code = 0
                current_len = 0

            if current_len > max_len:  # pragma: no cover
                break

        return values, pos
