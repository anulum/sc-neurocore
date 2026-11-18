# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ISI raster packet framing

"""Read and write the legacy varint and Huffman raster packets."""

from __future__ import annotations

import struct
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np

from .entropy import HuffmanEncoder


def encode_varint(value: int) -> bytes:
    """Encode one nonnegative integer as unsigned LEB128."""
    result = bytearray()
    while value >= 0x80:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value)
    return bytes(result)


def encode_packet(events: list[list[int]], rows: int, channels: int, entropy: str) -> bytes:
    """Encode ordered event times in the selected ISI wire format."""
    header = struct.pack("!II", rows, channels)
    if entropy == "huffman":
        counts = b"".join(encode_varint(len(times)) for times in events)
        first = b"".join(encode_varint(times[0]) for times in events if times)
        intervals = [
            time - previous for times in events for previous, time in zip(times, times[1:])
        ]
        return (
            b"\x01"
            + header
            + struct.pack("!I", len(counts))
            + counts
            + struct.pack("!I", len(first))
            + first
            + HuffmanEncoder().encode(intervals)
        )
    if rows >= 1 << 24:
        raise ValueError(
            "varint packets require T < 2**24 to avoid the Huffman marker collision; "
            "use entropy='huffman' for longer rasters"
        )
    packet = bytearray(header)
    for times in events:
        packet.extend(encode_varint(len(times)))
        if times:
            packet.extend(encode_varint(times[0]))
        for previous, time in zip(times, times[1:]):
            packet.extend(encode_varint(time - previous))
    return bytes(packet)


class _Reader:
    """Bound each field read to its packet section."""

    def __init__(self, data: memoryview) -> None:
        self.remaining = data

    def take(self, size: int) -> memoryview:
        if size > len(self.remaining):
            raise ValueError("truncated ISI packet")
        field, self.remaining = self.remaining[:size], self.remaining[size:]
        return field

    def uint32(self) -> int:
        return int.from_bytes(self.take(4), "big")

    def varint(self) -> int:
        value = 0
        shift = 0
        while True:
            byte = self.take(1)[0]
            if shift == 28 and byte > 15:
                raise ValueError("ISI varint exceeds uint32")
            value |= (byte & 0x7F) << shift
            if byte < 0x80:
                if byte == 0 and shift != 0:
                    raise ValueError("nonminimal ISI varint")
                return value
            shift += 7

    def count(self, limit: int) -> int:
        count = self.varint()
        if count > limit:
            raise ValueError("ISI event count exceeds the number of time bins")
        return count

    def section(self) -> _Reader:
        return _Reader(self.take(self.uint32()))

    def finish(self) -> None:
        if self.remaining:
            raise ValueError("unexpected trailing ISI packet data")


def _times(first: int, intervals: Iterable[int], bins: int) -> Iterator[int]:
    if first >= bins:
        raise ValueError("ISI timestamp is outside the raster")
    yield first
    time = first
    for interval in intervals:
        if interval <= 0:
            raise ValueError("ISI intervals must be positive")
        time += interval
        if time >= bins:
            raise ValueError("ISI timestamp is outside the raster")
        yield time


def _varint_events(reader: _Reader, channels: int, bins: int) -> Iterator[list[int]]:
    for _ in range(channels):
        count = reader.count(bins)
        if count > len(reader.remaining):
            raise ValueError("truncated ISI channel")
        if count:
            yield list(_times(reader.varint(), (reader.varint() for _ in range(count - 1)), bins))
        else:
            yield []
    reader.finish()


def _huffman_events(reader: _Reader, channels: int, bins: int) -> Iterator[list[int]]:
    counts, first = reader.section(), reader.section()
    if len(counts.remaining) < channels:
        raise ValueError("truncated ISI count section")
    counts_by_channel = [counts.count(bins) for _ in range(channels)]
    counts.finish()
    body = reader.remaining
    intervals, consumed = HuffmanEncoder().decode(
        body, sum(max(0, count - 1) for count in counts_by_channel)
    )
    reader.take(consumed)
    reader.finish()
    cursor = iter(intervals)
    for count in counts_by_channel:
        if count:
            yield list(_times(first.varint(), (next(cursor) for _ in range(count - 1)), bins))
        else:
            yield []
    first.finish()


def decode_packet(data: bytes, rows: int, channels: int, precision: int) -> np.ndarray[Any, Any]:
    """Validate a complete ISI packet and reconstruct its binary output raster."""
    reader = _Reader(memoryview(data))
    huffman = data[:1] == b"\x01"
    if huffman:
        reader.take(1)
    original_rows, original_channels = reader.uint32(), reader.uint32()
    bins = original_rows // precision
    if channels != original_channels or rows not in (original_rows, bins):
        raise ValueError("ISI dimensions do not match the packet")
    decoder = _huffman_events if huffman else _varint_events
    events = list(decoder(reader, channels, bins))
    raster = np.zeros((rows, channels), dtype=np.int8)
    for channel, times in enumerate(events):
        if times:
            raster[times, channel] = 1
    return raster
