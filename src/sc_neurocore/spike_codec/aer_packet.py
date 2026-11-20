# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Address-event raster packet grammar

"""Encode and parse AERX and AERI raster containers."""

from __future__ import annotations

import struct
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .codec import _dimension

_Raster = NDArray[np.int8]
_EVENTS = b"AERX"
_SILENCES = b"AERI"
_HEADER = struct.Struct("!4sIIIB")
_DELTA = struct.Struct("!H")


def _encode(samples: _Raster, timestamp_bits: int, neuron_bits: int) -> tuple[bytes, int, int]:
    rows, channels = samples.shape
    spikes = int(samples.sum())
    inverted = spikes > samples.size // 2
    events = _dimension(samples.size - spikes if inverted else spikes)
    symbols = np.nonzero(samples == int(not inverted))
    width = (max(1, channels.bit_length(), neuron_bits) + 7) // 8
    packet = bytearray(
        _HEADER.pack(_SILENCES if inverted else _EVENTS, rows, channels, events, width)
    )
    maximum = (1 << timestamp_bits) - 1
    escape = _DELTA.pack(maximum) + b"\xff" * width
    previous = 0
    for timestamp, neuron in zip(*symbols, strict=True):
        current = int(timestamp)
        delta = current - previous
        if delta > maximum:
            repeats, remainder = divmod(delta - 1, maximum)
            packet.extend(escape * repeats)
            delta = remainder + 1
        packet.extend(_DELTA.pack(delta))
        packet.extend(int(neuron).to_bytes(width, "big"))
        previous = current
    return bytes(packet), spikes, events


@dataclass(frozen=True, slots=True)
class _Packet:
    rows: int
    channels: int
    events: int
    width: int
    inverted: bool
    body: memoryview

    @classmethod
    def parse(cls, data: bytes) -> _Packet:
        if data[:4] not in (_EVENTS, _SILENCES):
            raise ValueError("Invalid header magic for AER")
        if len(data) < _HEADER.size:
            raise ValueError("truncated AER header")
        magic, rows, channels, events, width = _HEADER.unpack_from(data)
        if not width or (width < 4 and channels >= 1 << (width * 8)):
            raise ValueError("AER address width cannot store the channel count and escape")
        if events > rows * channels:
            raise ValueError("AER event count exceeds raster positions")
        body = memoryview(data)[_HEADER.size :]
        stride = _DELTA.size + width
        if len(body) % stride or len(body) // stride < events:
            raise ValueError("invalid AER event body length")
        return cls(rows, channels, events, width, magic == _SILENCES, body)

    def positions(self) -> Iterator[tuple[int, int]]:
        time = 0
        count = 0
        previous: tuple[int, int] | None = None
        escaped = False
        stride = _DELTA.size + self.width
        marker = b"\xff" * self.width
        for index in range(0, len(self.body), stride):
            if count == self.events:
                raise ValueError("trailing AER event bytes")
            delta = _DELTA.unpack_from(self.body, index)[0]
            time += delta
            if time > 0xFFFFFFFF:
                raise ValueError("AER timestamp overflows uint32")
            if time >= self.rows:
                raise ValueError("AER timestamp is outside the raster")
            address = self.body[index + _DELTA.size : index + stride]
            if address == marker:
                if not delta:
                    raise ValueError("AER timestamp escape must advance time")
                escaped = True
                continue
            neuron = int.from_bytes(address, "big")
            if neuron > 0xFFFFFFFF:
                raise ValueError("AER neuron address exceeds uint32")
            if neuron >= self.channels:
                raise ValueError("AER neuron address is outside the raster")
            point = (time, neuron)
            if previous is not None and point <= previous:
                raise ValueError("AER events must increase by timestamp then address")
            previous = point
            escaped = False
            count += 1
            yield point
        if escaped or count != self.events:
            raise ValueError("incomplete AER events or timestamp escape")


def _decode(data: bytes, rows: int, channels: int) -> _Raster:
    packet = _Packet.parse(data)
    rows = rows or packet.rows
    channels = channels or packet.channels
    if rows > packet.rows or channels > packet.channels:
        raise ValueError("requested raster shape exceeds AER header")
    for _ in packet.positions():
        pass
    raster = np.full((rows, channels), int(packet.inverted), dtype=np.int8)
    target = int(not packet.inverted)
    for time, neuron in packet.positions():
        if time < rows and neuron < channels:
            raster[time, neuron] = target
    return raster
