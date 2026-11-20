# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Raster frame and SSCF packet grammar

"""Encode and parse the existing binary raster frame wire format."""

from __future__ import annotations

import struct
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_Raster = NDArray[np.int8]
_FRAME = struct.Struct("!HH")
_MAGIC = b"SSCF"
_CONTAINER = struct.Struct("!4sHIHI")


def _frame_shape(data: memoryview, start: int = 0) -> tuple[int, int]:
    if len(data) - start < _FRAME.size:
        raise ValueError("truncated raster frame header")
    channels, rows = _FRAME.unpack_from(data, start)
    return rows, channels


def _encode_frame(samples: _Raster, stored_rows: int) -> tuple[bytes, int]:
    _, channels = samples.shape
    active = samples.any(axis=0)
    bitmap = np.packbits(~active, bitorder="little")
    packed = np.packbits(samples[:, active].T, axis=1, bitorder="little")
    stride = (stored_rows + 7) // 8
    if packed.shape[1] < stride:
        padded = np.zeros((packed.shape[0], stride), dtype=np.uint8)
        padded[:, : packed.shape[1]] = packed
        packed = padded
    return (
        _FRAME.pack(channels, stored_rows) + bitmap.tobytes() + packed.tobytes(order="C"),
        int(np.count_nonzero(active)),
    )


@dataclass(frozen=True, slots=True)
class _FrameView:
    rows: int
    channels: int
    active: NDArray[np.intp]
    masks: NDArray[np.uint8]

    def check_padding(self, rows: int) -> None:
        complete, bits = divmod(rows, 8)
        if complete < self.masks.shape[1] and (
            np.any(self.masks[:, complete] >> bits) or np.any(self.masks[:, complete + 1 :])
        ):
            raise ValueError("nonzero raster frame padding")

    def decode_into(self, target: _Raster) -> None:
        rows = target.shape[0]
        values = np.unpackbits(self.masks, axis=1, count=rows, bitorder="little")
        target[:, self.active] = values.T


def _read_frame(data: memoryview, start: int) -> tuple[_FrameView, int]:
    rows, channels = _frame_shape(data, start)
    bitmap_start = start + _FRAME.size
    masks_start = bitmap_start + (channels + 7) // 8
    if len(data) < masks_start:
        raise ValueError("truncated raster frame bitmap")
    bitmap = np.frombuffer(data[bitmap_start:masks_start], dtype=np.uint8)
    active = np.flatnonzero(np.unpackbits(bitmap, count=channels, bitorder="little") == 0)
    stride = (rows + 7) // 8
    end = masks_start + active.size * stride
    if len(data) < end:
        raise ValueError("truncated raster frame mask")
    masks = np.frombuffer(data[masks_start:end], dtype=np.uint8).reshape(active.size, stride)
    return _FrameView(rows, channels, active, masks), end


def _decode_frame(packet: bytes) -> _Raster:
    frame, end = _read_frame(memoryview(packet), 0)
    if end != len(packet):
        raise ValueError("trailing raster frame bytes")
    frame.check_padding(frame.rows)
    raster = np.zeros((frame.rows, frame.channels), dtype=np.int8)
    frame.decode_into(raster)
    return raster


def _encode_container(samples: _Raster, window: int) -> tuple[bytes, int, int, int]:
    rows, channels = samples.shape
    count = (rows + window - 1) // window
    packet = bytearray(_CONTAINER.pack(_MAGIC, window, rows, channels, count))
    active_total = 0
    largest = 0
    for start in range(0, rows, window):
        data, active = _encode_frame(samples[start : start + window], window)
        packet.extend(data)
        active_total += active
        largest = max(largest, len(data))
    return bytes(packet), int(samples.sum()), active_total, largest


def _container_header(packet: bytes) -> tuple[int, int, int]:
    if not packet.startswith(_MAGIC):
        raise ValueError("Invalid header magic for SSCF")
    if len(packet) < _CONTAINER.size:
        raise ValueError("truncated SSCF header")
    _, window, rows, channels, count = _CONTAINER.unpack_from(packet)
    if not window or count != (rows + window - 1) // window:
        raise ValueError("invalid SSCF frame count or window size")
    frame_minimum = _FRAME.size + (channels + 7) // 8
    if count > (len(packet) - _CONTAINER.size) // frame_minimum:
        raise ValueError("truncated SSCF frames")
    return window, rows, channels


def _frames(
    packet: bytes, window: int, rows: int, channels: int
) -> Iterator[tuple[int, _FrameView]]:
    start = _CONTAINER.size
    data = memoryview(packet)
    for first_row in range(0, rows, window):
        frame, start = _read_frame(data, start)
        if (frame.rows, frame.channels) != (window, channels):
            raise ValueError("raster frame dimensions do not match SSCF header")
        frame.check_padding(min(window, rows - first_row))
        yield first_row, frame
    if start != len(packet):
        raise ValueError("trailing SSCF bytes")


def _decode_container(packet: bytes, rows: int, channels: int) -> _Raster:
    window, recorded_rows, recorded_channels = _container_header(packet)
    rows = rows or recorded_rows
    if rows > recorded_rows or (channels and channels != recorded_channels):
        raise ValueError("requested raster shape does not match SSCF header")
    for _ in _frames(packet, window, recorded_rows, recorded_channels):
        pass
    raster = np.zeros((rows, recorded_channels), dtype=np.int8)
    for start, frame in _frames(packet, window, recorded_rows, recorded_channels):
        if start >= rows:
            break
        frame.decode_into(raster[start : start + window])
    return raster
