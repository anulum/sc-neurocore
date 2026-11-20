# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Legacy BCI raster input

"""Recover legacy BCI raster files into bounded time-by-channel windows."""

from __future__ import annotations

import struct
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

import numpy as np

from .samples import SampleInput, Samples

_LENGTH = struct.Struct("<I")


def _legacy_windows(
    stream: BinaryIO, sizes: Iterable[int], rows: int, channels: int
) -> Iterator[Samples]:
    total = rows * channels
    consumed = 0
    value = remaining = 0
    for size in sizes:
        window = np.empty((size, channels), dtype=np.uint8)
        flat = window.reshape(-1)
        offset = 0
        while offset < len(flat):
            if remaining == 0:
                pair = stream.read(2)
                if len(pair) != 2:
                    raise ValueError("truncated legacy RLE run")
                value, remaining = pair
                if value > 1 or remaining == 0 or remaining > total - consumed:
                    raise ValueError("invalid legacy RLE run")
            count = min(remaining, len(flat) - offset)
            flat[offset : offset + count] = value
            offset += count
            consumed += count
            remaining -= count
        yield window
    if consumed != total or stream.read(1):
        raise ValueError("legacy RLE runs do not match the declared raster")


@contextmanager
def open_legacy_raster(
    path: Path, rows: int | None, channels: int | None, sample_rate_hz: int | None
) -> Iterator[SampleInput]:
    """Open binary BCI telemetry with explicitly supplied time-by-channel geometry.

    Parameters
    ----------
    path
        File written by the legacy one-dimensional BCI spike codec.
    rows, channels
        Positive time rows and channel count. The old format stores only
        their product. A single channel mask has one time row.
    sample_rate_hz
        Optional acquisition rate in Hz supplied by the caller.

    Yields
    ------
    SampleInput
        Binary raster reader with forward-only bounded windows.

    Raises
    ------
    ValueError
        If the geometry, encoded length, runs or end of input disagree.
        Empty legacy buffers cannot be imported as a non-empty recording.
    """
    if rows is None or channels is None or rows < 1 or channels < 1:
        raise ValueError("legacy RLE needs positive rows and channels")
    with path.open("rb") as stream:
        prefix = stream.read(_LENGTH.size)
        if len(prefix) != _LENGTH.size or _LENGTH.unpack(prefix)[0] != rows * channels:
            raise ValueError("legacy RLE length does not match the supplied geometry")
        yield SampleInput(
            rows,
            channels,
            sample_rate_hz,
            lambda sizes: _legacy_windows(stream, sizes, rows, channels),
        )
