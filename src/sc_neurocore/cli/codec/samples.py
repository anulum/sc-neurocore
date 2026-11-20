# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Codec file input and array output

"""Read sample windows and write decoded NumPy arrays without loading a recording."""

from __future__ import annotations

import wave
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import BinaryIO, TextIO
from zipfile import ZIP_STORED, ZipFile

import numpy as np
from numpy.typing import NDArray

Samples = NDArray[np.generic]


@dataclass(frozen=True, slots=True)
class SampleInput:
    """Shape, sample rate and forward reader of a numeric recording."""

    rows: int
    channels: int
    sample_rate_hz: int | None
    windows: Callable[[Iterable[int]], Iterator[Samples]]


def _npy_windows(array: Samples, sizes: Iterable[int]) -> Iterator[Samples]:
    offset = 0
    for rows in sizes:
        yield array[offset : offset + rows]
        offset += rows


def _pcm_windows(reader: wave.Wave_read, sizes: Iterable[int]) -> Iterator[Samples]:
    channels = reader.getnchannels()
    for rows in sizes:
        data = reader.readframes(rows)
        if len(data) != rows * channels * 2:
            raise ValueError("truncated PCM16 recording")
        yield np.frombuffer(data, dtype="<i2").reshape(rows, channels)


def _event_rows(stream: TextIO, rows: int, channels: int) -> Iterator[tuple[int, int]]:
    if stream.readline(64).strip() != "sample,channel":
        raise ValueError("event CSV requires a sample,channel header")
    previous = (-1, -1)
    while line := stream.readline(64):
        if len(line) == 64 or line.count(",") != 1:
            raise ValueError("event CSV requires one sample and channel per line")
        left, right = line.split(",")
        point = int(left), int(right)
        if point <= previous or not (0 <= point[0] < rows and 0 <= point[1] < channels):
            raise ValueError("event indices must be sorted, unique and inside the recording")
        previous = point
        yield point


def _event_windows(
    stream: TextIO, sizes: Iterable[int], rows: int, channels: int
) -> Iterator[Samples]:
    events = _event_rows(stream, rows, channels)
    point = next(events, None)
    offset = 0
    for size in sizes:
        window = np.zeros((size, channels), dtype=np.int8)
        stop = offset + size
        while point is not None and point[0] < stop:
            window[point[0] - offset, point[1]] = 1
            point = next(events, None)
        yield window
        offset = stop


@contextmanager
def open_sample_input(
    path: Path,
    sample_rate_hz: int | None,
    *,
    event_rows: int | None,
    event_channels: int | None,
) -> Iterator[SampleInput]:
    """Open numeric NPY, PCM16 WAV or sorted event CSV data as time rows."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        if event_rows is None or event_channels is None:
            raise ValueError("event CSV needs --rows and --channels")
        with path.open(encoding="utf-8") as stream:
            yield SampleInput(
                event_rows,
                event_channels,
                sample_rate_hz,
                lambda sizes: _event_windows(stream, sizes, event_rows, event_channels),
            )
        return
    if event_rows is not None or event_channels is not None:
        raise ValueError("--rows and --channels apply only to event CSV input")
    if suffix == ".npy":
        array = np.lib.format.open_memmap(path, mode="r", max_header_size=4096)
        if array.ndim not in (1, 2) or array.dtype.kind not in "biuf":
            raise ValueError("NPY input must be a one- or two-dimensional numeric array")
        shaped = array[:, None] if array.ndim == 1 else array
        yield SampleInput(
            shaped.shape[0],
            shaped.shape[1],
            sample_rate_hz,
            lambda sizes: _npy_windows(shaped, sizes),
        )
    elif suffix == ".wav":
        with wave.open(str(path), "rb") as reader:
            if reader.getsampwidth() != 2 or reader.getcomptype() != "NONE":
                raise ValueError("WAV input must contain uncompressed signed 16-bit samples")
            rate = reader.getframerate()
            if sample_rate_hz is not None and rate != sample_rate_hz:
                raise ValueError("specified sample rate disagrees with the WAV header")
            yield SampleInput(
                reader.getnframes(),
                reader.getnchannels(),
                rate,
                lambda sizes: _pcm_windows(reader, sizes),
            )
    else:
        raise ValueError("sample input must be NPY, PCM16 WAV or event CSV")


@dataclass(frozen=True, slots=True)
class OutputField:
    """Name and array layout of a single decoded result."""

    name: str
    rows: int
    channels: int
    dtype: str


def _write_fields(
    fields: tuple[OutputField, ...],
    files: tuple[BinaryIO, ...],
    windows: Iterable[Mapping[str, Samples]],
) -> None:
    counts = dict.fromkeys((field.name for field in fields), 0)
    for field, stream in zip(fields, files, strict=True):
        np.lib.format.write_array_header_2_0(
            stream,
            {"shape": (field.rows, field.channels), "fortran_order": False, "descr": field.dtype},
        )
    for window in windows:
        for field, stream in zip(fields, files, strict=True):
            data = window[field.name]
            if data.ndim != 2 or data.shape[1] != field.channels:
                raise ValueError(f"decoded {field.name} has an incompatible channel layout")
            counts[field.name] += data.shape[0]
            if counts[field.name] > field.rows:
                raise ValueError(f"decoded {field.name} exceeds its declared row count")
            stream.write(data.astype(field.dtype, copy=False).tobytes(order="C"))
    for field in fields:
        if counts[field.name] != field.rows:
            raise ValueError(f"decoded {field.name} has an incomplete row count")


def write_decoded_samples(
    path: Path,
    fields: tuple[OutputField, ...],
    windows: Iterable[Mapping[str, Samples]],
    *,
    archive: bool,
) -> None:
    """Write the selected fields as a single NPY array or a bounded-memory NPZ archive."""
    if not archive:
        with path.open("xb") as stream:
            _write_fields(fields, (stream,), windows)
        return
    with TemporaryDirectory(prefix="arrays-", dir=path.parent) as directory:
        members = tuple(Path(directory) / f"{field.name}.npy" for field in fields)
        with ExitStack() as stack:
            streams = tuple(stack.enter_context(member.open("xb")) for member in members)
            _write_fields(fields, streams, windows)
        with ZipFile(path, "x", compression=ZIP_STORED, allowZip64=True) as output:
            for member in members:
                output.write(member, member.name)
