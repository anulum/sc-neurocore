# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Framed codec files

"""Store public codec packets with bounded framing, layout and integrity."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import struct
import sys
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import BinaryIO

_MAGIC = b"SC-CODEC\x01"
_LENGTH = struct.Struct("!I")
_MAX_HEADER = 4096
_MAX_PACKET = 16 * 1024 * 1024
MAX_FRAME_VALUES = 1 << 20


@dataclass(frozen=True, slots=True)
class CodecFileHeader:
    """Record the packet family, independent frame geometry and decoder options."""

    codec: str
    variant: str
    rows: int
    channels: int
    frame_rows: int
    sample_rate_hz: int | None
    options: dict[str, str | int]

    def __post_init__(self) -> None:
        """Reject dimensions outside the file reader's allocation bounds."""
        if self.rows < 1 or self.rows > sys.maxsize // max(self.channels, 1) // 8:
            raise ValueError("invalid recording row count")
        if not 1 <= self.channels <= 65535:
            raise ValueError("channel count must be between 1 and 65535")
        if not 1 <= self.frame_rows <= min(65535, MAX_FRAME_VALUES // self.channels):
            raise ValueError("frame exceeds the codec file's row or sample limit")
        if self.sample_rate_hz is not None and not 1 <= self.sample_rate_hz <= (1 << 31) - 1:
            raise ValueError("sample rate must be between 1 and 2147483647 Hz")

    def frame_sizes(self) -> Iterator[int]:
        """Yield expected source row counts, including the final partial frame."""
        for start in range(0, self.rows, self.frame_rows):
            yield min(self.frame_rows, self.rows - start)


def _exact(stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("truncated codec file")
    return data


def _bounded_size(size: int, limit: int, field: str) -> int:
    if not 1 <= size <= limit:
        raise ValueError(f"invalid codec file {field} size")
    return size


def _header(data: bytes) -> CodecFileHeader:
    payload: object = json.loads(data)
    if not isinstance(payload, dict) or set(payload) != {
        "codec",
        "variant",
        "rows",
        "channels",
        "frame_rows",
        "sample_rate_hz",
        "options",
    }:
        raise ValueError("invalid codec file metadata fields")
    codec, variant = payload["codec"], payload["variant"]
    rows, channels, frame_rows = payload["rows"], payload["channels"], payload["frame_rows"]
    rate, options = payload["sample_rate_hz"], payload["options"]
    if (
        not isinstance(codec, str)
        or not isinstance(variant, str)
        or any(type(item) is not int for item in (rows, channels, frame_rows))
        or (rate is not None and type(rate) is not int)
        or not isinstance(options, dict)
        or any(
            not isinstance(key, str) or type(value) not in (str, int)
            for key, value in options.items()
        )
    ):
        raise ValueError("invalid codec file metadata types")
    return CodecFileHeader(codec, variant, rows, channels, frame_rows, rate, options)


def read_codec_header(stream: BinaryIO) -> tuple[CodecFileHeader, bytes]:
    """Read the bounded metadata and retain its exact bytes for verification."""
    magic = _exact(stream, len(_MAGIC))
    if magic != _MAGIC:
        raise ValueError("unrecognised codec file format or version")
    length = _exact(stream, _LENGTH.size)
    size = _bounded_size(_LENGTH.unpack(length)[0], _MAX_HEADER, "header")
    data = _exact(stream, size)
    return _header(data), magic + length + data


def iter_codec_packets(
    stream: BinaryIO, header: CodecFileHeader, prefix: bytes
) -> Iterator[tuple[int, bytes]]:
    """Yield bounded frames and verify the file digest and end of input."""
    digest = hashlib.sha256(prefix)
    for rows in header.frame_sizes():
        length = _exact(stream, _LENGTH.size)
        size = _bounded_size(_LENGTH.unpack(length)[0], _MAX_PACKET, "packet")
        packet = _exact(stream, size)
        digest.update(length)
        digest.update(packet)
        yield rows, packet
    if not hmac.compare_digest(_exact(stream, digest.digest_size), digest.digest()):
        raise ValueError("codec file checksum mismatch")
    if stream.read(1):
        raise ValueError("trailing bytes after codec file checksum")


def write_codec_file(stream: BinaryIO, header: CodecFileHeader, packets: Iterable[bytes]) -> str:
    """Write framed packets and return the digest covering metadata and payload."""
    metadata = json.dumps(asdict(header), separators=(",", ":"), sort_keys=True).encode("utf-8")
    prefix = _MAGIC + _LENGTH.pack(_bounded_size(len(metadata), _MAX_HEADER, "header")) + metadata
    digest = hashlib.sha256(prefix)
    stream.write(prefix)
    for _, packet in zip(header.frame_sizes(), packets, strict=True):
        length = _LENGTH.pack(_bounded_size(len(packet), _MAX_PACKET, "packet"))
        digest.update(length)
        digest.update(packet)
        stream.write(length)
        stream.write(packet)
    stream.write(digest.digest())
    return digest.hexdigest()


@contextmanager
def atomic_codec_output(path: Path, source: Path, overwrite: bool) -> Iterator[Path]:
    """Stage a complete sibling file and publish atomically after successful work."""
    if path.resolve() == source.resolve() or (path.exists() and source.samefile(path)):
        raise ValueError("input and output refer to the same file")
    if not overwrite and (path.exists() or path.is_symlink()):
        raise FileExistsError(f"output already exists: {path}")
    with TemporaryDirectory(prefix=".sc-neurocore-codec-", dir=path.parent) as directory:
        pending = Path(directory) / "output"
        yield pending
        with pending.open("r+b") as stream:
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(pending, path)
        else:
            os.link(pending, path)
