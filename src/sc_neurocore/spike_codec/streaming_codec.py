# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Streaming spike codec: fixed-latency, causal

"""Compress binary raster windows as independently decodable frames."""

from __future__ import annotations

from collections.abc import Generator, Iterable
from dataclasses import dataclass
from importlib import import_module
from operator import index
from typing import Any, Protocol, cast

import numpy as np

from .codec import CompressionResult, _dimension, _ENGINE, _installed_native_extension
from .raster_frame import (
    _MAGIC,
    _Raster,
    _decode_container,
    _decode_frame,
    _encode_container,
    _encode_frame,
    _frame_shape,
)

_NATIVE_SYMBOLS = (
    "raster_stream_encode_frame",
    "raster_stream_decode_frame",
    "raster_stream_encode_raster",
    "raster_stream_decode_raster",
)


class _NativeRasterStream(Protocol):
    def raster_stream_encode_frame(self, samples: _Raster) -> bytes: ...
    def raster_stream_decode_frame(self, packet: bytes) -> _Raster: ...
    def raster_stream_encode_raster(
        self, samples: _Raster, window_size: int
    ) -> tuple[bytes, int, int, int]: ...
    def raster_stream_decode_raster(self, packet: bytes, rows: int, channels: int) -> _Raster: ...


def _native(backend: str) -> _NativeRasterStream | None:
    if backend not in ("auto", "python", "rust"):
        raise ValueError("backend must be auto, python or rust")
    if backend == "python":
        return None
    try:
        module = import_module(_ENGINE)
    except ImportError:
        if _installed_native_extension():
            raise
        module = None
    if module is not None and all(
        callable(getattr(module, name, None)) for name in _NATIVE_SYMBOLS
    ):
        return cast(_NativeRasterStream, module)
    if backend == "rust":
        raise ValueError("backend 'rust' needs sc_neurocore_engine with the raster stream codec")
    return None


def _window(value: int) -> int:
    size = index(value)
    if not 1 <= size <= 0xFFFF:
        raise ValueError("raster window size must be in 1..65535")
    return size


def _raster(value: np.ndarray[Any, Any], row_limit: int = 0xFFFFFFFF) -> np.ndarray[Any, Any]:
    raster = np.asarray(value)
    if raster.ndim != 2:
        raise ValueError("raster must have shape (time, channels)")
    rows, channels = raster.shape
    if rows > row_limit or channels > 0xFFFF:
        raise ValueError("raster dimensions exceed the packet field widths")
    return raster


def _binary(raster: np.ndarray[Any, Any]) -> _Raster:
    if raster.dtype != np.bool_ and not np.all((raster == 0) | (raster == 1)):
        raise ValueError("raster must contain only zeros and ones")
    return np.ascontiguousarray(raster, dtype=np.int8)


@dataclass
class StreamingCompressionResult(CompressionResult):
    """Compression result with statistics over the independent frames."""

    window_size: int = 0
    n_frames: int = 0
    mean_active_channels: float = 0.0
    max_frame_bytes: int = 0
    codec_type: str = "streaming"


class StreamingSpikeCodec:
    """Encode binary rasters with fixed-size, independently decodable frames.

    Parameters
    ----------
    window_size : int
        Time samples per complete frame, in 1 through 65535.
        At 20 kHz a 20-sample window spans one millisecond of acquisition.
    backend : str
        'python', 'rust', or 'auto'. Auto uses the native codec when all
        four packet operations are installed and otherwise uses Python.

    Notes
    -----
    Whole-raster SSCF packets pad the last window and store the original
    length. The pull API emits an unpadded standalone final frame.
    """

    HEADER_MAGIC = _MAGIC

    def __init__(self, window_size: int = 20, *, backend: str = "auto") -> None:
        self.window_size = _window(window_size)
        self._engine = _native(backend)

    @property
    def backend(self) -> str:
        """Return the selected complete implementation, 'python' or 'rust'."""
        return "rust" if self._engine is not None else "python"

    def compress(self, spikes: np.ndarray[Any, Any]) -> tuple[bytes, StreamingCompressionResult]:
        """Encode a binary time-by-channel raster as a complete SSCF packet.

        Parameters
        ----------
        spikes : ndarray
            A two-dimensional raster containing zeros and ones.

        Returns
        -------
        tuple of bytes and StreamingCompressionResult
            The encoded container and statistics over its original samples.

        Raises
        ------
        ValueError
            If the samples, dimensions or window size cannot be encoded.
        """
        window = _window(self.window_size)
        samples = _binary(_raster(spikes))
        rows, channels = samples.shape
        if self._engine is None:
            packet, spikes_count, active, max_size = _encode_container(samples, window)
        else:
            packet, spikes_count, active, max_size = self._engine.raster_stream_encode_raster(
                samples, window
            )
        frames = (rows + window - 1) // window
        bits = len(packet) * 8
        return packet, StreamingCompressionResult(
            original_bits=rows * channels,
            compressed_bits=bits,
            compression_ratio=rows * channels / bits,
            n_spikes=spikes_count,
            n_neurons=channels,
            n_timesteps=rows,
            lossless=True,
            window_size=window,
            n_frames=frames,
            mean_active_channels=active / frames if frames else 0.0,
            max_frame_bytes=max_size,
        )

    def decompress(self, data: bytes, T: int = 0, N: int = 0) -> _Raster:
        """Decode a complete SSCF packet or a prefix of its recorded rows.

        Parameters
        ----------
        data : bytes
            One SSCF packet including all frames.
        T : int
            Number of leading rows to return. Zero selects all recorded rows.
        N : int
            Expected channel count. Zero selects the recorded count.

        Returns
        -------
        ndarray
            The decoded time-by-channel int8 raster.

        Raises
        ------
        ValueError
            If dimensions are inconsistent, framing is incomplete, or data
            following the last frame or padding bits are nonempty.
        """
        rows, channels = _dimension(T), _dimension(N)
        if self._engine is None:
            return _decode_container(data, rows, channels)
        return self._engine.raster_stream_decode_raster(data, rows, channels)

    def compress_frame(self, window: np.ndarray[Any, Any]) -> bytes:
        """Encode one standalone binary frame with its actual dimensions.

        Parameters
        ----------
        window : ndarray
            A time-by-channel raster with at most 65535 rows and channels.

        Returns
        -------
        bytes
            One frame without a container header or final-window padding.
        """
        return self._pack_frame(_binary(_raster(window, 0xFFFF)))

    def _pack_frame(self, samples: _Raster) -> bytes:
        if self._engine is None:
            return _encode_frame(samples, samples.shape[0])[0]
        return self._engine.raster_stream_encode_frame(samples)

    def decompress_frame(self, frame: bytes) -> _Raster:
        """Decode one complete standalone frame.

        Parameters
        ----------
        frame : bytes
            An independent frame with its own channel count and row count.

        Returns
        -------
        ndarray
            An int8 raster at the frame's recorded dimensions.

        Raises
        ------
        ValueError
            If the frame is truncated or has trailing bytes or nonzero padding.
        """
        if self._engine is None:
            return _decode_frame(frame)
        return self._engine.raster_stream_decode_frame(frame)

    def iter_compress(self, chunks: Iterable[np.ndarray[Any, Any]]) -> Generator[bytes, None, None]:
        """Emit independent frames as the consumer requests them.

        Parameters
        ----------
        chunks : iterable of ndarray
            Binary time-by-channel chunks with a constant channel count.

        Yields
        ------
        bytes
            A complete standalone frame. The last frame has its actual size.

        Notes
        -----
        A single window of samples is buffered in addition to the chunk supplied
        by the caller. Closing the generator discards its unfinished window and
        stops pulling input. A new iterator starts a new stream.
        """
        size = _window(self.window_size)
        filled = 0
        buffer = None
        for chunk in chunks:
            raster = _raster(chunk)
            rows, channels = raster.shape
            if buffer is None:
                buffer = np.empty((size, channels), dtype=np.int8)
            if channels != buffer.shape[1]:
                raise ValueError("raster stream channel count changed")
            start = 0
            while start < rows:
                take = min(size - filled, rows - start)
                buffer[filled : filled + take] = _binary(raster[start : start + take])
                filled += take
                start += take
                if filled == size:
                    yield self._pack_frame(buffer)
                    filled = 0
        if buffer is not None and filled:
            yield self._pack_frame(buffer[:filled])

    def iter_decompress(self, frames: Iterable[bytes]) -> Generator[_Raster, None, None]:
        """Decode one independent frame per downstream request.

        Parameters
        ----------
        frames : iterable of bytes
            Complete frames with a constant channel count and at most
            the configured window size. Only the last frame may be short.

        Yields
        ------
        ndarray
            One decoded frame. No padding is added to the last frame.

        Notes
        -----
        Closing the iterator stops reading input. On a malformed frame the
        iterator raises and closes. Previously yielded frames remain usable.
        Create a new iterator at the next known frame boundary to recover.
        """
        size = _window(self.window_size)
        channels: int | None = None
        short = False
        for packet in frames:
            if short:
                raise ValueError("raster frame followed a short final frame")
            rows, count = _frame_shape(memoryview(packet))
            if rows > size or (channels is not None and count != channels):
                raise ValueError("raster stream frame dimensions changed")
            channels = count
            short = rows < size
            yield self.decompress_frame(packet)
