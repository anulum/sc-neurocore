# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike train compression codec

"""Inter-spike interval compression of binary rasters."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.machinery import PathFinder
from operator import index
from pkgutil import extend_path
from typing import Any, Protocol, cast

import numpy as np

from .isi_packet import decode_packet, encode_packet, encode_varint

_ENGINE_PACKAGE = "sc_neurocore_engine"
_ENGINE = f"{_ENGINE_PACKAGE}.sc_neurocore_engine"
_NATIVE_SYMBOLS = ("isi_encode_raster", "isi_decode_raster")
_ENTROPY_MODES = ("auto", "varint", "huffman")


class _NativeIsi(Protocol):
    def isi_encode_raster(
        self, raster: np.ndarray[Any, Any], precision: int, entropy: str
    ) -> tuple[bytes, int]: ...
    def isi_decode_raster(
        self, packet: bytes, rows: int, channels: int, precision: int
    ) -> np.ndarray[Any, Any]: ...


def _installed_native_extension() -> bool:
    # Inspect the package path without executing a failed source-tree bridge again.
    locations: list[str] = extend_path([], _ENGINE_PACKAGE)
    return PathFinder.find_spec(_ENGINE, locations) is not None


def _native(backend: str) -> _NativeIsi | None:
    if backend not in ("auto", "python", "rust"):
        raise ValueError("backend must be auto, python or rust")
    if backend == "python":
        return None
    try:
        engine = import_module(_ENGINE)
    except ImportError:
        if _installed_native_extension():
            raise
        engine = None
    if engine is not None and all(
        callable(getattr(engine, name, None)) for name in _NATIVE_SYMBOLS
    ):
        return cast(_NativeIsi, engine)
    if backend == "rust":
        raise ValueError("backend 'rust' needs sc_neurocore_engine with the ISI codec")
    return None


def _dimension(value: int) -> int:
    value = index(value)
    if not 0 <= value < 1 << 32:
        raise ValueError("ISI dimensions must fit uint32")
    return value


@dataclass
class CompressionResult:
    """Result of spike train compression."""

    original_bits: int
    compressed_bits: int
    compression_ratio: float
    n_spikes: int
    n_neurons: int
    n_timesteps: int
    lossless: bool

    def summary(self) -> str:
        """Return a human-readable one-line summary of the codec statistics."""
        mode = "lossless" if self.lossless else "lossy"
        return (
            f"SpikeCodec ({mode}): {self.compression_ratio:.1f}x compression, "
            f"{self.original_bits} -> {self.compressed_bits} bits, "
            f"{self.n_spikes} spikes across {self.n_neurons} neurons x {self.n_timesteps} steps"
        )


class SpikeCodec:
    """Compress binary raster columns as inter-spike intervals.

    Parameters
    ----------
    mode : str
        'lossless' preserves the input. 'lossy' allows timing quantisation.
    timing_precision : int
        Number of input samples in a complete quantised bin for lossy mode.
        A final partial bin is dropped.
    entropy : str
        'varint', 'huffman', or 'auto'. Auto uses Huffman above 3% encoded
        event density, measured against the original input shape.
    backend : str
        'python', 'rust', or 'auto'. Auto uses the Rust codec when available.
        Native codec errors propagate to the caller.

    Notes
    -----
    Varint requires fewer than 2**24 rows to avoid the Huffman discriminator.
    Quantised packets retain the original dimensions and store bin indices.
    """

    def __init__(
        self,
        mode: str = "lossless",
        timing_precision: int = 1,
        entropy: str = "auto",
        *,
        backend: str = "auto",
    ) -> None:
        if mode not in ("lossless", "lossy"):
            raise ValueError("mode must be lossless or lossy")
        if entropy not in _ENTROPY_MODES:
            raise ValueError("entropy must be auto, varint or huffman")
        if index(timing_precision) < 1:
            raise ValueError("timing_precision must be positive")
        self.mode = mode
        self.timing_precision = timing_precision
        self.entropy = entropy
        self._engine = _native(backend)

    @property
    def backend(self) -> str:
        """Return the selected implementation, 'python' or 'rust'."""
        return "rust" if self._engine is not None else "python"

    def _precision(self, rows: int) -> int:
        if self.mode != "lossy":
            return 1
        precision = index(self.timing_precision)
        if precision < 1:
            raise ValueError("timing_precision must be positive")
        # Larger precisions have the same zero-bin geometry and need not cross FFI.
        return min(precision, rows + 1)

    def _pick_entropy(self, n_spikes: int, total_bins: int) -> str:
        """Choose the wire format for an encoded event density."""
        if self.entropy != "auto":
            return self.entropy
        return "huffman" if n_spikes / max(total_bins, 1) > 0.03 else "varint"

    def compress(self, spikes: np.ndarray[Any, Any]) -> tuple[bytes, CompressionResult]:
        """Compress a binary bool or int8 raster of shape (time, channels).

        Returns
        -------
        tuple of bytes and CompressionResult
            Wire packet and statistics for the original shape and encoded events.
        """
        rows, channels = map(_dimension, spikes.shape)
        precision = self._precision(rows)
        if self.entropy not in _ENTROPY_MODES:
            raise ValueError("entropy must be auto, varint or huffman")
        if self._engine is not None:
            raster = np.ascontiguousarray(
                spikes if spikes.dtype == np.int8 else spikes > 0, dtype=np.int8
            )
            packet, event_count = self._engine.isi_encode_raster(raster, precision, self.entropy)
        else:
            active = spikes > 0
            if precision > 1:
                bins = rows // precision
                active = active[: bins * precision].reshape(bins, precision, channels).any(axis=1)
            events = [np.flatnonzero(column).tolist() for column in active.T]
            event_count = sum(map(len, events))
            packet = encode_packet(
                events, rows, channels, self._pick_entropy(event_count, rows * channels)
            )
        bits = len(packet) * 8
        return packet, CompressionResult(
            original_bits=rows * channels,
            compressed_bits=bits,
            compression_ratio=rows * channels / bits,
            n_spikes=event_count,
            n_neurons=channels,
            n_timesteps=rows,
            lossless=self.mode == "lossless",
        )

    def decompress(self, data: bytes, T: int, N: int) -> np.ndarray[Any, Any]:
        """Decode a complete packet to an int8 raster at the requested dimensions.

        N must match the header. T may be the original row count or, with the
        matching lossy timing configuration, the number of complete bins.
        Events are at bin indices in either layout.
        """
        rows, channels = _dimension(T), _dimension(N)
        # The cap preserves all plausible wire precisions, including for compact output.
        precision = self._precision((1 << 32) - 1)
        if self._engine is not None:
            return self._engine.isi_decode_raster(data, rows, channels, precision)
        return decode_packet(data, rows, channels, precision)

    @staticmethod
    def _encode_varint(value: int) -> bytes:
        """Encode unsigned LEB128 for the waveform section writers."""
        return encode_varint(value)

    @staticmethod
    def _decode_varint(data: bytes, pos: int) -> tuple[int, int]:
        """Decode variable-length integer, return (value, new_position)."""
        value = 0
        shift = 0
        while pos < len(data):
            byte = data[pos]
            pos += 1
            value |= (byte & 0x7F) << shift
            if not (byte & 0x80):
                break
            shift += 7
        return value, pos
