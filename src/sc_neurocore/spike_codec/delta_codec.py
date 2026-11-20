# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Inter-channel delta compression

"""Encode correlated binary raster channels as group-local XOR residuals."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from importlib import import_module
from operator import index
from typing import Any, Protocol, cast

import numpy as np

from .codec import CompressionResult, SpikeCodec, _ENGINE, _dimension, _installed_native_extension

_HEADER = struct.Struct("!4sHH")
_MAGIC = b"DSCX"
_NATIVE_SYMBOLS = ("delta_encode_raster", "delta_decode_raster")


class _NativeDelta(Protocol):
    def delta_encode_raster(
        self, raster: np.ndarray[Any, Any], group_size: int, precision: int, entropy: str
    ) -> tuple[bytes, int, int]: ...

    def delta_decode_raster(
        self, packet: bytes, rows: int, channels: int, precision: int
    ) -> np.ndarray[Any, Any]: ...


def _native(backend: str) -> _NativeDelta | None:
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
        return cast(_NativeDelta, module)
    if backend == "rust":
        raise ValueError("backend 'rust' needs sc_neurocore_engine with the DSCX codec")
    return None


def _group_size(value: int) -> int:
    size = index(value)
    if not 1 <= size <= 256:
        raise ValueError("group_size must be in 1..256 for uint8 reference indices")
    return size


def _group_count(channels: int, size: int) -> int:
    count = (channels + size - 1) // size
    if count > 0xFFFF:
        raise ValueError("DSCX group count exceeds uint16")
    return count


def _parse_packet(packet: bytes, channels: int) -> tuple[int, bytes, bytes]:
    if packet[:4] != _MAGIC:
        raise ValueError("Invalid header magic for DSCX")
    if len(packet) < _HEADER.size:
        raise ValueError("truncated DSCX header")
    _, size, count = _HEADER.unpack_from(packet)
    size = _group_size(size)
    if count != _group_count(channels, size):
        raise ValueError("DSCX group count does not match the channel count")
    end = _HEADER.size + count
    if len(packet) < end:
        raise ValueError("truncated DSCX references")
    references = packet[_HEADER.size : end]
    for start, reference in zip(range(0, channels, size), references):
        if reference >= min(size, channels - start):
            raise ValueError("DSCX reference index is outside its group")
    return size, references, packet[end:]


@dataclass
class DeltaCompressionResult(CompressionResult):
    """Report group geometry and mean non-reference residual sparsity."""

    n_groups: int = 0
    group_size: int = 0
    mean_delta_sparsity: float = 0.0
    codec_type: str = "delta"


class DeltaSpikeCodec:
    """Compress spatially correlated channels as inter-channel XOR residuals.

    Parameters
    ----------
    group_size : int
        Channels per group, from 1 to 256. Each DSCX reference index is
        an unsigned byte relative to its group.
    mode : str
        'lossless' preserves the input. 'lossy' quantises residual timing.
    timing_precision : int
        Input samples per complete bin in lossy mode.
    backend : str
        'python', 'rust', or 'auto'. Auto uses the complete native codec
        when available and falls back to Python on older wheels.

    Notes
    -----
    Each group keeps its most active channel, choosing the first on ties.
    The remaining channels carry XOR residuals. The embedded ISI body retains
    the original shape. Lossy decoding requires the same timing precision.
    """

    HEADER_MAGIC = _MAGIC

    def __init__(
        self,
        group_size: int = 8,
        mode: str = "lossless",
        timing_precision: int = 1,
        *,
        backend: str = "auto",
    ) -> None:
        self.group_size = _group_size(group_size)
        self.base_codec = SpikeCodec(mode=mode, timing_precision=timing_precision, backend="python")
        self._engine = _native(backend)

    @property
    def backend(self) -> str:
        """Return the selected complete implementation, 'python' or 'rust'."""
        return "rust" if self._engine is not None else "python"

    def _encode_python(
        self, raster: np.ndarray[Any, Any], size: int, groups: int
    ) -> tuple[bytes, int, int]:
        _, channels = raster.shape
        residual = np.empty_like(raster)
        references = bytearray()
        reference_spikes = 0
        for start in range(0, channels, size):
            group = raster[:, start : start + size]
            counts = group.sum(axis=0)
            reference = int(counts.argmax())
            references.append(reference)
            reference_spikes += int(counts[reference])
            output = residual[:, start : start + size]
            np.bitwise_xor(group, group[:, reference : reference + 1], out=output)
            output[:, reference] = group[:, reference]
        body, _ = self.base_codec.compress(residual)
        packet = _HEADER.pack(_MAGIC, size, groups) + references + body
        return packet, int(raster.sum()), int(residual.sum()) - reference_spikes

    def compress(self, spikes: np.ndarray[Any, Any]) -> tuple[bytes, DeltaCompressionResult]:
        """Encode a binary raster of shape (time, channels).

        Parameters
        ----------
        spikes : ndarray
            Binary time-by-channel input with zero and one values.

        Returns
        -------
        tuple of bytes and DeltaCompressionResult
            Complete DSCX packet and statistics for the original raster.

        Raises
        ------
        ValueError
            If the input is nonbinary or the shape or grouping cannot be encoded.
        """
        size = _group_size(self.group_size)
        raster = np.asarray(spikes)
        rows, channels = map(_dimension, raster.shape)
        groups = _group_count(channels, size)
        if self._engine is None or raster.dtype != np.int8:
            if raster.dtype != np.bool_ and not np.all((raster == 0) | (raster == 1)):
                raise ValueError("DSCX raster must contain only zeros and ones")
        raster = np.ascontiguousarray(raster, dtype=np.int8)
        if self._engine is None:
            packet, n_spikes, residual_count = self._encode_python(raster, size, groups)
        else:
            packet, n_spikes, residual_count = self._engine.delta_encode_raster(
                raster, size, self.base_codec._precision(rows), self.base_codec.entropy
            )
        residual_channels = channels - groups
        sparsity = 0.0
        if residual_channels:
            sparsity = 1.0 - residual_count / residual_channels / max(rows, 1)
        bits = len(packet) * 8
        return packet, DeltaCompressionResult(
            original_bits=rows * channels,
            compressed_bits=bits,
            compression_ratio=rows * channels / bits,
            n_spikes=n_spikes,
            n_neurons=channels,
            n_timesteps=rows,
            lossless=self.base_codec.mode == "lossless",
            n_groups=groups,
            group_size=size,
            mean_delta_sparsity=sparsity,
        )

    def decompress(self, data: bytes, T: int, N: int) -> np.ndarray[Any, Any]:
        """Decode a validated DSCX packet.

        Parameters
        ----------
        data : bytes
            Complete DSCX header and residual ISI packet.
        T, N : int
            Requested time and channel dimensions. With matching lossy
            precision, T may be the original length or the complete bin count.

        Returns
        -------
        ndarray
            Restored int8 raster of shape (T, N).

        Raises
        ------
        ValueError
            If the packet grammar, group geometry or output dimensions are invalid.
        """
        rows, channels = _dimension(T), _dimension(N)
        if self._engine is not None:
            return self._engine.delta_decode_raster(
                data, rows, channels, self.base_codec._precision(0xFFFFFFFF)
            )
        size, references, body = _parse_packet(data, channels)
        raster = self.base_codec.decompress(body, rows, channels)
        for start, reference in zip(range(0, channels, size), references):
            group = raster[:, start : start + size]
            reference_column = group[:, reference : reference + 1].copy()
            np.bitwise_xor(group, reference_column, out=group)
            group[:, reference : reference + 1] = reference_column
        return raster
