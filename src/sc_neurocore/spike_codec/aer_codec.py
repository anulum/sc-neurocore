# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — AER spike codec: address-event representation encoding

"""Losslessly encode binary spike rasters as address or silence events."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from operator import index
from typing import Any, Protocol, cast

import numpy as np

from .aer_packet import _EVENTS, _SILENCES, _Raster, _decode, _encode
from .codec import CompressionResult, _ENGINE, _dimension, _installed_native_extension

_NATIVE_SYMBOLS = ("address_event_encode_raster", "address_event_decode_raster")


class _NativeEvents(Protocol):
    def address_event_encode_raster(
        self, samples: _Raster, timestamp_bits: int, neuron_bits: int
    ) -> tuple[bytes, int, int]: ...
    def address_event_decode_raster(self, data: bytes, rows: int, channels: int) -> _Raster: ...


def _native(backend: str) -> _NativeEvents | None:
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
        return cast(_NativeEvents, module)
    if backend == "rust":
        raise ValueError("backend 'rust' needs sc_neurocore_engine with the address-event codec")
    return None


def _widths(timestamp_bits: int, neuron_bits: int) -> tuple[int, int]:
    timestamp, neuron = index(timestamp_bits), index(neuron_bits)
    if not 1 <= timestamp <= 16 or not 0 <= neuron <= 2040:
        raise ValueError("invalid AER timestamp or neuron bit width")
    return timestamp, neuron


@dataclass
class AERCompressionResult(CompressionResult):
    """Compression result with address-event counts and byte costs."""

    n_events: int = 0
    bytes_per_event: float = 0.0
    codec_type: str = "aer"


class AERSpikeCodec:
    """Encode a raster as ordered spike events, or silences above 50% density.

    Parameters
    ----------
    timestamp_bits : int
        Escape gap width in 1 through 16. The default emits an escape for
        gaps above 65535 rows. Wire delta slots always use two bytes.
    neuron_bits : int
        Minimum address width in 0 through 2040. Zero selects the minimum
        safe width. The encoder reserves the all-ones address for escapes.
    backend : str
        'python', 'rust', or 'auto'. Auto selects a complete installed
        native backend when available.

    Notes
    -----
    Each raster position contributes at most one event. Equal timestamps
    are ordered by increasing channel. The packets record sample-index
    deltas and carry neither an absolute start time nor a sample rate.
    """

    HEADER_MAGIC = _EVENTS
    HEADER_MAGIC_INV = _SILENCES

    def __init__(
        self, timestamp_bits: int = 16, neuron_bits: int = 0, *, backend: str = "auto"
    ) -> None:
        self.timestamp_bits, self.neuron_bits = _widths(timestamp_bits, neuron_bits)
        self._engine = _native(backend)

    @property
    def backend(self) -> str:
        """Return the selected complete implementation, 'python' or 'rust'."""
        return "rust" if self._engine is not None else "python"

    def compress(self, spikes: np.ndarray[Any, Any]) -> tuple[bytes, AERCompressionResult]:
        """Encode an entire binary time-by-channel raster.

        Parameters
        ----------
        spikes : ndarray
            Two-dimensional input with only zeros and ones.

        Returns
        -------
        tuple of bytes and AERCompressionResult
            Complete packet and statistics for the original raster.

        Raises
        ------
        ValueError
            If samples or dimensions cannot be represented.
        """
        timestamp, neuron = _widths(self.timestamp_bits, self.neuron_bits)
        raster = np.asarray(spikes)
        if raster.ndim != 2:
            raise ValueError("AER raster must have shape (time, channels)")
        rows, channels = map(_dimension, raster.shape)
        if self._engine is None or raster.dtype != np.int8:
            if raster.dtype != np.bool_ and not np.all((raster == 0) | (raster == 1)):
                raise ValueError("AER raster must contain only zeros and ones")
        samples = np.ascontiguousarray(raster, dtype=np.int8)
        if self._engine is None:
            packet, n_spikes, events = _encode(samples, timestamp, neuron)
        else:
            packet, n_spikes, events = self._engine.address_event_encode_raster(
                samples, timestamp, neuron
            )
        bits = len(packet) * 8
        return packet, AERCompressionResult(
            original_bits=rows * channels,
            compressed_bits=bits,
            compression_ratio=rows * channels / bits,
            n_spikes=n_spikes,
            n_neurons=channels,
            n_timesteps=rows,
            lossless=True,
            n_events=events,
            bytes_per_event=len(packet) / max(events, 1),
        )

    def decompress(self, data: bytes, T: int = 0, N: int = 0) -> _Raster:
        """Validate and decode an AERX or AERI raster packet.

        Parameters
        ----------
        data : bytes
            Complete container with all events and timestamp escapes.
        T, N : int
            Optional leading time and channel dimensions. Zero chooses the
            corresponding recorded dimension.

        Returns
        -------
        ndarray
            Time-by-channel int8 raster.

        Raises
        ------
        ValueError
            If a field, event, length or requested dimension is invalid.
        """
        rows, channels = _dimension(T), _dimension(N)
        if self._engine is None:
            return _decode(data, rows, channels)
        return self._engine.address_event_decode_raster(data, rows, channels)
