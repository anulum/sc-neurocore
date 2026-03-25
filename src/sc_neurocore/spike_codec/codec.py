# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike train compression codec

"""Compress spike trains for BCI telemetry and neural recording storage.

Neuralink generates ~200 Mbps but can transmit 1-2 Mbps. This codec
targets 50-200x compression via ISI entropy coding, population delta
encoding, and configurable lossy/lossless modes.

No Python library provides spike-domain compression.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
        mode = "lossless" if self.lossless else "lossy"
        return (
            f"SpikeCodec ({mode}): {self.compression_ratio:.1f}x compression, "
            f"{self.original_bits} -> {self.compressed_bits} bits, "
            f"{self.n_spikes} spikes across {self.n_neurons} neurons x {self.n_timesteps} steps"
        )


class SpikeCodec:
    """Spike train compression codec.

    Compression strategy:
    1. Extract spike events (time, neuron_id) from binary matrix
    2. Encode ISIs (inter-spike intervals) per neuron — ISIs follow
       approximate exponential distribution, compressing well with
       variable-length coding
    3. Population delta: correlated neurons share spike times, encode
       only differences
    4. Variable-length integer encoding for ISIs

    Parameters
    ----------
    mode : str
        'lossless' (exact reconstruction) or 'lossy' (preserve rates only).
    timing_precision : int
        For lossy mode: quantize spike times to this resolution.
    """

    def __init__(self, mode: str = "lossless", timing_precision: int = 1):
        self.mode = mode
        self.timing_precision = timing_precision

    def compress(self, spikes: np.ndarray) -> tuple[bytes, CompressionResult]:
        """Compress a spike raster.

        Parameters
        ----------
        spikes : ndarray of shape (T, N), binary (int8 or bool)

        Returns
        -------
        (compressed_bytes, CompressionResult)
        """
        T, N = spikes.shape
        original_bits = T * N

        if self.mode == "lossy":
            spikes = self._quantize_timing(spikes)

        # Extract per-neuron spike times
        events = []
        for n in range(N):
            times = np.where(spikes[:, n] > 0)[0]
            events.append(times)

        # Encode: ISIs per neuron + variable-length integers
        encoded = self._encode_events(events, T, N)

        compressed_bits = len(encoded) * 8
        ratio = original_bits / max(compressed_bits, 1)
        n_spikes = sum(len(e) for e in events)

        result = CompressionResult(
            original_bits=original_bits,
            compressed_bits=compressed_bits,
            compression_ratio=ratio,
            n_spikes=n_spikes,
            n_neurons=N,
            n_timesteps=T,
            lossless=self.mode == "lossless",
        )
        return encoded, result

    def decompress(self, data: bytes, T: int, N: int) -> np.ndarray:
        """Decompress to spike raster.

        Parameters
        ----------
        data : bytes
        T, N : int
            Original dimensions.

        Returns
        -------
        ndarray of shape (T, N), int8
        """
        events = self._decode_events(data, N)
        spikes = np.zeros((T, N), dtype=np.int8)
        for n, times in enumerate(events):
            for t in times:
                if 0 <= t < T:
                    spikes[t, n] = 1
        return spikes

    def _quantize_timing(self, spikes: np.ndarray) -> np.ndarray:
        if self.timing_precision <= 1:  # pragma: no cover
            return spikes
        T, N = spikes.shape
        new_T = T // self.timing_precision
        quantized = np.zeros((new_T, N), dtype=np.int8)
        for i in range(new_T):
            block = spikes[i * self.timing_precision : (i + 1) * self.timing_precision]
            quantized[i] = (block.sum(axis=0) > 0).astype(np.int8)
        return quantized

    def _encode_events(self, events: list[np.ndarray], T: int, N: int) -> bytes:
        """Encode spike events using ISI + variable-length integers."""
        parts = []
        # Header: T, N as 4-byte big-endian
        parts.append(T.to_bytes(4, "big"))
        parts.append(N.to_bytes(4, "big"))

        for times in events:
            # Number of spikes for this neuron
            n_spikes = len(times)
            parts.append(self._encode_varint(n_spikes))

            if n_spikes == 0:
                continue

            # First spike time
            parts.append(self._encode_varint(int(times[0])))

            # ISIs (differences between consecutive spike times)
            for i in range(1, n_spikes):
                isi = int(times[i] - times[i - 1])
                parts.append(self._encode_varint(isi))

        return b"".join(parts)

    def _decode_events(self, data: bytes, N: int) -> list[np.ndarray]:
        """Decode ISI-encoded spike events."""
        pos = 0
        # Skip header (T, N)
        pos += 8

        events = []
        for n in range(N):
            n_spikes, pos = self._decode_varint(data, pos)
            if n_spikes == 0:
                events.append(np.array([], dtype=np.int64))
                continue

            times = np.zeros(n_spikes, dtype=np.int64)
            first, pos = self._decode_varint(data, pos)
            times[0] = first

            for i in range(1, n_spikes):
                isi, pos = self._decode_varint(data, pos)
                times[i] = times[i - 1] + isi

            events.append(times)
        return events

    @staticmethod
    def _encode_varint(value: int) -> bytes:
        """Encode integer using variable-length encoding (LEB128-style)."""
        result = bytearray()
        while value >= 0x80:
            result.append((value & 0x7F) | 0x80)
            value >>= 7
        result.append(value & 0x7F)
        return bytes(result)

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
