# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike train compression codec

"""ISI spike train compression with configurable entropy backend.

Per-neuron inter-spike interval encoding. Two backends:
  'varint' (default): LEB128 variable-length integers. Simple, fast.
  'huffman': Adaptive Huffman coding on ISI distribution. 30-60%
             smaller than varint on medium-to-dense data because
             frequent short ISIs get 2-4 bit codes.

For better compression on structured data, see PredictiveSpikeCodec
(temporal prediction), DeltaSpikeCodec (inter-channel correlation),
or AERSpikeCodec (event encoding).
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any
import numpy as np

from .entropy import HuffmanEncoder


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
    """ISI spike train codec with configurable entropy backend.

    Compression strategy:
    1. Extract per-neuron spike times from binary raster
    2. Compute inter-spike intervals (ISIs) per neuron
    3. Encode ISIs with chosen backend:
       'varint': LEB128 variable-length integers (fast, simple)
       'huffman': Adaptive Huffman (30-60% smaller on dense data)

    Each neuron is encoded independently. No inter-channel modeling.
    For inter-channel compression, use DeltaSpikeCodec.

    Parameters
    ----------
    mode : str
        'lossless' (exact reconstruction) or 'lossy' (preserve rates only).
    timing_precision : int
        For lossy mode: quantize spike times to this resolution.
    entropy : str
        'varint' (default) or 'huffman'.
    """

    def __init__(self, mode: str = "lossless", timing_precision: int = 1, entropy: str = "auto"):
        self.mode = mode
        self.timing_precision = timing_precision
        self.entropy = entropy
        self._huffman = HuffmanEncoder()

    def compress(self, spikes: np.ndarray[Any, Any]) -> tuple[bytes, CompressionResult]:
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

    def decompress(self, data: bytes, T: int, N: int) -> np.ndarray[Any, Any]:
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

    def _quantize_timing(self, spikes: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        if self.timing_precision <= 1:  # pragma: no cover
            return spikes
        T, N = spikes.shape
        new_T = T // self.timing_precision
        quantized = np.zeros((new_T, N), dtype=np.int8)
        for i in range(new_T):
            block = spikes[i * self.timing_precision : (i + 1) * self.timing_precision]
            quantized[i] = (block.sum(axis=0) > 0).astype(np.int8)
        return quantized

    def _pick_entropy(self, n_spikes: int, total_bins: int) -> str:
        """Auto-select entropy backend based on data density."""
        if self.entropy in ("varint", "huffman"):
            return self.entropy
        # auto: huffman for dense data (>3% spikes), varint for sparse
        density = n_spikes / max(total_bins, 1)
        return "huffman" if density > 0.03 else "varint"

    def _encode_events(self, events: list[np.ndarray[Any, Any]], T: int, N: int) -> bytes:
        """Encode spike events using ISI + auto-selected entropy backend."""
        n_spikes = sum(len(e) for e in events)
        backend = self._pick_entropy(n_spikes, T * N)
        if backend == "huffman":
            return self._encode_events_huffman(events, T, N)

        parts = []
        # Header: T, N as 4-byte big-endian + entropy flag
        parts.append(T.to_bytes(4, "big"))
        parts.append(N.to_bytes(4, "big"))

        for times in events:
            n_spikes = len(times)
            parts.append(self._encode_varint(n_spikes))

            if n_spikes == 0:
                continue

            parts.append(self._encode_varint(int(times[0])))

            for i in range(1, n_spikes):
                isi = int(times[i] - times[i - 1])
                parts.append(self._encode_varint(isi))

        return b"".join(parts)

    def _encode_events_huffman(self, events: list[np.ndarray[Any, Any]], T: int, N: int) -> bytes:
        """Encode events using Huffman-coded ISIs."""
        # Collect all ISI values first (for building Huffman table)
        all_isis = []
        spike_counts = []
        first_times = []

        for times in events:
            n_spikes = len(times)
            spike_counts.append(n_spikes)
            if n_spikes == 0:
                continue
            first_times.append(int(times[0]))
            for i in range(1, n_spikes):
                all_isis.append(int(times[i] - times[i - 1]))

        # Header: magic(1) + T(4) + N(4)
        header = b"\x01"  # entropy=huffman flag
        header += T.to_bytes(4, "big") + N.to_bytes(4, "big")

        # Spike counts + first times as varint (small overhead)
        count_parts = []
        for n_spikes in spike_counts:
            count_parts.append(self._encode_varint(n_spikes))
        first_parts = []
        for ft in first_times:
            first_parts.append(self._encode_varint(ft))

        count_data = b"".join(count_parts)
        first_data = b"".join(first_parts)

        # Huffman-encode all ISIs as one stream
        assert self._huffman is not None
        huff_data = self._huffman.encode(all_isis)

        # Pack: header + count_data_len(4) + count_data + first_data_len(4) + first_data + huff_data
        import struct

        return (
            header
            + struct.pack("!I", len(count_data))
            + count_data
            + struct.pack("!I", len(first_data))
            + first_data
            + huff_data
        )

    def _decode_events(self, data: bytes, N: int) -> list[np.ndarray[Any, Any]]:
        """Decode ISI-encoded spike events (auto-detects entropy backend)."""
        if data[0:1] == b"\x01":
            return self._decode_events_huffman(data, N)

        pos = 0
        pos += 8  # skip header (T, N)

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

    def _decode_events_huffman(self, data: bytes, N: int) -> list[np.ndarray[Any, Any]]:
        """Decode Huffman-coded ISI events."""
        import struct

        pos = 1  # skip magic byte
        pos += 8  # skip T, N (already known from outer header)

        # Read spike counts
        count_len = struct.unpack("!I", data[pos : pos + 4])[0]
        pos += 4
        count_data = data[pos : pos + count_len]
        pos += count_len

        spike_counts = []
        cpos = 0
        for _ in range(N):
            n, cpos = self._decode_varint(count_data, cpos)
            spike_counts.append(n)

        # Read first times
        first_len = struct.unpack("!I", data[pos : pos + 4])[0]
        pos += 4
        first_data = data[pos : pos + first_len]
        pos += first_len

        first_times = []
        fpos = 0
        for sc in spike_counts:
            if sc > 0:
                ft, fpos = self._decode_varint(first_data, fpos)
                first_times.append(ft)

        # Decode Huffman ISIs
        total_isis = sum(max(0, sc - 1) for sc in spike_counts)
        huff = HuffmanEncoder()
        isis, _ = huff.decode(data[pos:], total_isis)

        # Reconstruct events
        events = []
        isi_idx = 0
        ft_idx = 0
        for sc in spike_counts:
            if sc == 0:
                events.append(np.array([], dtype=np.int64))
                continue
            times = np.zeros(sc, dtype=np.int64)
            times[0] = first_times[ft_idx]
            ft_idx += 1
            for i in range(1, sc):
                times[i] = times[i - 1] + isis[isi_idx]
                isi_idx += 1
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
