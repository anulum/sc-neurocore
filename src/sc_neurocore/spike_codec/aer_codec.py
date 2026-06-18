# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — AER spike codec: address-event representation encoding

"""AER spike compression with adaptive density handling.

Architecture:
    1. Measure spike density
    2. If density <= 50%: encode spike events (standard AER)
    3. If density > 50%: invert matrix, encode silence events
       (O(n_gaps) bytes when most channels are firing)
    4. Delta-code timestamps, variable-width neuron IDs

Compatible with the AER-over-UDP protocol in comm/aer_udp.py.

Target: neuromorphic chip-to-chip (Loihi, SpiNNaker, BrainScaleS),
event cameras (DVS), and event-driven simulators.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from typing import Any
import numpy as np

from .codec import CompressionResult


@dataclass
class AERCompressionResult(CompressionResult):
    """Compression result with AER codec metrics."""

    n_events: int = 0
    bytes_per_event: float = 0.0
    codec_type: str = "aer"


class AERSpikeCodec:
    """AER spike codec: event-list encoding for sparse spike data.

    Converts spike raster (T, N) to a compact stream of (timestamp,
    neuron_id) events. Delta-encodes timestamps for further compression.

    Parameters
    ----------
    timestamp_bits : int
        Bits for delta-coded timestamps. 16 = max gap of 65535 samples.
        Larger windows between spikes use escape codes.
    neuron_bits : int
        Bits for neuron ID. Auto-sized from N if 0.
    """

    HEADER_MAGIC = b"AERX"
    HEADER_MAGIC_INV = b"AERI"  # Inverted: encoding silences, not spikes

    def __init__(self, timestamp_bits: int = 16, neuron_bits: int = 0):
        self.timestamp_bits = timestamp_bits
        self.neuron_bits = neuron_bits

    def compress(self, spikes: np.ndarray[Any, Any]) -> tuple[bytes, AERCompressionResult]:
        """Compress spike raster to AER event stream.

        Parameters
        ----------
        spikes : ndarray of shape (T, N), binary

        Returns
        -------
        (compressed_bytes, AERCompressionResult)
        """
        spikes = np.asarray(spikes, dtype=np.int8)
        T, N = spikes.shape
        original_bits = T * N

        # Adaptive: if >50% density, invert (encode silences instead of spikes)
        n_ones = int(np.sum(spikes))
        density = n_ones / max(T * N, 1)
        inverted = density > 0.5
        encode_matrix = 1 - spikes if inverted else spikes

        # Extract events as (timestamp, neuron_id) sorted by time then neuron
        times, neurons = np.nonzero(encode_matrix)
        # Already sorted by time (row-major), then by neuron within same time
        n_events = len(times)

        neuron_bits = (
            self.neuron_bits if self.neuron_bits > 0 else max(1, int(np.ceil(np.log2(max(N, 2)))))
        )
        neuron_bytes = (neuron_bits + 7) // 8
        # Escape marker is all-1s bytes. If max valid ID (N-1) fills all
        # bits in neuron_bytes, bump size to avoid escape collision.
        while (1 << (neuron_bytes * 8)) - 1 <= (N - 1):
            neuron_bytes += 1

        # Header: magic(4) + T(4) + N(4) + n_events(4) + neuron_bytes(1) = 17 bytes
        magic = self.HEADER_MAGIC_INV if inverted else self.HEADER_MAGIC
        header = magic + struct.pack("!IIIB", T, N, n_events, neuron_bytes)

        if n_events == 0:
            encoded = header
        else:
            # Delta-encode timestamps
            parts = []
            prev_t = 0
            ts_max = (1 << self.timestamp_bits) - 1

            for i in range(n_events):
                t = int(times[i])
                nid = int(neurons[i])
                dt = t - prev_t

                # Emit escape codes for large gaps
                while dt > ts_max:
                    parts.append(struct.pack("!H", ts_max))
                    parts.append(b"\xff" * neuron_bytes)  # escape marker
                    dt -= ts_max

                parts.append(struct.pack("!H", dt))
                parts.append(nid.to_bytes(neuron_bytes, "big"))
                prev_t = t

            encoded = header + b"".join(parts)

        compressed_bits = len(encoded) * 8
        ratio = original_bits / max(compressed_bits, 1)
        bpe = len(encoded) / max(n_events, 1)

        return encoded, AERCompressionResult(
            original_bits=original_bits,
            compressed_bits=compressed_bits,
            compression_ratio=ratio,
            n_spikes=n_ones,
            n_neurons=N,
            n_timesteps=T,
            lossless=True,
            n_events=n_events,
            bytes_per_event=bpe,
            codec_type="aer",
        )

    def decompress(self, data: bytes, T: int = 0, N: int = 0) -> np.ndarray[Any, Any]:
        """Decompress AER event stream to spike raster.

        Parameters
        ----------
        data : bytes
        T, N : int (optional, read from header if 0)

        Returns
        -------
        ndarray of shape (T, N), int8
        """
        magic = data[:4]
        if magic not in (self.HEADER_MAGIC, self.HEADER_MAGIC_INV):
            raise ValueError(
                f"Invalid header magic: {magic!r}, expected {self.HEADER_MAGIC!r} or {self.HEADER_MAGIC_INV!r}"
            )
        inverted = magic == self.HEADER_MAGIC_INV

        T_stored, N_stored, n_events, neuron_bytes = struct.unpack("!IIIB", data[4:17])
        if T == 0:
            T = T_stored
        if N == 0:
            N = N_stored
        escape_marker = b"\xff" * neuron_bytes

        decoded = np.zeros((T, N), dtype=np.int8)
        offset = 17
        current_t = 0
        events_read = 0

        while events_read < n_events and offset + 2 + neuron_bytes <= len(data):
            dt = struct.unpack("!H", data[offset : offset + 2])[0]
            nid_bytes = data[offset + 2 : offset + 2 + neuron_bytes]
            offset += 2 + neuron_bytes

            if nid_bytes == escape_marker:
                current_t += dt
                continue

            current_t += dt
            nid = int.from_bytes(nid_bytes, "big")

            if 0 <= current_t < T and 0 <= nid < N:
                decoded[current_t, nid] = 1
            events_read += 1

        if inverted:
            return 1 - decoded
        return decoded
