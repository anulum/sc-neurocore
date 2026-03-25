# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — AER spike codec: address-event representation encoding

"""AER spike compression: event-based encoding for neuromorphic systems.

Architecture:
    1. Scan spike raster for active (timestamp, neuron_id) pairs
    2. Encode as compact AER event stream: (timestamp, neuron_id) tuples
    3. Timestamp delta encoding: store differences between consecutive events
    4. Neuron ID encoding: variable-width based on N

Compatible with the AER-over-UDP protocol in comm/aer_udp.py.
Uses the same event semantics but without socket overhead.

Target: neuromorphic chip-to-chip communication (Loihi, SpiNNaker,
BrainScaleS), where event-based encoding is the native data format.
Also natural for event cameras (DVS) and event-driven simulators.

Compression depends on sparsity: O(n_spikes) bytes vs O(T*N) raw.
For typical cortical firing rates (0.5-5 Hz at 20 kHz), this gives
100-1000x compression without any information loss.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

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

    def __init__(self, timestamp_bits: int = 16, neuron_bits: int = 0):
        self.timestamp_bits = timestamp_bits
        self.neuron_bits = neuron_bits

    def compress(self, spikes: np.ndarray) -> tuple[bytes, AERCompressionResult]:
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

        # Extract events as (timestamp, neuron_id) sorted by time then neuron
        times, neurons = np.nonzero(spikes)
        # Already sorted by time (row-major), then by neuron within same time
        n_events = len(times)

        neuron_bits = self.neuron_bits if self.neuron_bits > 0 else max(1, int(np.ceil(np.log2(max(N, 2)))))
        neuron_bytes = (neuron_bits + 7) // 8

        # Header: magic(4) + T(4) + N(4) + n_events(4) + neuron_bits(1) = 17 bytes
        header = self.HEADER_MAGIC + struct.pack("!IIIB", T, N, n_events, neuron_bits)

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
            n_spikes=n_events,
            n_neurons=N,
            n_timesteps=T,
            lossless=True,
            n_events=n_events,
            bytes_per_event=bpe,
            codec_type="aer",
        )

    def decompress(self, data: bytes, T: int = 0, N: int = 0) -> np.ndarray:
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
        if magic != self.HEADER_MAGIC:
            raise ValueError(f"Invalid header magic: {magic!r}, expected {self.HEADER_MAGIC!r}")

        T_stored, N_stored, n_events, neuron_bits = struct.unpack("!IIIB", data[4:17])
        if T == 0:
            T = T_stored
        if N == 0:
            N = N_stored

        neuron_bytes = (neuron_bits + 7) // 8
        ts_max = (1 << self.timestamp_bits) - 1
        escape_marker = b"\xff" * neuron_bytes

        spikes = np.zeros((T, N), dtype=np.int8)
        offset = 17
        current_t = 0
        events_read = 0

        while events_read < n_events and offset + 2 + neuron_bytes <= len(data):
            dt = struct.unpack("!H", data[offset : offset + 2])[0]
            nid_bytes = data[offset + 2 : offset + 2 + neuron_bytes]
            offset += 2 + neuron_bytes

            if nid_bytes == escape_marker:
                # Escape: just advance time, no event
                current_t += dt
                continue

            current_t += dt
            nid = int.from_bytes(nid_bytes, "big")

            if 0 <= current_t < T and 0 <= nid < N:
                spikes[current_t, nid] = 1
            events_read += 1

        return spikes
