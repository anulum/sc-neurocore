# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Delta spike codec: inter-channel correlation compression

"""Delta spike compression: exploit spatial correlation between channels.

Architecture:
    1. Group channels by spatial proximity (configurable group_size)
    2. Within each group, pick reference channel (highest spike count)
    3. XOR all other channels against the reference
    4. ISI-compress: reference channels raw, delta channels as XOR residuals
    5. Header stores group layout for decoder

Target: neural probes (Neuropixels 384ch, Utah 96-128ch) where nearby
electrodes record overlapping populations. Spatial correlation makes
inter-channel XOR sparser than individual channels.

Also effective for any recording with population synchrony (bursts,
oscillations, up/down states).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from typing import Any
import numpy as np

from .codec import SpikeCodec, CompressionResult


@dataclass
class DeltaCompressionResult(CompressionResult):
    """Compression result with delta coding metrics."""

    n_groups: int = 0
    group_size: int = 0
    mean_delta_sparsity: float = 0.0
    codec_type: str = "delta"


class DeltaSpikeCodec:
    """Delta spike codec: compress inter-channel XOR residuals.

    Channels are grouped spatially. Within each group, one reference
    channel is transmitted raw; others are XOR'd against the reference
    and ISI-compressed. When channels are correlated, the XOR residuals
    are much sparser than the raw data.

    Parameters
    ----------
    group_size : int
        Channels per group. Larger groups = more sharing but weaker
        correlation with distant channels. 4-16 typical for probes.
    mode : str
        'lossless' or 'lossy' for the underlying ISI codec.
    timing_precision : int
        For lossy mode: quantize timing resolution.
    """

    HEADER_MAGIC = b"DSCX"  # Delta Spike Codec XOR

    def __init__(
        self,
        group_size: int = 8,
        mode: str = "lossless",
        timing_precision: int = 1,
    ):
        self.group_size = group_size
        self.base_codec = SpikeCodec(mode=mode, timing_precision=timing_precision)

    def compress(self, spikes: np.ndarray[Any, Any]) -> tuple[bytes, DeltaCompressionResult]:
        """Compress spike raster using inter-channel delta coding.

        Parameters
        ----------
        spikes : ndarray of shape (T, N), binary (int8 or bool)

        Returns
        -------
        (compressed_bytes, DeltaCompressionResult)
        """
        spikes = np.asarray(spikes, dtype=np.int8)
        T, N = spikes.shape
        original_bits = T * N

        n_groups = (N + self.group_size - 1) // self.group_size

        # Build delta matrix: replace non-reference channels with XOR residuals
        delta_matrix = np.empty_like(spikes)
        ref_indices = np.empty(n_groups, dtype=np.int32)
        delta_spike_counts = []

        for g in range(n_groups):
            start = g * self.group_size
            end = min(start + self.group_size, N)
            group = spikes[:, start:end]

            # Reference = channel with most spikes (best predictor for group)
            spike_counts = group.sum(axis=0)
            ref_local = int(np.argmax(spike_counts))
            ref_indices[g] = ref_local

            ref_channel = group[:, ref_local]
            for c in range(end - start):
                if c == ref_local:
                    delta_matrix[:, start + c] = group[:, c]
                else:
                    delta = group[:, c] ^ ref_channel
                    delta_matrix[:, start + c] = delta
                    delta_spike_counts.append(int(delta.sum()))

        # ISI-compress the delta matrix
        delta_data, _ = self.base_codec.compress(delta_matrix)

        # Header: magic(4) + group_size(2) + n_groups(2) + ref_indices(n_groups bytes)
        header = self.HEADER_MAGIC
        header += struct.pack("!HH", self.group_size, n_groups)
        header += ref_indices.astype(np.uint8).tobytes()
        encoded = header + delta_data

        compressed_bits = len(encoded) * 8
        ratio = original_bits / max(compressed_bits, 1)
        n_spikes = int(np.sum(spikes))

        mean_delta_sparsity = 0.0
        if delta_spike_counts:
            raw_per_channel = n_spikes / max(N, 1)
            mean_delta = np.mean(delta_spike_counts)
            mean_delta_sparsity = 1.0 - (mean_delta / max(T, 1))  # type: ignore[assignment]

        return encoded, DeltaCompressionResult(
            original_bits=original_bits,
            compressed_bits=compressed_bits,
            compression_ratio=ratio,
            n_spikes=n_spikes,
            n_neurons=N,
            n_timesteps=T,
            lossless=self.base_codec.mode == "lossless",
            n_groups=n_groups,
            group_size=self.group_size,
            mean_delta_sparsity=mean_delta_sparsity,
            codec_type="delta",
        )

    def decompress(self, data: bytes, T: int, N: int) -> np.ndarray[Any, Any]:
        """Decompress delta-coded spike raster.

        Parameters
        ----------
        data : bytes
        T, N : int
            Original dimensions.

        Returns
        -------
        ndarray of shape (T, N), int8
        """
        magic = data[:4]
        if magic != self.HEADER_MAGIC:
            raise ValueError(f"Invalid header magic: {magic!r}, expected {self.HEADER_MAGIC!r}")

        group_size, n_groups = struct.unpack("!HH", data[4:8])
        ref_indices = np.frombuffer(data[8 : 8 + n_groups], dtype=np.uint8).astype(np.int32)
        delta_data = data[8 + n_groups :]

        delta_matrix = self.base_codec.decompress(delta_data, T, N)

        spikes = np.empty_like(delta_matrix)
        for g in range(n_groups):
            start = g * group_size
            end = min(start + group_size, N)
            ref_local = int(ref_indices[g])

            ref_channel = delta_matrix[:, start + ref_local]
            for c in range(end - start):
                if c == ref_local:
                    spikes[:, start + c] = delta_matrix[:, start + c]
                else:
                    spikes[:, start + c] = delta_matrix[:, start + c] ^ ref_channel

        return spikes
