# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Streaming spike codec: fixed-latency, causal

"""Streaming spike compression with bounded latency per window.

Architecture:
    1. Divide spike raster into fixed-size time windows (e.g. 20 samples = 1ms at 20kHz)
    2. Each window compressed independently as a frame
    3. Frames are self-contained: no dependency on past frames
    4. Bounded worst-case latency: window_size / sample_rate
    5. Frame format: frame_header + per-channel spike bitmask

Within each window, channels are packed as bitmasks (one bit per timestep).
For window_size=20, each channel needs 20 bits = 3 bytes. Silent channels
are run-length encoded (skip count). Active channels store raw bitmask.

Target: real-time BCI decoding where latency matters more than compression
ratio. Also suitable for online spike sorting and closed-loop experiments.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from typing import Any
import numpy as np

from .codec import CompressionResult


@dataclass
class StreamingCompressionResult(CompressionResult):
    """Compression result with streaming codec metrics."""

    window_size: int = 0
    n_frames: int = 0
    mean_active_channels: float = 0.0
    max_frame_bytes: int = 0
    codec_type: str = "streaming"


def _pack_window(window: np.ndarray[Any, Any]) -> bytes:
    """Pack a (W, N) spike window into compact frame bytes.

    Format per frame:
        n_channels: uint16
        window_size: uint16
        For each channel:
            If silent (no spikes): store nothing, mark in skip bitmap
            If active: store raw bitmask (ceil(W/8) bytes)

        skip_bitmap: ceil(N/8) bytes — 1 = silent, 0 = active
        active_bitmasks: concatenated, ceil(W/8) bytes each
    """
    W, N = window.shape
    bitmask_bytes = (W + 7) // 8

    skip_bits = bytearray((N + 7) // 8)
    active_data = bytearray()
    active_count = 0

    for ch in range(N):
        col = window[:, ch]
        if not np.any(col):
            # Mark as silent
            skip_bits[ch // 8] |= 1 << (ch % 8)
        else:
            active_count += 1
            # Pack spike times as bitmask
            packed = 0
            for t in range(W):
                if col[t]:
                    packed |= 1 << t
            active_data.extend(packed.to_bytes(bitmask_bytes, "little"))

    header = struct.pack("!HH", N, W)
    return header + bytes(skip_bits) + bytes(active_data)


def _unpack_window(frame: bytes, offset: int) -> tuple[np.ndarray[Any, Any], int]:
    """Unpack one frame. Returns (window, new_offset)."""
    N, W = struct.unpack("!HH", frame[offset : offset + 4])
    offset += 4

    skip_bytes = (N + 7) // 8
    bitmask_bytes = (W + 7) // 8

    skip_bits = frame[offset : offset + skip_bytes]
    offset += skip_bytes

    window = np.zeros((W, N), dtype=np.int8)

    for ch in range(N):
        is_silent = (skip_bits[ch // 8] >> (ch % 8)) & 1
        if is_silent:
            continue
        packed = int.from_bytes(frame[offset : offset + bitmask_bytes], "little")
        offset += bitmask_bytes
        for t in range(W):
            if (packed >> t) & 1:
                window[t, ch] = 1

    return window, offset


class StreamingSpikeCodec:
    """Streaming spike codec: fixed-latency, independently decodable frames.

    Each time window is compressed as a self-contained frame. No inter-frame
    dependencies. Worst-case latency = window_size samples.

    Parameters
    ----------
    window_size : int
        Samples per frame. 20 = 1ms at 20kHz (typical BCI).
        Smaller = lower latency but less compression.
    """

    HEADER_MAGIC = b"SSCF"  # Streaming Spike Codec Frames

    def __init__(self, window_size: int = 20):
        self.window_size = window_size

    def compress(self, spikes: np.ndarray[Any, Any]) -> tuple[bytes, StreamingCompressionResult]:
        """Compress spike raster into independently decodable frames.

        Parameters
        ----------
        spikes : ndarray of shape (T, N), binary

        Returns
        -------
        (compressed_bytes, StreamingCompressionResult)
        """
        spikes = np.asarray(spikes, dtype=np.int8)
        T, N = spikes.shape
        original_bits = T * N

        n_frames = (T + self.window_size - 1) // self.window_size
        frames = []
        active_counts = []
        max_frame_size = 0

        for i in range(n_frames):
            start = i * self.window_size
            end = min(start + self.window_size, T)
            window = spikes[start:end]

            # Pad last window if needed
            if window.shape[0] < self.window_size:
                pad = np.zeros((self.window_size - window.shape[0], N), dtype=np.int8)
                window = np.vstack([window, pad])

            frame = _pack_window(window)
            frames.append(frame)

            active = int(np.any(window, axis=0).sum())
            active_counts.append(active)
            if len(frame) > max_frame_size:
                max_frame_size = len(frame)

        # Global header: magic(4) + window_size(2) + T(4) + N(2) + n_frames(4)
        header = self.HEADER_MAGIC + struct.pack("!HIHI", self.window_size, T, N, n_frames)
        encoded = header + b"".join(frames)

        compressed_bits = len(encoded) * 8
        ratio = original_bits / max(compressed_bits, 1)

        return encoded, StreamingCompressionResult(
            original_bits=original_bits,
            compressed_bits=compressed_bits,
            compression_ratio=ratio,
            n_spikes=int(np.sum(spikes)),
            n_neurons=N,
            n_timesteps=T,
            lossless=True,
            window_size=self.window_size,
            n_frames=n_frames,
            mean_active_channels=float(np.mean(active_counts)) if active_counts else 0.0,
            max_frame_bytes=max_frame_size,
            codec_type="streaming",
        )

    def decompress(self, data: bytes, T: int = 0, N: int = 0) -> np.ndarray[Any, Any]:
        """Decompress streaming frames to spike raster.

        T and N are read from the header if not provided (or if 0).

        Parameters
        ----------
        data : bytes
        T, N : int (optional, read from header)

        Returns
        -------
        ndarray of shape (T, N), int8
        """
        magic = data[:4]
        if magic != self.HEADER_MAGIC:
            raise ValueError(f"Invalid header magic: {magic!r}, expected {self.HEADER_MAGIC!r}")

        window_size, T_stored, N_stored, n_frames = struct.unpack("!HIHI", data[4:16])
        if T == 0:
            T = T_stored
        if N == 0:
            N = N_stored

        offset = 16
        windows = []
        for _ in range(n_frames):
            window, offset = _unpack_window(data, offset)
            windows.append(window)

        if not windows:  # pragma: no cover — T=0 edge case
            return np.zeros((T, N), dtype=np.int8)

        full = np.vstack(windows)
        return full[:T]

    def compress_frame(self, window: np.ndarray[Any, Any]) -> bytes:
        """Compress a single time window (for real-time streaming).

        Parameters
        ----------
        window : ndarray of shape (W, N), binary

        Returns
        -------
        bytes — single frame, independently decodable
        """
        return _pack_window(np.asarray(window, dtype=np.int8))

    def decompress_frame(self, frame: bytes) -> np.ndarray[Any, Any]:
        """Decompress a single frame.

        Parameters
        ----------
        frame : bytes

        Returns
        -------
        ndarray of shape (W, N), int8
        """
        window, _ = _unpack_window(frame, 0)
        return window
