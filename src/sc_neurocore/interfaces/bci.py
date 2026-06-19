# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Brain-Computer Interface signal encoder

"""Encode continuous neural signals (EEG, LFP, intracortical) into
spike trains and stochastic bitstreams for SC processing.

Uses framework-native encoding (seeded RNG for reproducibility, Sobol
quasi-random for low-discrepancy encoding). Supports windowed encoding
for streaming BCI pipelines.

For spike compression/telemetry, see spike_codec (6 codecs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..encoding.encoders import rate_encode


@dataclass
class BCIEncoder:
    """Encode continuous neural signals into spike trains.

    Replaces the old BCIDecoder (misleading name — it encodes, not decodes).
    Uses seeded RNG for deterministic, reproducible encoding.

    Parameters
    ----------
    n_channels : int
        Number of recording channels.
    sampling_rate : int
        Input signal sampling rate (Hz).
    window_ms : float
        Encoding window duration in milliseconds.
    seed : int
        RNG seed for reproducibility.
    """

    n_channels: int
    sampling_rate: int = 20000
    window_ms: float = 1.0
    seed: int = 42

    def encode(self, signal: np.ndarray[Any, Any], T: int = 20) -> np.ndarray[Any, Any]:
        """Encode a signal block into spike trains via rate coding.

        Parameters
        ----------
        signal : ndarray of shape (n_channels,) or (n_channels, n_samples)
            Continuous neural signal. Multi-sample input is averaged
            per channel to get firing probabilities.
        T : int
            Number of output timesteps per window.

        Returns
        -------
        ndarray of shape (T, n_channels), int8 binary
        """
        if signal.ndim > 1:
            probs = signal.mean(axis=1)
        else:
            probs = signal.copy()

        probs = self._normalize(probs)
        return rate_encode(probs, T, seed=self.seed)

    def encode_stream(self, signal: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Encode a multi-window signal stream.

        Parameters
        ----------
        signal : ndarray of shape (n_channels, total_samples)
            Full recording. Split into windows of window_ms duration.

        Returns
        -------
        ndarray of shape (total_T, n_channels), int8 binary
        """
        samples_per_window = max(1, int(self.sampling_rate * self.window_ms / 1000))
        n_windows = signal.shape[1] // samples_per_window
        T_per_window = max(1, samples_per_window // 10)

        chunks = []
        for w in range(n_windows):
            start = w * samples_per_window
            end = start + samples_per_window
            window = signal[:, start:end]
            chunk = self.encode(window, T=T_per_window)
            chunks.append(chunk)

        if not chunks:
            return np.zeros((0, self.n_channels), dtype=np.int8)

        return np.vstack(chunks)

    @staticmethod
    def _normalize(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Normalize to [0, 1] for probability encoding."""
        vmin, vmax = values.min(), values.max()
        if vmax - vmin < 1e-10:
            return np.full_like(values, 0.5)
        normalised: np.ndarray[Any, Any] = (values - vmin) / (vmax - vmin)
        return normalised

    # --- Backward-compatible API (old BCIDecoder methods) ---

    def normalize_signal(self, signal: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Normalize signal to [0, 1]. Legacy API — use _normalize()."""
        s_min, s_max = np.min(signal), np.max(signal)
        if s_max - s_min == 0:
            return np.zeros_like(signal)
        scaled: np.ndarray[Any, Any] = (signal - s_min) / (s_max - s_min)
        return scaled

    def encode_to_bitstream(
        self, signal: np.ndarray[Any, Any], length: int = 256
    ) -> np.ndarray[Any, Any]:
        """Legacy API. Encodes (channels, time) → (channels, length).

        New code should use .encode() which returns (T, channels).
        """
        if signal.ndim > 1:
            mean_vals = np.mean(signal, axis=1)
        else:
            mean_vals = signal

        if len(mean_vals) != self.n_channels:
            raise ValueError(f"Signal has {len(mean_vals)} channels, expected {self.n_channels}")

        probs = self.normalize_signal(mean_vals)
        rng = np.random.RandomState(self.seed)
        bits = (rng.random((self.n_channels, length)) < probs[:, None]).astype(np.uint8)
        return bits


class BCIDecoder(BCIEncoder):
    """Legacy alias. Use BCIEncoder instead."""

    def __init__(self, channels: int, sampling_rate: int = 1000, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(n_channels=channels, sampling_rate=sampling_rate, **kwargs)
