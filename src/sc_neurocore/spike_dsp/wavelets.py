# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike-domain wavelet decomposition

"""Spike-domain wavelet decomposition using multi-scale spike filtering."""

from __future__ import annotations

from typing import Any
import numpy as np


def spike_wavelet_decompose(
    spikes: np.ndarray[Any, Any],
    n_scales: int = 4,
    base_window: int = 4,
) -> list[np.ndarray[Any, Any]]:
    """Decompose spike train into frequency bands via multi-scale filtering.

    Uses cascaded moving-average filters at doubling window sizes:
    scale 0: window=base_window, scale 1: window=2*base_window, etc.
    Each scale captures a different frequency band.

    Parameters
    ----------
    spikes : ndarray of shape (T,) or (T, N)
    n_scales : int
        Number of wavelet scales.
    base_window : int
        Window size for finest scale.

    Returns
    -------
    list of ndarray
        One array per scale, shape (T,) or (T, N). Binary spike representation
        of activity at each frequency band.
    """
    if spikes.ndim == 1:
        spikes = spikes[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    T, N = spikes.shape
    scales = []

    for s in range(n_scales):
        window = base_window * (2**s)
        # Moving average at this scale
        smoothed = np.zeros((T, N), dtype=np.float64)
        for t in range(T):
            start = max(0, t - window + 1)
            smoothed[t] = spikes[start : t + 1].mean(axis=0)

        # Difference between adjacent scales = bandpass
        if s == 0:
            band = smoothed
        else:
            prev_window = base_window * (2 ** (s - 1))
            prev_smoothed = np.zeros((T, N), dtype=np.float64)
            for t in range(T):
                start = max(0, t - prev_window + 1)
                prev_smoothed[t] = spikes[start : t + 1].mean(axis=0)
            band = np.abs(prev_smoothed - smoothed)

        # Threshold to binary
        threshold = max(band.mean() * 0.5, 1e-8)
        binary_band = (band > threshold).astype(np.int8)

        scales.append(binary_band[:, 0] if squeeze else binary_band)

    return scales
