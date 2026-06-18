# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike-domain spectral analysis

"""Spike-domain FFT and power spectrum via time-coded spiking Fourier transform.

Converts spike train to instantaneous firing rate, computes FFT, returns
frequency components as spike-rate amplitudes. The firing-rate conversion
uses a sliding window (kernel-based) that stays in the integer domain.
"""

from __future__ import annotations

from typing import Any
import numpy as np


def spike_fft(
    spikes: np.ndarray[Any, Any],
    dt: float = 0.001,
    window_size: int = 50,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Compute FFT of a spike train.

    Parameters
    ----------
    spikes : ndarray of shape (T,) or (T, N)
        Binary spike train(s).
    dt : float
        Timestep in seconds.
    window_size : int
        Sliding window for instantaneous rate estimation.

    Returns
    -------
    (frequencies, magnitudes) tuple
        frequencies: ndarray of shape (F,)
        magnitudes: ndarray of shape (F,) or (F, N)
    """
    if spikes.ndim == 1:
        spikes = spikes[:, np.newaxis]
    T, N = spikes.shape

    # Compute instantaneous firing rate via sliding window
    rates = np.zeros((T, N), dtype=np.float64)
    for t in range(T):
        start = max(0, t - window_size + 1)
        rates[t] = spikes[start : t + 1].mean(axis=0) / dt

    # FFT
    fft_result = np.fft.rfft(rates, axis=0)
    magnitudes = np.abs(fft_result)
    frequencies = np.fft.rfftfreq(T, d=dt)

    if N == 1:
        magnitudes = magnitudes[:, 0]
    return frequencies, magnitudes


def spike_power_spectrum(
    spikes: np.ndarray[Any, Any],
    dt: float = 0.001,
    window_size: int = 50,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Compute power spectral density of a spike train.

    Returns
    -------
    (frequencies, psd) tuple
    """
    freqs, mags = spike_fft(spikes, dt, window_size)
    psd = mags**2
    return freqs, psd
