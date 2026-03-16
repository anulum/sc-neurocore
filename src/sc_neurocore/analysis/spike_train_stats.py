# SPDX-License-Identifier: AGPL-3.0-or-later
"""Spike train statistics — standard neuroscience analysis toolkit.

Provides the metrics researchers expect when comparing SNN simulators:
CV (coefficient of variation), Fano factor, cross-correlation,
power spectral density, pairwise correlation, and PSTH.

All functions accept spike trains as 1-D numpy arrays of 0/1 (binary)
or spike-time arrays (float, in seconds).
"""

from __future__ import annotations

import numpy as np


def spike_times(binary_train: np.ndarray, dt: float = 0.001) -> np.ndarray:
    """Extract spike times (seconds) from a binary 0/1 array."""
    return np.where(binary_train > 0)[0] * dt


def isi(binary_train: np.ndarray, dt: float = 0.001) -> np.ndarray:
    """Inter-spike intervals (seconds) from a binary train."""
    times = spike_times(binary_train, dt)
    if times.size < 2:
        return np.array([], dtype=np.float64)
    return np.diff(times)


def firing_rate(binary_train: np.ndarray, dt: float = 0.001) -> float:
    """Mean firing rate (Hz)."""
    duration = binary_train.size * dt
    if duration <= 0:
        return 0.0
    return float(np.sum(binary_train) / duration)


def cv_isi(binary_train: np.ndarray, dt: float = 0.001) -> float:
    """Coefficient of variation of ISI. CV=1 for Poisson, <1 for regular."""
    intervals = isi(binary_train, dt)
    if intervals.size < 2:
        return float("nan")
    mu = intervals.mean()
    if mu == 0:
        return float("nan")
    return float(intervals.std() / mu)


def fano_factor(
    binary_train: np.ndarray, window_ms: float = 50.0, dt: float = 0.001
) -> float:
    """Fano factor: variance/mean of spike counts in sliding windows."""
    window_steps = max(1, int(window_ms / (dt * 1000)))
    n = binary_train.size
    if n < window_steps:
        return float("nan")
    n_windows = n // window_steps
    counts = binary_train[: n_windows * window_steps].reshape(n_windows, window_steps).sum(axis=1)
    mu = counts.mean()
    if mu == 0:
        return float("nan")
    return float(counts.var() / mu)


def spike_count(binary_train: np.ndarray) -> int:
    """Total spike count."""
    return int(np.sum(binary_train))


def psth(
    trials: list[np.ndarray], bin_ms: float = 10.0, dt: float = 0.001
) -> tuple[np.ndarray, np.ndarray]:
    """Peri-stimulus time histogram across trials.

    Returns (rates_hz, bin_centers_ms).
    """
    if not trials:
        return np.array([]), np.array([])
    max_len = max(t.size for t in trials)
    bin_steps = max(1, int(bin_ms / (dt * 1000)))
    n_bins = max_len // bin_steps
    if n_bins == 0:
        return np.array([]), np.array([])
    counts = np.zeros(n_bins, dtype=np.float64)
    for trial in trials:
        trimmed = trial[: n_bins * bin_steps]
        if trimmed.size == 0:
            continue
        reshaped = trimmed.reshape(-1, bin_steps) if trimmed.size >= bin_steps else trimmed[None, :]
        if reshaped.shape[0] <= n_bins:
            counts[: reshaped.shape[0]] += reshaped.sum(axis=1)
    rates = counts / (len(trials) * bin_ms / 1000.0)
    centers = (np.arange(n_bins) + 0.5) * bin_ms
    return rates, centers


def cross_correlation(
    train_a: np.ndarray, train_b: np.ndarray, max_lag_ms: float = 50.0, dt: float = 0.001
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-correlogram between two binary spike trains.

    Returns (correlation, lags_ms).
    """
    max_lag = int(max_lag_ms / (dt * 1000))
    n = min(train_a.size, train_b.size)
    a = train_a[:n].astype(np.float64) - train_a[:n].mean()
    b = train_b[:n].astype(np.float64) - train_b[:n].mean()
    lags = np.arange(-max_lag, max_lag + 1)
    cc = np.zeros(len(lags), dtype=np.float64)
    norm = np.sqrt(np.sum(a**2) * np.sum(b**2))
    if norm == 0:
        return cc, lags * dt * 1000
    for i, lag in enumerate(lags):
        if lag >= 0:
            cc[i] = np.sum(a[: n - lag] * b[lag:n])
        else:
            cc[i] = np.sum(a[-lag:n] * b[: n + lag])
    cc /= norm
    return cc, lags * dt * 1000


def pairwise_correlation(trains: list[np.ndarray], dt: float = 0.001) -> np.ndarray:
    """Pairwise Pearson correlation matrix across neurons."""
    n = len(trains)
    if n == 0:
        return np.array([[]])
    min_len = min(t.size for t in trains)
    mat = np.array([t[:min_len].astype(np.float64) for t in trains])
    return np.corrcoef(mat)


def power_spectrum(
    binary_train: np.ndarray, dt: float = 0.001
) -> tuple[np.ndarray, np.ndarray]:
    """Power spectral density of a binary spike train.

    Returns (psd, freqs_hz).
    """
    n = binary_train.size
    if n < 2:
        return np.array([]), np.array([])
    x = binary_train.astype(np.float64) - binary_train.mean()
    fft_vals = np.fft.rfft(x)
    psd = np.abs(fft_vals) ** 2 / n
    freqs = np.fft.rfftfreq(n, d=dt)
    return psd, freqs


def burst_detection(
    binary_train: np.ndarray, dt: float = 0.001, max_isi_ms: float = 10.0, min_spikes: int = 3
) -> list[tuple[float, float, int]]:
    """Detect bursts: consecutive spikes with ISI < max_isi_ms.

    Returns list of (start_time_s, end_time_s, spike_count).
    """
    times = spike_times(binary_train, dt)
    if times.size < min_spikes:
        return []
    max_isi = max_isi_ms / 1000.0
    intervals = np.diff(times)
    bursts = []
    i = 0
    while i < intervals.size:
        if intervals[i] < max_isi:
            start = i
            while i < intervals.size and intervals[i] < max_isi:
                i += 1
            n_spikes = i - start + 1
            if n_spikes >= min_spikes:
                bursts.append((float(times[start]), float(times[i]), n_spikes))
        else:
            i += 1
    return bursts
