# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Cross-correlation, synchrony, and covariance measures

"""Cross-correlation, synchrony, and covariance measures for spike trains."""

from __future__ import annotations

import os as _os
from typing import Any

import numpy as np

from .basic import spike_times, bin_spike_train

# ---------------------------------------------------------------------------
# Rust Acceleration
# ---------------------------------------------------------------------------

_HAS_RUST = False
_ssc = None

if not _os.environ.get("SC_NEUROCORE_NO_RUST"):
    try:
        from sc_neurocore.analysis.spike_stats import spike_stats_core as _ssc
        _HAS_RUST = True
    except ImportError:
        pass


def cross_correlation(
    train_a: np.ndarray[Any, Any],
    train_b: np.ndarray[Any, Any],
    max_lag_ms: float = 50.0,
    dt: float = 0.001,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
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


def pairwise_correlation(
    trains: list[np.ndarray[Any, Any]], dt: float = 0.001
) -> np.ndarray[Any, Any]:
    """Pairwise Pearson correlation matrix across neurons."""
    n = len(trains)
    if n == 0:
        return np.array([[]])
    min_len = min(t.size for t in trains)
    mat = np.array([t[:min_len].astype(np.float64) for t in trains])
    return np.corrcoef(mat)


def event_synchronization(
    train_a: np.ndarray[Any, Any],
    train_b: np.ndarray[Any, Any],
    dt: float = 0.001,
    tau_ms: float = 5.0,
) -> float:
    """Quian Quiroga et al. 2002 -- event synchronization."""
    ta = spike_times(train_a, dt)
    tb = spike_times(train_b, dt)
    na, nb = ta.size, tb.size
    if na == 0 or nb == 0:
        return 0.0
    tau = tau_ms / 1000.0
    if _HAS_RUST and _ssc is not None:
        return float(_ssc.py_event_synchronization(
            np.ascontiguousarray(ta, dtype=np.float64),
            np.ascontiguousarray(tb, dtype=np.float64),
            tau,
        ))
    count = 0
    for i in range(na):
        for j in range(nb):
            if abs(ta[i] - tb[j]) < tau:
                count += 1
    return float(count / (na * nb) ** 0.5)


def spike_train_coherence(
    train_a: np.ndarray[Any, Any], train_b: np.ndarray[Any, Any], dt: float = 0.001
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Magnitude-squared coherence between two binary spike trains.

    Returns (coherence, freqs_hz).
    """
    n = min(train_a.size, train_b.size)
    if n < 2:
        return np.array([]), np.array([])
    a = train_a[:n].astype(np.float64) - train_a[:n].mean()
    b = train_b[:n].astype(np.float64) - train_b[:n].mean()
    fa = np.fft.rfft(a)
    fb = np.fft.rfft(b)
    pab = fa * np.conj(fb)
    paa = np.abs(fa) ** 2
    pbb = np.abs(fb) ** 2
    denom = paa * pbb
    denom[denom == 0] = 1e-30
    coh = np.abs(pab) ** 2 / denom
    freqs = np.fft.rfftfreq(n, d=dt)
    return coh, freqs


def spike_time_tiling_coefficient(
    train_a: np.ndarray[Any, Any],
    train_b: np.ndarray[Any, Any],
    dt_param: float = 0.001,
    delta_ms: float = 5.0,
) -> float:
    """Spike Time Tiling Coefficient (STTC). Cutts & Eglen 2014.

    Corrects for firing rate bias unlike simple coincidence measures.
    """
    delta = delta_ms / 1000.0
    ta = spike_times(train_a, dt_param)
    tb = spike_times(train_b, dt_param)
    duration = max(train_a.size, train_b.size) * dt_param
    if ta.size == 0 or tb.size == 0:
        return 0.0

    def _tile_fraction(times: np.ndarray[Any, Any]) -> float:
        covered = 0.0
        intervals: list[tuple[Any, Any]] = []
        for t in times:
            intervals.append((t - delta, t + delta))
        intervals.sort()
        merged = [intervals[0]]
        for lo, hi in intervals[1:]:
            if lo <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))
        for lo, hi in merged:
            lo_c = max(lo, 0.0)
            hi_c = min(hi, duration)
            if hi_c > lo_c:
                covered += hi_c - lo_c
        return min(covered / duration, 1.0) if duration > 0 else 0.0

    def _coincidence_fraction(
        times_ref: np.ndarray[Any, Any], times_target: np.ndarray[Any, Any]
    ) -> float:
        count = 0
        for t in times_ref:
            if np.any(np.abs(times_target - t) <= delta):
                count += 1
        return count / len(times_ref) if len(times_ref) > 0 else 0.0

    ta_frac = _tile_fraction(ta)
    tb_frac = _tile_fraction(tb)
    pa = _coincidence_fraction(ta, tb)
    pb = _coincidence_fraction(tb, ta)

    def _sttc_term(p: float, t: float) -> float:
        if abs(1.0 - t) < 1e-15:
            return 0.0
        return (p - t) / (1.0 - p * t) if abs(1.0 - p * t) > 1e-15 else 0.0

    return float(0.5 * (_sttc_term(pa, tb_frac) + _sttc_term(pb, ta_frac)))


def covariance_matrix(
    trains: list[np.ndarray[Any, Any]], bin_size: int = 10
) -> np.ndarray[Any, Any]:
    """Spike count covariance matrix across neurons. de la Rocha et al. 2007."""
    binned = [bin_spike_train(t, bin_size).astype(np.float64) for t in trains]
    min_bins = min(b.size for b in binned)
    mat = np.array([b[:min_bins] for b in binned])
    return np.cov(mat) if mat.shape[0] > 1 else np.array([[mat.var()]])


def autocorrelation_time(
    binary_train: np.ndarray[Any, Any], dt: float = 0.001, max_lag_ms: float = 100.0
) -> float:
    """Autocorrelation time (seconds). Integral of normalized autocorrelation until first zero crossing."""
    max_lag = int(max_lag_ms / (dt * 1000))
    x = binary_train.astype(np.float64) - binary_train.mean()
    var = np.sum(x**2)
    if var == 0:
        return 0.0
    tau = 0.0
    for lag in range(1, min(max_lag, x.size)):
        ac = np.sum(x[: x.size - lag] * x[lag:]) / var
        if ac < 0:
            break
        tau += ac * dt
    return float(tau)


def noise_correlation(
    trains: list[np.ndarray[Any, Any]], bin_size: int = 50
) -> np.ndarray[Any, Any]:
    """Noise correlation (trial-to-trial variability correlation). Averbeck & Lee 2006.

    Uses residuals after subtracting mean across neurons.
    """
    binned = [bin_spike_train(t, bin_size).astype(np.float64) for t in trains]
    min_bins = min(b.size for b in binned)
    mat = np.array([b[:min_bins] for b in binned])
    residuals = mat - mat.mean(axis=0, keepdims=True)
    n = len(trains)
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            std_i = residuals[i].std()
            std_j = residuals[j].std()
            if std_i > 0 and std_j > 0:
                corr[i, j] = corr[j, i] = np.mean(residuals[i] * residuals[j]) / (std_i * std_j)
    return corr


def signal_correlation(
    trains: list[np.ndarray[Any, Any]], bin_size: int = 50
) -> np.ndarray[Any, Any]:
    """Signal correlation (tuning similarity). Pearson correlation of mean responses."""
    binned = [bin_spike_train(t, bin_size).astype(np.float64) for t in trains]
    min_bins = min(b.size for b in binned)
    mat = np.array([b[:min_bins] for b in binned])
    return np.corrcoef(mat)


def spike_count_covariance(
    trains: list[np.ndarray[Any, Any]], window: int = 50
) -> np.ndarray[Any, Any]:
    """Windowed spike count covariance. Kohn & Smith 2005."""
    return covariance_matrix(trains, bin_size=window)


def joint_psth(
    train_a: np.ndarray[Any, Any], train_b: np.ndarray[Any, Any], bin_size: int = 10
) -> np.ndarray[Any, Any]:
    """Joint PSTH (JPSTH) matrix. Aertsen et al. 1989.

    Returns 2D histogram of binned spike counts (neuron_a x neuron_b).
    """
    ca = bin_spike_train(train_a, bin_size).astype(np.float64)
    cb = bin_spike_train(train_b, bin_size).astype(np.float64)
    n = min(ca.size, cb.size)
    ca, cb = ca[:n], cb[:n]
    ca -= ca.mean()
    cb -= cb.mean()
    return np.outer(ca, cb) / n


def coincidence_index(
    train_a: np.ndarray[Any, Any],
    train_b: np.ndarray[Any, Any],
    dt: float = 0.001,
    delta_ms: float = 2.0,
) -> float:
    """Coincidence index (kappa). Joris et al. 2006.

    Corrects raw coincidence count for expected coincidences from rate.
    """
    ta = spike_times(train_a, dt)
    tb = spike_times(train_b, dt)
    if ta.size == 0 or tb.size == 0:
        return 0.0
    delta = delta_ms / 1000.0
    duration = max(train_a.size, train_b.size) * dt
    raw_coinc = 0
    for t in ta:
        if np.any(np.abs(tb - t) <= delta):
            raw_coinc += 1
    expected = 2.0 * delta * ta.size * tb.size / duration if duration > 0 else 0.0
    norm = 0.5 * (ta.size + tb.size)
    if norm <= expected:
        return 0.0
    return float((raw_coinc - expected) / (norm - expected))
