# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Cross-correlation, synchrony, and covariance measures

"""Cross-correlation, synchrony, and covariance measures for spike trains."""

from __future__ import annotations

import math
import os as _os
from typing import Any

import numpy as np

from .basic import bin_spike_train, spike_times

_HAS_RUST = False
_ssc = None

if not _os.environ.get("SC_NEUROCORE_NO_RUST"):
    try:
        from sc_neurocore.analysis.spike_stats import (  # type: ignore[attr-defined,no-redef]  # optional native PyO3 extension
            spike_stats_core as _ssc,
        )

        _HAS_RUST = True
    except ImportError:
        pass


def cross_correlation(
    train_a: np.ndarray[Any, Any],
    train_b: np.ndarray[Any, Any],
    max_lag_ms: float = 50.0,
    dt: float = 0.001,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Compute the normalised cross-correlogram between two spike trains.

    Parameters
    ----------
    train_a : numpy.ndarray
        First one-dimensional spike train.
    train_b : numpy.ndarray
        Second one-dimensional spike train.
    max_lag_ms : float
        Maximum lag magnitude in milliseconds; must be finite and non-negative.
    dt : float
        Sampling timestep in seconds; must be finite and positive.

    Returns
    -------
    tuple of numpy.ndarray
        ``(correlation, lags_ms)`` spanning ``[-max_lag, +max_lag]`` timesteps;
        lags whose magnitude reaches or exceeds the shared train length carry
        zero overlap and evaluate to ``0.0``.

    Raises
    ------
    ValueError
        If ``dt`` is not positive and finite, ``max_lag_ms`` is negative or
        non-finite, or either spike train is not one-dimensional or contains
        non-finite or non-real values.
    """
    if (
        isinstance(dt, bool)
        or not isinstance(dt, (int, float, np.integer, np.floating))
        or not math.isfinite(float(dt))
        or float(dt) <= 0.0
    ):
        raise ValueError(f"dt must be finite and positive, got {dt!r}")
    if (
        isinstance(max_lag_ms, bool)
        or not isinstance(max_lag_ms, (int, float, np.integer, np.floating))
        or not math.isfinite(float(max_lag_ms))
        or float(max_lag_ms) < 0.0
    ):
        raise ValueError(f"max_lag_ms must be finite and non-negative, got {max_lag_ms!r}")
    dt_f = float(dt)
    max_lag_ms_f = float(max_lag_ms)
    raw_a = np.asarray(train_a)
    raw_b = np.asarray(train_b)
    if raw_a.ndim != 1 or raw_b.ndim != 1:
        raise ValueError("spike trains must be one-dimensional arrays")
    if (
        raw_a.dtype == np.object_
        or raw_b.dtype == np.object_
        or np.iscomplexobj(raw_a)
        or np.iscomplexobj(raw_b)
    ):
        raise ValueError("spike trains must contain only finite real values")
    arr_a = raw_a.astype(np.float64)
    arr_b = raw_b.astype(np.float64)
    if not (np.all(np.isfinite(arr_a)) and np.all(np.isfinite(arr_b))):
        raise ValueError("spike trains must contain only finite values")

    max_lag = int(max_lag_ms_f / (dt_f * 1000.0))
    lags = np.arange(-max_lag, max_lag + 1)
    cc = np.zeros(len(lags), dtype=np.float64)
    lags_ms = lags * dt_f * 1000.0
    n = min(arr_a.size, arr_b.size)
    if n == 0:
        return cc, lags_ms

    a = arr_a[:n] - raw_a[:n].mean()
    b = arr_b[:n] - raw_b[:n].mean()
    norm = float(np.sqrt(np.sum(a**2) * np.sum(b**2)))
    if norm == 0.0:
        return cc, lags_ms

    for idx, lag in enumerate(lags):
        lag_steps = int(lag)
        overlap = n - abs(lag_steps)
        if overlap <= 0:
            continue
        if lag_steps >= 0:
            cc[idx] = float(np.sum(a[:overlap] * b[lag_steps : lag_steps + overlap]))
        else:
            offset = -lag_steps
            cc[idx] = float(np.sum(a[offset : offset + overlap] * b[:overlap]))
    cc /= norm
    return cc, lags_ms


def pairwise_correlation(
    trains: list[np.ndarray[Any, Any]], dt: float = 0.001
) -> np.ndarray[Any, Any]:
    """Compute the pairwise Pearson correlation matrix across spike trains.

    Parameters
    ----------
    trains : list of numpy.ndarray
        Binary spike trains aligned to a shared sampling clock.
    dt : float
        Sampling timestep in seconds (retained for API symmetry).

    Returns
    -------
    numpy.ndarray
        Symmetric ``(n_neurons, n_neurons)`` correlation matrix, or ``(1, 0)``
        when ``trains`` is empty.
    """
    _ = dt
    n = len(trains)
    if n == 0:
        return np.array([[]])
    min_len = min(t.size for t in trains)
    mat = np.array([t[:min_len].astype(np.float64) for t in trains])
    return np.asarray(np.corrcoef(mat))


def event_synchronization(
    train_a: np.ndarray[Any, Any],
    train_b: np.ndarray[Any, Any],
    dt: float = 0.001,
    tau_ms: float = 5.0,
) -> float:
    """Compute event synchronisation (Quian Quiroga et al., 2002).

    Parameters
    ----------
    train_a : numpy.ndarray
        First binary spike train.
    train_b : numpy.ndarray
        Second binary spike train.
    dt : float
        Sampling timestep in seconds.
    tau_ms : float
        Coincidence tolerance window in milliseconds.

    Returns
    -------
    float
        Normalised event coincidence count in ``[0, 1]`` for sparse trains.
    """
    ta = spike_times(train_a, dt)
    tb = spike_times(train_b, dt)
    na, nb = ta.size, tb.size
    if na == 0 or nb == 0:
        return 0.0
    tau = tau_ms / 1000.0
    if _HAS_RUST and _ssc is not None:
        return float(
            _ssc.py_event_synchronization(
                np.ascontiguousarray(ta, dtype=np.float64),
                np.ascontiguousarray(tb, dtype=np.float64),
                tau,
            )
        )
    count = 0
    for i in range(na):
        for j in range(nb):
            if abs(ta[i] - tb[j]) < tau:
                count += 1
    return float(count / (na * nb) ** 0.5)


def spike_train_coherence(
    train_a: np.ndarray[Any, Any], train_b: np.ndarray[Any, Any], dt: float = 0.001
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Compute magnitude-squared spectral coherence between two spike trains.

    Parameters
    ----------
    train_a : numpy.ndarray
        First binary spike train.
    train_b : numpy.ndarray
        Second binary spike train.
    dt : float
        Sampling timestep in seconds.

    Returns
    -------
    tuple of numpy.ndarray
        ``(coherence, freqs_hz)`` across real FFT frequencies, or two empty
        arrays when fewer than two samples are shared.
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
    """Compute the Spike Time Tiling Coefficient (Cutts & Eglen, 2014).

    Parameters
    ----------
    train_a : numpy.ndarray
        First binary spike train.
    train_b : numpy.ndarray
        Second binary spike train.
    dt_param : float
        Sampling timestep in seconds.
    delta_ms : float
        Symmetric coincidence half-window in milliseconds.

    Returns
    -------
    float
        Rate-corrected tiling correlation in ``[-1, 1]``.
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
    """Compute the binned spike-count covariance matrix (de la Rocha et al., 2007).

    Parameters
    ----------
    trains : list of numpy.ndarray
        Binary spike trains across neurons.
    bin_size : int
        Number of timesteps per count bin.

    Returns
    -------
    numpy.ndarray
        ``(n_neurons, n_neurons)`` sample covariance matrix across bins.
    """
    binned = [bin_spike_train(t, bin_size).astype(np.float64) for t in trains]
    min_bins = min(b.size for b in binned)
    mat = np.array([b[:min_bins] for b in binned])
    return np.cov(mat) if mat.shape[0] > 1 else np.array([[mat.var()]])


def autocorrelation_time(
    binary_train: np.ndarray[Any, Any], dt: float = 0.001, max_lag_ms: float = 100.0
) -> float:
    """Integrate normalised autocorrelation up to its first negative crossing.

    Parameters
    ----------
    binary_train : numpy.ndarray
        Binary spike train.
    dt : float
        Sampling timestep in seconds.
    max_lag_ms : float
        Maximum integration lag in milliseconds.

    Returns
    -------
    float
        Estimated autocorrelation timescale in seconds.
    """
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
    """Compute residual noise correlation across binned spike trains (Averbeck & Lee, 2006).

    Parameters
    ----------
    trains : list of numpy.ndarray
        Binary spike trains across neurons.
    bin_size : int
        Number of timesteps per count bin.

    Returns
    -------
    numpy.ndarray
        Symmetric ``(n_neurons, n_neurons)`` residual correlation matrix.
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
    """Compute binned signal Pearson correlation across spike trains.

    Parameters
    ----------
    trains : list of numpy.ndarray
        Binary spike trains across neurons.
    bin_size : int
        Number of timesteps per count bin.

    Returns
    -------
    numpy.ndarray
        ``(n_neurons, n_neurons)`` Pearson correlation matrix of binned counts.
    """
    binned = [bin_spike_train(t, bin_size).astype(np.float64) for t in trains]
    min_bins = min(b.size for b in binned)
    mat = np.array([b[:min_bins] for b in binned])
    return np.asarray(np.corrcoef(mat))


def spike_count_covariance(
    trains: list[np.ndarray[Any, Any]], window: int = 50
) -> np.ndarray[Any, Any]:
    """Compute windowed spike-count covariance (Kohn & Smith, 2005).

    Parameters
    ----------
    trains : list of numpy.ndarray
        Binary spike trains across neurons.
    window : int
        Number of timesteps per count window.

    Returns
    -------
    numpy.ndarray
        ``(n_neurons, n_neurons)`` windowed spike-count covariance matrix.
    """
    return covariance_matrix(trains, bin_size=window)


def joint_psth(
    train_a: np.ndarray[Any, Any], train_b: np.ndarray[Any, Any], bin_size: int = 10
) -> np.ndarray[Any, Any]:
    """Compute the mean-centred Joint PSTH matrix (Aertsen et al., 1989).

    Parameters
    ----------
    train_a : numpy.ndarray
        First binary spike train.
    train_b : numpy.ndarray
        Second binary spike train.
    bin_size : int
        Number of timesteps per count bin.

    Returns
    -------
    numpy.ndarray
        Two-dimensional ``(n_bins, n_bins)`` outer-product covariance matrix.
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
    """Compute the rate-corrected coincidence index kappa (Joris et al., 2006).

    Parameters
    ----------
    train_a : numpy.ndarray
        First binary spike train.
    train_b : numpy.ndarray
        Second binary spike train.
    dt : float
        Sampling timestep in seconds.
    delta_ms : float
        Coincidence half-window in milliseconds.

    Returns
    -------
    float
        Coincidence index corrected for chance coincidences, clipped below at ``0.0``.
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
