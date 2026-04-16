# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Information-theoretic measures for spike trains

"""Information-theoretic measures for spike trains."""

from __future__ import annotations

import os as _os
from typing import Any, Callable

import numpy as np

from .basic import bin_spike_train

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


def mutual_information(
    train_a: np.ndarray[Any, Any], train_b: np.ndarray[Any, Any], bin_size: int = 10
) -> float:
    """Mutual information between two binned spike trains (bits).

    MI = H(A) + H(B) - H(A,B) using binned spike counts.
    """
    ca = bin_spike_train(train_a, bin_size)
    cb = bin_spike_train(train_b, bin_size)
    n = min(ca.size, cb.size)
    ca, cb = ca[:n], cb[:n]

    def _entropy(x: np.ndarray[Any, Any]) -> float:
        vals, counts = np.unique(x, return_counts=True)
        p = counts / counts.sum()
        return float(-np.sum(p * np.log2(p + 1e-30)))

    ha = _entropy(ca)
    hb = _entropy(cb)
    joint = ca * (cb.max() + 1) + cb
    hab = _entropy(joint)
    return max(0.0, ha + hb - hab)


def transfer_entropy(
    source: np.ndarray[Any, Any], target: np.ndarray[Any, Any], bin_size: int = 10, lag: int = 1
) -> float:
    """Transfer entropy from source to target spike train (bits).

    TE = H(target_future | target_past) - H(target_future | target_past, source_past)
    """
    cs = bin_spike_train(source, bin_size)
    ct = bin_spike_train(target, bin_size)
    n = min(cs.size, ct.size)
    if n <= lag:
        return 0.0
    cs, ct = cs[:n], ct[:n]
    t_past = ct[:-lag]
    t_future = ct[lag:]
    s_past = cs[:-lag]
    n_pts = t_past.size

    def _cond_entropy(future: np.ndarray[Any, Any], *pasts: np.ndarray[Any, Any]) -> float:
        joint = future.copy()
        for p in pasts:
            joint = joint * (p.max() + 1) + p
        vals, counts = np.unique(joint, return_counts=True)
        h_joint = float(-np.sum(counts / n_pts * np.log2(counts / n_pts + 1e-30)))
        past_joint = pasts[0].copy()
        for p in pasts[1:]:
            past_joint = past_joint * (p.max() + 1) + p
        vals2, counts2 = np.unique(past_joint, return_counts=True)
        h_past = float(-np.sum(counts2 / n_pts * np.log2(counts2 / n_pts + 1e-30)))
        return h_joint - h_past

    h1 = _cond_entropy(t_future, t_past)
    h2 = _cond_entropy(t_future, t_past, s_past)
    return max(0.0, float(h1 - h2))


def spike_train_entropy(
    binary_train: np.ndarray[Any, Any], bin_size: int = 10, word_length: int = 4
) -> float:
    """Spike train entropy via binary word analysis. Strong et al. 1998."""
    binned = (bin_spike_train(binary_train, bin_size) > 0).astype(np.uint8)
    n = binned.size
    if n < word_length:
        return float("nan")
    if _HAS_RUST and _ssc is not None:
        return float(_ssc.py_spike_train_entropy(np.ascontiguousarray(binned), word_length))
    n_words = n - word_length + 1
    words = np.zeros(n_words, dtype=np.int64)
    for i in range(n_words):
        w = 0
        for j in range(word_length):
            w = w * 2 + int(binned[i + j])
        words[i] = w
    _, counts = np.unique(words, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p + 1e-30)))


def noise_entropy(
    binary_train: np.ndarray[Any, Any], n_trials: int = 10, bin_size: int = 10, word_length: int = 4
) -> float:
    """Noise entropy estimate via splitting train into pseudo-trials. de Ruyter van Steveninck et al. 1997.

    Splits the train into n_trials segments, computes entropy per segment, averages.
    """
    n = binary_train.size
    trial_len = n // n_trials
    if trial_len < bin_size * word_length:
        return float("nan")
    entropies = []
    for t in range(n_trials):
        seg = binary_train[t * trial_len : (t + 1) * trial_len]
        h = spike_train_entropy(seg, bin_size, word_length)
        if not np.isnan(h):
            entropies.append(h)
    if not entropies:
        return float("nan")
    return float(np.mean(entropies))


def stimulus_specific_information(
    spike_counts: np.ndarray[Any, Any], stimulus_ids: np.ndarray[Any, Any]
) -> float:
    """Stimulus-specific information (SSI). Butts 2003.

    spike_counts: array of spike counts per trial.
    stimulus_ids: corresponding stimulus labels.
    Returns SSI in bits.
    """
    unique_stim = np.unique(stimulus_ids)
    n_total = len(spike_counts)
    if n_total == 0:
        return 0.0
    overall_mean = spike_counts.mean()
    if overall_mean <= 0:
        return 0.0
    ssi = 0.0
    for s in unique_stim:
        mask = stimulus_ids == s
        n_s = mask.sum()
        if n_s == 0:
            continue
        p_s = n_s / n_total
        mean_s = spike_counts[mask].mean()
        if mean_s > 0:
            ssi += p_s * mean_s * np.log2(mean_s / overall_mean) / overall_mean
    return float(max(0.0, ssi))


def kozachenko_leonenko_mi(x: np.ndarray[Any, Any], y: np.ndarray[Any, Any], k: int = 3) -> float:
    """Kozachenko-Leonenko k-NN mutual information estimator. Kraskov et al. 2004."""
    n = min(x.size, y.size)
    if n < k + 1:
        return 0.0
    xf = np.ascontiguousarray(x[:n], dtype=np.float64)
    yf = np.ascontiguousarray(y[:n], dtype=np.float64)
    if _HAS_RUST and _ssc is not None:
        return float(_ssc.py_kozachenko_leonenko_mi(xf, yf, k))
    xf = xf.reshape(-1, 1)
    yf = yf.reshape(-1, 1)
    xy = np.hstack([xf, yf])

    from scipy.special import digamma

    def _kth_dist(data: np.ndarray[Any, Any], idx: int, kk: int) -> float:
        dists = np.max(np.abs(data - data[idx]), axis=1)
        dists[idx] = np.inf
        return float(np.partition(dists, kk - 1)[kk - 1])

    psi_k = float(digamma(k))
    psi_n = float(digamma(n))
    nx_sum = 0.0
    ny_sum = 0.0
    for i in range(n):
        eps = _kth_dist(xy, i, k)
        nx = np.sum(np.abs(xf - xf[i]).ravel() < eps) - 1
        ny = np.sum(np.abs(yf - yf[i]).ravel() < eps) - 1
        nx_sum += digamma(nx + 1)
        ny_sum += digamma(ny + 1)
    return float(max(0.0, psi_k + psi_n - nx_sum / n - ny_sum / n))


def time_rescaling_ks_test(
    times: np.ndarray[Any, Any],
    rate_func: Callable[[float], float],
    t_start: float = 0.0,
    t_end: float = 1.0,
) -> tuple[float, bool]:
    """Time-rescaling KS test for point process goodness-of-fit. Brown et al. 2002.

    rate_func(t) -> float: conditional intensity function.
    Returns (ks_statistic, passes_at_95pct).
    """
    if times.size < 5:
        return 1.0, False
    sorted_t = np.sort(times[(times >= t_start) & (times <= t_end)])
    n = sorted_t.size
    rescaled = np.zeros(n)
    for i in range(n):
        lo = t_start if i == 0 else sorted_t[i - 1]
        hi = sorted_t[i]
        n_quad = 20
        t_quad = np.linspace(lo, hi, n_quad)
        rates = np.array([rate_func(t) for t in t_quad])
        rescaled[i] = np.trapezoid(rates, t_quad)
    transformed = 1.0 - np.exp(-rescaled)
    transformed.sort()
    ecdf = np.arange(1, n + 1) / n
    ks = np.max(np.abs(ecdf - transformed))
    critical_95 = 1.36 / np.sqrt(n)  # Kolmogorov-Smirnov 95% critical value
    return float(ks), bool(ks < critical_95)
