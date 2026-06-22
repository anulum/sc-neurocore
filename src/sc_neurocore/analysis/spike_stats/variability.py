# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike train variability and regularity measures

"""Spike train variability and regularity measures."""

from __future__ import annotations

import os as _os
from typing import Any

import numpy as np

from .basic import isi, spike_times

# ---------------------------------------------------------------------------
# Rust Acceleration
# ---------------------------------------------------------------------------

_HAS_RUST = False
_ssc = None

if not _os.environ.get("SC_NEUROCORE_NO_RUST"):
    try:
        from sc_neurocore.analysis.spike_stats import spike_stats_core as _ssc  # type: ignore[attr-defined,no-redef]

        _HAS_RUST = True
    except ImportError:
        pass


def cv_isi(binary_train: np.ndarray[Any, Any], dt: float = 0.001) -> float:
    """Coefficient of variation of ISI. CV=1 for Poisson, <1 for regular."""
    intervals = isi(binary_train, dt)
    if intervals.size < 2:
        return float("nan")
    mu = intervals.mean()
    if mu == 0:
        return float("nan")
    return float(intervals.std() / mu)


def cv2(binary_train: np.ndarray[Any, Any], dt: float = 0.001) -> float:
    """Local coefficient of variation CV2. Holt et al. 1996.

    CV2 = mean(2|ISI_{i+1} - ISI_i| / (ISI_{i+1} + ISI_i)).
    Less sensitive to firing rate changes than global CV.
    """
    intervals = isi(binary_train, dt)
    if intervals.size < 2:
        return float("nan")
    diffs = np.abs(np.diff(intervals))
    sums = intervals[:-1] + intervals[1:]
    valid = sums > 0
    if not valid.any():
        return float("nan")
    return float(np.mean(2.0 * diffs[valid] / sums[valid]))


def local_variation(binary_train: np.ndarray[Any, Any], dt: float = 0.001) -> float:
    """Local variation LV. Shinomoto et al. 2003.

    LV = (3/(N-1)) * sum((ISI_i - ISI_{i+1})^2 / (ISI_i + ISI_{i+1})^2).
    LV=1 for Poisson, <1 for regular, >1 for bursty.
    """
    intervals = isi(binary_train, dt)
    n = intervals.size
    if n < 2:
        return float("nan")
    diffs = np.diff(intervals)
    sums = intervals[:-1] + intervals[1:]
    valid = sums > 0
    if not valid.any():
        return float("nan")
    return float(3.0 / (n - 1) * np.sum((diffs[valid] / sums[valid]) ** 2))


def lvr(
    binary_train: np.ndarray[Any, Any], dt: float = 0.001, refractoriness_ms: float = 2.0
) -> float:
    """Revised local variation LvR. Shinomoto et al. 2009.

    Corrects LV for refractoriness: LvR = mean(3(1 - 4*ISI_i*ISI_{i+1}/(ISI_i+ISI_{i+1})^2)(1 + 4*R/(ISI_i+ISI_{i+1}))).
    """
    intervals = isi(binary_train, dt)
    n = intervals.size
    if n < 2:
        return float("nan")
    r = refractoriness_ms / 1000.0
    result = 0.0
    count = 0
    for i in range(n - 1):
        s = intervals[i] + intervals[i + 1]
        if s <= 0:
            continue
        ratio = 4.0 * intervals[i] * intervals[i + 1] / (s * s)
        result += (1.0 - ratio) * (1.0 + 4.0 * r / s)
        count += 1
    if count == 0:
        return float("nan")
    return float(3.0 * result / count)


def fano_factor(
    binary_train: np.ndarray[Any, Any], window_ms: float = 50.0, dt: float = 0.001
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


def isi_entropy(binary_train: np.ndarray[Any, Any], dt: float = 0.001, bins: int = 20) -> float:
    """Shannon entropy of the ISI distribution (bits).

    Higher entropy = more irregular. Poisson has maximum entropy
    for a given rate.
    """
    intervals = isi(binary_train, dt)
    if intervals.size < 2:
        return float("nan")
    value_range = float(intervals.max() - intervals.min())
    # A (numerically) constant ISI distribution carries zero entropy. Guard on
    # the histogram's own resolution limit -- a range narrower than `bins` ULPs
    # of the largest interval -- so a regular train short-circuits to 0.0
    # instead of tripping np.histogram's "too many bins for data range" error.
    if value_range < bins * float(np.spacing(intervals.max())):
        return 0.0
    bin_width = value_range / bins
    hist, _ = np.histogram(intervals, bins=bins, density=True)
    hist = hist[hist > 0]
    p = hist * bin_width
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def lempel_ziv_complexity(binary_train: np.ndarray[Any, Any]) -> float:
    """Lempel-Ziv 1976 complexity. Normalized by N/log2(N)."""
    n = binary_train.size
    if n == 0:
        return 0.0
    s = (binary_train > 0).astype(np.uint8)
    if _HAS_RUST and _ssc is not None:
        return float(_ssc.py_lempel_ziv_complexity(np.ascontiguousarray(s)))
    s = s.astype(np.int8)
    complexity = 1
    l = 1
    k = 1
    k_max = 1
    while l + k <= n:
        if s[l + k - 1] == s[k - 1]:
            k += 1
        else:
            k_max = max(k_max, k)
            k = 1
            if k_max > k:
                k_max = k
            complexity += 1
            l += k_max
            k = 1
            k_max = 1
    complexity += 1
    norm = n / np.log2(max(n, 2))
    return float(complexity / norm)


def approximate_entropy(
    binary_train: np.ndarray[Any, Any], m: int = 2, r_factor: float = 0.2
) -> float:
    """Approximate entropy (ApEn). Pincus 1991."""
    x = binary_train.astype(np.float64)
    n = x.size
    if n < m + 2:
        return float("nan")
    r = r_factor * x.std()
    if r <= 0:
        r = 0.01
    if _HAS_RUST and _ssc is not None:
        return float(_ssc.py_approximate_entropy(np.ascontiguousarray(x), m, r))

    def _phi(dim: int) -> float:
        # _phi is only called with dim in {m, m+1}; the n < m + 2 guard above
        # ensures n >= m + 2 > dim, so n - dim + 1 >= 2 and the template list is
        # never empty.
        templates = np.array([x[i : i + dim] for i in range(n - dim + 1)])
        count = np.zeros(len(templates))
        for i in range(len(templates)):
            dists = np.max(np.abs(templates - templates[i]), axis=1)
            count[i] = np.sum(dists <= r)
        count /= len(templates)
        return float(np.mean(np.log(count + 1e-30)))

    return float(_phi(m) - _phi(m + 1))


def sample_entropy(binary_train: np.ndarray[Any, Any], m: int = 2, r_factor: float = 0.2) -> float:
    """Sample entropy (SampEn). Richman & Moorman 2000."""
    x = binary_train.astype(np.float64)
    n = x.size
    if n < m + 2:
        return float("nan")
    r = r_factor * x.std()
    if r <= 0:
        r = 0.01
    if _HAS_RUST and _ssc is not None:
        return float(_ssc.py_sample_entropy(np.ascontiguousarray(x), m, r))

    def _count_matches(dim: int) -> int:
        templates = np.array([x[i : i + dim] for i in range(n - dim)])
        total = 0
        for i in range(len(templates)):
            dists = np.max(np.abs(templates[i + 1 :] - templates[i]), axis=1)
            total += int(np.sum(dists <= r))
        return total

    a = _count_matches(m + 1)
    b = _count_matches(m)
    if b == 0:
        return float("nan")
    return float(-np.log((a + 1e-30) / (b + 1e-30)))


def permutation_entropy(
    binary_train: np.ndarray[Any, Any], order: int = 3, delay: int = 1
) -> float:
    """Bandt-Pompe permutation entropy. Bandt & Pompe 2002."""
    x = binary_train.astype(np.float64)
    n = x.size
    if n < order * delay:
        return float("nan")
    if _HAS_RUST and _ssc is not None:
        return float(_ssc.py_permutation_entropy(np.ascontiguousarray(x), order, delay))
    # n_patterns = n - (order - 1) * delay; the n < order * delay guard above
    # ensures n >= order * delay, so n_patterns >= delay >= 1 and at least one
    # ordinal pattern always exists.
    n_patterns = n - (order - 1) * delay
    patterns = np.zeros(n_patterns, dtype=np.int64)
    for i in range(n_patterns):
        window = x[i : i + order * delay : delay]
        rank = np.argsort(np.argsort(window))
        key = 0
        for j, r in enumerate(rank):
            key += int(r) * (order**j)
        patterns[i] = key
    _, counts = np.unique(patterns, return_counts=True)
    p = counts / counts.sum()
    h = -np.sum(p * np.log2(p + 1e-30))
    h_max = np.log2(float(np.prod(np.arange(1, order + 1))))
    return float(h / h_max) if h_max > 0 else 0.0


def hurst_exponent(binary_train: np.ndarray[Any, Any], min_window: int = 10) -> float:
    """Hurst exponent via detrended fluctuation analysis (DFA). Peng et al. 1994.

    H > 0.5: long-range positive correlation. H < 0.5: anti-correlated.
    """
    x = binary_train.astype(np.float64)
    n = x.size
    if n < 4 * min_window:
        return float("nan")
    y = np.cumsum(x - x.mean())
    scales = []
    flucts = []
    s = min_window
    while s <= n // 4:
        scales.append(s)
        n_seg = n // s
        f2 = 0.0
        for seg in range(n_seg):
            chunk = y[seg * s : (seg + 1) * s]
            t = np.arange(s, dtype=np.float64)
            coeffs = np.polyfit(t, chunk, 1)
            trend = np.polyval(coeffs, t)
            f2 += np.mean((chunk - trend) ** 2)
        f2 /= n_seg
        flucts.append(np.sqrt(f2))
        # int(1.5*s) > s for every integer s >= 2 (since 0.5*s >= 1), and a DFA
        # window of size 1 cannot be detrended (polyfit would be singular), so s
        # is never 1 here; the scale strictly increases and cannot stagnate.
        s = int(s * 1.5)
    if len(scales) < 2:
        return float("nan")
    log_s = np.log(np.array(scales, dtype=np.float64))
    log_f = np.log(np.array(flucts, dtype=np.float64) + 1e-30)
    coeffs = np.polyfit(log_s, log_f, 1)
    return float(coeffs[0])


def allan_factor(
    binary_train: np.ndarray[Any, Any], dt: float = 0.001, n_scales: int = 10
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Allan factor for spike trains. Allan 1966, adapted for point processes.

    Returns (af_values, window_sizes_s). AF > 1 indicates fractal clustering.
    """
    n = binary_train.size
    max_w = n // 4
    if max_w < 2:
        return np.array([]), np.array([])
    windows = np.unique(np.logspace(np.log10(2), np.log10(max_w), n_scales).astype(int))
    af = np.zeros(len(windows))
    for i, w in enumerate(windows):
        # Every window w comes from logspace(2, max_w) with max_w = n // 4, so
        # w <= n // 4 and n_bins = n // w >= n // (n // 4) >= 4 > 2 for all n;
        # the count-array always has enough bins to difference.
        n_bins = n // w
        counts = binary_train[: n_bins * w].reshape(n_bins, w).sum(axis=1).astype(np.float64)
        diffs = np.diff(counts)
        mean_count = counts.mean()
        if mean_count == 0:
            af[i] = float("nan")
        else:
            af[i] = np.mean(diffs**2) / (2.0 * mean_count)
    return af, windows * dt


def rescaled_range(binary_train: np.ndarray[Any, Any], min_window: int = 10) -> float:
    """Hurst exponent via rescaled range (R/S) analysis. Hurst 1951.

    Classic alternative to DFA. Returns H from log-log fit of R/S vs scale.
    """
    x = binary_train.astype(np.float64)
    n = x.size
    if n < 4 * min_window:
        return float("nan")
    scales = []
    rs_vals = []
    s = min_window
    while s <= n // 2:
        n_seg = n // s
        rs_seg = []
        for seg in range(n_seg):
            chunk = x[seg * s : (seg + 1) * s]
            mean_c = chunk.mean()
            y = np.cumsum(chunk - mean_c)
            r = y.max() - y.min()
            std_c = chunk.std()
            if std_c > 0:
                rs_seg.append(r / std_c)
        if rs_seg:
            scales.append(s)
            rs_vals.append(np.mean(rs_seg))
        # Guarantee a strictly increasing scale: int(1.5*s) stalls at s only for
        # s == 1 (size-1 segments yield zero variance and no R/S sample), which
        # would otherwise loop forever, so force at least a unit step.
        prev_s = s
        s = int(s * 1.5)
        if s <= prev_s:
            s = prev_s + 1
    if len(scales) < 2:
        return float("nan")
    log_s = np.log(np.array(scales, dtype=np.float64))
    log_rs = np.log(np.array(rs_vals, dtype=np.float64) + 1e-30)
    coeffs = np.polyfit(log_s, log_rs, 1)
    return float(coeffs[0])


def complexity_pdf(
    binary_train: np.ndarray[Any, Any], dt: float = 0.001, bins: int = 20
) -> np.ndarray[Any, Any]:
    """ISI probability density function via histogram. Abeles 1982."""
    intervals = isi(binary_train, dt)
    if intervals.size < 2:
        return np.array([], dtype=np.float64)
    if intervals.max() - intervals.min() < 1e-12:
        return np.array([], dtype=np.float64)
    hist, edges = np.histogram(intervals, bins=bins, density=True)
    return hist.astype(np.float64)


def optimal_bin_width(binary_train: np.ndarray[Any, Any], dt: float = 0.001) -> float:
    """Shimazaki-Shinomoto 2007 optimal histogram bin width for firing rate.

    Minimizes MISE cost C(delta) = (2*mean - var) / (N * delta)^2 over candidate deltas.
    Returns optimal bin width in seconds.
    """
    times = spike_times(binary_train, dt)
    n = times.size
    if n < 2:
        return float("nan")
    duration = binary_train.size * dt
    d_min = max(dt, duration / max(n, 1))
    d_max = duration
    n_candidates = 50
    deltas = np.linspace(d_min, d_max / 2, n_candidates)
    best_cost = np.inf
    best_delta = deltas[0]
    for delta in deltas:
        edges = np.arange(0, duration + delta, delta)
        counts = np.histogram(times, bins=edges)[0].astype(np.float64)
        k = counts.mean()
        v = counts.var()
        cost = (2.0 * k - v) / (delta * delta) if delta > 0 else np.inf
        if cost < best_cost:
            best_cost = cost
            best_delta = delta
    return float(best_delta)


def optimal_kernel_bandwidth(binary_train: np.ndarray[Any, Any], dt: float = 0.001) -> float:
    """Silverman's rule-of-thumb bandwidth for ISI kernel density. Silverman 1986.

    h = 0.9 * min(std, IQR/1.34) * N^{-1/5}.
    """
    intervals = isi(binary_train, dt)
    n = intervals.size
    if n < 2:
        return float("nan")
    std = intervals.std()
    q75, q25 = np.percentile(intervals, [75, 25])
    iqr = q75 - q25
    spread = min(std, iqr / 1.34) if iqr > 0 else std
    if spread <= 0:
        return float("nan")
    return float(0.9 * spread * n ** (-0.2))
