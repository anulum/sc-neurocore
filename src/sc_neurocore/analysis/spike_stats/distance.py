# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike train distance and similarity metrics

"""Spike train distance and similarity metrics."""

from __future__ import annotations

import os as _os
from typing import Any, Callable

import numpy as np

from .basic import isi
from .rate import instantaneous_rate

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


def van_rossum_distance(
    train_a: np.ndarray[Any, Any],
    train_b: np.ndarray[Any, Any],
    dt: float = 0.001,
    tau_ms: float = 10.0,
) -> float:
    """Van Rossum 2001 -- exponential-kernel spike train distance."""
    a = np.ascontiguousarray(train_a, dtype=np.float64)
    b = np.ascontiguousarray(train_b, dtype=np.float64)
    if _HAS_RUST and _ssc is not None:
        return float(_ssc.py_van_rossum_distance(a, b, dt, tau_ms))
    tau = tau_ms / 1000.0
    n = min(a.size, b.size)
    t = np.arange(n) * dt
    decay = np.exp(-t / tau) if tau > 0 else np.zeros(n)
    fa = np.convolve(a[:n], decay[:n], mode="full")[:n]
    fb = np.convolve(b[:n], decay[:n], mode="full")[:n]
    return float(np.sqrt(np.sum((fa - fb) ** 2) * dt / tau))


def victor_purpura_distance(
    times_a: np.ndarray[Any, Any], times_b: np.ndarray[Any, Any], cost_per_s: float = 1000.0
) -> float:
    """Victor-Purpura 1996 -- edit distance between spike time arrays."""
    a = np.ascontiguousarray(times_a, dtype=np.float64)
    b = np.ascontiguousarray(times_b, dtype=np.float64)
    if _HAS_RUST and _ssc is not None:
        return float(_ssc.py_victor_purpura_distance(a, b, cost_per_s))
    na, nb = len(a), len(b)
    if na == 0:
        return float(nb)
    if nb == 0:
        return float(na)
    d = np.zeros((na + 1, nb + 1), dtype=np.float64)
    for i in range(na + 1):
        d[i, 0] = float(i)
    for j in range(nb + 1):
        d[0, j] = float(j)
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            shift_cost = cost_per_s * abs(a[i - 1] - b[j - 1])
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + shift_cost)
    return float(d[na, nb])


def isi_distance(
    train_a: np.ndarray[Any, Any], train_b: np.ndarray[Any, Any], dt: float = 0.001
) -> float:
    """ISI-distance (Kreuz et al. 2007) -- ratio-based ISI comparison."""
    isi_a = isi(train_a, dt)
    isi_b = isi(train_b, dt)
    n = min(isi_a.size, isi_b.size)
    if n == 0:
        return float("nan")
    ratios = np.zeros(n)
    for i in range(n):
        a, b = isi_a[i], isi_b[i]
        if a == 0 and b == 0:
            ratios[i] = 0.0
        elif a <= b:
            ratios[i] = a / b - 1.0 if b > 0 else 0.0
        else:
            ratios[i] = -(b / a - 1.0) if a > 0 else 0.0
    return float(np.abs(ratios).mean())


def spike_distance(
    times_a: np.ndarray[Any, Any],
    times_b: np.ndarray[Any, Any],
    t_start: float = 0.0,
    t_end: float = 1.0,
) -> float:
    """SPIKE-distance. Kreuz et al. 2013."""
    a = np.ascontiguousarray(
        np.sort(times_a[(times_a >= t_start) & (times_a <= t_end)]), dtype=np.float64
    )
    b = np.ascontiguousarray(
        np.sort(times_b[(times_b >= t_start) & (times_b <= t_end)]), dtype=np.float64
    )
    if _HAS_RUST and _ssc is not None:
        return float(_ssc.py_spike_distance(a, b, t_start, t_end))
    if a.size == 0 and b.size == 0:
        return 0.0
    if a.size == 0 or b.size == 0:
        return 1.0
    n_eval = 100
    eval_times = np.linspace(t_start, t_end, n_eval)
    s_vals = np.zeros(n_eval)
    for k, t in enumerate(eval_times):
        idx_a = np.searchsorted(a, t, side="right")
        idx_b = np.searchsorted(b, t, side="right")
        prev_a = a[max(0, idx_a - 1)] if a.size > 0 else t_start
        next_a = a[min(idx_a, a.size - 1)] if a.size > 0 else t_end
        prev_b = b[max(0, idx_b - 1)] if b.size > 0 else t_start
        next_b = b[min(idx_b, b.size - 1)] if b.size > 0 else t_end
        isi_a = max(next_a - prev_a, 1e-30)
        isi_b = max(next_b - prev_b, 1e-30)
        da = min(abs(t - prev_a), abs(t - next_a))
        db = min(abs(t - prev_b), abs(t - next_b))
        s_vals[k] = abs(da / isi_a - db / isi_b)
    return float(s_vals.mean())


def _local_isi(times: np.ndarray[Any, Any], idx: int) -> float:
    """Nearest-neighbour ISI at index idx."""
    if times.size < 2:
        return 1.0
    if idx == 0:
        return float(times[1] - times[0])
    if idx >= times.size - 1:
        return float(times[-1] - times[-2])
    return float(min(times[idx] - times[idx - 1], times[idx + 1] - times[idx]))


def spike_sync(
    times_a: np.ndarray[Any, Any],
    times_b: np.ndarray[Any, Any],
    t_start: float = 0.0,
    t_end: float = 1.0,
) -> float:
    """SPIKE-synchronization. Kreuz et al. 2015."""
    a = np.ascontiguousarray(
        np.sort(times_a[(times_a >= t_start) & (times_a <= t_end)]), dtype=np.float64
    )
    b = np.ascontiguousarray(
        np.sort(times_b[(times_b >= t_start) & (times_b <= t_end)]), dtype=np.float64
    )
    if _HAS_RUST and _ssc is not None:
        return float(_ssc.py_spike_sync(a, b, t_start, t_end))
    if a.size == 0 or b.size == 0:
        return 0.0
    total_coincidences = 0
    total_possible = a.size + b.size
    for i in range(a.size):
        diffs = np.abs(b - a[i])
        j = int(np.argmin(diffs))
        isi_a = _local_isi(a, i)
        isi_b = _local_isi(b, j)
        tau = min(isi_a, isi_b) / 2.0
        if tau > 0 and diffs[j] < tau:
            total_coincidences += 1
    for j in range(b.size):
        diffs = np.abs(a - b[j])
        i = int(np.argmin(diffs))
        isi_a = _local_isi(a, i)
        isi_b = _local_isi(b, j)
        tau = min(isi_a, isi_b) / 2.0
        if tau > 0 and diffs[i] < tau:
            total_coincidences += 1
    if total_possible == 0:
        return 0.0
    return float(total_coincidences / total_possible)


def spike_sync_profile(
    times_a: np.ndarray[Any, Any],
    times_b: np.ndarray[Any, Any],
    n_bins: int = 50,
    t_start: float = 0.0,
    t_end: float = 1.0,
) -> np.ndarray[Any, Any]:
    """Binned SPIKE-synchronization profile. Kreuz et al. 2015."""
    edges = np.linspace(t_start, t_end, n_bins + 1)
    profile = np.zeros(n_bins)
    for k in range(n_bins):
        mask_a = (times_a >= edges[k]) & (times_a < edges[k + 1])
        mask_b = (times_b >= edges[k]) & (times_b < edges[k + 1])
        sub_a = times_a[mask_a]
        sub_b = times_b[mask_b]
        if sub_a.size + sub_b.size > 0:
            profile[k] = spike_sync(sub_a, sub_b, edges[k], edges[k + 1])
    return profile


def spike_profile(
    times_a: np.ndarray[Any, Any],
    times_b: np.ndarray[Any, Any],
    n_bins: int = 50,
    t_start: float = 0.0,
    t_end: float = 1.0,
) -> np.ndarray[Any, Any]:
    """Binned SPIKE-distance profile. Kreuz et al. 2013."""
    edges = np.linspace(t_start, t_end, n_bins + 1)
    profile = np.zeros(n_bins)
    for k in range(n_bins):
        mask_a = (times_a >= edges[k]) & (times_a < edges[k + 1])
        mask_b = (times_b >= edges[k]) & (times_b < edges[k + 1])
        sub_a = times_a[mask_a]
        sub_b = times_b[mask_b]
        profile[k] = spike_distance(sub_a, sub_b, edges[k], edges[k + 1])
    return profile


def isi_profile(
    binary_train_a: np.ndarray[Any, Any],
    binary_train_b: np.ndarray[Any, Any],
    dt: float = 0.001,
    n_bins: int = 50,
) -> np.ndarray[Any, Any]:
    """Binned ISI-distance profile. Kreuz et al. 2007."""
    n = min(binary_train_a.size, binary_train_b.size)
    bin_size = max(1, n // n_bins)
    profile = np.zeros(n_bins)
    for k in range(n_bins):
        start = k * bin_size
        end = min(start + bin_size, n)
        if start >= n:
            break
        profile[k] = isi_distance(binary_train_a[start:end], binary_train_b[start:end], dt)
    return profile


def adaptive_spike_distance(
    times_a: np.ndarray[Any, Any],
    times_b: np.ndarray[Any, Any],
    t_start: float = 0.0,
    t_end: float = 1.0,
    cost: float = 0.0,
) -> float:
    """Adaptive SPIKE-distance with cost parameter interpolating ISI and SPIKE. Kreuz et al. 2013.

    cost=0: pure SPIKE-distance. cost=1: ISI-like weighting.
    """
    sd = spike_distance(times_a, times_b, t_start, t_end)
    ta = times_a[(times_a >= t_start) & (times_a <= t_end)]
    tb = times_b[(times_b >= t_start) & (times_b <= t_end)]
    isi_a = np.diff(np.sort(ta)) if ta.size > 1 else np.array([t_end - t_start])
    isi_b = np.diff(np.sort(tb)) if tb.size > 1 else np.array([t_end - t_start])
    mean_a = isi_a.mean() if isi_a.size > 0 else 1.0
    mean_b = isi_b.mean() if isi_b.size > 0 else 1.0
    ratio = abs(mean_a - mean_b) / max(mean_a + mean_b, 1e-30)
    return float((1.0 - cost) * sd + cost * ratio)


def schreiber_similarity(
    train_a: np.ndarray[Any, Any],
    train_b: np.ndarray[Any, Any],
    dt: float = 0.001,
    sigma_ms: float = 5.0,
) -> float:
    """Schreiber et al. 2003 -- spike train similarity via smoothed correlation.

    Convolves each train with Gaussian kernel, returns Pearson correlation.
    """
    ra = instantaneous_rate(train_a, dt, "gaussian", sigma_ms)
    rb = instantaneous_rate(train_b, dt, "gaussian", sigma_ms)
    n = min(ra.size, rb.size)
    ra, rb = ra[:n], rb[:n]
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt(np.sum(ra**2) * np.sum(rb**2))
    if denom == 0:
        return 0.0
    return float(np.sum(ra * rb) / denom)


def hunter_milton_similarity(
    times_a: np.ndarray[Any, Any], times_b: np.ndarray[Any, Any], dt_max: float = 0.01
) -> float:
    """Hunter-Milton 2003 similarity."""
    a = np.ascontiguousarray(times_a, dtype=np.float64)
    b = np.ascontiguousarray(times_b, dtype=np.float64)
    if _HAS_RUST and _ssc is not None:
        return float(_ssc.py_hunter_milton(a, b, dt_max))
    if a.size == 0 or b.size == 0:
        return 0.0
    count = 0
    total = a.size + b.size
    for t in a:
        if np.min(np.abs(b - t)) < dt_max:
            count += 1
    for t in b:
        if np.min(np.abs(a - t)) < dt_max:
            count += 1
    return float(count / total)


def earth_movers_distance(
    times_a: np.ndarray[Any, Any],
    times_b: np.ndarray[Any, Any],
    t_start: float = 0.0,
    t_end: float = 1.0,
    n_bins: int = 100,
) -> float:
    """Earth mover's distance between spike time distributions. Rubner et al. 1998."""
    edges = np.linspace(t_start, t_end, n_bins + 1)
    ha = np.histogram(times_a, bins=edges)[0].astype(np.float64)
    hb = np.histogram(times_b, bins=edges)[0].astype(np.float64)
    sa = ha.sum()
    sb = hb.sum()
    if sa > 0:
        ha /= sa
    if sb > 0:
        hb /= sb
    return float(np.sum(np.abs(np.cumsum(ha) - np.cumsum(hb))) * (t_end - t_start) / n_bins)


def multi_neuron_victor_purpura(
    spike_times_list: list[np.ndarray[Any, Any]], cost_per_s: float = 1000.0
) -> np.ndarray[Any, Any]:
    """All-pairs Victor-Purpura distance matrix."""
    if _HAS_RUST and _ssc is not None:
        arrs = [np.ascontiguousarray(s, dtype=np.float64) for s in spike_times_list]
        flat = _ssc.py_multi_neuron_vp(arrs, cost_per_s)
        n = len(spike_times_list)
        return np.asarray(flat).reshape(n, n)
    n = len(spike_times_list)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = victor_purpura_distance(spike_times_list[i], spike_times_list[j], cost_per_s)
            mat[i, j] = mat[j, i] = d
    return mat


def generalized_victor_purpura(
    times_a: np.ndarray[Any, Any],
    times_b: np.ndarray[Any, Any],
    cost_func: Callable[[float], float] | None = None,
) -> float:
    """Generalized Victor-Purpura with arbitrary cost function. Victor & Purpura 1997.

    cost_func(dt) returns the cost of shifting a spike by dt seconds.
    Default: linear cost q*|dt| with q=1000.
    """
    if cost_func is None:

        def cost_func(delta_t: float) -> float:
            return 1000.0 * abs(delta_t)

    na, nb = len(times_a), len(times_b)
    if na == 0:
        return float(nb)
    if nb == 0:
        return float(na)
    d = np.zeros((na + 1, nb + 1))
    for i in range(na + 1):
        d[i, 0] = float(i)
    for j in range(nb + 1):
        d[0, j] = float(j)
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            shift = cost_func(times_a[i - 1] - times_b[j - 1])
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + shift)
    return float(d[na, nb])


def spike_distance_matrix(
    spike_times_list: list[np.ndarray[Any, Any]],
    metric: str = "spike_distance",
    t_start: float = 0.0,
    t_end: float = 1.0,
) -> np.ndarray[Any, Any]:
    """All-pairs spike train distance matrix.

    metric: 'spike_distance', 'spike_sync', 'victor_purpura'.
    """
    _F = Callable[[np.ndarray[Any, Any], np.ndarray[Any, Any]], float]
    funcs: dict[str, _F] = {
        "spike_distance": lambda a, b: spike_distance(a, b, t_start, t_end),
        "spike_sync": lambda a, b: 1.0 - spike_sync(a, b, t_start, t_end),
        "victor_purpura": lambda a, b: victor_purpura_distance(a, b),
    }
    f: _F = funcs.get(metric, funcs["spike_distance"])
    n = len(spike_times_list)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = f(spike_times_list[i], spike_times_list[j])
            mat[i, j] = mat[j, i] = d
    return mat
