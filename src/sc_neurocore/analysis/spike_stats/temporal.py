# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Temporal pattern detection: bursts, latency, onset,

"""Temporal pattern detection: bursts, latency, onset, change points."""

from __future__ import annotations

from typing import Any
import numpy as np

from .basic import spike_times, bin_spike_train


def burst_detection(
    binary_train: np.ndarray[Any, Any],
    dt: float = 0.001,
    max_isi_ms: float = 10.0,
    min_spikes: int = 3,
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


def first_spike_latency(binary_train: np.ndarray[Any, Any], dt: float = 0.001) -> float:
    """Time to first spike (seconds). Returns nan if no spike."""
    idx = np.argmax(binary_train > 0)
    if binary_train[idx] == 0:
        return float("nan")
    return float(idx * dt)


def response_onset(
    binary_train: np.ndarray[Any, Any],
    baseline_steps: int = 100,
    dt: float = 0.001,
    threshold_sigma: float = 3.0,
) -> float:
    """Detect response onset as first bin exceeding baseline + threshold_sigma * std.

    Returns onset time (seconds), or nan if no response detected.
    """
    if binary_train.size <= baseline_steps:
        return float("nan")
    baseline_rate = binary_train[:baseline_steps].mean()
    baseline_std = binary_train[:baseline_steps].std()
    if baseline_std == 0:
        baseline_std = 1e-10
    threshold = baseline_rate + threshold_sigma * baseline_std
    post = binary_train[baseline_steps:]
    above = np.where(post > threshold)[0]
    if above.size == 0:
        return float("nan")
    return float((baseline_steps + above[0]) * dt)


def change_point_detection(
    binary_train: np.ndarray[Any, Any], bin_size: int = 50, threshold: float = 3.0
) -> list[int]:
    """CUSUM-based change point detection in firing rate. Page 1954.

    Returns list of bin indices where significant rate changes occur.
    """
    counts = bin_spike_train(binary_train, bin_size).astype(np.float64)
    n = counts.size
    if n < 5:
        return []
    mean_rate = counts.mean()
    std_rate = counts.std()
    if std_rate < 1e-10:
        return []
    cusum_pos = np.zeros(n)
    cusum_neg = np.zeros(n)
    change_points = []
    for i in range(1, n):
        cusum_pos[i] = max(0, cusum_pos[i - 1] + (counts[i] - mean_rate) / std_rate)
        cusum_neg[i] = max(0, cusum_neg[i - 1] - (counts[i] - mean_rate) / std_rate)
        if cusum_pos[i] > threshold or cusum_neg[i] > threshold:
            change_points.append(i)
            cusum_pos[i] = 0
            cusum_neg[i] = 0
    return change_points
