# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike-triggered analysis and receptive field estimation

"""Spike-triggered analysis and receptive field estimation."""

from __future__ import annotations

from typing import Any
import numpy as np


def spike_triggered_average(
    stimulus: np.ndarray[Any, Any], binary_train: np.ndarray[Any, Any], window_steps: int = 50
) -> np.ndarray[Any, Any]:
    """Spike-triggered average (STA) of a stimulus signal.

    Returns the average stimulus snippet preceding each spike.
    """
    times = np.where(binary_train > 0)[0]
    valid = times[times >= window_steps]
    if valid.size == 0:
        return np.zeros(window_steps, dtype=np.float64)
    snippets = np.array([stimulus[t - window_steps : t] for t in valid])
    triggered_average: np.ndarray[Any, Any] = snippets.mean(axis=0)
    return triggered_average


def spike_triggered_covariance(
    stimulus: np.ndarray[Any, Any], binary_train: np.ndarray[Any, Any], window_steps: int = 50
) -> np.ndarray[Any, Any]:
    """Spike-triggered covariance (STC). Schwartz et al. 2006.

    Returns covariance matrix of stimulus snippets preceding spikes.
    """
    times = np.where(binary_train > 0)[0]
    valid = times[times >= window_steps]
    if valid.size < 3:
        return np.eye(window_steps)
    snippets = np.array([stimulus[t - window_steps : t].astype(np.float64) for t in valid])
    return np.cov(snippets.T)


def spatial_information(
    binary_train: np.ndarray[Any, Any],
    positions: np.ndarray[Any, Any],
    n_bins: int = 20,
    dt: float = 0.001,
) -> float:
    """Spatial information (bits/spike). Skaggs et al. 1993.

    positions: 1D array of position values (same length as binary_train).
    SI = sum(p_i * r_i/r_mean * log2(r_i/r_mean)).
    """
    n = min(binary_train.size, positions.size)
    if n < 10:
        return 0.0
    pos = positions[:n]
    spk = binary_train[:n].astype(np.float64)
    edges = np.linspace(pos.min(), pos.max() + 1e-10, n_bins + 1)
    occupancy = np.zeros(n_bins)
    spike_counts = np.zeros(n_bins)
    for k in range(n_bins):
        mask = (pos >= edges[k]) & (pos < edges[k + 1])
        occupancy[k] = mask.sum() * dt
        spike_counts[k] = spk[mask].sum()
    total_occ = occupancy.sum()
    if total_occ <= 0:
        return 0.0
    p_occ = occupancy / total_occ
    rates = np.zeros(n_bins)
    for k in range(n_bins):
        rates[k] = spike_counts[k] / occupancy[k] if occupancy[k] > 0 else 0.0
    mean_rate = spk.sum() / (n * dt) if n > 0 else 0.0
    if mean_rate <= 0:
        return 0.0
    si = 0.0
    for k in range(n_bins):
        if rates[k] > 0 and p_occ[k] > 0:
            si += p_occ[k] * rates[k] / mean_rate * np.log2(rates[k] / mean_rate)
    return float(max(0.0, si))


def place_field_detection(
    binary_train: np.ndarray[Any, Any],
    positions: np.ndarray[Any, Any],
    n_bins: int = 50,
    threshold_std: float = 2.0,
    dt: float = 0.001,
) -> list[tuple[float, float]]:
    """Detect place fields as contiguous bins with rate > mean + threshold_std * std. O'Keefe & Dostrovsky 1971.

    Returns list of (field_start, field_end) position values.
    """
    n = min(binary_train.size, positions.size)
    if n < 10:
        return []
    pos = positions[:n]
    spk = binary_train[:n].astype(np.float64)
    edges = np.linspace(pos.min(), pos.max() + 1e-10, n_bins + 1)
    rates = np.zeros(n_bins)
    for k in range(n_bins):
        mask = (pos >= edges[k]) & (pos < edges[k + 1])
        occ = mask.sum() * dt
        rates[k] = spk[mask].sum() / occ if occ > 0 else 0.0
    thresh = rates.mean() + threshold_std * rates.std()
    fields = []
    in_field = False
    start = 0.0
    for k in range(n_bins):
        if rates[k] > thresh and not in_field:
            in_field = True
            start = edges[k]
        elif rates[k] <= thresh and in_field:
            in_field = False
            fields.append((start, edges[k]))
    if in_field:
        fields.append((start, edges[-1]))
    return fields


def tuning_curve(
    binary_train: np.ndarray[Any, Any],
    stimulus_values: np.ndarray[Any, Any],
    n_bins: int = 20,
    dt: float = 0.001,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Compute tuning curve: mean firing rate vs stimulus value. Dayan & Abbott 2001.

    Returns (mean_rates, bin_centers).
    """
    n = min(binary_train.size, stimulus_values.size)
    if n < 5:
        return np.array([]), np.array([])
    stim = stimulus_values[:n]
    spk = binary_train[:n].astype(np.float64)
    edges = np.linspace(stim.min(), stim.max() + 1e-10, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    rates = np.zeros(n_bins)
    for k in range(n_bins):
        mask = (stim >= edges[k]) & (stim < edges[k + 1])
        occ = mask.sum() * dt
        rates[k] = spk[mask].sum() / occ if occ > 0 else 0.0
    return rates, centers
