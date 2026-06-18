# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Instantaneous and population firing rate estimation

"""Instantaneous and population firing rate estimation."""

from __future__ import annotations

from typing import Any
import numpy as np


def instantaneous_rate(
    binary_train: np.ndarray[Any, Any],
    dt: float = 0.001,
    kernel: str = "gaussian",
    sigma_ms: float = 10.0,
) -> np.ndarray[Any, Any]:
    """Instantaneous firing rate via kernel convolution (Hz).

    Kernels: 'gaussian', 'exponential', 'rectangular'.
    """
    n = binary_train.size
    sigma_steps = max(1, int(sigma_ms / (dt * 1000)))
    if kernel == "gaussian":
        hw = 3 * sigma_steps
        x = np.arange(-hw, hw + 1, dtype=np.float64)
        k = np.exp(-0.5 * (x / sigma_steps) ** 2)
    elif kernel == "exponential":
        hw = 5 * sigma_steps
        x = np.arange(0, hw, dtype=np.float64)
        k = np.exp(-x / sigma_steps)
    elif kernel == "rectangular":
        hw = sigma_steps
        k = np.ones(2 * hw + 1, dtype=np.float64)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    k /= k.sum() * dt
    return np.convolve(binary_train.astype(np.float64), k, mode="same")


def population_rate(
    trains: list[np.ndarray[Any, Any]], dt: float = 0.001, sigma_ms: float = 10.0
) -> np.ndarray[Any, Any]:
    """Population-level instantaneous firing rate (Hz).

    Sums all trains then applies Gaussian kernel smoothing.
    """
    if not trains:
        return np.array([])
    min_len = min(t.size for t in trains)
    total = np.zeros(min_len, dtype=np.float64)
    for t in trains:
        total += t[:min_len].astype(np.float64)
    return instantaneous_rate(total, dt=dt, kernel="gaussian", sigma_ms=sigma_ms)


def psth(
    trials: list[np.ndarray[Any, Any]], bin_ms: float = 10.0, dt: float = 0.001
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
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
