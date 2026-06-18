# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Point process models and hazard functions

"""Point process models and hazard functions."""

from __future__ import annotations

from typing import Any
import numpy as np

from .basic import isi


def conditional_intensity(
    binary_train: np.ndarray[Any, Any], dt: float = 0.001, window_ms: float = 50.0
) -> np.ndarray[Any, Any]:
    """Conditional intensity function estimate (Hz). Brown et al. 2004.

    Moving-window MLE of the Poisson rate at each time step.
    """
    w = max(1, int(window_ms / (dt * 1000)))
    x = binary_train.astype(np.float64)
    kernel = np.ones(w) / (w * dt)
    return np.convolve(x, kernel, mode="same")


def isi_hazard_function(
    binary_train: np.ndarray[Any, Any], dt: float = 0.001, bins: int = 30
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """ISI hazard function h(t) = f(t) / S(t). Tuckwell 1988.

    Returns (hazard, bin_centers) where hazard is the failure rate at each ISI duration.
    """
    intervals = isi(binary_train, dt)
    if intervals.size < 5:
        return np.array([]), np.array([])
    hist, edges = np.histogram(intervals, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    pdf = hist.astype(np.float64) / (intervals.size * (edges[1] - edges[0]))
    survivor = 1.0 - np.cumsum(pdf) * (edges[1] - edges[0])
    survivor = np.clip(survivor, 1e-30, None)
    hazard = pdf / survivor
    return hazard, centers


def isi_survivor_function(
    binary_train: np.ndarray[Any, Any], dt: float = 0.001, bins: int = 30
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """ISI survivor function S(t) = P(ISI > t). Tuckwell 1988.

    Returns (survivor, bin_centers).
    """
    intervals = isi(binary_train, dt)
    if intervals.size < 2:
        return np.array([]), np.array([])
    sorted_isi = np.sort(intervals)
    n = sorted_isi.size
    edges = np.linspace(0, sorted_isi[-1], bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    survivor = np.array([np.sum(sorted_isi > t) / n for t in centers])
    return survivor, centers


def renewal_density(
    binary_train: np.ndarray[Any, Any], dt: float = 0.001, bins: int = 30
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Renewal density h(t) from ISI distribution. Cox 1962.

    Returns (density, bin_centers). Density normalized by mean rate.
    """
    intervals = isi(binary_train, dt)
    if intervals.size < 5:
        return np.array([]), np.array([])
    hist, edges = np.histogram(intervals, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    mean_rate = 1.0 / intervals.mean() if intervals.mean() > 0 else 1.0
    return hist / mean_rate, centers
