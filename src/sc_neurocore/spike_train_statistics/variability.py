# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike-train variability: ISI CV, Fano factor, permutation entropy

"""Spike-train variability: ISI coefficient of variation, Fano factor and permutation entropy.

Every function takes a binary spike train: a one-dimensional array with one
entry per time step, where a positive entry marks a spike.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .basic import _require_finite_train, _require_positive_dt, isi


def cv_isi(binary_train: np.ndarray[Any, Any], dt: float = 0.001) -> float:
    """Return the coefficient of variation of the inter-spike intervals.

    A Poisson train gives a value near 1 and a regular train a value below 1.

    Parameters
    ----------
    binary_train : numpy.ndarray
        One entry per time step; a positive entry marks a spike.
    dt : float, default=0.001
        Non-negative time step in seconds.

    Returns
    -------
    float
        Standard deviation of the intervals divided by their mean; ``nan``
        when the train has fewer than two intervals or their mean is zero.

    Raises
    ------
    ValueError
        If ``binary_train`` contains non-finite values or ``dt`` is negative,
        boolean, or non-finite.
    """
    intervals = isi(binary_train, dt)
    if intervals.size < 2:
        return float("nan")
    mu = intervals.mean()
    if mu == 0:
        return float("nan")
    return float(intervals.std() / mu)


def fano_factor(
    binary_train: np.ndarray[Any, Any], window_ms: float = 50.0, dt: float = 0.001
) -> float:
    """Return the Fano factor of the spike counts in consecutive windows.

    Parameters
    ----------
    binary_train : numpy.ndarray
        One entry per time step; each entry is counted as that many spikes.
    window_ms : float, default=50.0
        Strictly positive window length in milliseconds; at least one time step
        is used.
    dt : float, default=0.001
        Strictly positive time step in seconds.

    Returns
    -------
    float
        Variance of the window counts divided by their mean; steps after the
        last complete window are dropped. ``nan`` when the train is shorter
        than one window or the mean count is zero.

    Raises
    ------
    ValueError
        If ``binary_train`` contains non-finite values, or ``window_ms`` or
        ``dt`` is boolean, non-finite, or not positive.
    """
    arr = _require_finite_train(binary_train)
    if (
        isinstance(window_ms, bool)
        or not isinstance(window_ms, (int, float, np.integer, np.floating))
        or not math.isfinite(float(window_ms))
        or float(window_ms) <= 0.0
    ):
        raise ValueError(f"window_ms must be a finite positive real number, got {window_ms!r}")
    step_s = _require_positive_dt(dt)
    window_steps = max(1, int(float(window_ms) / (step_s * 1000.0)))
    n = arr.size
    if n < window_steps:
        return float("nan")
    n_windows = n // window_steps
    counts = arr[: n_windows * window_steps].reshape(n_windows, window_steps).sum(axis=1)
    mu = counts.mean()
    if mu == 0:
        return float("nan")
    return float(counts.var() / mu)


def permutation_entropy(
    binary_train: np.ndarray[Any, Any], order: int = 3, delay: int = 1
) -> float:
    """Return the normalised Bandt-Pompe permutation entropy (Bandt & Pompe 2002).

    Each window of ``order`` samples spaced ``delay`` steps apart is reduced to
    the ranks of its samples. Tied samples are ranked by position, earlier
    first. The Shannon entropy of the rank patterns is divided by its maximum,
    ``log2(order!)``.

    Parameters
    ----------
    binary_train : numpy.ndarray
        One entry per time step.
    order : int, default=3
        Samples per window; must be an integer at least 1.
    delay : int, default=1
        Steps between the samples of a window; must be an integer at least 1.

    Returns
    -------
    float
        Entropy in ``[0, 1]``; ``nan`` when the train has fewer than
        ``order * delay`` samples, and ``0.0`` when ``order`` is 1.

    Raises
    ------
    ValueError
        If ``binary_train`` contains non-finite values, or ``order`` or
        ``delay`` is boolean, non-integer, or less than 1.
    """
    if (
        isinstance(order, bool)
        or isinstance(delay, bool)
        or not isinstance(order, (int, np.integer))
        or not isinstance(delay, (int, np.integer))
    ):
        raise ValueError(
            f"order and delay must be positive integers, got order={order!r}, delay={delay!r}"
        )
    order_i = int(order)
    delay_i = int(delay)
    if order_i < 1 or delay_i < 1:
        raise ValueError(
            f"order and delay must be at least 1, got order={order_i}, delay={delay_i}"
        )
    arr = _require_finite_train(binary_train)
    x = np.asarray(arr, dtype=np.float64)
    if x.size < order_i * delay_i:
        return float("nan")
    windows = np.lib.stride_tricks.sliding_window_view(x, (order_i - 1) * delay_i + 1)[:, ::delay_i]
    ranks = np.argsort(np.argsort(windows, axis=1, kind="stable"), axis=1, kind="stable")
    _, counts = np.unique(ranks, axis=0, return_counts=True)
    p = counts / counts.sum()
    h = float(np.sum(p * np.log2(1.0 / p)))
    h_max = math.log2(math.factorial(order_i))
    return h / h_max if h_max > 0 else 0.0
