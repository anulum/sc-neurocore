# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Basic spike-train operations: spike times, ISI, rate, count, binning

"""Basic spike-train operations: spike times, inter-spike intervals, rate, count and binning.

Every function takes a binary spike train: a one-dimensional array with one
entry per time step, where a positive entry marks a spike. ``dt`` is the time
step in seconds.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _require_finite_train(binary_train: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Validate that ``binary_train`` contains only finite real or integer samples."""
    arr = np.asarray(binary_train)
    if arr.dtype == np.object_ or np.iscomplexobj(arr) or not np.all(np.isfinite(arr)):
        raise ValueError("binary_train must contain only finite real numeric values")
    return arr


def _require_non_negative_dt(dt: float) -> float:
    """Validate that ``dt`` is a finite non-negative real number (not ``bool``)."""
    if (
        isinstance(dt, bool)
        or not isinstance(dt, (int, float, np.integer, np.floating))
        or not math.isfinite(float(dt))
        or float(dt) < 0.0
    ):
        raise ValueError(f"dt must be a finite non-negative real number, got {dt!r}")
    return float(dt)


def _require_positive_dt(dt: float) -> float:
    """Validate that ``dt`` is a finite strictly positive real number (not ``bool``)."""
    if (
        isinstance(dt, bool)
        or not isinstance(dt, (int, float, np.integer, np.floating))
        or not math.isfinite(float(dt))
        or float(dt) <= 0.0
    ):
        raise ValueError(f"dt must be a finite positive real number, got {dt!r}")
    return float(dt)


def spike_times(binary_train: np.ndarray[Any, Any], dt: float = 0.001) -> np.ndarray[Any, Any]:
    """Return the spike times of a binary spike train, in seconds.

    Parameters
    ----------
    binary_train : numpy.ndarray
        One entry per time step; a positive entry marks a spike.
    dt : float, default=0.001
        Non-negative time step in seconds.

    Returns
    -------
    numpy.ndarray
        Index of every spiking step multiplied by ``dt``, in increasing order.

    Raises
    ------
    ValueError
        If ``binary_train`` contains non-finite values or ``dt`` is negative,
        boolean, or non-finite.
    """
    arr = _require_finite_train(binary_train)
    step_s = _require_non_negative_dt(dt)
    return np.where(arr > 0)[0] * step_s


def isi(binary_train: np.ndarray[Any, Any], dt: float = 0.001) -> np.ndarray[Any, Any]:
    """Return the inter-spike intervals of a binary spike train, in seconds.

    Parameters
    ----------
    binary_train : numpy.ndarray
        One entry per time step; a positive entry marks a spike.
    dt : float, default=0.001
        Non-negative time step in seconds.

    Returns
    -------
    numpy.ndarray
        Differences between consecutive spike times; an empty ``float64``
        array when the train holds fewer than two spikes.

    Raises
    ------
    ValueError
        If ``binary_train`` contains non-finite values or ``dt`` is negative,
        boolean, or non-finite.
    """
    times = spike_times(binary_train, dt)
    if times.size < 2:
        return np.array([], dtype=np.float64)
    return np.diff(times)


def firing_rate(binary_train: np.ndarray[Any, Any], dt: float = 0.001) -> float:
    """Return the mean firing rate of a binary spike train, in hertz.

    Parameters
    ----------
    binary_train : numpy.ndarray
        One entry per time step; each entry is counted as that many spikes.
    dt : float, default=0.001
        Non-negative time step in seconds.

    Returns
    -------
    float
        Sum of the train divided by its duration ``binary_train.size * dt``;
        ``0.0`` when that duration is zero.

    Raises
    ------
    ValueError
        If ``binary_train`` contains non-finite values or ``dt`` is negative,
        boolean, or non-finite.
    """
    arr = _require_finite_train(binary_train)
    step_s = _require_non_negative_dt(dt)
    duration = arr.size * step_s
    if duration <= 0.0:
        return 0.0
    return float(np.sum(arr) / duration)


def spike_count(binary_train: np.ndarray[Any, Any]) -> int:
    """Return the total number of spikes in a binary spike train.

    Parameters
    ----------
    binary_train : numpy.ndarray
        One entry per time step; each entry is counted as that many spikes.

    Returns
    -------
    int
        Sum of the train.

    Raises
    ------
    ValueError
        If ``binary_train`` contains non-finite values.
    """
    arr = _require_finite_train(binary_train)
    return int(np.sum(arr))


def bin_spike_train(binary_train: np.ndarray[Any, Any], bin_size: int = 10) -> np.ndarray[Any, Any]:
    """Return the spike count in each consecutive bin of ``bin_size`` steps.

    Parameters
    ----------
    binary_train : numpy.ndarray
        One entry per time step; each entry is counted as that many spikes.
    bin_size : int, default=10
        Number of time steps per bin; must be a positive integer.

    Returns
    -------
    numpy.ndarray
        One count per complete bin; steps after the last complete bin are
        dropped. When the train is shorter than one bin, a single-element
        array holding the count of the whole train.

    Raises
    ------
    ValueError
        If ``binary_train`` contains non-finite values or ``bin_size`` is
        boolean, non-integer, or less than 1.
    """
    arr = _require_finite_train(binary_train)
    if (
        isinstance(bin_size, bool)
        or not isinstance(bin_size, (int, np.integer))
        or int(bin_size) < 1
    ):
        raise ValueError(f"bin_size must be a positive integer, got {bin_size!r}")
    step_bin = int(bin_size)
    n = arr.size
    n_bins = n // step_bin
    if n_bins == 0:
        return np.array([int(arr.sum())])
    trimmed = arr[: n_bins * step_bin]
    binned_counts: np.ndarray[Any, Any] = trimmed.reshape(n_bins, step_bin).sum(axis=1)
    return binned_counts
