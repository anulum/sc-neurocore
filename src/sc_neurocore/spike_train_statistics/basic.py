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

from typing import Any

import numpy as np


def spike_times(binary_train: np.ndarray[Any, Any], dt: float = 0.001) -> np.ndarray[Any, Any]:
    """Return the spike times of a binary spike train, in seconds.

    Parameters
    ----------
    binary_train : numpy.ndarray
        One entry per time step; a positive entry marks a spike.
    dt : float
        Time step in seconds.

    Returns
    -------
    numpy.ndarray
        Index of every spiking step multiplied by ``dt``, in increasing order.
    """
    return np.where(binary_train > 0)[0] * dt


def isi(binary_train: np.ndarray[Any, Any], dt: float = 0.001) -> np.ndarray[Any, Any]:
    """Return the inter-spike intervals of a binary spike train, in seconds.

    Parameters
    ----------
    binary_train : numpy.ndarray
        One entry per time step; a positive entry marks a spike.
    dt : float
        Time step in seconds.

    Returns
    -------
    numpy.ndarray
        Differences between consecutive spike times; an empty ``float64``
        array when the train holds fewer than two spikes.
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
    dt : float
        Time step in seconds.

    Returns
    -------
    float
        Sum of the train divided by its duration ``binary_train.size * dt``;
        ``0.0`` when that duration is not positive.
    """
    duration = binary_train.size * dt
    if duration <= 0:
        return 0.0
    return float(np.sum(binary_train) / duration)


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
    """
    return int(np.sum(binary_train))


def bin_spike_train(binary_train: np.ndarray[Any, Any], bin_size: int = 10) -> np.ndarray[Any, Any]:
    """Return the spike count in each consecutive bin of ``bin_size`` steps.

    Parameters
    ----------
    binary_train : numpy.ndarray
        One entry per time step; each entry is counted as that many spikes.
    bin_size : int
        Number of time steps per bin; must be positive.

    Returns
    -------
    numpy.ndarray
        One count per complete bin; steps after the last complete bin are
        dropped. When the train is shorter than one bin, a single-element
        array holding the count of the whole train.
    """
    n = binary_train.size
    n_bins = n // bin_size
    if n_bins == 0:
        return np.array([int(binary_train.sum())])
    trimmed = binary_train[: n_bins * bin_size]
    binned_counts: np.ndarray[Any, Any] = trimmed.reshape(n_bins, bin_size).sum(axis=1)
    return binned_counts
