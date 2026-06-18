# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Basic spike train operations: spike extraction, ISI,

"""Basic spike train operations: spike extraction, ISI, rate, count, binning."""

from __future__ import annotations

from typing import Any
import numpy as np


def spike_times(binary_train: np.ndarray[Any, Any], dt: float = 0.001) -> np.ndarray[Any, Any]:
    """Extract spike times (seconds) from a binary 0/1 array."""
    return np.where(binary_train > 0)[0] * dt


def isi(binary_train: np.ndarray[Any, Any], dt: float = 0.001) -> np.ndarray[Any, Any]:
    """Inter-spike intervals (seconds) from a binary train."""
    times = spike_times(binary_train, dt)
    if times.size < 2:
        return np.array([], dtype=np.float64)
    return np.diff(times)


def firing_rate(binary_train: np.ndarray[Any, Any], dt: float = 0.001) -> float:
    """Mean firing rate (Hz)."""
    duration = binary_train.size * dt
    if duration <= 0:
        return 0.0
    return float(np.sum(binary_train) / duration)


def spike_count(binary_train: np.ndarray[Any, Any]) -> int:
    """Return the total number of spikes in a binary spike train."""
    return int(np.sum(binary_train))


def bin_spike_train(binary_train: np.ndarray[Any, Any], bin_size: int = 10) -> np.ndarray[Any, Any]:
    """Bin a binary spike train into spike counts per bin."""
    n = binary_train.size
    n_bins = n // bin_size
    if n_bins == 0:
        return np.array([int(binary_train.sum())])
    trimmed = binary_train[: n_bins * bin_size]
    return trimmed.reshape(n_bins, bin_size).sum(axis=1)
