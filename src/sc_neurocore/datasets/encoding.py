# SPDX-License-Identifier: AGPL-3.0-or-later
"""Spike encoding utilities for converting analog signals to spike trains."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def poisson_encode(
    rates: npt.ArrayLike,
    T: int,
    dt_ms: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Convert firing-rate array to Poisson spike trains.

    Parameters
    ----------
    rates : array_like, shape (N,)
        Firing probabilities per timestep, clipped to [0, 1].
    T : int
        Number of timesteps.
    dt_ms : float
        Timestep duration in ms (scales rates linearly).
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    spikes : ndarray, shape (T, N), dtype bool
    """
    rng = np.random.default_rng(seed)
    rates = np.asarray(rates, dtype=np.float64)
    scaled = np.clip(rates * (dt_ms / 1.0), 0.0, 1.0)
    return rng.random((T, rates.shape[0])) < scaled


def latency_encode(
    values: npt.ArrayLike,
    T: int,
    tau: float = 5.0,
) -> np.ndarray:
    """Convert normalized values [0, 1] to first-spike-time encoded trains.

    Higher values spike earlier. Each neuron fires exactly once.

    Parameters
    ----------
    values : array_like, shape (N,)
        Input values in [0, 1].
    T : int
        Number of timesteps.
    tau : float
        Time constant controlling the spike-time spread.

    Returns
    -------
    spikes : ndarray, shape (T, N), dtype bool
    """
    values = np.asarray(values, dtype=np.float64)
    # spike_time = tau * (1 - value); higher value => earlier spike
    spike_times = np.clip(tau * (1.0 - values), 0, T - 1).astype(int)
    spikes = np.zeros((T, values.shape[0]), dtype=bool)
    neuron_idx = np.arange(values.shape[0])
    spikes[spike_times, neuron_idx] = True
    return spikes
