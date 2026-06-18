# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Complete spike encoding zoo

"""7 spike encoding schemes: rate, latency, delta, phase, burst, rank-order, sigma-delta.

No framework provides all 7 in one place with consistent API.
"""

from __future__ import annotations

from typing import Any
import numpy as np


def rate_encode(values: np.ndarray[Any, Any], T: int, seed: int = 42) -> np.ndarray[Any, Any]:
    """Rate coding: spike probability proportional to value.

    Parameters
    ----------
    values : ndarray of shape (N,)
        Input values in [0, 1].
    T : int
        Number of timesteps.
    seed : int

    Returns
    -------
    ndarray of shape (T, N), binary
    """
    rng = np.random.RandomState(seed)
    rates = np.clip(values, 0, 1)
    return (rng.random((T, len(rates))) < rates[np.newaxis, :]).astype(np.int8)


def latency_encode(values: np.ndarray[Any, Any], T: int) -> np.ndarray[Any, Any]:
    """Latency (Time-to-First-Spike) coding: higher value = earlier spike.

    Parameters
    ----------
    values : ndarray of shape (N,)
        Input values in [0, 1].
    T : int

    Returns
    -------
    ndarray of shape (T, N), binary
    """
    spikes = np.zeros((T, len(values)), dtype=np.int8)
    for i, v in enumerate(values):
        if v > 0:
            t_spike = max(0, int((1.0 - np.clip(v, 0, 1)) * (T - 1)))
            spikes[t_spike, i] = 1
    return spikes


def delta_encode(values: np.ndarray[Any, Any], threshold: float = 0.1) -> np.ndarray[Any, Any]:
    """Delta coding: spike when change exceeds threshold.

    Parameters
    ----------
    values : ndarray of shape (T,) or (T, N)
        Time-varying input signal.
    threshold : float

    Returns
    -------
    ndarray of same shape, binary
    """
    if values.ndim == 1:
        values = values[:, np.newaxis]
    diff = np.abs(np.diff(values, axis=0, prepend=values[:1]))
    spikes: np.ndarray[Any, Any] = (diff > threshold).astype(np.int8)
    return spikes


def phase_encode(values: np.ndarray[Any, Any], T: int, n_phases: int = 8) -> np.ndarray[Any, Any]:
    """Phase coding: value encoded as spike phase within oscillation cycle.

    Parameters
    ----------
    values : ndarray of shape (N,)
        Input values in [0, 1].
    T : int
    n_phases : int
        Number of phase bins per cycle.

    Returns
    -------
    ndarray of shape (T, N), binary
    """
    spikes = np.zeros((T, len(values)), dtype=np.int8)
    for i, v in enumerate(values):
        phase = int(np.clip(v, 0, 1) * (n_phases - 1))
        for t in range(phase, T, n_phases):
            spikes[t, i] = 1
    return spikes


def burst_encode(values: np.ndarray[Any, Any], T: int, max_burst: int = 5) -> np.ndarray[Any, Any]:
    """Burst coding: value encoded as burst length (consecutive spikes).

    Parameters
    ----------
    values : ndarray of shape (N,)
        Input values in [0, 1].
    T : int
    max_burst : int
        Maximum burst length for value=1.

    Returns
    -------
    ndarray of shape (T, N), binary
    """
    spikes = np.zeros((T, len(values)), dtype=np.int8)
    for i, v in enumerate(values):
        burst_len = max(1, int(np.clip(v, 0, 1) * max_burst))
        for t in range(min(burst_len, T)):
            spikes[t, i] = 1
    return spikes


def rank_order_encode(values: np.ndarray[Any, Any], T: int) -> np.ndarray[Any, Any]:
    """Rank-order coding: neurons fire in order of decreasing value.

    Parameters
    ----------
    values : ndarray of shape (N,)
    T : int

    Returns
    -------
    ndarray of shape (T, N), binary
    """
    N = len(values)
    spikes = np.zeros((T, N), dtype=np.int8)
    order = np.argsort(-values)  # highest first
    for rank, neuron_idx in enumerate(order):
        t = min(rank, T - 1)
        if values[neuron_idx] > 0:
            spikes[t, neuron_idx] = 1
    return spikes


def sigma_delta_encode(
    values: np.ndarray[Any, Any], threshold: float = 0.1
) -> np.ndarray[Any, Any]:
    """Sigma-delta coding: integrate error, spike when threshold exceeded.

    Parameters
    ----------
    values : ndarray of shape (T,) or (T, N)
        Time-varying signal.
    threshold : float

    Returns
    -------
    ndarray of same shape, binary
    """
    if values.ndim == 1:
        values = values[:, np.newaxis]
    T, N = values.shape
    spikes = np.zeros((T, N), dtype=np.int8)
    integrator = np.zeros(N)
    reconstructed = np.zeros(N)

    for t in range(T):
        error = values[t] - reconstructed
        integrator += error
        fire = np.abs(integrator) >= threshold
        spikes[t] = fire.astype(np.int8)
        reconstructed += np.sign(integrator) * fire * threshold
        integrator -= np.sign(integrator) * fire * threshold

    return spikes
