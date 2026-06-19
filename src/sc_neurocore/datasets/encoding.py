# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike encoding utilities for converting analog signals

"""Spike encoding utilities for converting analog signals to spike trains."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def poisson_encode(
    rates: npt.ArrayLike,
    T: int,
    dt_ms: float = 1.0,
    seed: int | None = None,
) -> np.ndarray[Any, Any]:
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
    strict: bool = True,
) -> np.ndarray[Any, Any]:
    """Convert normalised values in [0, 1] to first-spike-time trains.

    Higher values spike earlier. Each neuron fires exactly once.

    Parameters
    ----------
    values : array_like, shape (N,)
        Input values, expected in ``[0, 1]``.
    T : int
        Number of timesteps.
    tau : float
        Time constant controlling the spike-time spread.
    strict : bool
        If True (default), raise ``ValueError`` when any value lies
        outside ``[0, 1]``. If False, silently clip the resulting
        spike times to ``[0, T-1]`` (the legacy behaviour). The
        clip happens regardless of ``strict``; this flag controls
        only whether the function raises before clipping.

    Returns
    -------
    spikes : ndarray, shape (T, N), dtype bool

    Raises
    ------
    ValueError
        If ``strict=True`` (default) and any element of ``values``
        is outside ``[0, 1]``.
    """
    values = np.asarray(values, dtype=np.float64)
    if strict and (values.min() < 0.0 or values.max() > 1.0):
        bad_min = float(values.min())
        bad_max = float(values.max())
        raise ValueError(
            f"latency_encode: values must be in [0, 1] when strict=True; "
            f"got min={bad_min}, max={bad_max}. Pass strict=False to "
            f"accept the legacy silent-clip behaviour."
        )
    # spike_time = tau * (1 - value); higher value => earlier spike
    spike_times = np.clip(tau * (1.0 - values), 0, T - 1).astype(int)
    spikes = np.zeros((T, values.shape[0]), dtype=bool)
    neuron_idx = np.arange(values.shape[0])
    spikes[spike_times, neuron_idx] = True
    return spikes
