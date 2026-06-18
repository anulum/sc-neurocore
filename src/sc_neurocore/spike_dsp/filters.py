# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike-domain FIR/IIR filters

"""Spike-domain filtering: process spike trains without converting to float.

FIR: weighted sum of delayed spike trains via coincidence counting.
IIR: recursive filtering with leaky integrator (LIF-based).

All operations stay in spike/binary domain for hardware deployment.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any
import numpy as np


@dataclass
class SpikeFIR:
    """Finite Impulse Response filter in spike domain.

    Computes weighted sum of delayed input spike trains.
    Output at time t = sum(coefficients[k] * input[t-k]) > threshold.

    Parameters
    ----------
    coefficients : ndarray
        FIR tap weights.
    threshold : float
        Output spike threshold (on weighted sum).
    """

    coefficients: np.ndarray[Any, Any]
    threshold: float = 0.5

    def filter(self, spikes: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Apply FIR filter to spike train.

        Parameters
        ----------
        spikes : ndarray of shape (T,) or (T, N)

        Returns
        -------
        ndarray of same shape, binary
        """
        if spikes.ndim == 1:
            spikes = spikes[:, np.newaxis]
        T, N = spikes.shape
        K = len(self.coefficients)
        output = np.zeros_like(spikes, dtype=np.int8)

        for t in range(K, T):
            weighted = np.zeros(N, dtype=np.float64)
            for k, c in enumerate(self.coefficients):
                weighted += c * spikes[t - k].astype(np.float64)
            output[t] = (weighted >= self.threshold).astype(np.int8)

        return output if output.shape[1] > 1 else output[:, 0]


@dataclass
class SpikeIIR:
    """Infinite Impulse Response filter using leaky integration.

    Membrane-like dynamics: state = decay * state + input_spike.
    Fires when state exceeds threshold. Natural IIR in spike domain.

    Parameters
    ----------
    decay : float
        Decay factor per timestep (0-1).
    threshold : float
    gain : float
        Input spike gain.
    """

    decay: float = 0.9
    threshold: float = 1.0
    gain: float = 0.5

    def filter(self, spikes: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Apply IIR filter to spike train."""
        if spikes.ndim == 1:
            spikes = spikes[:, np.newaxis]
        T, N = spikes.shape
        state = np.zeros(N, dtype=np.float64)
        output = np.zeros_like(spikes, dtype=np.int8)

        for t in range(T):
            state = self.decay * state + self.gain * spikes[t].astype(np.float64)
            fire = state >= self.threshold
            output[t] = fire.astype(np.int8)
            state[fire] = 0.0

        return output if output.shape[1] > 1 else output[:, 0]


def spike_convolve(
    spikes: np.ndarray[Any, Any], kernel: np.ndarray[Any, Any], threshold: float = 0.5
) -> np.ndarray[Any, Any]:
    """Convolve a spike train with a kernel in spike domain.

    Parameters
    ----------
    spikes : ndarray of shape (T,)
    kernel : ndarray of shape (K,)
    threshold : float

    Returns
    -------
    ndarray of shape (T,), binary
    """
    fir = SpikeFIR(coefficients=kernel, threshold=threshold)
    return fir.filter(spikes)
