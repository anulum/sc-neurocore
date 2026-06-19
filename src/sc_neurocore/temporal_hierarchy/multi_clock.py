# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Multi-timescale / multi-clock SNN

"""Multi-timescale SNN with per-synapse learnable time constants and
multi-clock scheduling.

HetSynLayer: per-synapse time constants following log-normal distribution
(matching Allen Institute data). Different synapses integrate over different
temporal windows, enabling a single layer to capture both fast transients
and slow trends.

MultiClockSNN: different layers run at different temporal resolutions.
Fast sensory layers tick every step, slow cognitive layers tick every
N steps. Clock-domain crossing buffers handle inter-layer communication.

Reference: HetSyn (NeurIPS 2025)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ClockDomain:
    """One clock domain in a multi-clock SNN.

    Parameters
    ----------
    name : str
    tick_interval : int
        Steps between updates (1 = every step, 10 = every 10th step).
    layers : list of str
        Layer names assigned to this clock domain.
    """

    name: str
    tick_interval: int = 1
    layers: list[str] = field(default_factory=list)


class HetSynLayer:
    """Layer with heterogeneous per-synapse time constants.

    Each synapse has its own tau, initialized log-normally (mean=5ms, std=1ms).
    The synaptic trace at each synapse decays at its own rate:
      trace[i,j] = exp(-dt/tau[i,j]) * trace[i,j] + input_spike[j]

    Parameters
    ----------
    n_inputs : int
    n_neurons : int
    tau_mean : float
        Mean synaptic time constant (ms).
    tau_std : float
        Std of log(tau) for log-normal initialization.
    threshold : float
    seed : int
    """

    def __init__(
        self,
        n_inputs: int,
        n_neurons: int,
        tau_mean: float = 5.0,
        tau_std: float = 1.0,
        threshold: float = 1.0,
        seed: int = 42,
    ):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.threshold = threshold

        rng = np.random.RandomState(seed)
        # Per-synapse time constants (log-normal)
        log_tau = np.log(tau_mean) + tau_std * rng.randn(n_neurons, n_inputs)
        self.tau = np.exp(log_tau)
        self.tau = np.clip(self.tau, 0.5, 100.0)

        self.W = rng.randn(n_neurons, n_inputs) * np.sqrt(2.0 / n_inputs)
        self._traces = np.zeros((n_neurons, n_inputs))
        self._v = np.zeros(n_neurons)

    def step(self, x: np.ndarray[Any, Any], dt: float = 1.0) -> np.ndarray[Any, Any]:
        """Process one timestep.

        Parameters
        ----------
        x : ndarray of shape (n_inputs,)
        dt : float

        Returns
        -------
        ndarray of shape (n_neurons,), binary spikes
        """
        decay = np.exp(-dt / self.tau)
        self._traces = decay * self._traces + x[np.newaxis, :]
        current = (self.W * self._traces).sum(axis=1)
        self._v += current
        spikes = (self._v >= self.threshold).astype(np.float64)
        self._v -= spikes * self.threshold
        return spikes

    def reset(self) -> None:
        self._traces = np.zeros((self.n_neurons, self.n_inputs))
        self._v = np.zeros(self.n_neurons)

    @property
    def tau_stats(self) -> dict[str, float]:
        return {
            "mean": float(self.tau.mean()),
            "std": float(self.tau.std()),
            "min": float(self.tau.min()),
            "max": float(self.tau.max()),
            "median": float(np.median(self.tau)),
        }


class MultiClockSNN:
    """Multi-clock SNN with different temporal resolutions per layer.

    Parameters
    ----------
    layers : list of HetSynLayer
        Network layers.
    clock_domains : list of ClockDomain
        Clock domain assignments.
    """

    def __init__(
        self,
        layers: list[HetSynLayer],
        layer_names: list[str],
        clock_intervals: list[int] | None = None,
    ):
        self.layers = layers
        self.layer_names = layer_names
        if clock_intervals is None:
            clock_intervals = [1] * len(layers)
        self.clock_intervals = clock_intervals
        self._step_count = 0
        self._last_outputs: list[np.ndarray[Any, Any]] = [np.zeros(l.n_neurons) for l in layers]

    def step(self, x: np.ndarray[Any, Any], dt: float = 1.0) -> np.ndarray[Any, Any]:
        """Process one global timestep.

        Layers only update when their clock ticks.
        Between ticks, they hold their last output.

        Parameters
        ----------
        x : ndarray of shape (n_input,)

        Returns
        -------
        ndarray of shape (n_output,), final layer spikes
        """
        self._step_count += 1
        h = x.astype(np.float64)

        for i, (layer, interval) in enumerate(zip(self.layers, self.clock_intervals)):
            if self._step_count % interval == 0:
                spikes = layer.step(h, dt=dt * interval)
                self._last_outputs[i] = spikes
            h = self._last_outputs[i]

        return h

    def run(self, inputs: np.ndarray[Any, Any], dt: float = 1.0) -> np.ndarray[Any, Any]:
        """Run full sequence.

        Parameters
        ----------
        inputs : ndarray of shape (T, n_input)

        Returns
        -------
        ndarray of shape (T, n_output)
        """
        self.reset()
        T = inputs.shape[0]
        n_out = self.layers[-1].n_neurons
        outputs = np.zeros((T, n_out))
        for t in range(T):
            outputs[t] = self.step(inputs[t], dt)
        return outputs

    def reset(self) -> None:
        self._step_count = 0
        for i, layer in enumerate(self.layers):
            layer.reset()
            self._last_outputs[i] = np.zeros(layer.n_neurons)
