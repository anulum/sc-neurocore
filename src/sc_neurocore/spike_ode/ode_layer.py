# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spiking Neural ODE layer

"""Continuous-depth SNN layer combining ODE solver with spike events.

Solves the LIF membrane ODE continuously, detects threshold crossings
as events, emits spikes, resets, continues. Adaptive step-size Euler
with event detection.

The implementation is an event-detected continuous-depth SNN layer for
experiments that need ODE integration and spike-reset semantics in one
component.

Reference: EventProp (Wunderlich & Pehle 2021)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ODELIFDynamics:
    """LIF membrane ODE dynamics.

    dv/dt = -(v - v_rest) / tau_mem + I(t) / C_mem

    Parameters
    ----------
    tau_mem : float
        Membrane time constant (ms).
    v_rest : float
    v_threshold : float
    v_reset : float
    C_mem : float
        Membrane capacitance (normalized).
    """

    tau_mem: float = 20.0
    v_rest: float = 0.0
    v_threshold: float = 1.0
    v_reset: float = 0.0
    C_mem: float = 1.0

    def dvdt(self, v: np.ndarray[Any, Any], I: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Compute membrane voltage derivative."""
        return -(v - self.v_rest) / self.tau_mem + I / self.C_mem


class SpikingODELayer:
    """Spiking Neural ODE layer with event-driven integration.

    Integrates the membrane ODE with adaptive Euler stepping.
    Detects threshold crossings via bisection, emits spikes, resets.

    Parameters
    ----------
    n_inputs : int
    n_neurons : int
    dynamics : ODELIFDynamics
    dt_init : float
        Initial integration step size.
    dt_min : float
        Minimum step size.
    max_steps_per_interval : int
        Max ODE steps per simulation interval.
    seed : int
    """

    def __init__(
        self,
        n_inputs: int,
        n_neurons: int,
        dynamics: ODELIFDynamics | None = None,
        dt_init: float = 0.1,
        dt_min: float = 0.001,
        max_steps_per_interval: int = 100,
        seed: int = 42,
    ):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.dynamics = dynamics or ODELIFDynamics()
        self.dt_init = dt_init
        self.dt_min = dt_min
        self.max_steps = max_steps_per_interval

        rng = np.random.RandomState(seed)
        self.W = rng.randn(n_neurons, n_inputs) * np.sqrt(2.0 / n_inputs)
        self._v = np.full(n_neurons, self.dynamics.v_rest)

    def step(self, x: np.ndarray[Any, Any], interval: float = 1.0) -> np.ndarray[Any, Any]:
        """Integrate ODE over one interval, return spike counts.

        Parameters
        ----------
        x : ndarray of shape (n_inputs,)
            Input (constant over interval).
        interval : float
            Duration of this interval (ms).

        Returns
        -------
        ndarray of shape (n_neurons,)
            Spike count per neuron during interval.
        """
        I = self.W @ x
        spike_counts = np.zeros(self.n_neurons)
        t = 0.0
        dt = self.dt_init
        steps = 0

        while t < interval and steps < self.max_steps:
            dt = min(dt, interval - t)
            if dt < self.dt_min:
                break

            # Euler step
            dv = self.dynamics.dvdt(self._v, I)
            v_new = self._v + dt * dv

            # Event detection: threshold crossing
            crossed = v_new >= self.dynamics.v_threshold
            if crossed.any():
                # Bisection to find exact crossing time
                for _ in range(5):  # 5 bisection steps
                    dt_half = dt / 2
                    v_mid = self._v + dt_half * dv
                    still_crossed = v_mid >= self.dynamics.v_threshold
                    if still_crossed.any():
                        dt = dt_half
                        v_new = v_mid
                    else:
                        break

                spike_counts[crossed] += 1
                v_new[crossed] = self.dynamics.v_reset

            self._v = v_new  # type: ignore[assignment]
            t += dt
            steps += 1

            # Adaptive step: increase if no spikes, decrease near threshold
            distance_to_thresh = self.dynamics.v_threshold - self._v
            min_dist = distance_to_thresh.min()
            if min_dist < 0.1 * self.dynamics.v_threshold:
                dt = max(dt * 0.5, self.dt_min)
            else:
                dt = min(dt * 1.5, self.dt_init)

        return spike_counts

    def forward(self, inputs: np.ndarray[Any, Any], interval: float = 1.0) -> np.ndarray[Any, Any]:
        """Process a sequence of inputs.

        Parameters
        ----------
        inputs : ndarray of shape (T, n_inputs)
        interval : float
            Duration per input step.

        Returns
        -------
        ndarray of shape (T, n_neurons), spike counts per interval
        """
        self.reset()
        T = inputs.shape[0]
        outputs = np.zeros((T, self.n_neurons))
        for t in range(T):
            outputs[t] = self.step(inputs[t], interval)
        return outputs

    def reset(self) -> None:
        """Reset membrane voltages to the configured resting potential."""
        self._v = np.full(self.n_neurons, self.dynamics.v_rest)

    @property
    def voltage(self) -> np.ndarray[Any, Any]:
        """Return a defensive copy of the current membrane voltages."""
        return self._v.copy()
