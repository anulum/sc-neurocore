# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Auto-critical spiking reservoir computing

"""Liquid State Machine with mean-field auto-criticality tuning.

The critical weight W_c = theta / (2 * beta * N) where theta is threshold,
beta is leak, N is neuron count. At criticality, exactly half the neurons
fire in each refractory period, maximizing computational capacity.

Zero hyperparameter tuning: `AutoCriticalReservoir(n_neurons=1000)` — done.

Reference: Scientific Reports 2025 — mean-field analytical framework
for configuring spiking reservoirs at the critical regime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ReservoirMetrics:
    """Reservoir quality metrics."""

    firing_fraction: float  # fraction of neurons active
    criticality_error: float  # |firing_fraction - 0.5|
    kernel_quality: float  # linear separability of reservoir states
    spectral_radius: float

    def summary(self) -> str:
        return (
            f"Reservoir: firing={self.firing_fraction:.3f}, "
            f"criticality_err={self.criticality_error:.4f}, "
            f"kernel_q={self.kernel_quality:.3f}, "
            f"spectral_r={self.spectral_radius:.3f}"
        )


class AutoCriticalReservoir:
    """Spiking Liquid State Machine with automatic criticality tuning.

    Parameters
    ----------
    n_inputs : int
    n_neurons : int
        Reservoir size.
    n_outputs : int
        Readout dimension.
    threshold : float
        LIF spike threshold.
    leak : float
        Membrane leak factor (0-1). Higher = faster decay.
    connectivity : float
        Fraction of possible synapses that exist (sparsity).
    seed : int
    """

    def __init__(
        self,
        n_inputs: int,
        n_neurons: int = 1000,
        n_outputs: int = 10,
        threshold: float = 1.0,
        leak: float = 0.1,
        connectivity: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.n_outputs = n_outputs
        self.threshold = threshold
        self.leak = leak
        self.connectivity = connectivity

        rng = np.random.RandomState(seed)

        # Mean-field critical weight: W_c = theta / (2 * beta * N * p)
        effective_n = n_neurons * connectivity
        self.w_critical = threshold / max(2.0 * leak * effective_n, 1e-8)

        # Reservoir weights: sparse, scaled to W_critical
        mask = rng.random((n_neurons, n_neurons)) < connectivity
        np.fill_diagonal(mask, False)
        self.W_res = rng.randn(n_neurons, n_neurons) * self.w_critical
        self.W_res *= mask

        # Input weights
        self.W_in = rng.randn(n_neurons, n_inputs) * np.sqrt(2.0 / n_inputs)

        # Readout weights (trained by ridge regression)
        self.W_out = np.zeros((n_outputs, n_neurons))

        # State
        self._v = np.zeros(n_neurons)
        self._spikes = np.zeros(n_neurons)

    @property
    def spectral_radius(self) -> float:
        eigvals = np.abs(np.linalg.eigvals(self.W_res))
        return float(eigvals.max()) if len(eigvals) > 0 else 0.0

    def reset(self) -> None:
        self._v = np.zeros(self.n_neurons)
        self._spikes = np.zeros(self.n_neurons)

    def step(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Process one timestep, return reservoir state (spikes)."""
        current = self.W_in @ x + self.W_res @ self._spikes
        self._v = (1 - self.leak) * self._v + self.leak * current
        self._spikes = (self._v >= self.threshold).astype(np.float64)  # type: ignore[assignment]
        self._v -= self._spikes * self.threshold
        return self._spikes.copy()

    def run(self, inputs: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Run input sequence through reservoir, return state matrix.

        Parameters
        ----------
        inputs : ndarray of shape (T, n_inputs)

        Returns
        -------
        ndarray of shape (T, n_neurons)
        """
        self.reset()
        T = inputs.shape[0]
        states = np.zeros((T, self.n_neurons))
        for t in range(T):
            states[t] = self.step(inputs[t])
        return states

    def fit_readout(
        self, states: np.ndarray[Any, Any], targets: np.ndarray[Any, Any], ridge: float = 1e-4
    ) -> None:
        """Train readout via ridge regression.

        Parameters
        ----------
        states : ndarray of shape (T, n_neurons)
        targets : ndarray of shape (T, n_outputs)
        ridge : float
            Regularization strength.
        """
        # W_out = targets^T @ states @ (states^T @ states + ridge*I)^{-1}
        S = states
        reg = ridge * np.eye(self.n_neurons)
        self.W_out = np.linalg.solve(S.T @ S + reg, S.T @ targets).T

    def predict(self, states: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Predict from reservoir states."""
        return states @ self.W_out.T

    def train_and_predict(
        self,
        train_inputs: np.ndarray[Any, Any],
        train_targets: np.ndarray[Any, Any],
        test_inputs: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Full pipeline: run train, fit readout, run test, predict."""
        train_states = self.run(train_inputs)
        self.fit_readout(train_states, train_targets)
        test_states = self.run(test_inputs)
        return self.predict(test_states)

    def metrics(self, inputs: np.ndarray[Any, Any]) -> ReservoirMetrics:
        """Compute reservoir quality metrics."""
        states = self.run(inputs)
        firing_fraction = float(states.mean())
        criticality_error = abs(firing_fraction - 0.5)

        # Kernel quality: rank of state matrix normalized by timesteps
        rank = np.linalg.matrix_rank(states)
        kernel_quality = rank / max(states.shape[0], 1)

        return ReservoirMetrics(
            firing_fraction=firing_fraction,
            criticality_error=criticality_error,
            kernel_quality=kernel_quality,
            spectral_radius=self.spectral_radius,
        )
