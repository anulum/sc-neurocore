# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — e-prop: eligibility propagation for O(1) memory training

"""E-prop: biologically plausible online learning for recurrent SNNs.

Three-factor learning rule: dW = learning_signal * eligibility_trace * modulation.
Eligibility traces track per-synapse contribution to output without storing
full activation history. Memory: O(n_synapses) regardless of sequence length.

Reference: Bellec et al. 2020 — "A solution to the learning dilemma for
recurrent networks of spiking neurons" (Nature Communications)

Caveat: e-prop is an approximation of BPTT. Accuracy may be lower on
tasks requiring very long-range credit assignment (>100 steps).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EpropTrainer:
    """E-prop online trainer for a single-layer recurrent SNN.

    Parameters
    ----------
    n_inputs : int
        Input dimension.
    n_neurons : int
        Number of LIF neurons.
    n_outputs : int
        Output dimension.
    tau_mem : float
        Membrane time constant (ms).
    tau_trace : float
        Eligibility trace decay time constant (ms).
    threshold : float
        Spike threshold.
    lr : float
        Learning rate.
    dt : float
        Timestep (ms).
    """

    n_inputs: int
    n_neurons: int
    n_outputs: int
    tau_mem: float = 20.0
    tau_trace: float = 20.0
    threshold: float = 1.0
    lr: float = 0.01
    dt: float = 1.0

    # Learned weights
    W_in: np.ndarray[Any, Any] = field(init=False, repr=False)
    W_rec: np.ndarray[Any, Any] = field(init=False, repr=False)
    W_out: np.ndarray[Any, Any] = field(init=False, repr=False)

    # Internal state
    _v: np.ndarray[Any, Any] = field(init=False, repr=False)
    _spikes: np.ndarray[Any, Any] = field(init=False, repr=False)
    _trace_in: np.ndarray[Any, Any] = field(init=False, repr=False)
    _trace_rec: np.ndarray[Any, Any] = field(init=False, repr=False)
    _eligibility_in: np.ndarray[Any, Any] = field(init=False, repr=False)
    _eligibility_rec: np.ndarray[Any, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        rng = np.random.RandomState(42)
        scale_in = np.sqrt(2.0 / self.n_inputs)
        scale_rec = np.sqrt(2.0 / self.n_neurons)
        self.W_in = rng.randn(self.n_neurons, self.n_inputs) * scale_in
        self.W_rec = rng.randn(self.n_neurons, self.n_neurons) * scale_rec
        np.fill_diagonal(self.W_rec, 0)  # no self-connections
        self.W_out = rng.randn(self.n_outputs, self.n_neurons) * np.sqrt(2.0 / self.n_neurons)
        self.reset()

    def reset(self) -> None:
        """Reset all internal state and eligibility traces."""
        self._v = np.zeros(self.n_neurons)
        self._spikes = np.zeros(self.n_neurons)
        self._trace_in = np.zeros((self.n_neurons, self.n_inputs))
        self._trace_rec = np.zeros((self.n_neurons, self.n_neurons))
        self._eligibility_in = np.zeros((self.n_neurons, self.n_inputs))
        self._eligibility_rec = np.zeros((self.n_neurons, self.n_neurons))

    def step(
        self, x: np.ndarray[Any, Any], target: np.ndarray[Any, Any] | None = None
    ) -> dict[str, Any]:
        """Process one timestep with optional learning.

        Parameters
        ----------
        x : ndarray of shape (n_inputs,)
            Input spike vector or rates.
        target : ndarray of shape (n_outputs,), optional
            Target output for computing learning signal. If None, no learning.

        Returns
        -------
        dict with keys: 'spikes', 'output', 'loss' (if target given)
        """
        alpha = np.exp(-self.dt / self.tau_mem)
        kappa = np.exp(-self.dt / self.tau_trace)

        # LIF dynamics
        current = self.W_in @ x + self.W_rec @ self._spikes
        self._v = alpha * self._v + (1 - alpha) * current
        new_spikes = (self._v >= self.threshold).astype(np.float64)
        self._v -= new_spikes * self.threshold

        # Surrogate gradient: pseudo-derivative of spike function
        pseudo_deriv = 1.0 / (1.0 + np.abs(self._v - self.threshold) * 5) ** 2

        # Update eligibility traces (low-pass filtered outer products)
        self._trace_in = kappa * self._trace_in + np.outer(pseudo_deriv, x)
        self._trace_rec = kappa * self._trace_rec + np.outer(pseudo_deriv, self._spikes)
        self._eligibility_in = kappa * self._eligibility_in + self._trace_in
        self._eligibility_rec = kappa * self._eligibility_rec + self._trace_rec

        self._spikes = new_spikes

        # Readout
        output = self.W_out @ self._spikes

        result: dict[str, Any] = {"spikes": self._spikes.copy(), "output": output}

        if target is not None:
            error = output - target
            loss = 0.5 * float(np.sum(error**2))
            result["loss"] = loss

            # Learning signal: broadcast error through output weights
            learning_signal = self.W_out.T @ error  # (n_neurons,)

            # Three-factor update: learning_signal * eligibility
            dW_in = np.outer(learning_signal, np.ones(self.n_inputs)) * self._eligibility_in
            dW_rec = np.outer(learning_signal, np.ones(self.n_neurons)) * self._eligibility_rec
            dW_out = np.outer(error, self._spikes)

            self.W_in -= self.lr * dW_in
            self.W_rec -= self.lr * dW_rec
            np.fill_diagonal(self.W_rec, 0)
            self.W_out -= self.lr * dW_out

        return result

    def train_sequence(self, inputs: np.ndarray[Any, Any], targets: np.ndarray[Any, Any]) -> float:
        """Train on one sequence, return mean loss.

        Parameters
        ----------
        inputs : ndarray of shape (T, n_inputs)
        targets : ndarray of shape (T, n_outputs)

        Returns
        -------
        float
            Mean loss over sequence.
        """
        self.reset()
        total_loss = 0.0
        T: int = int(inputs.shape[0])
        for t in range(T):
            result = self.step(inputs[t], target=targets[t])
            total_loss += float(result.get("loss", 0.0))
        return total_loss / T

    def predict_sequence(self, inputs: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Run inference on a sequence without learning.

        Parameters
        ----------
        inputs : ndarray of shape (T, n_inputs)

        Returns
        -------
        ndarray of shape (T, n_outputs)
        """
        self.reset()
        T = inputs.shape[0]
        outputs = np.zeros((T, self.n_outputs))
        for t in range(T):
            result = self.step(inputs[t])
            outputs[t] = result["output"]
        return outputs

    @property
    def memory_per_step(self) -> int:
        """Memory usage per timestep in parameters (O(1) in T)."""
        return (
            self.n_neurons  # membrane voltages
            + self.n_neurons  # spikes
            + self.n_neurons * self.n_inputs * 2  # traces + eligibilities (in)
            + self.n_neurons * self.n_neurons * 2  # traces + eligibilities (rec)
        )
