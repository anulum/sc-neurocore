# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Online LIF layer and generic online trainer

"""Online learning building blocks: OnlineLIFLayer with eligibility-based updates,
and OnlineTrainer that composes layers into a trainable feedforward network.

All computation is O(1) in sequence length — no stored activations, no BPTT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class OnlineLIFLayer:
    """Single LIF layer with online (eligibility-based) learning.

    Parameters
    ----------
    n_inputs : int
    n_neurons : int
    tau_mem : float
        Membrane time constant.
    threshold : float
    lr : float
        Learning rate for local weight updates.
    """

    n_inputs: int
    n_neurons: int
    tau_mem: float = 20.0
    threshold: float = 1.0
    lr: float = 0.01
    dt: float = 1.0

    W: np.ndarray[Any, Any] = field(init=False, repr=False)
    _v: np.ndarray[Any, Any] = field(init=False, repr=False)
    _spikes: np.ndarray[Any, Any] = field(init=False, repr=False)
    _trace: np.ndarray[Any, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        rng = np.random.RandomState(42)
        self.W = rng.randn(self.n_neurons, self.n_inputs) * np.sqrt(2.0 / self.n_inputs)
        self.reset()

    def reset(self) -> None:
        self._v = np.zeros(self.n_neurons)
        self._spikes = np.zeros(self.n_neurons)
        self._trace = np.zeros((self.n_neurons, self.n_inputs))

    def step(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Forward one timestep. Returns spike vector."""
        alpha = np.exp(-self.dt / self.tau_mem)
        current = self.W @ x
        self._v = alpha * self._v + (1 - alpha) * current
        self._spikes = (self._v >= self.threshold).astype(np.float64)
        self._v -= self._spikes * self.threshold

        # Update eligibility trace
        pseudo = 1.0 / (1.0 + np.abs(self._v - self.threshold) * 5) ** 2
        self._trace = 0.95 * self._trace + np.outer(pseudo, x)
        return self._spikes

    def apply_learning_signal(self, signal: np.ndarray[Any, Any]) -> None:
        """Apply a top-down learning signal to update weights.

        Parameters
        ----------
        signal : ndarray of shape (n_neurons,)
            Per-neuron learning signal (e.g., error backprojected from output).
        """
        dW = np.outer(signal, np.ones(self.n_inputs)) * self._trace
        self.W -= self.lr * dW


@dataclass
class OnlineTrainer:
    """Feedforward online trainer: stacks OnlineLIFLayers with eligibility learning.

    Parameters
    ----------
    layer_sizes : list of int
        [n_input, n_hidden1, ..., n_output]
    tau_mem : float
    threshold : float
    lr : float
    """

    layer_sizes: list[int]
    tau_mem: float = 20.0
    threshold: float = 1.0
    lr: float = 0.01

    layers: list[OnlineLIFLayer] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.layers = []
        for i in range(len(self.layer_sizes) - 1):
            self.layers.append(
                OnlineLIFLayer(
                    n_inputs=self.layer_sizes[i],
                    n_neurons=self.layer_sizes[i + 1],
                    tau_mem=self.tau_mem,
                    threshold=self.threshold,
                    lr=self.lr,
                )
            )

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()

    def step(self, x: np.ndarray[Any, Any], target: np.ndarray[Any, Any] | None = None) -> dict[str, Any]:
        """Forward one timestep through all layers with optional learning.

        Parameters
        ----------
        x : ndarray of shape (n_input,)
        target : ndarray of shape (n_output,), optional

        Returns
        -------
        dict with 'output' (final layer spikes) and optionally 'loss'
        """
        h = x
        for layer in self.layers:
            h = layer.step(h)

        result: dict[str, Any] = {"output": h.copy()}

        if target is not None:
            error = h - target
            result["loss"] = 0.5 * float(np.sum(error**2))
            # Propagate learning signal backward through layers
            signal = error
            for layer in reversed(self.layers):
                layer.apply_learning_signal(signal)
                signal = layer.W.T @ signal  # project to previous layer

        return result

    def train_sequence(self, inputs: np.ndarray[Any, Any], targets: np.ndarray[Any, Any]) -> float:
        """Train on one sequence, return mean loss."""
        self.reset()
        total_loss = 0.0
        T: int = int(inputs.shape[0])
        for t in range(T):
            result = self.step(inputs[t], target=targets[t])
            total_loss += float(result.get("loss", 0.0))
        return total_loss / T

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def memory_per_step(self) -> int:
        """Total parameters stored per timestep (O(1) in T)."""
        return sum(
            layer.n_neurons + layer.n_neurons + layer.n_neurons * layer.n_inputs
            for layer in self.layers
        )
