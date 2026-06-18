# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hardware-aware training with defect injection

"""Hardware-aware SC layer with memristive defect modeling.

Simulates stuck-at faults and weight variability during training,
enabling the network to learn around hardware defects.

    from sc_neurocore.layers.hardware_aware import HardwareAwareSCLayer

    layer = HardwareAwareSCLayer(n_inputs=8, n_neurons=4, stuck_rate=0.05)
    output = layer.forward([0.3, 0.5, 0.7, 0.2, 0.8, 0.1, 0.6, 0.4])
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any
import numpy as np

from .vectorized_layer import VectorizedSCLayer


@dataclass
class HardwareAwareSCLayer:
    """SC layer with memristive hardware defect injection.

    Parameters
    ----------
    n_inputs : int
        Number of input channels.
    n_neurons : int
        Number of output neurons.
    length : int
        Bitstream length.
    stuck_rate : float
        Fraction of synapses with stuck-at faults (0 or 1). Default 0.05.
    variability : float
        Additive weight noise std. Default 0.02.
    seed : int
        Random seed for defect generation.
    """

    n_inputs: int
    n_neurons: int
    length: int = 1024
    stuck_rate: float = 0.05
    variability: float = 0.02
    seed: int = 42

    def __post_init__(self) -> None:
        """Build the backing layer and inject stuck-at and variability defects."""
        self._layer = VectorizedSCLayer(
            n_inputs=self.n_inputs,
            n_neurons=self.n_neurons,
            length=self.length,
            use_gpu=False,
        )
        rng = np.random.RandomState(self.seed)
        shape = (self.n_neurons, self.n_inputs)

        # Stuck-at mask: True where synapse is stuck
        self.stuck_mask = rng.random(shape) < self.stuck_rate
        self.stuck_values = rng.choice([0.0, 1.0], size=shape)

        # Apply stuck-at defects to initial weights
        self._apply_defects()

    def _apply_defects(self) -> None:
        self._layer.weights[self.stuck_mask] = self.stuck_values[self.stuck_mask]
        if self.variability > 0:
            noise = np.random.RandomState(self.seed + 1).normal(
                0, self.variability, self._layer.weights.shape
            )
            mask = ~self.stuck_mask
            self._layer.weights[mask] = np.clip(self._layer.weights[mask] + noise[mask], 0.0, 1.0)
        self._layer._refresh_packed_weights()

    def forward(self, input_values: list[float] | np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Run a forward pass through the defect-injected stochastic layer."""
        return self._layer.forward(input_values)  # type: ignore[arg-type]

    def update_weights(self, gradient: np.ndarray[Any, Any], lr: float = 0.01) -> None:
        """Update weights with gradient, respecting stuck-at mask.

        Stuck synapses receive zero gradient — the network learns
        around the defects.
        """
        masked_gradient = gradient.copy()
        masked_gradient[self.stuck_mask] = 0.0
        self._layer.weights -= lr * masked_gradient
        self._layer.weights = np.clip(self._layer.weights, 0.0, 1.0)
        self._apply_defects()

    @property
    def weights(self) -> np.ndarray[Any, Any]:
        """Return the current defect-affected weight matrix."""
        return self._layer.weights

    @property
    def n_stuck(self) -> int:
        """Return the number of stuck-at synapses."""
        return int(self.stuck_mask.sum())

    @property
    def stuck_fraction(self) -> float:
        """Return the fraction of synapses that are stuck-at."""
        return float(self.stuck_mask.mean())
