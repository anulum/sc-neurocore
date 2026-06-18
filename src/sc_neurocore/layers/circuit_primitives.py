# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Neural circuit primitives (lateral inhibition, WTA)

"""Neural circuit primitives for sparse coding and competitive learning.

from sc_neurocore.layers.circuit_primitives import LateralInhibition, WinnerTakeAll

# Lateral inhibition: each neuron suppresses its neighbors
li = LateralInhibition(n_neurons=10, inhibition_strength=0.3, radius=2)
output = li.apply(firing_rates)

# Winner-take-all: only the strongest neuron(s) survive
wta = WinnerTakeAll(n_neurons=10, k=1)
winners = wta.apply(firing_rates)
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any
import numpy as np


@dataclass
class LateralInhibition:
    """Lateral inhibition: each neuron inhibits its neighbors.

    Models the surround suppression found in retinal ganglion cells,
    cortical simple cells, and throughout sensory processing.

    The inhibition kernel is a Gaussian centered on each neuron with
    width `radius`, producing a Mexican-hat (center-surround) response
    when combined with the neuron's own excitation.
    """

    n_neurons: int
    inhibition_strength: float = 0.3
    radius: int = 2

    def __post_init__(self) -> None:
        """Build the lateral-inhibition kernel matrix."""
        # Build inhibition kernel matrix
        kernel = np.zeros((self.n_neurons, self.n_neurons))
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                d = min(abs(i - j), self.n_neurons - abs(i - j))  # circular distance
                if 0 < d <= self.radius:
                    kernel[i, j] = self.inhibition_strength * np.exp(
                        -(d**2) / (2 * (self.radius / 2) ** 2)
                    )
        self._kernel = kernel

    def apply(self, rates: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Apply lateral inhibition to firing rates.

        Parameters
        ----------
        rates : np.ndarray, shape (n_neurons,)
            Input firing rates or probabilities.

        Returns
        -------
        np.ndarray, shape (n_neurons,)
            Inhibited firing rates, clipped to [0, inf).
        """
        inhibition = self._kernel @ rates
        inhibited: np.ndarray[Any, Any] = np.maximum(rates - inhibition, 0.0)
        return inhibited


@dataclass
class WinnerTakeAll:
    """k-Winner-Take-All circuit.

    Only the top-k neurons remain active; all others are suppressed to zero.
    Models competitive dynamics in cortical columns and basal ganglia
    action selection.

    With k=1, this is a hard argmax over the population.
    """

    n_neurons: int
    k: int = 1

    def apply(self, rates: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Apply k-WTA to firing rates.

        Parameters
        ----------
        rates : np.ndarray, shape (n_neurons,)
            Input firing rates.

        Returns
        -------
        np.ndarray, shape (n_neurons,)
            Only top-k values survive; rest are zero.
        """
        if self.k >= self.n_neurons:
            return rates.copy()
        top_k = np.argsort(rates)[-self.k :]
        result = np.zeros_like(rates)
        result[top_k] = rates[top_k]
        return result

    def winners(self, rates: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Return indices of the k winning neurons."""
        return np.argsort(rates)[-self.k :][::-1]
