# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations
from typing import Any, Optional
from dataclasses import dataclass
import numpy as np

from ..neurons.stochastic_lif import StochasticLIFNeuron
from ..utils.bitstreams import BitstreamEncoder


@dataclass
class SCRecurrentLayer:
    """
    Stochastic Computing Recurrent Neural Network (RNN) / Reservoir Layer.

    Inputs: [Batch, Time, Features] or just sequential vector inputs.
    Internal State: Neurons connect to themselves (or each other).
    """

    n_inputs: int
    n_neurons: int
    feedback_strength: float = 0.5
    input_strength: float = 0.5
    spectral_radius: float = 0.9  # For reservoir initialization
    length: int = 1024
    seed: Optional[int] = None

    def __post_init__(self):  # type: ignore
        rng = np.random.RandomState(self.seed)

        self.W_in = rng.uniform(0, 1, (self.n_neurons, self.n_inputs)) * self.input_strength
        self.W_rec = rng.uniform(0, 0.2, (self.n_neurons, self.n_neurons))

        # Neurons
        self.neurons = [
            StochasticLIFNeuron(seed=self.seed + i if self.seed else None)
            for i in range(self.n_neurons)
        ]

        # Previous State (Firing Rate / Probability)
        self.state = np.zeros(self.n_neurons)

        # Encoder for state feedback
        self.state_encoders = [
            BitstreamEncoder(x_min=0, x_max=1, length=self.length) for _ in range(self.n_neurons)
        ]
        self.input_encoders = [
            BitstreamEncoder(x_min=0, x_max=1, length=self.length) for _ in range(self.n_inputs)
        ]

    def step(self, input_vector: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """
        Process one time step (e.g., one frame of audio).
        Input: (n_inputs,)
        Output: (n_neurons,) - New State
        """
        currents = np.dot(self.W_in, input_vector) + np.dot(self.W_rec, self.state)
        new_rates = np.clip(currents, 0.0, 1.0)

        self.state = new_rates
        return self.state

    def reset(self):  # type: ignore
        self.state = np.zeros(self.n_neurons)
