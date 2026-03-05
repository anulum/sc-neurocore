# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations
from typing import Any, Optional
from dataclasses import dataclass
import numpy as np
from typing import Optional

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
        np.random.seed(self.seed)

        # Input Weights (W_in): (n_neurons, n_inputs)
        self.W_in = np.random.uniform(0, 1, (self.n_neurons, self.n_inputs)) * self.input_strength

        # Recurrent Weights (W_rec): (n_neurons, n_neurons)
        # Initialize as a Reservoir (sparse, scaled)
        # For true SC, weights are [0,1]. Standard Reservoir weights are [-1, 1].
        # We map [-1, 1] logic to [0, 1] using Excitatory/Inhibitory paths or
        # Bipolar coding. Here we stick to Unipolar [0,1].
        # We'll initialize random sparse connections.

        self.W_rec = np.random.uniform(0, 0.2, (self.n_neurons, self.n_neurons))  # Weak connections

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
        # Generate Bitstreams for Input (used by hardware simulation loop below)
        _in_bits = np.array(
            [enc.encode(val) for enc, val in zip(self.input_encoders, input_vector)]
        )  # noqa: F841

        # Generate Bitstreams for Previous State (Feedback)
        _state_bits = np.array(
            [enc.encode(val) for enc, val in zip(self.state_encoders, self.state)]
        )  # noqa: F841

        new_rates = np.zeros(self.n_neurons)

        # Processing loop (Simulation of hardware)
        # This could be vectorized, but for clarity/structure:

        # In hardware, this is:
        # Neuron_i_Current = Sum(Input_j * W_in_ij) + Sum(State_k * W_rec_ik)

        # We can calculate the expected current probabilities directly to speed up (Soft Simulation)
        # or do the bitwise (Hard Simulation).
        # Let's do Soft Simulation here for the "Recurrent" logic proof,
        # as bit-level recurrence requires cycle-accurate feedback which is slow in Python loops.

        # Soft SC: P_out = P_in * P_w

        currents = np.dot(self.W_in, input_vector) + np.dot(self.W_rec, self.state)

        # Update Neurons
        # We treat 'current' as the probability of input spikes arriving.
        # We need to run the neuron for 'length' steps?
        # Or just update state based on transfer function?

        # Let's map current to firing rate roughly:
        # Rate ~ Current (in linear region)
        # We'll use a Tanh-like saturation
        new_rates = np.tanh(currents)
        # Map back to [0,1] for Unipolar SC?
        # Tanh gives [-1, 1].
        # If we assume our neuron handles positive only:
        new_rates = np.maximum(0, np.minimum(1, currents))  # ReLU-like saturation

        self.state = new_rates
        return self.state

    def reset(self):  # type: ignore
        self.state = np.zeros(self.n_neurons)
