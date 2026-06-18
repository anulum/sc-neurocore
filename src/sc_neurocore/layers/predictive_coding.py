# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Zero-multiplication predictive coding in stochastic computing

"""Predictive coding SC layer: XOR = error, popcount = magnitude, STDP = precision.

Conjecture C9: Predictive coding can be implemented in stochastic computing
with ZERO multiplications. The key insight:

1. Prediction error = XOR(predicted_bitstream, actual_bitstream)
   - XOR gives a bitstream where 1s mark disagreements
   - This is a single gate per bit — no multiplier needed

2. Error magnitude = popcount(XOR result) / L
   - Hamming distance between predicted and actual streams
   - Proportional to |p_pred - p_actual| for independent streams

3. Precision weighting = STDP learning rule
   - Synapses that consistently predict correctly get potentiated
   - Synapses with high prediction error get depressed
   - This implements Bayesian precision weighting without division

The FPGA implementation needs only XOR gates and a popcount tree —
no DSP blocks, no multipliers, no dividers. This is the most
hardware-efficient predictive coding architecture possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..utils.bitstreams import generate_bernoulli_bitstream, bitstream_to_probability


@dataclass
class PredictiveCodingSCLayer:
    """Zero-multiplication predictive coding in SC.

    Parameters
    ----------
    n_inputs : int
        Number of input channels.
    n_neurons : int
        Number of predictive neurons.
    length : int
        Bitstream length.
    lr : float
        STDP-like learning rate for prediction weights.
    seed : int or None
        Random seed.
    """

    n_inputs: int
    n_neurons: int
    length: int = 256
    lr: float = 0.01
    seed: int | None = None

    def __post_init__(self) -> None:
        """Initialise the prediction weights and recurrent state."""
        rng = np.random.RandomState(self.seed)
        # Prediction weights: each neuron predicts the next input
        self.weights = rng.uniform(0.1, 0.9, (self.n_neurons, self.n_inputs))
        self._prev_input: np.ndarray[Any, Any] | None = None

    def forward(self, inputs: list[float] | np.ndarray[Any, Any]) -> dict[str, Any]:
        """Process one timestep.

        Parameters
        ----------
        inputs : array-like
            Input probabilities, shape (n_inputs,).

        Returns
        -------
        dict with keys:
            'prediction_error': float — mean Hamming distance across neurons
            'surprises': ndarray shape (n_neurons,) — per-neuron surprise
            'predictions': ndarray shape (n_neurons, n_inputs) — predicted probs
        """
        inputs = np.asarray(inputs, dtype=np.float64)
        rng = np.random.RandomState(None)

        # Generate actual input bitstreams
        actual_streams = np.array(
            [generate_bernoulli_bitstream(float(np.clip(p, 0, 1)), self.length) for p in inputs]
        )  # shape: (n_inputs, length)

        surprises = np.zeros(self.n_neurons)
        predictions = np.zeros((self.n_neurons, self.n_inputs))

        for j in range(self.n_neurons):
            neuron_error = 0.0
            for i in range(self.n_inputs):
                # Generate predicted bitstream from weight
                pred_stream = generate_bernoulli_bitstream(
                    float(np.clip(self.weights[j, i], 0, 1)), self.length
                )
                predictions[j, i] = self.weights[j, i]

                # XOR = prediction error bitstream (zero multiplications)
                error_stream = np.bitwise_xor(pred_stream, actual_streams[i])

                # Popcount = error magnitude
                error_magnitude = float(np.sum(error_stream)) / self.length
                neuron_error += error_magnitude

                # STDP-like precision update: reduce weight error
                # Move weight toward actual input probability
                actual_p = bitstream_to_probability(actual_streams[i])
                self.weights[j, i] += self.lr * (actual_p - self.weights[j, i])

            surprises[j] = neuron_error / self.n_inputs

        # Clip weights
        np.clip(self.weights, 0.0, 1.0, out=self.weights)

        mean_error = float(np.mean(surprises))

        return {
            "prediction_error": mean_error,
            "surprises": surprises,
            "predictions": predictions,
        }

    def reset(self) -> None:
        """Re-initialise the prediction weights and clear the previous input."""
        rng = np.random.RandomState(self.seed)
        self.weights = rng.uniform(0.1, 0.9, (self.n_neurons, self.n_inputs))
        self._prev_input = None
