# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC recurrent / reservoir layer (echo state network)

"""Stochastic-computing recurrent reservoir layer (echo state network)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..constants import (
    LAYER_DEFAULT_LENGTH,
    RESERVOIR_FEEDBACK_STRENGTH,
    RESERVOIR_INPUT_STRENGTH,
    RESERVOIR_SPECTRAL_RADIUS,
)
from ..neurons.stochastic_lif import StochasticLIFNeuron
from ..utils.bitstreams import BitstreamEncoder


@dataclass
class SCRecurrentLayer:
    """Stochastic-computing recurrent reservoir layer (echo state network).

    Spectral radius scaling follows Jaeger, GMD Report 148, 2001.

    Parameters
    ----------
    n_inputs : int
        Number of input channels (at least 1).
    n_neurons : int
        Number of reservoir neurons (at least 1).
    feedback_strength : float, default=RESERVOIR_FEEDBACK_STRENGTH
        Finite recurrent feedback gain.
    input_strength : float, default=RESERVOIR_INPUT_STRENGTH
        Finite input drive gain.
    spectral_radius : float, default=RESERVOIR_SPECTRAL_RADIUS
        Finite reservoir spectral radius target.
    length : int, default=LAYER_DEFAULT_LENGTH
        Positive stochastic bitstream length.
    seed : int, optional
        Optional PRNG seed.
    """

    n_inputs: int
    n_neurons: int
    feedback_strength: float = RESERVOIR_FEEDBACK_STRENGTH
    input_strength: float = RESERVOIR_INPUT_STRENGTH
    spectral_radius: float = RESERVOIR_SPECTRAL_RADIUS
    length: int = LAYER_DEFAULT_LENGTH
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate configuration parameters and initialise reservoir weights and encoders."""
        if self.n_inputs < 1 or self.n_neurons < 1:
            raise ValueError("n_inputs and n_neurons must be positive integers.")
        if self.length < 1:
            raise ValueError("length must be positive.")
        for name, val in (
            ("feedback_strength", self.feedback_strength),
            ("input_strength", self.input_strength),
            ("spectral_radius", self.spectral_radius),
        ):
            if isinstance(val, bool) or not math.isfinite(float(val)):
                raise ValueError(f"{name} must be finite, got {val!r}.")

        rng = np.random.RandomState(self.seed)
        self.W_in = rng.uniform(0, 1, (self.n_neurons, self.n_inputs)) * self.input_strength
        self.W_rec = rng.uniform(0, 0.2, (self.n_neurons, self.n_neurons))

        self.neurons = [
            StochasticLIFNeuron(seed=self.seed + i if self.seed else None)
            for i in range(self.n_neurons)
        ]
        self.state = np.zeros(self.n_neurons)

        self.state_encoders = [
            BitstreamEncoder(x_min=0.0, x_max=1.0, length=self.length)
            for _ in range(self.n_neurons)
        ]
        self.input_encoders = [
            BitstreamEncoder(x_min=0.0, x_max=1.0, length=self.length) for _ in range(self.n_inputs)
        ]

    def step(self, input_vector: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Advance the reservoir by one timestep on a validated finite input vector.

        Parameters
        ----------
        input_vector : numpy.ndarray
            Finite 1-D input array of shape ``(n_inputs,)``.

        Returns
        -------
        numpy.ndarray
            Updated reservoir state vector of shape ``(n_neurons,)`` in ``[0, 1]``.

        Raises
        ------
        ValueError
            If ``input_vector`` does not have shape ``(n_inputs,)`` or contains
            non-finite entries; ``self.state`` remains unchanged.
        """
        vec = np.asarray(input_vector, dtype=np.float64)
        if vec.shape != (self.n_inputs,):
            raise ValueError(f"input_vector must have shape ({self.n_inputs},), got {vec.shape}.")
        if not np.all(np.isfinite(vec)):
            raise ValueError("input_vector must contain only finite values.")

        currents = np.dot(self.W_in, vec) + np.dot(self.W_rec, self.state)
        new_rates: np.ndarray[Any, Any] = np.clip(currents, 0.0, 1.0)
        self.state = new_rates
        return self.state

    def reset(self) -> None:
        """Reset the reservoir state vector to zeros."""
        self.state = np.zeros(self.n_neurons)
