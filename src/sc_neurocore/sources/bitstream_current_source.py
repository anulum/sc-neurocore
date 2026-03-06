# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from collections.abc import Sequence
import numpy as np

from ..utils.bitstreams import BitstreamEncoder
from ..synapses.sc_synapse import BitstreamSynapse
from ..synapses.dot_product import BitstreamDotProduct


@dataclass
class BitstreamCurrentSource:
    """
    Multi-channel bitstream current source.

    - Takes scalar inputs x_i in [x_min, x_max]
    - Encodes each into a bitstream via BitstreamEncoder
    - Passes them through BitstreamSynapses
    - Uses BitstreamDotProduct to compute a scalar current I(t)
      for the neuron.

    For now we assume static inputs and weights over the full length,
    but you can extend this to time-varying later.
    """

    x_inputs: Sequence[float]
    x_min: float
    x_max: float
    weight_values: Sequence[float]
    w_min: float
    w_max: float
    length: int = 1024
    y_min: float = 0.0  # output current min
    y_max: float = 0.1  # output current max
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.n_inputs = len(self.x_inputs)
        if len(self.weight_values) != self.n_inputs:
            raise ValueError("x_inputs and weight_values must have same length.")

        # Encoders for input channels
        self._encoders: List[BitstreamEncoder] = []
        for i in range(self.n_inputs):
            self._encoders.append(
                BitstreamEncoder(
                    x_min=self.x_min,
                    x_max=self.x_max,
                    length=self.length,
                    seed=None if self.seed is None else self.seed + i,
                )
            )

        # Generate pre-synaptic bitstreams
        self.pre_matrix = np.zeros((self.n_inputs, self.length), dtype=np.uint8)
        for i, (enc, x) in enumerate(zip(self._encoders, self.x_inputs)):
            self.pre_matrix[i] = enc.encode(x)

        # Build synapses
        self.synapses: List[BitstreamSynapse] = []
        for i, w in enumerate(self.weight_values):
            self.synapses.append(
                BitstreamSynapse(
                    w_min=self.w_min,
                    w_max=self.w_max,
                    length=self.length,
                    w=w,
                    seed=None if self.seed is None else self.seed + 1000 + i,
                )
            )

        # Dot-product engine
        self.dot = BitstreamDotProduct(self.synapses)

        # Post-synaptic streams and scalar current
        self.post_matrix, self.current_scalar = self.dot.apply(
            self.pre_matrix, y_min=self.y_min, y_max=self.y_max
        )

        # We'll treat each timestep as one index in the bitstreams
        self._t = 0

    def reset(self) -> None:
        self._t = 0

    def step(self) -> float:
        """
        Return the current I_t at the current time index and advance.

        We approximate I_t by reading the t-th bit of each post-synaptic
        stream, then mapping their sum to [y_min, y_max].
        """
        idx = self._t
        if idx >= self.length:
            # Clamp at last timestep (or you can wrap)
            idx = self.length - 1

        # Retrieve bits from all post-synaptic streams at time idx
        bits = self.post_matrix[:, idx]

        # Sum bits and normalize
        n_ones = int(bits.sum())
        prob = n_ones / max(self.n_inputs, 1)

        # Map probability into [y_min, y_max]
        I_t = self.y_min + prob * (self.y_max - self.y_min)

        self._t += 1
        return float(I_t)

    def full_current_estimate(self) -> float:
        """
        Estimate average current over full bitstream duration
        using the dot-product's scalar value.
        """
        return float(self.current_scalar)
