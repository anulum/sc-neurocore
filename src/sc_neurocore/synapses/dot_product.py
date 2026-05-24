# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bitstream-level dot product via SC synapses

from __future__ import annotations
from typing import Any
from dataclasses import dataclass
from typing import List, Tuple
import math
import numpy as np

from .sc_synapse import BitstreamSynapse
from ..utils.bitstreams import (
    bitstream_to_probability,
    unipolar_prob_to_value,
)


@dataclass
class BitstreamDotProduct:
    """
    Bitstream-level dot product via SC synapses.

    For each input i, applies synapse_i (AND gate), then sums decoded
    probabilities: y ~ sum_i w_i * x_i.

    Example
    -------
    >>> import numpy as np
    >>> from sc_neurocore import BitstreamSynapse
    >>> syns = [BitstreamSynapse(w_min=0.0, w_max=1.0, w=0.5, length=256)
    ...         for _ in range(3)]
    >>> dp = BitstreamDotProduct(synapses=syns)
    >>> pre = np.ones((3, 256), dtype=np.uint8)
    >>> post_matrix, y_scalar = dp.apply(pre)
    >>> post_matrix.shape
    (3, 256)
    """

    synapses: List[BitstreamSynapse]

    def __post_init__(self) -> None:
        if not isinstance(self.synapses, list) or len(self.synapses) == 0:
            raise ValueError("synapses must be a non-empty list")
        if not all(isinstance(synapse, BitstreamSynapse) for synapse in self.synapses):
            raise ValueError("synapses must contain only BitstreamSynapse instances")
        length = self.synapses[0].length
        if any(synapse.length != length for synapse in self.synapses):
            raise ValueError("synapses must share a common bitstream length")

    @property
    def n_inputs(self) -> int:
        return len(self.synapses)

    def apply(
        self,
        pre_matrix: np.ndarray[Any, Any],
        y_min: float = 0.0,
        y_max: float = 1.0,
    ) -> Tuple[np.ndarray[Any, Any], float]:
        """
        Apply all synapses to the pre-synaptic bitstreams and compute
        a scalar 'dot-product-like' value.

        Parameters
        ----------
        pre_matrix : np.ndarray
            Shape (n_inputs, length), entries {0,1}.
        y_min, y_max : float
            Range in which the final scalar output is interpreted
            (e.g., current range for the neuron).

        Returns
        -------
        post_matrix : np.ndarray
            Post-synaptic bitstreams of shape (n_inputs, length).
        y_scalar : float
            Scalar result representing sum_i P(post_i=1) mapped into [y_min, y_max].
        """
        if not math.isfinite(y_min) or not math.isfinite(y_max) or y_min >= y_max:
            raise ValueError("y_min and y_max must be finite with y_min < y_max")
        if not isinstance(pre_matrix, np.ndarray):
            raise ValueError("pre_matrix must be a numpy array")
        if pre_matrix.ndim != 2:
            raise ValueError("pre_matrix must be a two-dimensional bitstream matrix")
        if pre_matrix.shape[0] != self.n_inputs:
            raise ValueError(
                f"pre_matrix expected {self.n_inputs} input bitstreams, got {pre_matrix.shape[0]}"
            )
        expected_length = self.synapses[0].length
        if pre_matrix.shape[1] != expected_length:
            raise ValueError(
                f"pre_matrix expected bitstream length {expected_length}, got {pre_matrix.shape[1]}"
            )
        if not np.all((pre_matrix == 0) | (pre_matrix == 1)):
            raise ValueError("pre_matrix must contain only binary values 0 or 1")

        post_matrix = np.zeros_like(pre_matrix, dtype=np.uint8)
        probs = []

        for i, syn in enumerate(self.synapses):
            post_i = syn.apply(pre_matrix[i])
            post_matrix[i] = post_i
            probs.append(bitstream_to_probability(post_i))

        # Dot-product in probability space (weights already baked into probs)
        y_prob_sum = float(sum(probs))

        # Normalize by number of inputs if desired
        # Here we just keep the sum and clamp into [0, 1]
        y_prob_clamped = max(min(y_prob_sum, 1.0), 0.0)

        # Map that into [y_min, y_max]
        y_scalar = unipolar_prob_to_value(y_prob_clamped, y_min, y_max)

        return post_matrix, y_scalar
