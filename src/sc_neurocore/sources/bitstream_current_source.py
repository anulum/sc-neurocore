# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Multi-channel bitstream current source

"""Multi-channel bitstream current source for neuron simulations.

This module encodes scalar inputs and weights into stochastic bitstreams,
combines them through SC synapse or bipolar XNOR semantics, and exposes the
realised per-cycle current trace consumed by deterministic simulation tests.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Optional
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
    - Decodes the realised post-synaptic bitstreams into a per-cycle
      current trace for neuron simulation.

    Static inputs and weights are encoded once at construction time.
    The stochastic realisation is then fixed until a new source is
    constructed with different parameters or seeds.
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
    sc_mode: str = "unipolar"

    def __post_init__(self) -> None:
        """Validate input/weight lengths and derive the input count."""
        self.n_inputs = len(self.x_inputs)
        if self.n_inputs < 1:
            raise ValueError("BitstreamCurrentSource requires at least one input.")
        if len(self.weight_values) != self.n_inputs:
            raise ValueError("x_inputs and weight_values must have same length.")
        if self.sc_mode not in {"unipolar", "bipolar"}:
            raise ValueError("sc_mode must be 'unipolar' or 'bipolar'.")

        # Encoders for input channels
        self._encoders: List[BitstreamEncoder] = []
        for i in range(self.n_inputs):
            self._encoders.append(
                BitstreamEncoder(
                    x_min=self.x_min,
                    x_max=self.x_max,
                    length=self.length,
                    seed=None if self.seed is None else self.seed + i,
                    mode="bipolar" if self.sc_mode == "bipolar" else "bernoulli",
                )
            )

        # Generate pre-synaptic bitstreams
        self.pre_matrix = np.zeros((self.n_inputs, self.length), dtype=np.uint8)
        for i, (enc, x) in enumerate(zip(self._encoders, self.x_inputs)):
            self.pre_matrix[i] = enc.encode(x)

        self.synapses: List[BitstreamSynapse] = []
        self.dot: BitstreamDotProduct | None = None
        if self.sc_mode == "bipolar":
            self.dot = None
            self.post_matrix, _ = self._apply_bipolar_xnor()
        else:
            # Build synapses
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

            # Post-synaptic streams; the scalar current is decoded below from
            # the same realised trace used by step().
            self.post_matrix, _ = self.dot.apply(
                self.pre_matrix, y_min=self.y_min, y_max=self.y_max
            )

        self._current_trace = self._decode_current_trace()
        self.current_scalar = float(self._current_trace.mean())

        # We'll treat each timestep as one index in the bitstreams
        self._t = 0

    def _apply_bipolar_xnor(self) -> tuple[np.ndarray[Any, Any], float]:
        weight_matrix = np.zeros((self.n_inputs, self.length), dtype=np.uint8)
        for i, w in enumerate(self.weight_values):
            weight_encoder = BitstreamEncoder(
                x_min=self.w_min,
                x_max=self.w_max,
                length=self.length,
                seed=None if self.seed is None else self.seed + 1000 + i,
                mode="bipolar",
            )
            weight_matrix[i] = weight_encoder.encode(w)

        post_matrix = (self.pre_matrix == weight_matrix).astype(np.uint8)
        products = 2.0 * post_matrix.mean(axis=1) - 1.0
        bipolar_mean = float(products.mean()) if products.size else 0.0
        current = self._map_bipolar_to_current(bipolar_mean)
        return post_matrix, current

    def _map_bipolar_to_current(self, value: float) -> float:
        clipped = max(min(value, 1.0), -1.0)
        return float(self.y_min + ((clipped + 1.0) / 2.0) * (self.y_max - self.y_min))

    def _decode_current_trace(self) -> np.ndarray[Any, Any]:
        if self.sc_mode == "bipolar":
            probs = self.post_matrix.mean(axis=0, dtype=np.float64)
            bipolar_values = (2.0 * probs) - 1.0
            clipped = np.clip(bipolar_values, -1.0, 1.0)
            bipolar_current: np.ndarray[Any, Any] = (
                self.y_min + ((clipped + 1.0) / 2.0) * (self.y_max - self.y_min)
            ).astype(np.float64, copy=False)
            return bipolar_current

        probs = self.post_matrix.mean(axis=0, dtype=np.float64)
        unipolar_current: np.ndarray[Any, Any] = (
            self.y_min + probs * (self.y_max - self.y_min)
        ).astype(np.float64, copy=False)
        return unipolar_current

    def reset(self) -> None:
        """Reset the realised current trace cursor to its first timestep."""
        self._t = 0

    def current_trace(self) -> np.ndarray[Any, Any]:
        """
        Return the realised per-cycle decoded current trace.

        The returned trace is computed from the fixed post-synaptic
        bitstreams generated during construction. It is therefore the
        exact sequence consumed by repeated ``step()`` calls, not a
        separate probability-space approximation.
        """
        return self._current_trace.copy()

    def step(self) -> float:
        """
        Return the current I_t at the current time index and advance.

        ``step()`` returns the t-th element of ``current_trace()`` and
        clamps at the final element after the bitstream is exhausted.
        """
        idx = self._t
        if idx >= self.length:
            # Clamp at last timestep (or you can wrap)
            idx = self.length - 1

        I_t = self._current_trace[idx]
        self._t += 1
        return float(I_t)

    def full_current_estimate(self) -> float:
        """Return the mean current over the realised bitstream duration."""
        return float(self.current_scalar)
