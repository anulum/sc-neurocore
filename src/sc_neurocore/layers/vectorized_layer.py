from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Sequence

from ..accel.vector_ops import pack_bitstream, vec_and, vec_popcount
from ..accel.gpu_backend import (
    HAS_CUPY,
    to_device,
    to_host,
    gpu_vec_mac,
)


@dataclass
class VectorizedSCLayer:
    """
    High-Performance SC Layer using packed bitwise operations.

    When CuPy is available the heavy AND + popcount path runs on the GPU;
    otherwise pure NumPy is used transparently.
    """

    n_inputs: int
    n_neurons: int
    length: int = 1024
    use_gpu: bool = True

    def __post_init__(self):
        self.weights = np.random.uniform(0.0, 1.0, (self.n_neurons, self.n_inputs))
        self.packed_weights = None
        self._on_gpu = self.use_gpu and HAS_CUPY
        self._refresh_packed_weights()

    def _refresh_packed_weights(self):
        w_probs = self.weights
        bits = (
            np.random.random((self.n_neurons, self.n_inputs, self.length)) < w_probs[:, :, None]
        ).astype(np.uint8)

        flat = bits.reshape(-1, self.length)
        packed_flat = pack_bitstream(flat)
        pw = packed_flat.reshape(self.n_neurons, self.n_inputs, -1)

        if self._on_gpu:  # pragma: no cover
            self.packed_weights = to_device(pw)
        else:
            self.packed_weights = pw

    def forward(self, input_values: Sequence[float]) -> np.ndarray:
        """Compute output firing rates for the layer."""
        in_probs = np.array(input_values)
        if in_probs.ndim != 1 or in_probs.shape[0] != self.n_inputs:
            raise ValueError(
                f"Expected 1-D input of length {self.n_inputs}, " f"got shape {in_probs.shape}"
            )
        input_bits = (np.random.random((self.n_inputs, self.length)) < in_probs[:, None]).astype(
            np.uint8
        )

        packed_inputs = pack_bitstream(input_bits)

        if self._on_gpu:  # pragma: no cover
            packed_inputs_dev = to_device(packed_inputs)
            counts = gpu_vec_mac(self.packed_weights, packed_inputs_dev)
            outputs = to_host(counts).astype(np.float64)
        else:
            products = vec_and(self.packed_weights, packed_inputs[None, :, :])
            flat_products = products.reshape(self.n_neurons, -1)
            outputs = np.zeros(self.n_neurons)
            for i in range(self.n_neurons):
                outputs[i] = vec_popcount(flat_products[i])

        return outputs / self.length
