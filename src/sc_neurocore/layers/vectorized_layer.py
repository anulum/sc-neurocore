# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations
from typing import Any
from dataclasses import dataclass
import numpy as np
from collections.abc import Sequence

from ..accel.vector_ops import pack_bitstream, vec_and, vec_popcount
from ..accel.gpu_backend import (
    HAS_CUPY,
    to_device,
    to_host,
    gpu_vec_mac,
)

try:
    import scipy.sparse as sp

    HAS_SCIPY_SPARSE = True
except ImportError:  # pragma: no cover
    HAS_SCIPY_SPARSE = False


@dataclass
class VectorizedSCLayer:
    """
    High-Performance SC Layer using packed bitwise operations.

    When CuPy is available the heavy AND + popcount path runs on the GPU;
    otherwise pure NumPy is used transparently.

    When ``sparse=True``, only a fraction ``connectivity`` of synapses are
    allocated. The connectivity mask is stored as a ``scipy.sparse.csr_matrix``
    and the forward pass skips zero-weight entries entirely.
    """

    n_inputs: int
    n_neurons: int
    length: int = 1024
    use_gpu: bool = True
    sparse: bool = False
    connectivity: float = 1.0

    def __post_init__(self):  # type: ignore
        if self.sparse and not HAS_SCIPY_SPARSE:
            raise ImportError("scipy is required for sparse=True")
        if not 0.0 < self.connectivity <= 1.0:
            raise ValueError(f"connectivity must be in (0, 1], got {self.connectivity}")

        self._on_gpu = self.use_gpu and HAS_CUPY

        if self.sparse:
            self._init_sparse()
        else:
            self.weights = np.random.uniform(0.0, 1.0, (self.n_neurons, self.n_inputs))
            self.packed_weights = None
            self._refresh_packed_weights()

    # -- Dense path (unchanged) ------------------------------------------------

    def _refresh_packed_weights(self):  # type: ignore
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

    # -- Sparse path -----------------------------------------------------------

    def _init_sparse(self):
        n_total = self.n_neurons * self.n_inputs
        n_nonzero = max(1, int(round(n_total * self.connectivity)))
        indices = np.random.choice(n_total, size=n_nonzero, replace=False)
        rows, cols = np.divmod(indices, self.n_inputs)
        weight_vals = np.random.uniform(0.0, 1.0, n_nonzero)

        self.mask_csr = sp.csr_matrix(
            (np.ones(n_nonzero, dtype=np.float32), (rows, cols)),
            shape=(self.n_neurons, self.n_inputs),
        )
        self.weights_csr = sp.csr_matrix(
            (weight_vals, (rows, cols)),
            shape=(self.n_neurons, self.n_inputs),
        )
        self._pack_sparse_weights()

    def _pack_sparse_weights(self):
        """Pack bitstreams only for non-zero synapses, stored in a flat array."""
        csr = self.weights_csr
        n_words = (self.length + 63) // 64
        self._sparse_packed = np.empty((csr.nnz, n_words), dtype=np.uint64)
        for k in range(csr.nnz):
            w = csr.data[k]
            bits = (np.random.random(self.length) < w).astype(np.uint8)
            self._sparse_packed[k] = pack_bitstream(bits)

    # -- Forward ---------------------------------------------------------------

    def forward(self, input_values: Sequence[float]) -> np.ndarray[Any, Any]:
        """Compute output firing rates for the layer."""
        in_probs = np.array(input_values)
        if in_probs.ndim != 1 or in_probs.shape[0] != self.n_inputs:
            raise ValueError(
                f"Expected 1-D input of length {self.n_inputs}, got shape {in_probs.shape}"
            )

        if self.sparse:
            return self._forward_sparse(in_probs)
        return self._forward_dense(in_probs)

    def _forward_dense(self, in_probs: np.ndarray) -> np.ndarray:
        input_bits = (np.random.random((self.n_inputs, self.length)) < in_probs[:, None]).astype(
            np.uint8
        )
        packed_inputs = pack_bitstream(input_bits)

        if self._on_gpu:  # pragma: no cover
            packed_inputs_dev = to_device(packed_inputs)
            counts = gpu_vec_mac(self.packed_weights, packed_inputs_dev)
            outputs = to_host(counts).astype(np.float64)
        else:
            products = vec_and(self.packed_weights, packed_inputs[None, :, :])  # type: ignore
            flat_products = products.reshape(self.n_neurons, -1)
            outputs = np.zeros(self.n_neurons)
            for i in range(self.n_neurons):
                outputs[i] = vec_popcount(flat_products[i])

        return outputs / self.length

    def _forward_sparse(self, in_probs: np.ndarray) -> np.ndarray:
        input_bits = (np.random.random((self.n_inputs, self.length)) < in_probs[:, None]).astype(
            np.uint8
        )
        packed_inputs = pack_bitstream(input_bits)

        csr = self.weights_csr
        outputs = np.zeros(self.n_neurons, dtype=np.float64)
        for row in range(self.n_neurons):
            start, end = csr.indptr[row], csr.indptr[row + 1]
            for idx in range(start, end):
                col = csr.indices[idx]
                product = vec_and(self._sparse_packed[idx], packed_inputs[col])
                outputs[row] += vec_popcount(product)

        return outputs / self.length
