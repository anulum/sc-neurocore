# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations
from typing import Any
from dataclasses import dataclass
import numpy as np
from collections.abc import Sequence

from ..accel.vector_ops import pack_bitstream, vec_and
from ..accel.gpu_backend import (
    HAS_CUPY,
    to_device,
    to_host,
    gpu_vec_mac,
)
from ..constants import LAYER_DEFAULT_LENGTH

try:
    import scipy.sparse as sp

    HAS_SCIPY_SPARSE = True
except ImportError:  # pragma: no cover
    HAS_SCIPY_SPARSE = False


def _popcount_rows(packed: np.ndarray) -> np.ndarray:
    """Vectorized Hamming-weight popcount across rows of a uint64 array."""
    x = packed.astype(np.uint64).copy()
    m1 = np.uint64(0x5555555555555555)
    m2 = np.uint64(0x3333333333333333)
    m4 = np.uint64(0x0F0F0F0F0F0F0F0F)
    h01 = np.uint64(0x0101010101010101)
    x -= (x >> np.uint64(1)) & m1
    x = (x & m2) + ((x >> np.uint64(2)) & m2)
    x = (x + (x >> np.uint64(4))) & m4
    x = (x * h01) >> np.uint64(56)
    return x.sum(axis=1).astype(np.float64)


@dataclass
class VectorizedSCLayer:
    """
    High-performance SC layer using packed bitwise operations.

    Uses GPU (CuPy) when available, otherwise pure NumPy.
    Optional sparse connectivity via ``scipy.sparse``.

    Example
    -------
    >>> import numpy as np
    >>> layer = VectorizedSCLayer(n_inputs=8, n_neurons=4, length=512)
    >>> out = layer.forward(np.random.rand(8))
    >>> out.shape
    (4,)
    >>> (out >= 0).all() and (out <= 1).all()
    True
    """

    n_inputs: int
    n_neurons: int
    length: int = LAYER_DEFAULT_LENGTH
    use_gpu: bool = True
    sparse: bool = False
    connectivity: float = 1.0

    def __post_init__(self) -> None:
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

    def _refresh_packed_weights(self) -> None:
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
            products = vec_and(self.packed_weights, packed_inputs[None, :, :])
            flat_products = products.reshape(self.n_neurons, -1)
            outputs = _popcount_rows(flat_products)

        return outputs / self.length

    def _forward_sparse(self, in_probs: np.ndarray) -> np.ndarray:
        input_bits = (np.random.random((self.n_inputs, self.length)) < in_probs[:, None]).astype(
            np.uint8
        )
        packed_inputs = pack_bitstream(input_bits)

        csr = self.weights_csr
        if csr.nnz == 0:  # pragma: no cover
            return np.zeros(self.n_neurons, dtype=np.float64)

        if self._on_gpu:  # pragma: no cover
            return self._forward_sparse_gpu(packed_inputs)

        gathered_inputs = packed_inputs[csr.indices]
        products = vec_and(self._sparse_packed, gathered_inputs)
        counts = _popcount_rows(products)

        outputs = np.zeros(self.n_neurons, dtype=np.float64)
        np.add.at(outputs, np.repeat(np.arange(self.n_neurons), np.diff(csr.indptr)), counts)

        return outputs / self.length

    def _forward_sparse_gpu(self, packed_inputs: np.ndarray) -> np.ndarray:  # pragma: no cover
        """CuPy CSR matmul path for sparse connectivity on GPU."""
        import cupy
        import cupyx.scipy.sparse as cusp

        csr = self.weights_csr
        w_gpu = cusp.csr_matrix(
            (
                cupy.asarray(csr.data.astype(np.float32)),
                cupy.asarray(csr.indices),
                cupy.asarray(csr.indptr),
            ),
            shape=csr.shape,
        )
        in_probs_flat = _popcount_rows(packed_inputs).astype(np.float32) / self.length
        in_gpu = cupy.asarray(in_probs_flat)
        out_gpu = w_gpu @ in_gpu
        return cupy.asnumpy(out_gpu).astype(np.float64)
