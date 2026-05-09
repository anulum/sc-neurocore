# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Vectorized Hamming-weight popcount across rows of a

from __future__ import annotations
from typing import Any
from dataclasses import dataclass, field
import numpy as np
from collections.abc import Mapping, Sequence

from ..accel.vector_ops import pack_bitstream, vec_and, vec_xnor
from ..accel.gpu_backend import (
    HAS_CUPY,
    to_device,
    to_host,
    gpu_vec_mac,
)
from ..constants import LAYER_DEFAULT_LENGTH

_scipy_sparse = None


def _get_scipy_sparse() -> Any:
    global _scipy_sparse
    if _scipy_sparse is None:
        import scipy.sparse

        _scipy_sparse = scipy.sparse
    return _scipy_sparse


def _has_scipy_sparse() -> bool:
    try:
        _get_scipy_sparse()
        return True
    except ImportError:  # pragma: no cover
        return False


def _popcount_rows(packed: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
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
    result: np.ndarray[Any, Any] = x.sum(axis=1).astype(np.float64)
    return result


def _bipolar_prob(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    result: np.ndarray[Any, Any] = ((values + 1.0) / 2.0).clip(0.0, 1.0)
    return result


def _mask_unused_tail_bits(packed: np.ndarray[Any, Any], length: int) -> np.ndarray[Any, Any]:
    valid_tail = length % 64
    if valid_tail == 0:
        return packed
    masked = packed.copy()
    mask = np.uint64((1 << valid_tail) - 1)
    masked[..., -1] = masked[..., -1] & mask
    return masked


def _as_float_array(value: Any, name: str) -> np.ndarray[Any, Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


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
    sc_mode: str = "unipolar"
    seed: int | None = None
    bias: Sequence[float] | None = None
    _rng: np.random.Generator | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        if self.n_inputs < 1:
            raise ValueError(f"n_inputs must be >= 1, got {self.n_inputs}")
        if self.n_neurons < 1:
            raise ValueError(f"n_neurons must be >= 1, got {self.n_neurons}")
        if self.length < 1:
            raise ValueError(f"length must be >= 1, got {self.length}")
        if self.sparse and not _has_scipy_sparse():
            raise ImportError("scipy is required for sparse=True")
        if not 0.0 < self.connectivity <= 1.0:
            raise ValueError(f"connectivity must be in (0, 1], got {self.connectivity}")
        if self.sc_mode not in {"unipolar", "bipolar"}:
            raise ValueError("sc_mode must be 'unipolar' or 'bipolar'")
        self.bias_values: np.ndarray[Any, Any] | None = None
        if self.bias is not None:
            bias = _as_float_array(self.bias, "bias")
            if bias.ndim != 1 or bias.shape[0] != self.n_neurons:
                raise ValueError(
                    f"bias must be a 1-D vector of length {self.n_neurons}, got {bias.shape}"
                )
            self.bias_values = bias
        self._rng = np.random.default_rng(self.seed) if self.seed is not None else None

        self._on_gpu = self.use_gpu and HAS_CUPY and self.sc_mode == "unipolar"

        if self.sparse:
            self._init_sparse()
        else:
            weight_low, weight_high = (-1.0, 1.0) if self.sc_mode == "bipolar" else (0.0, 1.0)
            self.weights = self._uniform(weight_low, weight_high, (self.n_neurons, self.n_inputs))
            self.packed_weights: np.ndarray[Any, Any] | None = None
            self._refresh_packed_weights()

    @classmethod
    def from_exported_weights(
        cls,
        exported_layer: Mapping[str, Any],
        *,
        length: int = LAYER_DEFAULT_LENGTH,
        use_gpu: bool = True,
        sparse: bool = False,
        connectivity: float = 1.0,
        sc_mode: str | None = None,
        seed: int | None = None,
    ) -> "VectorizedSCLayer":
        """Build a packed SC inference layer from ``to_sc_weights()`` output."""
        if "weight" not in exported_layer:
            raise ValueError("exported_layer must contain a 'weight' entry")
        weights = _as_float_array(exported_layer["weight"], "weight")
        if weights.ndim != 2:
            raise ValueError(f"weight must be a 2-D matrix, got {weights.shape}")

        exported_encoding = str(exported_layer.get("encoding", "unipolar"))
        resolved_mode = exported_encoding if sc_mode is None else sc_mode
        if resolved_mode != exported_encoding:
            raise ValueError(
                f"sc_mode {resolved_mode!r} does not match exported encoding {exported_encoding!r}"
            )
        if resolved_mode not in {"unipolar", "bipolar"}:
            raise ValueError("exported encoding must be 'unipolar' or 'bipolar'")

        if resolved_mode == "bipolar":
            if np.any(weights < -1.0) or np.any(weights > 1.0):
                raise ValueError("bipolar exported weights must be in [-1, 1]")
        elif np.any(weights < 0.0) or np.any(weights > 1.0):
            raise ValueError("unipolar exported weights must be in [0, 1]")

        bias = exported_layer.get("bias")
        layer = cls(
            n_inputs=int(weights.shape[1]),
            n_neurons=int(weights.shape[0]),
            length=length,
            use_gpu=use_gpu,
            sparse=sparse,
            connectivity=connectivity,
            sc_mode=resolved_mode,
            seed=seed,
            bias=bias,
        )
        if seed is not None:
            layer._rng = np.random.default_rng(seed)
        if sparse:
            sp = _get_scipy_sparse()
            layer.weights_csr = sp.csr_matrix(weights)
            layer._pack_sparse_weights()
        else:
            layer.weights = weights.copy()
            layer._refresh_packed_weights()
        return layer

    def _random(self, size: int | tuple[int, ...]) -> np.ndarray[Any, Any]:
        if self._rng is not None:
            return self._rng.random(size)
        result: np.ndarray[Any, Any] = np.random.random(size)
        return result

    def _uniform(
        self, low: float, high: float, size: int | tuple[int, ...]
    ) -> np.ndarray[Any, Any]:
        if self._rng is not None:
            return self._rng.uniform(low, high, size)
        result: np.ndarray[Any, Any] = np.random.uniform(low, high, size)
        return result

    def _choice(self, n_items: int, size: int) -> np.ndarray[Any, Any]:
        if self._rng is not None:
            return self._rng.choice(n_items, size=size, replace=False)
        result: np.ndarray[Any, Any] = np.random.choice(n_items, size=size, replace=False)
        return result

    def _apply_bias(self, outputs: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        if self.bias_values is None:
            return outputs
        result: np.ndarray[Any, Any] = outputs + self.bias_values
        return result

    # -- Dense path ------------------------------------------------------------

    def _refresh_packed_weights(self) -> None:
        w_probs = _bipolar_prob(self.weights) if self.sc_mode == "bipolar" else self.weights
        bits = (
            self._random((self.n_neurons, self.n_inputs, self.length)) < w_probs[:, :, None]
        ).astype(np.uint8)

        flat = bits.reshape(-1, self.length)
        packed_flat = pack_bitstream(flat)
        pw = packed_flat.reshape(self.n_neurons, self.n_inputs, -1)

        if self._on_gpu:  # pragma: no cover
            self.packed_weights = to_device(pw)
        else:
            self.packed_weights = pw

    # -- Sparse path -----------------------------------------------------------

    def _init_sparse(self) -> None:
        sp = _get_scipy_sparse()
        n_total = self.n_neurons * self.n_inputs
        n_nonzero = max(1, int(round(n_total * self.connectivity)))
        indices = self._choice(n_total, size=n_nonzero)
        rows, cols = np.divmod(indices, self.n_inputs)
        weight_low, weight_high = (-1.0, 1.0) if self.sc_mode == "bipolar" else (0.0, 1.0)
        weight_vals = self._uniform(weight_low, weight_high, n_nonzero)

        self.mask_csr = sp.csr_matrix(
            (np.ones(n_nonzero, dtype=np.float32), (rows, cols)),
            shape=(self.n_neurons, self.n_inputs),
        )
        self.weights_csr = sp.csr_matrix(
            (weight_vals, (rows, cols)),
            shape=(self.n_neurons, self.n_inputs),
        )
        self._pack_sparse_weights()

    def _pack_sparse_weights(self) -> None:
        """Pack bitstreams only for non-zero synapses, stored in a flat array."""
        csr = self.weights_csr
        n_words = (self.length + 63) // 64
        self._sparse_packed = np.empty((csr.nnz, n_words), dtype=np.uint64)
        for k in range(csr.nnz):
            w = csr.data[k]
            p = (w + 1.0) / 2.0 if self.sc_mode == "bipolar" else w
            bits = (self._random(self.length) < p).astype(np.uint8)
            self._sparse_packed[k] = pack_bitstream(bits)

    # -- Forward ---------------------------------------------------------------

    def forward(self, input_values: Sequence[float]) -> np.ndarray[Any, Any]:
        """Compute output firing rates for the layer."""
        in_probs = np.asarray(input_values, dtype=np.float64)
        if in_probs.ndim != 1 or in_probs.shape[0] != self.n_inputs:
            raise ValueError(
                f"Expected 1-D input of length {self.n_inputs}, got shape {in_probs.shape}"
            )
        if not np.all(np.isfinite(in_probs)):
            raise ValueError("Input contains NaN or Inf")
        if self.sc_mode == "bipolar":
            if np.any(in_probs < -1.0) or np.any(in_probs > 1.0):
                raise ValueError("Bipolar input values must be in [-1, 1]")
        elif np.any(in_probs < 0.0) or np.any(in_probs > 1.0):
            raise ValueError("Input probabilities must be in [0, 1]")

        if self.sparse:
            return self._forward_sparse(in_probs)
        return self._forward_dense(in_probs)

    def _forward_dense(self, in_probs: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        input_probs = _bipolar_prob(in_probs) if self.sc_mode == "bipolar" else in_probs
        input_bits = (self._random((self.n_inputs, self.length)) < input_probs[:, None]).astype(
            np.uint8
        )
        packed_inputs = pack_bitstream(input_bits)

        if self._on_gpu:  # pragma: no cover
            packed_inputs_dev = to_device(packed_inputs)
            counts = gpu_vec_mac(self.packed_weights, packed_inputs_dev)
            outputs = to_host(counts).astype(np.float64)
        else:
            assert self.packed_weights is not None
            if self.sc_mode == "bipolar":
                products = vec_xnor(self.packed_weights, packed_inputs[None, :, :])
                products = _mask_unused_tail_bits(products, self.length)
                flat_products = products.reshape(-1, products.shape[-1])
                counts = _popcount_rows(flat_products).reshape(self.n_neurons, self.n_inputs)
                outputs = ((2.0 * counts / self.length) - 1.0).sum(axis=1)
            else:
                products = vec_and(self.packed_weights, packed_inputs[None, :, :])
                flat_products = products.reshape(self.n_neurons, -1)
                outputs = _popcount_rows(flat_products)

        result = outputs if self.sc_mode == "bipolar" else outputs / self.length
        return self._apply_bias(result)

    def _forward_sparse(self, in_probs: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        input_probs = _bipolar_prob(in_probs) if self.sc_mode == "bipolar" else in_probs
        input_bits = (self._random((self.n_inputs, self.length)) < input_probs[:, None]).astype(
            np.uint8
        )
        packed_inputs = pack_bitstream(input_bits)

        csr = self.weights_csr
        if csr.nnz == 0:
            return self._apply_bias(np.zeros(self.n_neurons, dtype=np.float64))

        if self._on_gpu:  # pragma: no cover
            return self._apply_bias(self._forward_sparse_gpu(packed_inputs))

        gathered_inputs = packed_inputs[csr.indices]
        if self.sc_mode == "bipolar":
            products = vec_xnor(self._sparse_packed, gathered_inputs)
            products = _mask_unused_tail_bits(products, self.length)
        else:
            products = vec_and(self._sparse_packed, gathered_inputs)
        counts = _popcount_rows(products)
        terms = (2.0 * counts / self.length) - 1.0 if self.sc_mode == "bipolar" else counts

        outputs = np.zeros(self.n_neurons, dtype=np.float64)
        np.add.at(outputs, np.repeat(np.arange(self.n_neurons), np.diff(csr.indptr)), terms)

        result = outputs if self.sc_mode == "bipolar" else outputs / self.length
        return self._apply_bias(result)

    def _forward_sparse_gpu(
        self, packed_inputs: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:  # pragma: no cover
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
        result: np.ndarray[Any, Any] = cupy.asnumpy(out_gpu).astype(np.float64)
        return result
