# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bipolar stochastic computing primitives

"""Bipolar SC: values in [-1, 1] encoded as bitstream probabilities.

Unipolar SC (AND gate) only handles [0, 1]. Bipolar SC uses XNOR:
  value v in [-1, 1] -> probability p = (v + 1) / 2
  XNOR(a, b) computes v_a * v_b in bipolar domain.

This enables signed weight multiplication for trained SNNs.

Gaines, B.R. (1969). Stochastic computing systems. Advances in
Information Systems Science, 2:37-172.

Alaghi, A. & Hayes, J.P. (2013). Survey of stochastic computing.
ACM TECS 12(2s):1-19.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
UInt8Array = NDArray[np.uint8]


def _validate_bitstream_length(length: int) -> None:
    if length <= 0:
        raise ValueError("bitstream length must be positive")


def _as_float_array(values: Any, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    if np.any(array < -1.0) or np.any(array > 1.0):
        raise ValueError(f"{name} values must be in [-1, 1]")
    return array


def _as_bit_array(bits: Any, name: str) -> UInt8Array:
    array = np.asarray(bits, dtype=np.uint8)
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must contain only 0/1 bits")
    return array


def bipolar_encode(value: float, L: int, rng: Any = None) -> UInt8Array:
    """Encode a bipolar value in [-1, 1] as a Bernoulli bitstream.

    p = (value + 1) / 2.  Bitstream has P(bit=1) = p.
    """
    _validate_bitstream_length(L)
    if not np.isfinite(value):
        raise ValueError("value contains NaN or Inf")
    if value < -1.0 or value > 1.0:
        raise ValueError("value must be in [-1, 1]")
    p = (value + 1.0) / 2.0
    if rng is None:
        rng = np.random.default_rng()
    result: UInt8Array = (rng.random(L) < p).astype(np.uint8)
    return result


def bipolar_decode(bits: np.ndarray[Any, Any]) -> float:
    """Decode a bitstream back to bipolar value in [-1, 1].

    value = 2 * mean(bits) - 1.
    """
    bit_array = _as_bit_array(bits, "bits")
    return float(2.0 * bit_array.mean() - 1.0)


def bipolar_multiply(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> UInt8Array:
    """XNOR gate: bipolar multiplication.

    XNOR(a, b) = NOT(XOR(a, b)) = 1 when a == b, 0 when a != b.
    E[XNOR] decodes to v_a * v_b in bipolar domain.
    """
    a_bits = _as_bit_array(a, "a")
    b_bits = _as_bit_array(b, "b")
    if a_bits.shape != b_bits.shape:
        raise ValueError("a and b must have the same shape")
    result: UInt8Array = (a_bits == b_bits).astype(np.uint8)
    return result


def bipolar_mac(
    inputs: np.ndarray[Any, Any],
    weights: np.ndarray[Any, Any],
    L: int,
    seed: int = 42,
) -> FloatArray:
    """Bipolar multiply-accumulate: weighted sum via XNOR + popcount.

    Parameters
    ----------
    inputs : (N,) float array, values in [-1, 1]
    weights : (M, N) float array, values in [-1, 1]
    L : int, bitstream length
    seed : int

    Returns
    -------
    (M,) float array, dot product results (sum of N bipolar products)
    """
    _validate_bitstream_length(L)
    inputs = _as_float_array(inputs, "inputs")
    weights = _as_float_array(weights, "weights")
    if inputs.ndim != 1:
        raise ValueError(f"inputs must have shape (N,), got {inputs.shape}")
    if weights.ndim != 2:
        raise ValueError(f"weights must have shape (M, N), got {weights.shape}")
    if weights.shape[1] != inputs.shape[0]:
        raise ValueError(
            f"weights shape {weights.shape} does not match inputs shape {inputs.shape}"
        )

    N = len(inputs)
    M = weights.shape[0]
    rng = np.random.default_rng(seed)

    # Encode inputs as bitstreams: (N, L)
    input_probs = (inputs + 1.0) / 2.0
    input_bits = (rng.random((N, L)) < input_probs[:, None]).astype(np.uint8)

    # Encode weights as bitstreams: (M, N, L)
    weight_probs = (weights + 1.0) / 2.0
    weight_bits = (rng.random((M, N, L)) < weight_probs[:, :, None]).astype(np.uint8)

    # XNOR multiplication: per-input bipolar product, then sum (dot product)
    outputs: FloatArray = np.zeros(M, dtype=np.float64)
    for j in range(M):
        xnor = (input_bits == weight_bits[j]).astype(np.float32)  # (N, L)
        # Per-input: average over L, decode to bipolar [-1, 1]
        per_input = 2.0 * xnor.mean(axis=1) - 1.0  # (N,)
        # Sum across inputs = dot product (matches w @ x)
        outputs[j] = per_input.sum()

    return outputs


def bipolar_sc_layer(
    inputs: np.ndarray[Any, Any],
    weights: np.ndarray[Any, Any],
    bias: np.ndarray[Any, Any] | None,
    L: int,
    seed: int = 42,
    activation: str = "relu",
) -> FloatArray:
    """Single SC layer: bipolar MAC + optional bias + activation.

    Parameters
    ----------
    inputs : (N,) float, normalised to [-1, 1]
    weights : (M, N) float, normalised to [-1, 1]
    bias : (M,) float or None
    L : bitstream length
    activation : "relu", "none", or "tanh"

    Returns
    -------
    (M,) float, layer output in [-1, 1]
    """
    if activation not in {"relu", "none", "tanh"}:
        raise ValueError("activation must be 'relu', 'none', or 'tanh'")
    out = bipolar_mac(inputs, weights, L, seed=seed)

    if bias is not None:
        bias_array = np.asarray(bias, dtype=np.float64)
        if bias_array.shape != out.shape:
            raise ValueError(
                f"bias shape {bias_array.shape} does not match output shape {out.shape}"
            )
        if not np.all(np.isfinite(bias_array)):
            raise ValueError("bias contains NaN or Inf")
        # Scale bias to bipolar range
        out = out + bias_array * 0.1  # damped bias to stay in [-1, 1]

    if activation == "relu":
        out = np.maximum(out, 0.0)
    elif activation == "tanh":
        out = np.tanh(out * 2.0)

    result: FloatArray = np.clip(out, -1.0, 1.0)
    return result


def float_to_bipolar_weights(weight_tensor: Any) -> FloatArray:
    """Normalise float weights to [-1, 1] for bipolar SC.

    Preserves sign information (unlike unipolar to_sc_weights).
    """
    w: FloatArray = (
        weight_tensor.detach().cpu().numpy().astype(np.float64)
        if hasattr(weight_tensor, "detach")
        else np.asarray(weight_tensor, dtype=np.float64)
    )
    if not np.all(np.isfinite(w)):
        raise ValueError("weight_tensor contains NaN or Inf")
    abs_max = max(np.abs(w).max(), 1e-8)
    result: FloatArray = w / abs_max
    return result
