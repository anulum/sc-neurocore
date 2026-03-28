# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
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

import numpy as np


def bipolar_encode(value: float, L: int, rng=None) -> np.ndarray:
    """Encode a bipolar value in [-1, 1] as a Bernoulli bitstream.

    p = (value + 1) / 2.  Bitstream has P(bit=1) = p.
    """
    p = np.clip((value + 1.0) / 2.0, 0.0, 1.0)
    if rng is None:
        rng = np.random.default_rng()
    return (rng.random(L) < p).astype(np.uint8)


def bipolar_decode(bits: np.ndarray) -> float:
    """Decode a bitstream back to bipolar value in [-1, 1].

    value = 2 * mean(bits) - 1.
    """
    return 2.0 * bits.mean() - 1.0


def bipolar_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """XNOR gate: bipolar multiplication.

    XNOR(a, b) = NOT(XOR(a, b)) = 1 when a == b, 0 when a != b.
    E[XNOR] decodes to v_a * v_b in bipolar domain.
    """
    return (a == b).astype(np.uint8)


def bipolar_mac(
    inputs: np.ndarray,
    weights: np.ndarray,
    L: int,
    seed: int = 42,
) -> np.ndarray:
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
    N = len(inputs)
    M = weights.shape[0]
    rng = np.random.default_rng(seed)

    # Encode inputs as bitstreams: (N, L)
    input_probs = np.clip((inputs + 1.0) / 2.0, 0.0, 1.0)
    input_bits = (rng.random((N, L)) < input_probs[:, None]).astype(np.uint8)

    # Encode weights as bitstreams: (M, N, L)
    weight_probs = np.clip((weights + 1.0) / 2.0, 0.0, 1.0)
    weight_bits = (rng.random((M, N, L)) < weight_probs[:, :, None]).astype(np.uint8)

    # XNOR multiplication: per-input bipolar product, then sum (dot product)
    outputs = np.zeros(M)
    for j in range(M):
        xnor = (input_bits == weight_bits[j]).astype(np.float32)  # (N, L)
        # Per-input: average over L, decode to bipolar [-1, 1]
        per_input = 2.0 * xnor.mean(axis=1) - 1.0  # (N,)
        # Sum across inputs = dot product (matches w @ x)
        outputs[j] = per_input.sum()

    return outputs


def bipolar_sc_layer(
    inputs: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray | None,
    L: int,
    seed: int = 42,
    activation: str = "relu",
) -> np.ndarray:
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
    out = bipolar_mac(inputs, weights, L, seed=seed)

    if bias is not None:
        # Scale bias to bipolar range
        out = out + bias * 0.1  # damped bias to stay in [-1, 1]

    if activation == "relu":
        out = np.maximum(out, 0.0)
    elif activation == "tanh":
        out = np.tanh(out * 2.0)

    return np.clip(out, -1.0, 1.0)


def float_to_bipolar_weights(weight_tensor) -> np.ndarray:
    """Normalise float weights to [-1, 1] for bipolar SC.

    Preserves sign information (unlike unipolar to_sc_weights).
    """
    w = weight_tensor.detach().cpu().numpy() if hasattr(weight_tensor, 'detach') else np.asarray(weight_tensor)
    abs_max = max(np.abs(w).max(), 1e-8)
    return w / abs_max
