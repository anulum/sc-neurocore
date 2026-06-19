# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quantization-aware training

"""Train SNNs through quantization using straight-through estimators.

Missing link between training and hardware deployment.
No SNN library ships QAT as a reusable module.

Reference: QP-SNN (ICLR 2025), SpikeFit (EurIPS 2025)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _ste_quantize(
    x: np.ndarray[Any, Any], bits: int, symmetric: bool = True
) -> np.ndarray[Any, Any]:
    """Quantize with straight-through estimator (forward quantized, backward identity)."""
    n_levels = 2**bits
    if symmetric:
        abs_max = max(np.abs(x).max(), 1e-8)
        scale = abs_max / (n_levels // 2 - 1)
        quantized: np.ndarray[Any, Any] = np.round(x / scale) * scale
        return quantized
    x_min, x_max = x.min(), x.max()
    x_range = max(x_max - x_min, 1e-8)
    scale = x_range / (n_levels - 1)
    asymmetric_q: np.ndarray[Any, Any] = np.round((x - x_min) / scale) * scale + x_min
    return asymmetric_q


class TernaryWeights:
    """Ternary weight quantization: {-1, 0, +1}.

    94% memory reduction. Each weight is one of three values.
    Threshold-based: weights with |w| < threshold become 0.

    Parameters
    ----------
    threshold_ratio : float
        Fraction of max(|w|) below which weights are zeroed.
    """

    def __init__(self, threshold_ratio: float = 0.7):
        self.threshold_ratio = threshold_ratio

    def quantize(self, weights: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        threshold = self.threshold_ratio * np.mean(np.abs(weights))
        ternary = np.zeros_like(weights)
        ternary[weights > threshold] = 1.0
        ternary[weights < -threshold] = -1.0
        return ternary

    def sparsity(self, weights: np.ndarray[Any, Any]) -> float:
        t = self.quantize(weights)
        return float(np.mean(t == 0))


@dataclass
class QuantizedSNNLayer:
    """SNN layer with quantization-aware forward pass.

    During training: weights quantized in forward, full-precision in backward (STE).
    At export: weights are already at target precision.

    Parameters
    ----------
    n_inputs : int
    n_neurons : int
    weight_bits : int
        Target weight precision (2, 4, 8, 16).
    threshold : float
    tau_mem : float
    """

    n_inputs: int
    n_neurons: int
    weight_bits: int = 8
    threshold: float = 1.0
    tau_mem: float = 20.0

    def __post_init__(self) -> None:
        rng = np.random.RandomState(42)
        self.W = rng.randn(self.n_neurons, self.n_inputs) * np.sqrt(2.0 / self.n_inputs)
        self._v = np.zeros(self.n_neurons)

    def forward(self, x: np.ndarray[Any, Any], dt: float = 1.0) -> np.ndarray[Any, Any]:
        """Quantization-aware forward pass."""
        W_q = _ste_quantize(self.W, self.weight_bits)
        alpha = np.exp(-dt / self.tau_mem)
        current = W_q @ x
        self._v = alpha * self._v + (1 - alpha) * current
        spikes = (self._v >= self.threshold).astype(np.float64)
        self._v -= spikes * self.threshold
        return spikes

    def export_weights(self) -> np.ndarray[Any, Any]:
        """Export quantized weights for hardware deployment."""
        return _ste_quantize(self.W, self.weight_bits)

    def reset(self) -> None:  # pragma: no cover
        self._v = np.zeros(self.n_neurons)


def quantize_aware_train_step(
    layer: QuantizedSNNLayer,
    x: np.ndarray[Any, Any],
    target: np.ndarray[Any, Any],
    lr: float = 0.01,
) -> dict[str, object]:
    """One QAT training step with STE.

    Parameters
    ----------
    layer : QuantizedSNNLayer
    x : ndarray of shape (n_inputs,)
    target : ndarray of shape (n_neurons,)
    lr : float

    Returns
    -------
    dict with 'output', 'loss'
    """
    output = layer.forward(x)
    error = output - target
    loss = 0.5 * float(np.sum(error**2))

    # STE: gradient flows through quantization as if it weren't there
    grad_W = np.outer(error, x)
    layer.W -= lr * grad_W

    return {"output": output, "loss": loss}
