# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — FPGA mismatch simulation for hardware-aware training

"""Simulate FPGA hardware imperfections during SNN training.

Real FPGA implementations suffer from:
- Q8.8 quantization noise (finite precision)
- Clock jitter (timing variation per cycle)
- Weight perturbation (process variation in LUT/BRAM)
- Threshold mismatch (per-neuron variation in comparators)
- Routing delay variation (path-dependent timing skew)

Training through these imperfections produces networks that tolerate
hardware mismatch at deployment time.

Calibrated against published data: ~20% coefficient of variation
for analog mixed-signal neuromorphic processors. Digital FPGAs have
lower variation (~1-5%) but Q8.8 quantization is the dominant error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FPGAMismatchModel:
    """Wraps weight matrices and neuron parameters with FPGA imperfections.

    Parameters
    ----------
    quantization_bits : int
        Fixed-point bit width (default 16 for Q8.8).
    weight_cv : float
        Coefficient of variation for weight perturbation (default 0.02 = 2%).
    threshold_cv : float
        Per-neuron threshold variation (default 0.05 = 5%).
    clock_jitter_pct : float
        Clock period variation (default 0.01 = 1%).
    seed : int
        Random seed for reproducibility.
    """

    quantization_bits: int = 16
    weight_cv: float = 0.02
    threshold_cv: float = 0.05
    clock_jitter_pct: float = 0.01
    seed: int = 42

    def __post_init__(self) -> None:
        self._rng = np.random.RandomState(self.seed)

    def quantize(self, values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Apply Q-format quantization noise."""
        fraction = self.quantization_bits // 2
        scale = 1 << fraction
        quantized = np.round(values * scale) / scale
        return quantized

    def perturb_weights(self, weights: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Add process variation noise to weights."""
        noise = self._rng.normal(0, self.weight_cv, weights.shape)
        return self.quantize(weights * (1.0 + noise))

    def perturb_thresholds(self, thresholds: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Add per-neuron threshold mismatch."""
        noise = self._rng.normal(0, self.threshold_cv, thresholds.shape)
        return self.quantize(thresholds * (1.0 + noise))

    def jitter_timing(self, n_steps: int) -> np.ndarray[Any, Any]:
        """Generate clock jitter: per-step timing variation."""
        jitter = self._rng.normal(1.0, self.clock_jitter_pct, n_steps)
        return np.clip(jitter, 0.9, 1.1)

    def apply_to_network_weights(
        self, weights: list[np.ndarray[Any, Any]]
    ) -> list[np.ndarray[Any, Any]]:
        """Apply all hardware imperfections to a list of weight matrices."""
        return [self.perturb_weights(w) for w in weights]

    def mismatch_report(self, weights: list[np.ndarray[Any, Any]]) -> dict[str, object]:
        """Report expected mismatch statistics for given weights."""
        perturbed = self.apply_to_network_weights(weights)
        total_params = sum(w.size for w in weights)
        total_error = sum(np.abs(w - p).sum() for w, p in zip(weights, perturbed))
        max_error = max(np.abs(w - p).max() for w, p in zip(weights, perturbed))
        return {
            "total_parameters": total_params,
            "mean_absolute_error": float(total_error / max(total_params, 1)),
            "max_absolute_error": float(max_error),
            "weight_cv": self.weight_cv,
            "threshold_cv": self.threshold_cv,
            "quantization_bits": self.quantization_bits,
        }
