# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L14 Transdimensional Integration Layer (Stochastic

"""
SCPN L14: Transdimensional Integration Layer (Stochastic Implementation)

Weighted aggregation across all lower layers to produce a unified
coherence metric. Acts as the integration hub of the SCPN stack.

I_global = sum_n w_n * M_n  (weighted layer metrics)

Ref: Paper 14 — Transdimensional Resonance.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any, Dict, Optional

import numpy as np

# Default weights for 13 lower-layer metrics (L1-L13)
_DEFAULT_WEIGHTS = np.array(
    [
        0.10,
        0.08,
        0.06,
        0.10,
        0.08,  # L1-L5
        0.06,
        0.08,
        0.08,
        0.07,
        0.07,  # L6-L10
        0.06,
        0.08,
        0.08,  # L11-L13
    ]
)


@dataclass
class L14_StochasticParameters:
    n_dimensions: int = 13  # one per lower layer
    bitstream_length: int = 1024
    integration_weights: Optional[np.ndarray] = None
    temporal_coupling: float = 0.1  # from L13
    resonance_lock_tolerance: float = 1e-6
    rng_seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.integration_weights is None:
            if self.n_dimensions == len(_DEFAULT_WEIGHTS):
                self.integration_weights = _DEFAULT_WEIGHTS.copy()
            else:
                self.integration_weights = np.ones(self.n_dimensions, dtype=np.float64)
        else:
            self.integration_weights = np.asarray(self.integration_weights, dtype=np.float64)


class L14_IntegrationLayer:
    """Weighted integration across SCPN layer metrics."""

    def __init__(self, params: Optional[L14_StochasticParameters] = None):
        self.params = params or L14_StochasticParameters()
        self._validate_params(self.params)
        self._rng = np.random.default_rng(self.params.rng_seed)
        self.integration_weights = self._normalised_weights(self.params.integration_weights)
        self.layer_metrics = np.zeros(self.params.n_dimensions)
        self.integrated_coherence = 0.5
        self.resonance_determinant = 1.0
        self.resonance_lock = False
        self.time = 0.0

    def step(
        self,
        dt: float,
        layer_metrics: Optional[Dict[str, float]] = None,
        l13_input: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._validate_step_inputs(dt, layer_metrics, l13_input)
        self.time += dt

        if layer_metrics is not None:
            values = self._metric_vector(layer_metrics, self.params.n_dimensions)
            self.layer_metrics[: len(values)] = values

        if l13_input is not None and "source_field" in l13_input:
            source_drive = self._finite_mean(l13_input["source_field"], "source_field")
            self.layer_metrics[-1] = np.clip(
                (1.0 - self.params.temporal_coupling) * self.layer_metrics[-1]
                + self.params.temporal_coupling * source_drive,
                0.0,
                1.0,
            )

        self.integrated_coherence = float(np.dot(self.integration_weights, self.layer_metrics))
        self.integrated_coherence = float(np.clip(self.integrated_coherence, 0.0, 1.0))
        resonance_matrix = np.diag(self.layer_metrics - self.integrated_coherence)
        self.resonance_determinant = float(np.linalg.det(resonance_matrix))
        self.resonance_lock = abs(self.resonance_determinant) <= self.params.resonance_lock_tolerance

        activation = np.full(self.params.n_dimensions, self.integrated_coherence)
        activation = np.clip(activation, 0, 1).astype(np.float64, copy=False)

        rands = self._rng.random((self.params.n_dimensions, self.params.bitstream_length))
        output_bitstreams = (rands < activation[:, None]).astype(np.uint8)

        return {
            "integrated_coherence": self.integrated_coherence,
            "layer_metrics": self.layer_metrics.copy(),
            "resonance_determinant": self.resonance_determinant,
            "resonance_lock": self.resonance_lock,
            "output_bitstreams": output_bitstreams,
        }

    def get_global_metric(self) -> float:
        return self.integrated_coherence

    @staticmethod
    def _validate_params(params: L14_StochasticParameters) -> None:
        if (
            not isinstance(params.n_dimensions, int)
            or isinstance(params.n_dimensions, bool)
            or params.n_dimensions <= 0
        ):
            raise ValueError("n_dimensions must be a positive integer")
        if (
            not isinstance(params.bitstream_length, int)
            or isinstance(params.bitstream_length, bool)
            or params.bitstream_length <= 0
        ):
            raise ValueError("bitstream_length must be a positive integer")
        weights = np.asarray(params.integration_weights, dtype=np.float64)
        if weights.shape != (params.n_dimensions,):
            raise ValueError("integration_weights must contain one value per dimension")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or float(np.sum(weights)) <= 0.0:
            raise ValueError("integration_weights must be finite, non-negative, and non-zero")
        if not math.isfinite(float(params.temporal_coupling)) or params.temporal_coupling < 0.0:
            raise ValueError("temporal_coupling must be finite and non-negative")
        if (
            not math.isfinite(float(params.resonance_lock_tolerance))
            or params.resonance_lock_tolerance <= 0.0
        ):
            raise ValueError("resonance_lock_tolerance must be finite and positive")
        if params.rng_seed is not None:
            if isinstance(params.rng_seed, bool) or not isinstance(params.rng_seed, int):
                raise ValueError("rng_seed must be a non-negative integer or None")
            if params.rng_seed < 0:
                raise ValueError("rng_seed must be a non-negative integer or None")

    @classmethod
    def _validate_step_inputs(
        cls,
        dt: float,
        layer_metrics: Optional[Dict[str, float]],
        l13_input: Optional[Dict[str, Any]],
    ) -> None:
        if not math.isfinite(float(dt)) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if layer_metrics is not None:
            cls._metric_vector(layer_metrics, len(layer_metrics))
        if l13_input is not None and "source_field" in l13_input:
            cls._finite_mean(l13_input["source_field"], "source_field")

    @staticmethod
    def _normalised_weights(weights: np.ndarray) -> np.ndarray:
        values = np.asarray(weights, dtype=np.float64)
        return values / float(np.sum(values))

    @staticmethod
    def _metric_vector(layer_metrics: Dict[str, float], limit: int) -> np.ndarray:
        values = np.asarray(list(layer_metrics.values())[:limit], dtype=np.float64)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("layer_metrics must contain finite values")
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("layer_metrics must be within [0, 1]")
        return values

    @staticmethod
    def _finite_mean(values: Any, name: str) -> float:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must contain finite values")
        mean = float(np.mean(arr))
        if mean < 0.0 or mean > 1.0:
            raise ValueError(f"{name} must be within [0, 1]")
        return mean
