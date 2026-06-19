# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L14 Transdimensional Integration Layer (Stochastic

"""SCPN L14: transdimensional integration layer (stochastic implementation).

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
    """Stochastic configuration parameters for the SCPN transdimensional integration layer."""

    n_dimensions: int = 13  # one per lower layer
    bitstream_length: int = 1024
    integration_weights: Optional[np.ndarray[Any, Any]] = None
    temporal_coupling: float = 0.1  # from L13
    bridge_decoherence_coupling: float = 0.1
    resonance_lock_tolerance: float = 1e-6
    rng_seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate and finalise the layer parameters after construction."""
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
        if self.params.integration_weights is None:
            raise ValueError("integration_weights must be initialised by L14_StochasticParameters")
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
        """Advance the transdimensional integration layer one timestep and return its output state."""
        self._validate_step_inputs(dt, layer_metrics, l13_input)
        self.time += dt

        if layer_metrics is not None:
            values = self._metric_vector(layer_metrics, self.params.n_dimensions)
            self.layer_metrics[: len(values)] = values

        transdimensional_bridge_drive = 0.0
        holographic_protection_load = 0.0
        boundary_context_id: Optional[str] = None
        boundary_terminals: tuple[str, ...] = ()
        bridge_terminal_set: tuple[str, ...] = ()
        bridge_terminal_bandwidth = 1.0
        if l13_input is not None:
            bridge_effect = self._l13_bridge_effect(l13_input)
            transdimensional_bridge_drive = bridge_effect["bridge_drive"]
            holographic_protection_load = bridge_effect["protection_load"]
            boundary_context_id = bridge_effect["boundary_context_id"]
            boundary_terminals = bridge_effect["boundary_terminals"]
            bridge_terminal_set = bridge_effect["bridge_terminal_set"]
            bridge_terminal_bandwidth = bridge_effect["bridge_terminal_bandwidth"]
            source_drive = np.clip(
                transdimensional_bridge_drive
                - self.params.bridge_decoherence_coupling * holographic_protection_load,
                0.0,
                1.0,
            )
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
        self.resonance_lock = (
            abs(self.resonance_determinant) <= self.params.resonance_lock_tolerance
        )

        activation = np.full(self.params.n_dimensions, self.integrated_coherence)
        activation[...] = np.clip(activation, 0, 1).astype(np.float64, copy=False)

        rands = self._rng.random((self.params.n_dimensions, self.params.bitstream_length))
        output_bitstreams = (rands < activation[:, None]).astype(np.uint8)

        return {
            "integrated_coherence": self.integrated_coherence,
            "layer_metrics": self.layer_metrics.copy(),
            "resonance_determinant": self.resonance_determinant,
            "resonance_lock": self.resonance_lock,
            "transdimensional_bridge_drive": transdimensional_bridge_drive,
            "holographic_protection_load": holographic_protection_load,
            "boundary_context_id": boundary_context_id,
            "boundary_terminals": boundary_terminals,
            "bridge_terminal_set": bridge_terminal_set,
            "bridge_terminal_bandwidth": bridge_terminal_bandwidth,
            "output_bitstreams": output_bitstreams,
        }

    def get_global_metric(self) -> float:
        """Return the scalar global metric summarising this layer's state."""
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
        if (
            not np.all(np.isfinite(weights))
            or np.any(weights < 0.0)
            or float(np.sum(weights)) <= 0.0
        ):
            raise ValueError("integration_weights must be finite, non-negative, and non-zero")
        if not math.isfinite(float(params.temporal_coupling)) or params.temporal_coupling < 0.0:
            raise ValueError("temporal_coupling must be finite and non-negative")
        if (
            not math.isfinite(float(params.bridge_decoherence_coupling))
            or params.bridge_decoherence_coupling < 0.0
        ):
            raise ValueError("bridge_decoherence_coupling must be finite and non-negative")
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
        if l13_input is not None:
            cls._l13_bridge_effect(l13_input)

    @staticmethod
    def _normalised_weights(weights: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        values = np.asarray(weights, dtype=np.float64)
        return values / float(np.sum(values))

    @staticmethod
    def _metric_vector(layer_metrics: Dict[str, float], limit: int) -> np.ndarray[Any, Any]:
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
        if np.any(arr < 0.0) or np.any(arr > 1.0):
            raise ValueError(f"{name} values must be within [0, 1]")
        mean = float(np.mean(arr))
        return mean

    @classmethod
    def _l13_bridge_effect(cls, l13_input: Dict[str, Any]) -> Dict[str, Any]:
        bridge_context = cls._bridge_context(l13_input)
        if "source_sampling_signal" not in l13_input:
            if "source_field" in l13_input:
                return {
                    "bridge_drive": cls._finite_mean(l13_input["source_field"], "source_field")
                    * bridge_context["bridge_terminal_bandwidth"],
                    "protection_load": 0.0,
                    "boundary_context_id": bridge_context["boundary_context_id"],
                    "boundary_terminals": bridge_context["boundary_terminals"],
                    "bridge_terminal_set": bridge_context["bridge_terminal_set"],
                    "bridge_terminal_bandwidth": bridge_context["bridge_terminal_bandwidth"],
                }
            return {
                "bridge_drive": 0.0,
                "protection_load": 0.0,
                "boundary_context_id": bridge_context["boundary_context_id"],
                "boundary_terminals": bridge_context["boundary_terminals"],
                "bridge_terminal_set": bridge_context["bridge_terminal_set"],
                "bridge_terminal_bandwidth": bridge_context["bridge_terminal_bandwidth"],
            }

        source_sampling = cls._finite_mean(
            l13_input["source_sampling_signal"], "source_sampling_signal"
        )
        source_gain = cls._finite_scalar(
            l13_input.get("source_sampling_gain", 0.0), "source_sampling_gain"
        )
        binding_strength = cls._unit_scalar(
            l13_input.get("binding_strength", 0.0), "binding_strength"
        )
        protection_load = cls._finite_scalar(
            l13_input.get("temporal_decoherence_load", 0.0), "temporal_decoherence_load"
        )
        bridge_drive = float(
            np.clip(
                source_sampling + source_gain + binding_strength,
                0.0,
                1.0,
            )
            * bridge_context["bridge_terminal_bandwidth"]
        )
        return {
            "bridge_drive": bridge_drive,
            "protection_load": protection_load,
            "boundary_context_id": bridge_context["boundary_context_id"],
            "boundary_terminals": bridge_context["boundary_terminals"],
            "bridge_terminal_set": bridge_context["bridge_terminal_set"],
            "bridge_terminal_bandwidth": bridge_context["bridge_terminal_bandwidth"],
        }

    @staticmethod
    def _bridge_context(l13_input: Dict[str, Any]) -> Dict[str, Any]:
        has_context_id = "boundary_context_id" in l13_input
        has_terminals = "boundary_terminals" in l13_input
        if not has_context_id and not has_terminals:
            return {
                "boundary_context_id": None,
                "boundary_terminals": (),
                "bridge_terminal_set": (),
                "bridge_terminal_bandwidth": 1.0,
            }
        if not has_context_id or not has_terminals:
            raise ValueError("boundary context requires boundary_context_id and boundary_terminals")

        raw_context_id = l13_input["boundary_context_id"]
        terminals = tuple(l13_input["boundary_terminals"])
        if raw_context_id is None and not terminals:
            return {
                "boundary_context_id": None,
                "boundary_terminals": (),
                "bridge_terminal_set": (),
                "bridge_terminal_bandwidth": 1.0,
            }
        context_id = str(raw_context_id)
        if not context_id:
            raise ValueError("boundary_context_id must be non-empty")
        valid_terminals = {"T1", "T2", "T3", "T4", "T5", "T6", "T7"}
        if not terminals or any(terminal not in valid_terminals for terminal in terminals):
            raise ValueError("boundary_terminals must contain valid T1-T7 terminal identifiers")

        bridge_terminals = tuple(terminal for terminal in terminals if terminal in {"T2", "T5"})
        return {
            "boundary_context_id": context_id,
            "boundary_terminals": terminals,
            "bridge_terminal_set": bridge_terminals,
            "bridge_terminal_bandwidth": float(len(bridge_terminals) / 2.0),
        }

    @staticmethod
    def _finite_scalar(value: Any, name: str) -> float:
        values = np.asarray(value, dtype=np.float64)
        if values.shape != ():
            raise ValueError(f"{name} must be a finite scalar")
        scalar = float(values)
        if not math.isfinite(scalar) or scalar < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return scalar

    @classmethod
    def _unit_scalar(cls, value: Any, name: str) -> float:
        scalar = cls._finite_scalar(value, name)
        if scalar > 1.0:
            raise ValueError(f"{name} must be within [0, 1]")
        return scalar
