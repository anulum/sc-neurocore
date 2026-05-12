# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L15 Meta-Cognitive / Consilium Layer (Stochastic

"""
SCPN L15: Meta-Cognitive / Consilium Layer (Stochastic Implementation)

Self-monitoring layer: compares integrated coherence against a target
attractor and computes the deviation signal for L16 Director feedback.

Error = |I_target - I_actual|
GCI = 1 - Error  (Global Coherence Index)

Ref: Paper 15 — Consilium and Universal Metric.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class L15_StochasticParameters:
    n_monitors: int = 16  # one per SCPN layer
    bitstream_length: int = 1024
    target_coherence: float = 0.8
    smoothing_alpha: float = 0.1
    integration_coupling: float = 0.2  # from L14
    bridge_alignment_coupling: float = 0.1
    bridge_protection_coupling: float = 0.1
    rng_seed: Optional[int] = None


class L15_MetaLayer:
    """Self-monitoring meta-cognitive layer with GCI computation."""

    def __init__(self, params: Optional[L15_StochasticParameters] = None):
        self.params = params or L15_StochasticParameters()
        self._validate_params(self.params)
        self._rng = np.random.default_rng(self.params.rng_seed)
        self.umo_weights = np.full(self.params.n_monitors, 1.0 / self.params.n_monitors)
        self.gci = 0.5
        self.error_history = np.zeros(self.params.n_monitors)
        self.ethical_dissonance = 0.0
        self.free_energy = 0.0
        self.oversoul_attractor = 0.5
        self.time = 0.0

    def step(
        self,
        dt: float,
        l14_input: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        (
            actual,
            resonance_penalty,
            bridge_credit,
            bridge_penalty,
            boundary_context_id,
            boundary_terminals,
            consilium_terminal_set,
            consilium_terminal_bandwidth,
        ) = self._validate_step_inputs(
            dt,
            l14_input,
            self.params,
        )
        self.time += dt

        error = abs(self.params.target_coherence - actual)
        self.ethical_dissonance = float(
            np.clip(
                error
                + self.params.integration_coupling * resonance_penalty
                + bridge_penalty
                - bridge_credit,
                0.0,
                1.0,
            )
        )
        self.free_energy = self.ethical_dissonance**2
        self.oversoul_attractor = 1.0 - self.ethical_dissonance

        gain = self.params.smoothing_alpha * self.params.integration_coupling
        self.gci += gain * (self.oversoul_attractor - self.gci)
        self.gci = float(np.clip(self.gci, 0.0, 1.0))

        # Per-monitor error tracking (shift and append)
        self.error_history = np.roll(self.error_history, -1)
        self.error_history[-1] = error

        activation = np.full(self.params.n_monitors, np.clip(self.gci, 0, 1))
        rands = self._rng.random((self.params.n_monitors, self.params.bitstream_length))
        output_bitstreams = (rands < activation[:, None]).astype(np.uint8)

        return {
            "gci": self.gci,
            "error": error,
            "actual_coherence": actual,
            "error_trend": float(np.mean(self.error_history)),
            "ethical_dissonance": self.ethical_dissonance,
            "free_energy": self.free_energy,
            "oversoul_attractor": self.oversoul_attractor,
            "bridge_alignment_credit": bridge_credit,
            "bridge_protection_penalty": bridge_penalty,
            "boundary_context_id": boundary_context_id,
            "boundary_terminals": boundary_terminals,
            "consilium_terminal_set": consilium_terminal_set,
            "consilium_terminal_bandwidth": consilium_terminal_bandwidth,
            "umo_weights": self.umo_weights.copy(),
            "output_bitstreams": output_bitstreams,
        }

    def get_global_metric(self) -> float:
        return self.gci

    @staticmethod
    def _validate_params(params: L15_StochasticParameters) -> None:
        if params.n_monitors <= 0:
            raise ValueError("n_monitors must be positive")
        if params.bitstream_length <= 0:
            raise ValueError("bitstream_length must be positive")
        if not math.isfinite(params.target_coherence) or not 0.0 <= params.target_coherence <= 1.0:
            raise ValueError("target_coherence must be finite and in [0, 1]")
        if not math.isfinite(params.smoothing_alpha) or not 0.0 <= params.smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be finite and in [0, 1]")
        if (
            not math.isfinite(params.integration_coupling)
            or not 0.0 <= params.integration_coupling <= 1.0
        ):
            raise ValueError("integration_coupling must be finite and in [0, 1]")
        if (
            not math.isfinite(params.bridge_alignment_coupling)
            or not 0.0 <= params.bridge_alignment_coupling <= 1.0
        ):
            raise ValueError("bridge_alignment_coupling must be finite and in [0, 1]")
        if (
            not math.isfinite(params.bridge_protection_coupling)
            or not 0.0 <= params.bridge_protection_coupling <= 1.0
        ):
            raise ValueError("bridge_protection_coupling must be finite and in [0, 1]")
        if params.rng_seed is not None and (
            isinstance(params.rng_seed, bool)
            or not isinstance(params.rng_seed, int)
            or params.rng_seed < 0
        ):
            raise ValueError("rng_seed must be non-negative when provided")

    @staticmethod
    def _validate_step_inputs(
        dt: float, l14_input: Optional[Dict[str, Any]], params: L15_StochasticParameters
    ) -> tuple[float, float, float, float, Optional[str], tuple[str, ...], tuple[str, ...], float]:
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")

        if l14_input is None:
            return 0.5, 0.0, 0.0, 0.0, None, (), (), 1.0
        consilium_context = L15_MetaLayer._consilium_context(l14_input)

        if "integrated_coherence" not in l14_input:
            raise ValueError("l14_input must include integrated_coherence")

        actual = float(l14_input["integrated_coherence"])
        if not math.isfinite(actual) or not 0.0 <= actual <= 1.0:
            raise ValueError("l14 integrated_coherence must be finite and in [0, 1]")

        resonance_lock = l14_input.get("resonance_lock", True)
        if not isinstance(resonance_lock, bool):
            raise ValueError("l14 resonance_lock must be a boolean when provided")

        determinant = float(l14_input.get("resonance_determinant", 0.0))
        if not math.isfinite(determinant):
            raise ValueError("l14 resonance_determinant must be finite when provided")

        bridge_drive = L15_MetaLayer._unit_scalar(
            l14_input.get("transdimensional_bridge_drive", 0.0),
            "l14 transdimensional_bridge_drive",
        )
        protection_load = L15_MetaLayer._nonnegative_scalar(
            l14_input.get("holographic_protection_load", 0.0),
            "l14 holographic_protection_load",
        )
        resonance_penalty = 0.0 if resonance_lock else min(abs(determinant), 1.0)
        bridge_credit = (
            params.bridge_alignment_coupling
            * bridge_drive
            * consilium_context["consilium_terminal_bandwidth"]
        )
        bridge_penalty = params.bridge_protection_coupling * protection_load
        return (
            actual,
            resonance_penalty,
            bridge_credit,
            bridge_penalty,
            consilium_context["boundary_context_id"],
            consilium_context["boundary_terminals"],
            consilium_context["consilium_terminal_set"],
            consilium_context["consilium_terminal_bandwidth"],
        )

    @staticmethod
    def _consilium_context(l14_input: Dict[str, Any]) -> Dict[str, Any]:
        has_context_id = "boundary_context_id" in l14_input
        has_terminals = "boundary_terminals" in l14_input
        if not has_context_id and not has_terminals:
            return {
                "boundary_context_id": None,
                "boundary_terminals": (),
                "consilium_terminal_set": (),
                "consilium_terminal_bandwidth": 1.0,
            }
        if not has_context_id or not has_terminals:
            raise ValueError("boundary context requires boundary_context_id and boundary_terminals")

        context_id = str(l14_input["boundary_context_id"])
        if not context_id:
            raise ValueError("boundary_context_id must be non-empty")
        terminals = tuple(l14_input["boundary_terminals"])
        valid_terminals = {"T1", "T2", "T3", "T4", "T5", "T6", "T7"}
        if not terminals or any(terminal not in valid_terminals for terminal in terminals):
            raise ValueError("boundary_terminals must contain valid T1-T7 terminal identifiers")

        consilium_terminals = tuple(
            terminal for terminal in terminals if terminal in {"T2", "T4", "T5", "T6"}
        )
        return {
            "boundary_context_id": context_id,
            "boundary_terminals": terminals,
            "consilium_terminal_set": consilium_terminals,
            "consilium_terminal_bandwidth": float(len(consilium_terminals) / 4.0),
        }

    @staticmethod
    def _nonnegative_scalar(value: Any, name: str) -> float:
        values = np.asarray(value, dtype=np.float64)
        if values.shape != ():
            raise ValueError(f"{name} must be a finite scalar")
        scalar = float(values)
        if not math.isfinite(scalar) or scalar < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return scalar

    @classmethod
    def _unit_scalar(cls, value: Any, name: str) -> float:
        scalar = cls._nonnegative_scalar(value, name)
        if scalar > 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
        return scalar
