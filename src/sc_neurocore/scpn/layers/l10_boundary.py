# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L10 Boundary Firewall Layer (Stochastic Implementation)

"""SCPN L10: boundary firewall layer (stochastic implementation).

Topological boundary insulation with dissonance-triggered rejection.
Firewall strength decays under external noise (dissonance) and grows
under intentional steering.

Shielding ~ exp(-|∇V|² / σ)
D_topo = 1 - overlap(Ψ_local, Ψ_template)

Ref: Paper 10 — Projective Field Boundary Control.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class L10_StochasticParameters:
    """Stochastic configuration parameters for the L10 boundary firewall layer."""

    n_boundary_nodes: int = 100
    bitstream_length: int = 1024
    rejection_threshold: float = 0.4
    shielding_strength: float = 1.5
    steering_gain: float = 0.2
    memory_coupling: float = 0.1  # from L9
    qec_coupling: float = 0.2
    rng_seed: Optional[int] = None


class L10_BoundaryLayer:
    """Topological firewall with dissonance rejection."""

    def __init__(self, params: Optional[L10_StochasticParameters] = None):
        self.params = params or L10_StochasticParameters()
        self._validate_params(self.params)
        n = self.params.n_boundary_nodes
        self.firewall_strength = np.full(n, 0.9)
        self.intention = np.zeros(n)
        self.time = 0.0
        self._rng = np.random.default_rng(self.params.rng_seed)

    def step(
        self,
        dt: float,
        l9_input: Optional[Dict[str, Any]] = None,
        external_noise: Optional[np.ndarray[Any, Any]] = None,
    ) -> Dict[str, Any]:
        """Advance the firewall one timestep and return its boundary state."""
        if not math.isfinite(float(dt)) or float(dt) <= 0.0:
            raise ValueError("dt must be finite and positive")
        self.time += dt
        n = self.params.n_boundary_nodes

        noise = np.zeros(n)
        if external_noise is not None:
            noise = self._noise_vector(external_noise, n)

        if l9_input is not None and "retrieval_quality" in l9_input:
            retrieval_quality = self._retrieval_quality(l9_input["retrieval_quality"])
            self.intention = np.full(n, retrieval_quality * self.params.memory_coupling)

        qec_residual = np.zeros(n, dtype=np.float64)
        memory_complexity_flux = 0.0
        boundary_context = self._boundary_context(l9_input)
        if l9_input is not None:
            qec_residual = self._qec_residual(l9_input, n)
            memory_complexity_flux = self._memory_complexity_flux(l9_input)

        dissonance = (
            np.abs(noise - self.intention)
            + self.params.qec_coupling * qec_residual
            + memory_complexity_flux
        )
        rejection_excess = np.maximum(dissonance - self.params.rejection_threshold, 0.0)
        shielding_loss = rejection_excess * self.firewall_strength / self.params.shielding_strength
        d_strength = (
            -shielding_loss
            + self.params.steering_gain * self.intention
            - 0.01 * self.firewall_strength
        )
        self.firewall_strength = np.clip(self.firewall_strength + d_strength * dt, 0, 1)

        rands = self._rng.random((n, self.params.bitstream_length))
        output_bitstreams = (rands < self.firewall_strength[:, None]).astype(np.uint8)

        return {
            "firewall_strength": self.firewall_strength.copy(),
            "dissonance": float(np.mean(dissonance)),
            "integrity": self._integrity(),
            "qec_residual_load": float(np.mean(qec_residual)),
            "memory_complexity_flux": memory_complexity_flux,
            "boundary_complexity": float(np.mean(rejection_excess)),
            "boundary_context_id": boundary_context["ebs_id"],
            "boundary_terminals": boundary_context["terminal_set"],
            "topological_rejection_mask": rejection_excess > 0.0,
            "output_bitstreams": output_bitstreams,
        }

    def _integrity(self) -> float:
        return float(np.mean(self.firewall_strength))

    def get_global_metric(self) -> float:
        """Return the scalar boundary-integrity metric for this layer."""
        return self._integrity()

    @staticmethod
    def _validate_params(params: L10_StochasticParameters) -> None:
        if not isinstance(params.n_boundary_nodes, int) or isinstance(
            params.n_boundary_nodes, bool
        ):
            raise ValueError("n_boundary_nodes must be a positive integer")
        if params.n_boundary_nodes <= 0:
            raise ValueError("n_boundary_nodes must be positive")
        if not isinstance(params.bitstream_length, int) or isinstance(
            params.bitstream_length, bool
        ):
            raise ValueError("bitstream_length must be a positive integer")
        if params.bitstream_length <= 0:
            raise ValueError("bitstream_length must be positive")
        if (
            not math.isfinite(float(params.rejection_threshold))
            or not 0.0 <= params.rejection_threshold <= 1.0
        ):
            raise ValueError("rejection_threshold must be finite and in [0, 1]")
        if not math.isfinite(float(params.shielding_strength)) or params.shielding_strength <= 0.0:
            raise ValueError("shielding_strength must be finite and positive")
        if not math.isfinite(float(params.steering_gain)) or params.steering_gain < 0.0:
            raise ValueError("steering_gain must be finite and non-negative")
        if not math.isfinite(float(params.memory_coupling)) or params.memory_coupling < 0.0:
            raise ValueError("memory_coupling must be finite and non-negative")
        if (
            not math.isfinite(float(params.qec_coupling))
            or params.qec_coupling < 0.0
            or params.qec_coupling > 1.0
        ):
            raise ValueError("qec_coupling must be finite and in [0, 1]")
        if params.rng_seed is not None:
            if isinstance(params.rng_seed, bool) or not isinstance(params.rng_seed, int):
                raise ValueError("rng_seed must be a non-negative integer or None")
            if params.rng_seed < 0:
                raise ValueError("rng_seed must be a non-negative integer or None")

    @staticmethod
    def _retrieval_quality(value: Any) -> float:
        values = np.asarray(value, dtype=np.float64)
        if values.shape != ():
            raise ValueError("retrieval_quality must be a finite scalar")
        retrieval_quality = float(values)
        if not math.isfinite(retrieval_quality):
            raise ValueError("retrieval_quality must be a finite scalar")
        return retrieval_quality

    @staticmethod
    def _qec_residual(l9_input: Dict[str, Any], n_boundary_nodes: int) -> np.ndarray[Any, Any]:
        if "qec_syndrome" not in l9_input:
            return np.zeros(n_boundary_nodes, dtype=np.float64)

        syndrome = L10_BoundaryLayer._bounded_l9_vector(
            l9_input["qec_syndrome"], n_boundary_nodes, "qec_syndrome", pad_value=0.0
        )
        recovery = L10_BoundaryLayer._bounded_l9_vector(
            l9_input.get("recovery_operator", np.zeros(n_boundary_nodes)),
            n_boundary_nodes,
            "recovery_operator",
            pad_value=1.0,
        )
        return np.clip(syndrome * (1.0 - recovery), 0.0, 1.0)

    @staticmethod
    def _bounded_l9_vector(
        value: Any, n_boundary_nodes: int, name: str, *, pad_value: float
    ) -> np.ndarray[Any, Any]:
        values = np.asarray(value, dtype=np.float64).reshape(-1)
        if values.size == 0:
            raise ValueError(f"{name} must contain at least one value")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values")
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError(f"{name} values must be within [0, 1]")
        if values.size >= n_boundary_nodes:
            return values[:n_boundary_nodes].copy()
        return np.pad(values, (0, n_boundary_nodes - values.size), constant_values=pad_value)

    def _memory_complexity_flux(self, l9_input: Dict[str, Any]) -> float:
        if "memory_free_energy" not in l9_input:
            return 0.0
        values = np.asarray(l9_input["memory_free_energy"], dtype=np.float64)
        if values.shape != ():
            raise ValueError("memory_free_energy must be a finite scalar")
        free_energy = float(values)
        if not math.isfinite(free_energy) or free_energy < 0.0:
            raise ValueError("memory_free_energy must be finite and non-negative")
        return float(np.clip(free_energy, 0.0, 1.0) * self.params.qec_coupling)

    @staticmethod
    def _boundary_context(l9_input: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if l9_input is None:
            return {"ebs_id": None, "terminal_set": ()}
        has_context_id = "boundary_context_id" in l9_input
        has_terminals = "boundary_terminals" in l9_input
        if not has_context_id and not has_terminals:
            return {"ebs_id": None, "terminal_set": ()}
        if not has_context_id or not has_terminals:
            raise ValueError("boundary context requires boundary_context_id and boundary_terminals")
        raw_context_id = l9_input["boundary_context_id"]
        terminals = tuple(l9_input["boundary_terminals"])
        if raw_context_id is None and not terminals:
            return {"ebs_id": None, "terminal_set": ()}
        ebs_id = str(raw_context_id)
        if not ebs_id:
            raise ValueError("boundary_context_id must be non-empty")
        valid_terminals = {"T1", "T2", "T3", "T4", "T5", "T6", "T7"}
        if not terminals or any(terminal not in valid_terminals for terminal in terminals):
            raise ValueError("boundary_terminals must contain valid T1-T7 terminal identifiers")
        return {"ebs_id": ebs_id, "terminal_set": terminals}

    @staticmethod
    def _noise_vector(
        external_noise: np.ndarray[Any, Any], n_boundary_nodes: int
    ) -> np.ndarray[Any, Any]:
        values = np.asarray(external_noise, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(values)):
            raise ValueError("external_noise must contain only finite values")
        if values.size >= n_boundary_nodes:
            return values[:n_boundary_nodes].copy()
        return np.pad(values, (0, n_boundary_nodes - values.size))
