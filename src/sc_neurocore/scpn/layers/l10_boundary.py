# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L10 Boundary Firewall Layer (Stochastic Implementation)

"""
SCPN L10: Boundary Firewall Layer (Stochastic Implementation)

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
    n_boundary_nodes: int = 100
    bitstream_length: int = 1024
    rejection_threshold: float = 0.4
    shielding_strength: float = 1.5
    steering_gain: float = 0.2
    memory_coupling: float = 0.1  # from L9
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
        external_noise: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
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

        dissonance = np.abs(noise - self.intention)
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
            "output_bitstreams": output_bitstreams,
        }

    def _integrity(self) -> float:
        return float(np.mean(self.firewall_strength))

    def get_global_metric(self) -> float:
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
        if (
            not math.isfinite(float(params.shielding_strength))
            or params.shielding_strength <= 0.0
        ):
            raise ValueError("shielding_strength must be finite and positive")
        if not math.isfinite(float(params.steering_gain)) or params.steering_gain < 0.0:
            raise ValueError("steering_gain must be finite and non-negative")
        if not math.isfinite(float(params.memory_coupling)) or params.memory_coupling < 0.0:
            raise ValueError("memory_coupling must be finite and non-negative")
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
    def _noise_vector(external_noise: np.ndarray, n_boundary_nodes: int) -> np.ndarray:
        values = np.asarray(external_noise, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(values)):
            raise ValueError("external_noise must contain only finite values")
        if values.size >= n_boundary_nodes:
            return values[:n_boundary_nodes].copy()
        return np.pad(values, (0, n_boundary_nodes - values.size))
