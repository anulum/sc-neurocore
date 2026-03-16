# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

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


class L10_BoundaryLayer:
    """Topological firewall with dissonance rejection."""

    def __init__(self, params: Optional[L10_StochasticParameters] = None):
        self.params = params or L10_StochasticParameters()
        n = self.params.n_boundary_nodes
        self.firewall_strength = np.full(n, 0.9)
        self.intention = np.zeros(n)
        self.time = 0.0

    def step(
        self,
        dt: float,
        l9_input: Optional[Dict[str, Any]] = None,
        external_noise: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        self.time += dt
        n = self.params.n_boundary_nodes

        noise = np.zeros(n)
        if external_noise is not None:
            noise = (
                external_noise[:n]
                if len(external_noise) >= n
                else np.pad(external_noise, (0, n - len(external_noise)))
            )

        if l9_input is not None and "retrieval_quality" in l9_input:
            self.intention = np.full(n, l9_input["retrieval_quality"])

        dissonance = np.abs(noise - self.intention)
        d_strength = (
            -dissonance * self.firewall_strength
            + self.params.steering_gain * self.intention
            - 0.01 * self.firewall_strength
        )
        self.firewall_strength = np.clip(self.firewall_strength + d_strength * dt, 0, 1)

        rands = np.random.random((n, self.params.bitstream_length))
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
