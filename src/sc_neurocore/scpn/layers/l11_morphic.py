# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L11 Morphic Resonance / Noospheric Layer

"""
SCPN L11: Morphic Resonance / Noospheric Layer (Stochastic Implementation)

Ising-style spin-glass with memetic SIR dynamics for cultural/informational
field evolution.

H = -sum(J_ij σ_i σ_j) - sum(h_i σ_i)  (NTHS Hamiltonian)
dS/dt = -β S I / N  (memetic SIR)

Ref: Paper 11 — Noosphere-Technosphere Hybrid System.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class L11_StochasticParameters:
    n_nodes: int = 100
    bitstream_length: int = 1024
    j_coupling: float = 0.5
    h_bias: float = 0.1
    beta_infection: float = 0.2
    gamma_recovery: float = 0.05
    boundary_coupling: float = 0.1  # from L10


class L11_MorphicLayer:
    """Noospheric spin-glass with memetic spreading dynamics."""

    def __init__(self, params: Optional[L11_StochasticParameters] = None):
        self.params = params or L11_StochasticParameters()
        n = self.params.n_nodes
        self.spins = np.full(n, 0.5)
        self.info_density = np.zeros(n)
        self.time = 0.0

    def step(
        self,
        dt: float,
        l10_input: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.time += dt
        n = self.params.n_nodes

        field_input = np.zeros(n)
        if l10_input is not None and "integrity" in l10_input:
            field_input = np.full(n, l10_input["integrity"] * 0.1)

        mean_field = np.mean(self.spins)
        d_spin = (
            self.params.j_coupling * mean_field
            + self.params.h_bias
            + field_input
            - 0.1 * self.spins
        )
        self.spins = np.clip(self.spins + d_spin * dt, 0, 1)
        self.info_density = 0.9 * self.info_density + 0.1 * np.abs(self.spins - 0.5)

        rands = np.random.random((n, self.params.bitstream_length))
        output_bitstreams = (rands < self.spins[:, None]).astype(np.uint8)

        return {
            "spins": self.spins.copy(),
            "polarization": float(np.std(self.spins)),
            "info_saturation": float(np.mean(self.info_density)),
            "output_bitstreams": output_bitstreams,
        }

    def get_global_metric(self) -> float:
        return float(np.mean(self.spins))
