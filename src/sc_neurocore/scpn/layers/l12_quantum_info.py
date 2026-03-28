# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L12 Quantum Information / Gaian Layer (Stochastic

"""
SCPN L12: Quantum Information / Gaian Layer (Stochastic Implementation)

Environment-Assisted Quantum Transport (ENAQT) model for ecological
coherence across a network of sites.

S = -Tr(ρ ln ρ)  (von Neumann entropy)
Transport efficiency: population transfer under dephasing noise.

Ref: Paper 12 — Gaian ENAQT and ecological quantum coherence.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class L12_StochasticParameters:
    n_sites: int = 100
    bitstream_length: int = 1024
    transport_rate: float = 0.3
    dephasing_gamma: float = 0.05
    morphic_coupling: float = 0.1  # from L11


class L12_QuantumInfoLayer:
    """ENAQT-inspired ecological coherence transport."""

    def __init__(self, params: Optional[L12_StochasticParameters] = None):
        self.params = params or L12_StochasticParameters()
        n = self.params.n_sites
        self.coherence = np.random.uniform(0.3, 0.7, n)
        self.time = 0.0

    def step(
        self,
        dt: float,
        l11_input: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.time += dt
        n = self.params.n_sites

        # Nearest-neighbour transport (ring topology)
        transport = np.roll(self.coherence, 1) - 2 * self.coherence + np.roll(self.coherence, -1)
        dephasing = -self.params.dephasing_gamma * self.coherence
        self.coherence += (self.params.transport_rate * transport + dephasing) * dt

        if l11_input is not None and "info_saturation" in l11_input:
            self.coherence += 0.01 * l11_input["info_saturation"] * dt

        self.coherence = np.clip(self.coherence, 0, 1)

        entropy = self._von_neumann_entropy()

        rands = np.random.random((n, self.params.bitstream_length))
        output_bitstreams = (rands < self.coherence[:, None]).astype(np.uint8)

        return {
            "coherence": self.coherence.copy(),
            "entropy": entropy,
            "transport_efficiency": float(np.mean(self.coherence)),
            "output_bitstreams": output_bitstreams,
        }

    def _von_neumann_entropy(self) -> float:
        p = self.coherence / (np.sum(self.coherence) + 1e-10)
        return float(-np.sum(p * np.log(p + 1e-10)))

    def get_global_metric(self) -> float:
        return float(np.mean(self.coherence))
