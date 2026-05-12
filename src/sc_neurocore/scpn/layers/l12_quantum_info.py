# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
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
import math
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class L12_StochasticParameters:
    n_sites: int = 100
    bitstream_length: int = 1024
    transport_rate: float = 0.3
    dephasing_gamma: float = 0.05
    morphic_coupling: float = 0.1  # from L11
    rng_seed: Optional[int] = None


class L12_QuantumInfoLayer:
    """ENAQT-inspired ecological coherence transport."""

    def __init__(self, params: Optional[L12_StochasticParameters] = None):
        self.params = params or L12_StochasticParameters()
        self._validate_params(self.params)
        n = self.params.n_sites
        self._rng = np.random.default_rng(self.params.rng_seed)
        self.coherence = self._rng.uniform(0.3, 0.7, n)
        self.time = 0.0

    def step(
        self,
        dt: float,
        l11_input: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not math.isfinite(float(dt)) or float(dt) <= 0.0:
            raise ValueError("dt must be finite and positive")
        self.time += dt
        n = self.params.n_sites

        # Nearest-neighbour transport (ring topology)
        transport = np.roll(self.coherence, 1) - 2 * self.coherence + np.roll(self.coherence, -1)
        dephasing = -self.params.dephasing_gamma * self.coherence
        self.coherence += (self.params.transport_rate * transport + dephasing) * dt

        if l11_input is not None and "info_saturation" in l11_input:
            info_saturation = self._info_saturation(l11_input["info_saturation"])
            self.coherence += self.params.morphic_coupling * info_saturation * dt

        self.coherence = np.clip(self.coherence, 0, 1)

        entropy = self._von_neumann_entropy()

        rands = self._rng.random((n, self.params.bitstream_length))
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

    @staticmethod
    def _validate_params(params: L12_StochasticParameters) -> None:
        if not isinstance(params.n_sites, int) or isinstance(params.n_sites, bool):
            raise ValueError("n_sites must be a positive integer")
        if params.n_sites <= 0:
            raise ValueError("n_sites must be positive")
        if not isinstance(params.bitstream_length, int) or isinstance(
            params.bitstream_length, bool
        ):
            raise ValueError("bitstream_length must be a positive integer")
        if params.bitstream_length <= 0:
            raise ValueError("bitstream_length must be positive")
        if not math.isfinite(float(params.transport_rate)) or params.transport_rate < 0.0:
            raise ValueError("transport_rate must be finite and non-negative")
        if not math.isfinite(float(params.dephasing_gamma)) or params.dephasing_gamma < 0.0:
            raise ValueError("dephasing_gamma must be finite and non-negative")
        if not math.isfinite(float(params.morphic_coupling)) or params.morphic_coupling < 0.0:
            raise ValueError("morphic_coupling must be finite and non-negative")
        if params.rng_seed is not None:
            if isinstance(params.rng_seed, bool) or not isinstance(params.rng_seed, int):
                raise ValueError("rng_seed must be a non-negative integer or None")
            if params.rng_seed < 0:
                raise ValueError("rng_seed must be a non-negative integer or None")

    @staticmethod
    def _info_saturation(value: Any) -> float:
        values = np.asarray(value, dtype=np.float64)
        if values.shape != ():
            raise ValueError("info_saturation must be a finite scalar")
        info_saturation = float(values)
        if not math.isfinite(info_saturation):
            raise ValueError("info_saturation must be a finite scalar")
        return info_saturation
