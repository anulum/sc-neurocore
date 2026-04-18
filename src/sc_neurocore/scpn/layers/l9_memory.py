# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L9 Holographic Memory Layer (Stochastic Implementation)

"""
SCPN L9: Holographic Memory Layer (Stochastic Implementation)

Hopfield-style associative memory with TSVF-inspired forward/backward
overlap for memory retrieval quality.

E = -1/2 * sum_ij w_ij s_i s_j  (Hopfield energy)
Retrieval = <Phi|Psi> / <Phi|Phi>  (TSVF weak-value proxy)

Ref: Paper 9 — Memory Imprint-Existential Holograph.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class L9_StochasticParameters:
    n_memory_slots: int = 64
    bitstream_length: int = 1024
    retrieval_gain: float = 0.8
    imprint_rate: float = 0.3
    decay_rate: float = 0.02
    phase_field_coupling: float = 0.1  # from L8


class L9_MemoryLayer:
    """Hopfield associative memory with stochastic bitstream encoding."""

    def __init__(self, params: Optional[L9_StochasticParameters] = None):
        self.params = params or L9_StochasticParameters()
        n = self.params.n_memory_slots
        self.patterns = np.zeros((n, n))  # weight matrix
        self.state = np.random.choice([-1, 1], size=n).astype(np.float64)
        self.n_stored = 0
        self.time = 0.0

    def store(self, pattern: np.ndarray) -> None:
        """Hebbian imprint: W += pattern ⊗ pattern."""
        p = np.sign(pattern[: self.params.n_memory_slots])
        self.patterns += np.outer(p, p) / self.params.n_memory_slots
        np.fill_diagonal(self.patterns, 0)
        self.n_stored += 1

    def step(
        self,
        dt: float,
        l8_input: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.time += dt
        n = self.params.n_memory_slots

        # Hopfield dynamics: async update (random subset)
        update_mask = np.random.random(n) < 0.3
        h = self.patterns @ self.state
        self.state = np.where(update_mask, np.sign(h + 1e-10), self.state)

        # Retrieval quality: overlap with stored patterns
        activation = (self.state + 1) / 2  # map [-1,1] -> [0,1]

        if l8_input is not None and "cosmic_alignment" in l8_input:
            activation *= 0.9 + 0.1 * l8_input["cosmic_alignment"]

        activation = np.clip(activation, 0, 1)

        # Decay
        self.patterns *= 1.0 - self.params.decay_rate * dt

        rands = np.random.random((n, self.params.bitstream_length))
        output_bitstreams = (rands < activation[:, None]).astype(np.uint8)

        energy = -0.5 * float(self.state @ self.patterns @ self.state)

        return {
            "state": self.state.copy(),
            "energy": energy,
            "retrieval_quality": self._retrieval_quality(),
            "output_bitstreams": output_bitstreams,
        }

    def _retrieval_quality(self) -> float:
        if self.n_stored == 0:
            return 0.0
        h = self.patterns @ self.state
        return float(np.mean(np.sign(h) == np.sign(self.state)))

    def get_global_metric(self) -> float:
        return self._retrieval_quality()
