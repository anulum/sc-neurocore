# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""
SCPN L8: Cosmic Phase-Locking Layer (Stochastic Implementation)

Pulsar Timing Array synchronization: local oscillators phase-lock to
cosmic reference frequencies via Kuramoto coupling.

dTheta_n/dt = Omega_n + K_cosmic * sum(sin(Theta_m - Theta_n))

Ref: Paper 8 — Cosmic Phase-Locking and PTA synchronisation.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class L8_StochasticParameters:
    n_pulsars: int = 12
    bitstream_length: int = 1024
    k_cosmic: float = 0.05
    symbolic_coupling: float = 0.1  # from L7
    director_coupling: float = 0.15  # to L16
    pulsar_omegas: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.pulsar_omegas is None:
            self.pulsar_omegas = np.array(
                [1.6, 2.3, 0.8, 4.1, 1.1, 0.5, 3.2, 2.7, 1.9, 0.4, 5.5, 0.2]
            )


class L8_PhaseFieldLayer:
    """Stochastic cosmic phase-locking via Kuramoto-coupled PTA oscillators."""

    def __init__(self, params: Optional[L8_StochasticParameters] = None):
        self.params = params or L8_StochasticParameters()
        self.phases = np.random.uniform(0, 2 * np.pi, self.params.n_pulsars)
        self.time = 0.0

    def step(
        self,
        dt: float,
        l7_input: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        self.time += dt
        n = self.params.n_pulsars
        omegas = self.params.pulsar_omegas

        # Kuramoto coupling: phase differences
        phase_diff = self.phases[np.newaxis, :] - self.phases[:, np.newaxis]
        coupling = self.params.k_cosmic * np.sum(np.sin(phase_diff), axis=1) / n

        d_phase = omegas + coupling
        if l7_input is not None and "glyph_vector" in l7_input:
            drive = np.mean(l7_input["glyph_vector"])
            d_phase += self.params.symbolic_coupling * drive * np.sin(-self.phases)

        self.phases = (self.phases + d_phase * dt) % (2 * np.pi)

        activation = (1.0 + np.cos(self.phases)) / 2.0
        rands = np.random.random((n, self.params.bitstream_length))
        output_bitstreams = (rands < activation[:, None]).astype(np.uint8)

        return {
            "phases": self.phases.copy(),
            "cosmic_alignment": self._order_parameter(),
            "output_bitstreams": output_bitstreams,
        }

    def _order_parameter(self) -> float:
        return float(np.abs(np.mean(np.exp(1j * self.phases))))

    def get_global_metric(self) -> float:
        return self._order_parameter()
