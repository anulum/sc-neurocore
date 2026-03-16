# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quadratic Integrate-and-Fire — canonical Type-I excitability

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QuadraticIFNeuron:
    """Quadratic Integrate-and-Fire — canonical Type-I excitability.

    dv/dt = v² + I
    Reset when v >= v_peak.
    """

    v: float = -1.0
    v_reset: float = -1.0
    v_peak: float = 1.0
    dt: float = 0.01

    def step(self, current: float) -> int:
        self.v += (self.v**2 + current) * self.dt
        if self.v >= self.v_peak:
            self.v = self.v_reset
            return 1
        return 0

    def reset(self):
        self.v = self.v_reset
