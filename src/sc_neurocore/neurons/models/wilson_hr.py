# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wilson 1999 — polynomial cortical model

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WilsonHRNeuron:
    """Wilson 1999 — polynomial cortical model.

    dV/dt = -(17.81 + 47.71*V + 32.63*V^2)*(V - 0.55) - 26*R*(V + 0.92) + I
    dR/dt = (-R + 1.35*V + 1.03) / tau_R

    V in dimensionless units, spike at V > V_peak.
    """

    v: float = -0.7
    r: float = 0.1
    tau_r: float = 1.9
    v_peak: float = 0.4
    dt: float = 0.05

    def step(self, current: float) -> int:
        poly = -(17.81 + 47.71 * self.v + 32.63 * self.v**2) * (self.v - 0.55)
        syn = -26.0 * self.r * (self.v + 0.92)
        dv = (poly + syn + current) * self.dt
        dr = (-self.r + 1.35 * self.v + 1.03) / self.tau_r * self.dt
        self.v += dv
        self.r += dr
        if self.v >= self.v_peak:
            self.v = -0.7
            return 1
        return 0

    def reset(self) -> None:
        self.v = -0.7
        self.r = 0.1
