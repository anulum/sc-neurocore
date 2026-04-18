# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Resonate-and-Fire — subthreshold oscillation + threshold

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ResonateAndFireNeuron:
    """Resonate-and-Fire — subthreshold oscillation + threshold.

    Izhikevich 2001. Complex dynamics: z = x + i*y,
    dz/dt = (b + iω)z + I, fire when |z| > threshold.
    Implemented as 2 real ODEs.
    """

    x: float = 0.0
    y: float = 0.0
    b: float = -0.1
    omega: float = 1.0
    threshold: float = 1.0
    dt: float = 0.05

    def step(self, current: float) -> int:
        dx = (self.b * self.x - self.omega * self.y + current) * self.dt
        dy = (self.omega * self.x + self.b * self.y) * self.dt
        self.x += dx
        self.y += dy
        r = np.sqrt(self.x**2 + self.y**2)
        if r >= self.threshold:
            self.x = 0.0
            self.y = 0.0
            return 1
        return 0

    def reset(self) -> None:
        self.x = 0.0
        self.y = 0.0
