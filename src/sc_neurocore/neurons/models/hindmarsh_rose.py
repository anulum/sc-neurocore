# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HindmarshRoseNeuron:
    """Hindmarsh-Rose 1984 — 3D chaotic bursting model.

    dx/dt = y - x³ + bx² - z + I
    dy/dt = 1 - 5x² - y
    dz/dt = r(s(x - x_rest) - z)
    """

    x: float = -1.6
    y: float = -10.0
    z: float = 2.0
    b: float = 3.0
    r: float = 0.001
    s: float = 4.0
    x_rest: float = -1.6
    dt: float = 0.1
    x_threshold: float = 1.0

    def step(self, current: float) -> int:
        x_prev = self.x
        dx = (self.y - self.x**3 + self.b * self.x**2 - self.z + current) * self.dt
        dy = (1.0 - 5.0 * self.x**2 - self.y) * self.dt
        dz = self.r * (self.s * (self.x - self.x_rest) - self.z) * self.dt
        self.x += dx
        self.y += dy
        self.z += dz
        return 1 if (self.x >= self.x_threshold and x_prev < self.x_threshold) else 0

    def reset(self):
        self.x = -1.6
        self.y = -10.0
        self.z = 2.0
