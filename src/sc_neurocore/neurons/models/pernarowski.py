# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PernarowskiNeuron:
    """Pernarowski 1994 — simplified pancreatic beta cell burster.

    3 ODEs (V, w, z) with two slow variables. Captures square-wave
    and parabolic bursting depending on parameters.
    """

    v: float = -1.0
    w: float = 0.0
    z: float = 0.0
    alpha: float = 0.1
    beta: float = 0.5
    eps1: float = 0.1
    eps2: float = 0.001
    gamma: float = 0.5
    dt: float = 0.1
    v_threshold: float = 0.5

    def step(self, current: float = 0.0) -> int:
        v_prev = self.v
        f_v = self.v - self.v**3 / 3.0
        dv = (f_v - self.w - self.z + current) * self.dt
        dw = self.eps1 * (self.v - self.gamma * self.w + self.alpha) * self.dt
        dz = self.eps2 * (self.beta * (self.v + 0.7) - self.z) * self.dt
        self.v += dv
        self.w += dw
        self.z += dz
        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self):
        self.v, self.w, self.z = -1.0, 0.0, 0.0
