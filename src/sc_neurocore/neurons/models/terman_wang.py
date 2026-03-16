# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class TermanWangOscillator:
    """Terman & Wang 1995 — relaxation oscillator for LEGION networks.

    dv/dt = f(v) - w + I + rho
    dw/dt = epsilon * (g(v) - w)

    f(v) = 3*v - v^3 + 2                (cubic nullcline)
    g(v) = alpha * (1 + tanh(v/beta))    (sigmoid recovery)
    """

    v: float = -1.5
    w: float = -0.5
    alpha: float = 3.0
    beta: float = 0.2
    epsilon: float = 0.02
    rho: float = 0.0
    dt: float = 0.05
    v_peak: float = 1.5

    def step(self, current: float) -> int:
        f = 3.0 * self.v - self.v**3 + 2.0
        g = self.alpha * (1.0 + np.tanh(self.v / self.beta))
        dv = (f - self.w + current + self.rho) * self.dt
        dw = self.epsilon * (g - self.w) * self.dt
        v_prev = self.v
        self.v += dv
        self.w += dw
        return 1 if (self.v >= self.v_peak and v_prev < self.v_peak) else 0

    def reset(self):
        self.v = -1.5
        self.w = -0.5
