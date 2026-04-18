# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Theta neuron — canonical Type-I on the unit circle

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ThetaNeuron:
    """Theta neuron — canonical Type-I on the unit circle.

    dθ/dt = (1 - cos θ) + (1 + cos θ) · I
    Spike when θ crosses π.
    Ermentrout & Kopell 1986.

    Reference: Ermentrout, G.B. & Kopell, N. (1986). SIAM J. Appl. Math. 46:233–253.
    """

    theta: float = 0.0
    dt: float = 0.01

    def step(self, current: float) -> int:
        theta_prev = self.theta
        dtheta = ((1.0 - np.cos(self.theta)) + (1.0 + np.cos(self.theta)) * current) * self.dt
        self.theta += dtheta
        spike = 1 if (theta_prev < np.pi * 0.99 and self.theta >= np.pi * 0.99) else 0
        self.theta = ((self.theta + np.pi) % (2 * np.pi)) - np.pi
        return spike

    def reset(self) -> None:
        self.theta = 0.0
