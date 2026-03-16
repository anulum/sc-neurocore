# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class CazellesMapNeuron:
    """Cazelles et al. 2001 — simple 2D bursting map neuron.

    x(n+1) = f(x(n)) - y(n) + I
    y(n+1) = y(n) + epsilon * (x(n) - sigma)

    f(x) = a*x*(1 - x)    (logistic-like fast dynamics)

    Bursting arises from slow y modulation of fast x.
    """

    x: float = 0.1
    y: float = 0.0
    a: float = 3.8
    epsilon: float = 0.01
    sigma: float = 0.5
    x_threshold: float = 0.9

    def step(self, current: float) -> int:
        f = self.a * self.x * (1.0 - self.x)
        x_new = f - self.y + current
        y_new = self.y + self.epsilon * (self.x - self.sigma)
        self.x = np.clip(x_new, -2.0, 2.0)
        self.y = y_new
        return 1 if self.x >= self.x_threshold else 0

    def reset(self):
        self.x = 0.1
        self.y = 0.0
