# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Continuous rate model with sigmoidal transfer. Wilson &

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SigmoidRateNeuron:
    """Continuous rate model with sigmoidal transfer. Wilson & Cowan 1972 style.

    tau dr/dt = -r + sigma(beta * (input - theta))
    """

    r: float = 0.0
    tau: float = 10.0
    beta: float = 1.0
    theta: float = 0.0
    dt: float = 0.1

    def step(self, current: float) -> float:
        sigma = 1.0 / (1.0 + np.exp(-self.beta * (current - self.theta)))
        self.r += (-self.r + sigma) / self.tau * self.dt
        return self.r

    def reset(self) -> None:
        self.r = 0.0
