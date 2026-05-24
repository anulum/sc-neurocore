# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Continuous rate model with sigmoidal transfer. Wilson &

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class SigmoidRateNeuron:
    """Continuous rate model with sigmoidal transfer. Wilson & Cowan 1972 style.

    tau dr/dt = -r + sigma(beta * (input - theta))

    Reference: Wilson, H.R. & Cowan, J.D. (1972). Biophys. J. 12:1–24.
    """

    r: float = 0.0
    tau: float = 10.0
    beta: float = 1.0
    theta: float = 0.0
    dt: float = 0.1

    def __post_init__(self) -> None:
        for field in ("r", "beta", "theta"):
            if not math.isfinite(getattr(self, field)):
                raise ValueError(f"{field} must be finite")
        for field in ("tau", "dt"):
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be finite and positive")

    def step(self, current: float) -> float:
        if not math.isfinite(current):
            raise ValueError("current must be finite")

        sigma = 1.0 / (1.0 + np.exp(-self.beta * (current - self.theta)))
        self.r += (-self.r + sigma) / self.tau * self.dt
        return self.r

    def reset(self) -> None:
        self.r = 0.0
