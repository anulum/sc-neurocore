# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Cox 1955 — doubly stochastic Poisson (time-varying rate)

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class InhomogeneousPoissonNeuron:
    """Cox 1955 — doubly stochastic Poisson (time-varying rate)."""

    dt_ms: float = 1.0

    def step(self, rate_hz: float) -> int:
        p = max(0.0, rate_hz) * self.dt_ms / 1000.0
        return 1 if np.random.random() < p else 0

    def reset(self):
        pass
