# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Cox 1955 — doubly stochastic Poisson (time-varying rate)

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class InhomogeneousPoissonNeuron:
    """Cox 1955 — doubly stochastic Poisson (time-varying rate).

    Reference: Gerstner, W. et al. (2014). Neuronal Dynamics. Cambridge Univ. Press, §7.3.
    """

    dt_ms: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.dt_ms) or self.dt_ms <= 0.0:
            raise ValueError("dt_ms must be finite and positive")

    def _probability(self, rate_hz: float) -> float:
        if not math.isfinite(rate_hz):
            raise ValueError("rate_hz must be finite")
        hazard = max(0.0, rate_hz) * self.dt_ms / 1000.0
        return -math.expm1(-hazard)

    def step(self, rate_hz: float) -> int:
        p = self._probability(rate_hz)
        return 1 if np.random.random() < p else 0

    def reset(self) -> None:
        pass
