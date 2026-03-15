# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, field
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
