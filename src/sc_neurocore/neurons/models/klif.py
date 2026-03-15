# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class KLIFNeuron:
    """KLIF — LIF with learnable scaling factor k.

    V[t+1] = alpha * V[t] + k * I; spike when V >= threshold.
    The scaling factor k is a trainable parameter for SNN backprop.
    """

    v: float = 0.0
    k: float = 1.0  # learnable scaling factor
    tau: float = 10.0
    v_threshold: float = 1.0
    v_reset: float = 0.0
    dt: float = 1.0
    alpha: float = field(init=False)

    def __post_init__(self):
        self.alpha = np.exp(-self.dt / self.tau)

    def step(self, current: float) -> int:
        self.v = self.alpha * self.v + self.k * current
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            return 1
        return 0

    def reset(self):
        self.v = 0.0
