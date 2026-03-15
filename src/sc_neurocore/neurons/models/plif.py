# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ParametricLIFNeuron:
    """Fang et al. 2021 — Parametric LIF (PLIF) with learnable decay.

    V(t+1) = alpha * V(t) * (1 - spike(t)) + I(t)
    alpha  = sigmoid(a)    (learnable parameter)
    spike  = Theta(V - threshold)
    """

    v: float = 0.0
    a: float = 0.0
    threshold: float = 1.0
    dt: float = 1.0

    @property
    def alpha(self) -> float:
        return 1.0 / (1.0 + np.exp(-self.a))

    def step(self, current: float) -> int:
        spike = 1 if self.v >= self.threshold else 0
        self.v = self.alpha * self.v * (1 - spike) + current
        return 1 if self.v >= self.threshold else 0

    def reset(self):
        self.v = 0.0
