# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ThresholdLinearRateNeuron:
    """Threshold-linear (ReLU) rate neuron. Dayan & Abbott 2001.

    r = gain * max(0, input - theta)
    """

    r: float = 0.0
    theta: float = 0.0
    gain: float = 1.0

    def step(self, current: float) -> float:
        self.r = self.gain * max(0.0, current - self.theta)
        return self.r

    def reset(self):
        self.r = 0.0
