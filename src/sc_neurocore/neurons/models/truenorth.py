# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrueNorthNeuron:
    """Merolla 2014 — IBM TrueNorth digital neuron."""

    v: int = 0
    leak: int = 0
    threshold: int = 100
    v_reset: int = 0

    def step(self, weighted_input: int) -> int:
        self.v = self.v + weighted_input - self.leak
        if self.v >= self.threshold:
            self.v = self.v_reset
            return 1
        if self.v < -self.threshold:
            self.v = self.v_reset
        return 0

    def reset(self):
        self.v = 0
