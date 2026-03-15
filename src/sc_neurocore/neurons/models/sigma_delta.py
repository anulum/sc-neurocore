# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SigmaDeltaNeuron:
    """Yoon 2017 — event-driven sigma-delta encoding."""

    sigma: float = 0.0
    v_threshold: float = 1.0

    def step(self, current: float) -> int:
        self.sigma += current
        if self.sigma >= self.v_threshold:
            self.sigma -= self.v_threshold
            return 1
        elif self.sigma <= -self.v_threshold:
            self.sigma += self.v_threshold
            return -1
        return 0

    def reset(self):
        self.sigma = 0.0
