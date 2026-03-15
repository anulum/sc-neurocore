# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class SpiNNakerLIFNeuron:
    """SpiNNaker LIF — ARM Cortex-M4 digital. Furber 2014."""

    v: float = -70.0
    v_rest: float = -70.0
    v_reset: float = -70.0
    v_threshold: float = -50.0
    tau_m: float = 20.0
    i_offset: float = 0.0
    tau_refrac: float = 2.0
    refrac_count: float = 0.0
    dt: float = 1.0

    def step(self, current: float) -> int:
        if self.refrac_count > 0:
            self.refrac_count -= self.dt
            return 0
        self.v += (-(self.v - self.v_rest) + (current + self.i_offset)) / self.tau_m * self.dt
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            self.refrac_count = self.tau_refrac
            return 1
        return 0

    def reset(self):
        self.v = self.v_rest
        self.refrac_count = 0.0


# ── SPECIALIZED / MODERN ──────────────────────────────────────────
