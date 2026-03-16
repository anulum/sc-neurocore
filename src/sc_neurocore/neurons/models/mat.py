# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Kobayashi 2009 — Multi-timescale Adaptive Threshold

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class MATNeuron:
    """Kobayashi 2009 — Multi-timescale Adaptive Threshold."""

    v: float = -70.0
    theta1: float = 0.0
    theta2: float = 0.0
    v_rest: float = -70.0
    v_reset: float = -70.0
    v_threshold_base: float = -50.0
    tau_m: float = 10.0
    tau_1: float = 10.0
    tau_2: float = 200.0
    h1: float = 5.0
    h2: float = 3.0
    resistance: float = 1.0
    dt: float = 1.0

    def step(self, current: float) -> int:
        self.v += (-(self.v - self.v_rest) + self.resistance * current) / self.tau_m * self.dt
        self.theta1 *= np.exp(-self.dt / self.tau_1)
        self.theta2 *= np.exp(-self.dt / self.tau_2)
        threshold = self.v_threshold_base + self.theta1 + self.theta2
        if self.v >= threshold:
            self.v = self.v_reset
            self.theta1 += self.h1
            self.theta2 += self.h2
            return 1
        return 0

    def reset(self):
        self.v = self.v_rest
        self.theta1, self.theta2 = 0.0, 0.0
