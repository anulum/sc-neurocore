# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Fardet & Levina 2020 — LIF with metabolic energy constraint

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EnergyLIFNeuron:
    """Fardet & Levina 2020 — LIF with metabolic energy constraint."""

    v: float = -70.0
    epsilon: float = 1.0
    v_rest: float = -70.0
    v_reset: float = -70.0
    v_threshold: float = -50.0
    tau_m: float = 10.0
    tau_e: float = 500.0
    alpha: float = 0.1
    epsilon_0: float = 1.0
    resistance: float = 1.0
    dt: float = 1.0

    def step(self, current: float) -> int:
        effective_r = self.resistance * self.epsilon
        self.v += (-(self.v - self.v_rest) + effective_r * current) / self.tau_m * self.dt
        self.epsilon += (self.epsilon_0 - self.epsilon) / self.tau_e * self.dt
        if self.v >= self.v_threshold and self.epsilon > 0.1:
            self.v = self.v_reset
            self.epsilon -= self.alpha
            self.epsilon = max(0.0, self.epsilon)
            return 1
        return 0

    def reset(self):
        self.v = self.v_rest
        self.epsilon = self.epsilon_0
