# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SFANeuron:
    """Benda & Herz 2003 — Spike Frequency Adaptation IF."""

    v: float = -70.0
    g_sfa: float = 0.0
    v_rest: float = -70.0
    v_reset: float = -70.0
    v_threshold: float = -50.0
    tau_m: float = 10.0
    tau_sfa: float = 200.0
    delta_g: float = 0.5
    e_k: float = -80.0
    resistance: float = 1.0
    dt: float = 1.0

    def step(self, current: float) -> int:
        self.v += (
            (-(self.v - self.v_rest) - self.g_sfa * (self.v - self.e_k) + self.resistance * current)
            / self.tau_m
            * self.dt
        )
        self.g_sfa *= np.exp(-self.dt / self.tau_sfa)
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            self.g_sfa += self.delta_g
            return 1
        return 0

    def reset(self):
        self.v = self.v_rest
        self.g_sfa = 0.0
