# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Allen Institute GLIF5 — Generalized LIF, 5-level hierarchy

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class GLIFNeuron:
    """Allen Institute GLIF5 — Generalized LIF, 5-level hierarchy.

    Teeter et al. 2018, Nat Comm. Level 5: LIF + reset rules +
    instantaneous threshold + threshold adaptation + after-spike currents.
    """

    v: float = -70.0
    theta: float = -50.0
    theta_inf: float = -50.0
    i_asc1: float = 0.0
    i_asc2: float = 0.0
    v_rest: float = -70.0
    v_reset: float = -70.0
    tau_m: float = 10.0
    tau_theta: float = 100.0
    tau_asc1: float = 10.0
    tau_asc2: float = 200.0
    a_theta: float = 0.01
    delta_theta: float = 2.0
    r_asc1: float = 1.0
    r_asc2: float = 0.5
    resistance: float = 1.0
    dt: float = 1.0

    def step(self, current: float) -> int:
        dv = (
            (-(self.v - self.v_rest) + self.resistance * current + self.i_asc1 + self.i_asc2)
            / self.tau_m
            * self.dt
        )
        dtheta = (
            (self.theta_inf - self.theta + self.a_theta * (self.v - self.v_rest))
            / self.tau_theta
            * self.dt
        )
        self.i_asc1 *= np.exp(-self.dt / self.tau_asc1)
        self.i_asc2 *= np.exp(-self.dt / self.tau_asc2)
        self.v += dv
        self.theta += dtheta
        if self.v >= self.theta:
            self.v = self.v_reset
            self.theta += self.delta_theta
            self.i_asc1 += self.r_asc1
            self.i_asc2 += self.r_asc2
            return 1
        return 0

    def reset(self) -> None:
        self.v = self.v_rest
        self.theta = self.theta_inf
        self.i_asc1, self.i_asc2 = 0.0, 0.0
