# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Integrate-and-fire with dynamic threshold. Platkiewicz &

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdaptiveThresholdIFNeuron:
    """Integrate-and-fire with dynamic threshold. Platkiewicz & Bhatt 2010.

    C dV/dt = -g_L(V - V_rest) + I
    dtheta/dt = -(theta - theta_rest) / tau_theta
    On spike: V -> V_reset, theta += delta_theta

    Reference: Platkiewicz, J. & Brette, R. (2010). J. Neurosci. 30:6891–6902.
    """

    v: float = -65.0
    theta: float = -50.0
    v_rest: float = -65.0
    v_reset: float = -65.0
    theta_rest: float = -50.0
    delta_theta: float = 5.0
    tau_m: float = 10.0
    tau_theta: float = 50.0
    dt: float = 0.1

    def step(self, current: float) -> int:
        self.v += (-(self.v - self.v_rest) + current) / self.tau_m * self.dt
        self.theta += (-(self.theta - self.theta_rest)) / self.tau_theta * self.dt
        if self.v >= self.theta:
            self.v = self.v_reset
            self.theta += self.delta_theta
            return 1
        return 0

    def reset(self) -> None:
        self.v = self.v_rest
        self.theta = self.theta_rest
