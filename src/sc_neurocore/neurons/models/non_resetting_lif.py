# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Adaptive multi-timescale threshold (aMAT) variant —

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NonResettingLIFNeuron:
    """Adaptive multi-timescale threshold (aMAT) variant — non-resetting LIF.

    tau_m dV/dt = -(V - V_rest) + R*I
    On spike: threshold rises by delta_theta, V does NOT reset.
    dtheta/dt  = -(theta - theta_rest) / tau_theta

    Kobayashi et al. 2009, Jolivet et al. 2004.

    Reference: Gerstner, W. et al. (2014). Neuronal Dynamics. Cambridge Univ. Press, §1.3.
    """

    v: float = -65.0
    theta: float = -50.0
    v_rest: float = -65.0
    theta_rest: float = -50.0
    delta_theta: float = 5.0
    tau_m: float = 10.0
    tau_theta: float = 50.0
    r_m: float = 1.0
    dt: float = 0.1

    def step(self, current: float) -> int:
        self.v += (-(self.v - self.v_rest) + self.r_m * current) / self.tau_m * self.dt
        self.theta += (-(self.theta - self.theta_rest)) / self.tau_theta * self.dt
        if self.v >= self.theta:
            self.theta += self.delta_theta
            return 1
        return 0

    def reset(self) -> None:
        self.v = self.v_rest
        self.theta = self.theta_rest
