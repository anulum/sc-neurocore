# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tsodyks & Markram 1997 — LIF with short-term synaptic

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TsodyksMarkramNeuron:
    """Tsodyks & Markram 1997 — LIF with short-term synaptic plasticity.

    LIF membrane: tau_m dV/dt = -(V - V_rest) + R*I_syn + R*I_ext
    dx/dt = (1 - x)/tau_d - u*x*delta(spike_in)
    du/dt = (U - u)/tau_f + U*(1-u)*delta(spike_in)
    I_syn = A * u * x on presynaptic spike

    Reference: Tsodyks, M. et al. (1998). Neural Comput. 10:821–835.
    """

    v: float = -65.0
    x: float = 1.0
    u: float = 0.2
    v_rest: float = -65.0
    v_reset: float = -65.0
    v_threshold: float = -50.0
    tau_m: float = 20.0
    tau_d: float = 200.0
    tau_f: float = 600.0
    u_se: float = 0.2
    a_se: float = 50.0
    r_m: float = 1.0
    dt: float = 0.1

    def step(self, current: float, presynaptic_spike: bool = False) -> int:
        self.x += (1.0 - self.x) / self.tau_d * self.dt
        self.u += (self.u_se - self.u) / self.tau_f * self.dt

        i_syn = 0.0
        if presynaptic_spike:
            self.u += self.u_se * (1.0 - self.u)
            i_syn = self.a_se * self.u * self.x
            self.x -= self.u * self.x

        dv = (-(self.v - self.v_rest) + self.r_m * (i_syn + current)) / self.tau_m * self.dt
        self.v += dv
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            return 1
        return 0

    def reset(self) -> None:
        self.v = self.v_rest
        self.x = 1.0
        self.u = self.u_se
