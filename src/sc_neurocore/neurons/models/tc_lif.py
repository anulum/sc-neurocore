# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TwoCompartmentLIFNeuron:
    """Two-compartment LIF — Yang et al. AAAI 2024.

    Soma:     tau_s dV_s/dt = -(V_s - V_rest) + kappa*(V_d - V_s) + I_ext
    Dendrite: tau_d dV_d/dt = -(V_d - V_rest) + I_d
    Spike when V_s >= theta; V_s -> V_reset, V_d unchanged.
    Dendrite provides history-dependent input for sequential tasks.
    """

    v_s: float = 0.0
    v_d: float = 0.0
    v_rest: float = 0.0
    v_reset: float = 0.0
    theta: float = 1.0
    tau_s: float = 2.0
    tau_d: float = 20.0
    kappa: float = 0.5
    dt: float = 1.0

    def step(self, i_soma: float, i_dend: float = 0.0) -> int:
        dvd = (-(self.v_d - self.v_rest) + i_dend) / self.tau_d * self.dt
        self.v_d += dvd
        dvs = (
            (-(self.v_s - self.v_rest) + self.kappa * (self.v_d - self.v_s) + i_soma)
            / self.tau_s
            * self.dt
        )
        self.v_s += dvs
        if self.v_s >= self.theta:
            self.v_s = self.v_reset
            return 1
        return 0

    def reset(self):
        self.v_s = self.v_rest
        self.v_d = self.v_rest
