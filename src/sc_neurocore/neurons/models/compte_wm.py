# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Compte et al. 2000 — NMDA-based working memory neuron

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class CompteWMNeuron:
    """Compte et al. 2000 — NMDA-based working memory neuron.

    C dV/dt = -I_L - I_AMPA - I_NMDA - I_GABA + I_ext
    NMDA includes voltage-dependent Mg2+ block:
      B(V) = 1 / (1 + [Mg]/3.57 * exp(-0.062*V))
    ds_NMDA/dt = -s_NMDA/tau_NMDA + alpha*x*(1-s_NMDA)
    dx/dt      = -x/tau_x
    """

    v: float = -70.0
    s_ampa: float = 0.0
    s_nmda: float = 0.0
    x_nmda: float = 0.0
    s_gaba: float = 0.0
    g_l: float = 0.025
    g_ampa: float = 0.005
    g_nmda: float = 0.165
    g_gaba: float = 0.013
    e_l: float = -70.0
    e_exc: float = 0.0
    e_inh: float = -70.0
    c_m: float = 0.5
    mg: float = 1.0
    tau_ampa: float = 2.0
    tau_nmda: float = 100.0
    tau_x: float = 2.0
    alpha_nmda: float = 0.5
    v_threshold: float = -50.0
    v_reset: float = -55.0
    dt: float = 0.1

    def _mg_block(self, v: float) -> float:
        return 1.0 / (1.0 + self.mg / 3.57 * np.exp(-0.062 * v))

    def step(self, current: float, spike_in: bool = False) -> int:
        if spike_in:
            self.s_ampa += 1.0
            self.x_nmda += 1.0

        self.s_ampa *= np.exp(-self.dt / self.tau_ampa)
        self.s_nmda += (
            -self.s_nmda / self.tau_nmda + self.alpha_nmda * self.x_nmda * (1.0 - self.s_nmda)
        ) * self.dt
        self.x_nmda *= np.exp(-self.dt / self.tau_x)
        self.s_gaba *= np.exp(-self.dt / 5.0)

        b = self._mg_block(self.v)
        i_l = self.g_l * (self.v - self.e_l)
        i_ampa = self.g_ampa * self.s_ampa * (self.v - self.e_exc)
        i_nmda = self.g_nmda * b * self.s_nmda * (self.v - self.e_exc)
        i_gaba = self.g_gaba * self.s_gaba * (self.v - self.e_inh)

        self.v += (-i_l - i_ampa - i_nmda - i_gaba + current) / self.c_m * self.dt
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            self.s_gaba += 1.0
            return 1
        return 0

    def reset(self) -> None:
        self.v = self.e_l
        self.s_ampa = 0.0
        self.s_nmda = 0.0
        self.x_nmda = 0.0
        self.s_gaba = 0.0
