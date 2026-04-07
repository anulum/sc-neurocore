# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Dendritic NMDA Neuron (Jahr & Stevens 1990)

"""Two-compartment neuron with voltage-dependent NMDA Mg2+ block.

Implements a soma-dendrite pair where the dendrite has NMDA receptors
with the classical Jahr & Stevens (1990) magnesium block:

    B(V) = 1 / (1 + [Mg2+]/3.57 * exp(-0.062 * V))

The NMDA current is:

    I_NMDA = g_NMDA * glutamate * B(V_dend) * (V_dend - E_NMDA)

This enables coincidence detection: the dendrite only passes current
when both presynaptic glutamate AND postsynaptic depolarisation are present.

Dendrite dynamics:

    tau_d * dV_d/dt = -(V_d - E_L) + I_NMDA + g_c * (V_s - V_d)

Soma dynamics:

    tau_s * dV_s/dt = -(V_s - E_L) + I_ext + g_c * (V_d - V_s)

Reference: Jahr & Stevens (1990), Schiller et al. (2000).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class DendriticNMDANeuron:
    """Two-compartment neuron with NMDA Mg2+ block (Jahr & Stevens 1990).

    Parameters
    ----------
    g_nmda : float
        NMDA conductance. Default: 1.5.
    e_nmda : float
        NMDA reversal potential (mV). Default: 0.0.
    mg_conc : float
        Extracellular Mg2+ concentration (mM). Default: 1.0.
    g_coupling : float
        Soma-dendrite coupling conductance. Default: 0.5.
    tau_soma : float
        Soma time constant (ms). Default: 20.0.
    tau_dend : float
        Dendrite time constant (ms). Default: 50.0.
    theta : float
        Spike threshold (mV). Default: -50.0.
    dt : float
        Integration timestep (ms). Default: 0.1.
    """

    g_nmda: float = 1.5
    e_nmda: float = 0.0
    mg_conc: float = 1.0
    g_coupling: float = 0.5
    tau_soma: float = 20.0
    tau_dend: float = 50.0
    theta: float = -50.0
    dt: float = 0.1

    v_soma: float = -65.0
    v_dend: float = -65.0

    def mg_block(self, v: float) -> float:
        """Mg2+ block factor: B(V) = 1/(1 + [Mg]/3.57 * exp(-0.062*V))."""
        return 1.0 / (1.0 + (self.mg_conc / 3.57) * math.exp(-0.062 * v))

    def step(self, i_soma: float, glutamate: float) -> int:
        """Step with somatic input current and dendritic glutamate.

        Returns 1 if spike, 0 otherwise.
        """
        b = self.mg_block(self.v_dend)
        i_nmda = self.g_nmda * glutamate * b * (self.v_dend - self.e_nmda)

        dv_dend = (
            -self.v_dend - 65.0 + i_nmda + self.g_coupling * (self.v_soma - self.v_dend)
        ) / self.tau_dend
        self.v_dend += dv_dend * self.dt

        i_dend_to_soma = self.g_coupling * (self.v_dend - self.v_soma)
        dv_soma = (-self.v_soma - 65.0 + i_soma + i_dend_to_soma) / self.tau_soma
        self.v_soma += dv_soma * self.dt

        if self.v_soma >= self.theta:
            self.v_soma = -65.0
            return 1
        return 0

    def reset(self) -> None:
        """Reset state to resting potential."""
        self.v_soma = -65.0
        self.v_dend = -65.0
