# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Multicompartment MCN Neuron (Spiking-WM, PNAS 2025)

"""Multi-compartment neuron matching the Spiking-WM architecture.

Dual-dendrite model with basal and apical compartments. The apical dendrite
gates how strongly basal information influences the soma, enabling
nonlinear integration for long-term temporal memory in RL tasks.

Exact equations from arXiv:2503.00713 (Spiking-WM, PNAS 2025):

    tau_b * dV_b/dt = -V_b + x_b                                  (basal)
    tau_a * dV_a/dt = -V_a + x_a                                  (apical)
    tau   * dU/dt   = -U + sigma(V_a) * [g_B/g_L * (V_b - U) + I]  (soma)
    S[t] = Theta(U[t] - V_th)                                     (spike)
    U[t] <- U[t] * (1 - S[t])                                     (soft reset)

Default parameters from Table II: tau = tau_a = tau_b = 2.0,
g_B/g_L = 1.0, beta = 1.0 (sigmoid steepness), V_th = 1.0.

Reference: Brain-Cog-Lab, arXiv:2503.00713, PNAS 2025.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class MulticompartmentMCNNeuron:
    """Multi-compartment neuron (Spiking-WM, PNAS 2025).

    Parameters
    ----------
    tau : float
        Soma time constant. Default: 2.0.
    tau_b : float
        Basal dendrite time constant. Default: 2.0.
    tau_a : float
        Apical dendrite time constant. Default: 2.0.
    g_ratio : float
        Basal-to-soma conductance ratio (g_B/g_L). Default: 1.0.
    beta : float
        Sigmoid steepness for apical gating. Default: 1.0.
    v_th : float
        Spike threshold. Default: 1.0.
    dt : float
        Integration timestep. Default: 1.0.
    """

    tau: float = 2.0
    tau_b: float = 2.0
    tau_a: float = 2.0
    g_ratio: float = 1.0
    beta: float = 1.0
    v_th: float = 1.0
    dt: float = 1.0

    u: float = 0.0
    v_basal: float = 0.0
    v_apical: float = 0.0

    def _sigma(self, x: float) -> float:
        """Sigmoid gating: sigma(x) = 1/(1 + exp(-beta*x))."""
        return 1.0 / (1.0 + math.exp(-self.beta * x))

    def step_compartments(
        self,
        x_basal: float,
        x_apical: float,
        i_soma: float,
    ) -> int:
        """Step with basal input, apical input, and direct somatic input.

        Returns 1 if spike, 0 otherwise.
        """
        # Basal dendrite: tau_b * dV_b/dt = -V_b + x_b.
        dv_b = (-self.v_basal + x_basal) / self.tau_b
        self.v_basal += dv_b * self.dt

        # Apical dendrite: tau_a * dV_a/dt = -V_a + x_a.
        dv_a = (-self.v_apical + x_apical) / self.tau_a
        self.v_apical += dv_a * self.dt

        # Soma: tau * dU/dt = -U + sigma(V_a) * [g_ratio * (V_b - U) + I].
        gate = self._sigma(self.v_apical)
        du = (-self.u + gate * (self.g_ratio * (self.v_basal - self.u) + i_soma)) / self.tau
        self.u += du * self.dt

        # Spike: S = Theta(U - V_th), reset: U <- U * (1 - S).
        if self.u >= self.v_th:
            self.u = 0.0
            return 1
        return 0

    def step(self, current: float) -> int:
        """Simple step: input goes to basal dendrite only."""
        return self.step_compartments(current, 0.0, 0.0)

    def reset(self) -> None:
        """Reset state to initial conditions."""
        self.u = 0.0
        self.v_basal = 0.0
        self.v_apical = 0.0
