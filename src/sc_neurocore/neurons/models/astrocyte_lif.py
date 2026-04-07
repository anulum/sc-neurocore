# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Astrocyte-LIF Hybrid Neuron (Perea et al. 2009)

"""Astrocyte-LIF hybrid unit with calcium wave feedback.

Models the tripartite synapse: a glial astrocyte monitors extracellular
glutamate from a paired LIF neuron and provides slow homeostatic feedback
via calcium-dependent gliotransmitter release.

Equations:

    dCa/dt = -Ca/tau_ca + delta * S_pre(t)        (calcium rise on presynaptic spike)
    I_glio = g_glio * H(Ca - Ca_thresh)            (gliotransmitter release)
    dV/dt  = -(V - E_L)/tau_m + I_ext + I_glio    (LIF with glial feedback)

H is the Heaviside step function. Gliotransmitter release occurs when
calcium exceeds threshold, providing slow excitatory feedback.

Reference: Perea, Navarrete & Araque, "Tripartite synapses" (2009).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AstrocyteLIFNeuron:
    """Astrocyte-LIF hybrid with tripartite synapse (Perea et al. 2009).

    Parameters
    ----------
    tau_m : float
        Membrane time constant (ms). Default: 20.0.
    tau_ca : float
        Calcium decay time constant (ms). Default: 500.0.
    e_l : float
        Leak reversal potential (mV). Default: -65.0.
    theta : float
        Spike threshold (mV). Default: -50.0.
    v_reset : float
        Post-spike reset potential (mV). Default: -65.0.
    ca_delta : float
        Calcium increment per presynaptic spike. Default: 0.1.
    ca_thresh : float
        Calcium threshold for gliotransmitter release. Default: 0.5.
    g_glio : float
        Gliotransmitter current amplitude. Default: 2.0.
    dt : float
        Integration timestep (ms). Default: 0.1.
    """

    tau_m: float = 20.0
    tau_ca: float = 500.0
    e_l: float = -65.0
    theta: float = -50.0
    v_reset: float = -65.0
    ca_delta: float = 0.1
    ca_thresh: float = 0.5
    g_glio: float = 2.0
    dt: float = 0.1

    v: float = -65.0
    ca: float = 0.0

    def step_with_pre(self, i_ext: float, pre_spike: bool) -> int:
        """Step with external current and presynaptic spike indicator.

        Returns 1 if spike, 0 otherwise.
        """
        # Astrocyte calcium dynamics.
        dca = -self.ca / self.tau_ca
        if pre_spike:
            dca += self.ca_delta / self.dt
        self.ca += dca * self.dt
        self.ca = max(self.ca, 0.0)

        # Gliotransmitter release (Heaviside on calcium).
        i_glio = self.g_glio if self.ca > self.ca_thresh else 0.0

        # LIF membrane dynamics with glial feedback.
        dv = (-(self.v - self.e_l) + i_ext + i_glio) / self.tau_m
        self.v += dv * self.dt

        if self.v >= self.theta:
            self.v = self.v_reset
            return 1
        return 0

    def step(self, current: float) -> int:
        """Step without presynaptic spike info (no glial feedback)."""
        return self.step_with_pre(current, pre_spike=False)

    def reset(self) -> None:
        """Reset state to initial conditions."""
        self.v = self.e_l
        self.ca = 0.0
