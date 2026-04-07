# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Short-Term Plasticity Synapse (Tsodyks-Markram 1997)

"""Short-term plasticity (STP): facilitation and depression.

Tsodyks-Markram model of use-dependent synaptic dynamics on ms-to-s timescale.

Equations:

    dx/dt = (1 - x) / tau_d - u * x * delta(t_spike)
    du/dt = (U - u) / tau_f + U * (1 - u) * delta(t_spike)
    PSC = A * u * x * delta(t_spike)

x: available resources (depression variable, starts at 1.0).
u: release probability (facilitation variable, starts at U).

Depressing synapses (high U, fast tau_d, slow tau_f) model cortical
pyramidal-pyramidal connections. Facilitating synapses (low U, slow tau_d,
fast tau_f) model cortical pyramidal-interneuron connections.

Reference: Tsodyks & Markram (1997), Markram et al. (1998).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ShortTermPlasticitySynapse:
    """Short-term plasticity synapse (Tsodyks-Markram 1997).

    Parameters
    ----------
    x : float
        Available resources (depression). Default: 1.0.
    u : float
        Release probability (facilitation). Default: 0.5.
    u_base : float
        Baseline release probability U. Default: 0.5.
    tau_d : float
        Depression recovery time constant (ms). Default: 200.0.
    tau_f : float
        Facilitation decay time constant (ms). Default: 20.0.
    amplitude : float
        Maximum PSC amplitude. Default: 1.0.
    dt : float
        Integration timestep (ms). Default: 1.0.
    """

    x: float = 1.0
    u: float = 0.5
    u_base: float = 0.5
    tau_d: float = 200.0
    tau_f: float = 20.0
    amplitude: float = 1.0
    dt: float = 1.0

    @classmethod
    def new_depressing(cls) -> ShortTermPlasticitySynapse:
        """Create a depressing synapse (cortical pyr-pyr)."""
        return cls(
            x=1.0,
            u=0.5,
            u_base=0.5,
            tau_d=200.0,
            tau_f=20.0,
            amplitude=1.0,
        )

    @classmethod
    def new_facilitating(cls) -> ShortTermPlasticitySynapse:
        """Create a facilitating synapse (cortical pyr-interneuron)."""
        return cls(
            x=1.0,
            u=0.1,
            u_base=0.1,
            tau_d=50.0,
            tau_f=500.0,
            amplitude=1.0,
        )

    def step(self, pre_spike: bool) -> float:
        """Advance one timestep. Returns post-synaptic current.

        Between spikes, x recovers toward 1 and u decays toward U.
        On a presynaptic spike: u is facilitated, PSC = A*u*x, then x is depressed.
        """
        # Recover between spikes.
        self.x += (1.0 - self.x) / self.tau_d * self.dt
        self.u += (self.u_base - self.u) / self.tau_f * self.dt

        if pre_spike:
            # Facilitation: increase release probability.
            self.u += self.u_base * (1.0 - self.u)
            # Compute PSC before depression.
            psc = self.amplitude * self.u * self.x
            # Depression: consume resources.
            self.x -= self.u * self.x
            self.x = max(self.x, 0.0)
            return psc
        return 0.0

    def reset(self) -> None:
        """Reset state to initial conditions."""
        self.x = 1.0
        self.u = self.u_base
