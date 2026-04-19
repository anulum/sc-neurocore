# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Exact LIF Solver (event-driven, zero discretization error)

"""Exact analytical integration for Leaky Integrate-and-Fire neurons.

Computes spike times and membrane evolution without ODE discretization,
using the closed-form solution of the LIF differential equation:

    tau * dV/dt = -(V - V_rest) + R * I

Solution: V(t) = V_rest + (V_0 - V_rest) * exp(-t/tau) + R*I*(1 - exp(-t/tau))
Spike time: t_spike = -tau * ln((V_rest + R*I - V_thresh) / (V_rest + R*I - V_0))
"""

from __future__ import annotations

import math
from dataclasses import dataclass



@dataclass
class ExactLIFSolver:
    """Event-driven exact integration for LIF neurons.

    Reference: Rotter, S. & Diesmann, M. (1999). Biol. Cybern. 81:381–402.
    """

    tau: float = 20.0
    v_rest: float = -65.0
    v_thresh: float = -50.0
    v_reset: float = -65.0
    r_m: float = 1.0

    def evolve_to_time(self, v0: float, t: float, current: float) -> float:
        """Compute V(t) given initial voltage v0 and constant current.

        V(t) = V_rest + (v0 - V_rest) * exp(-t/tau) + R*I*(1 - exp(-t/tau))
        """
        decay = math.exp(-t / self.tau)
        v_inf = self.v_rest + self.r_m * current
        return v_inf + (v0 - v_inf) * decay

    def next_spike_time(self, v0: float, current: float) -> float | None:
        """Compute time until next spike under constant current.

        Returns None if the neuron will never reach threshold.
        t_spike = -tau * ln((V_inf - V_thresh) / (V_inf - v0))
        where V_inf = V_rest + R*I
        """
        v_inf = self.v_rest + self.r_m * current
        if v_inf <= self.v_thresh:
            return None  # current insufficient to reach threshold
        if v0 >= self.v_thresh:
            return 0.0  # already at or above threshold

        ratio = (v_inf - self.v_thresh) / (v_inf - v0)
        if ratio <= 0:
            return None
        return -self.tau * math.log(ratio)

    def isi(self, current: float) -> float | None:
        """Compute the inter-spike interval for constant current.

        ISI = -tau * ln((V_inf - V_thresh) / (V_inf - V_reset))
        """
        return self.next_spike_time(self.v_reset, current)

    def firing_rate(self, current: float) -> float:
        """Compute steady-state firing rate (Hz) for constant current.

        f = 1 / ISI, or 0 if sub-threshold.
        """
        t = self.isi(current)
        if t is None or t <= 0:
            return 0.0
        return 1000.0 / t  # t is in ms → rate in Hz

    def simulate(
        self,
        current: float,
        t_end: float,
        v0: float | None = None,
    ) -> tuple[list[float], list[float]]:
        """Simulate LIF with constant current from t=0 to t=t_end.

        Returns (spike_times, voltage_at_spikes).
        """
        v = v0 if v0 is not None else self.v_rest
        t = 0.0
        spike_times: list[float] = []

        while t < t_end:
            ts = self.next_spike_time(v, current)
            if ts is None or t + ts > t_end:
                break
            t += ts
            spike_times.append(t)
            v = self.v_reset

        return spike_times, [self.v_thresh] * len(spike_times)
