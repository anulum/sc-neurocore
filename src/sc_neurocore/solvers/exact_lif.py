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
from typing import Any


def _finite_float(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real value")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real value") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real value")
    return result


def _positive_float(name: str, value: Any) -> float:
    result = _finite_float(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


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

    def __post_init__(self) -> None:
        """Validate and normalise the membrane parameters."""
        self.tau = _positive_float("tau", self.tau)
        self.v_rest = _finite_float("v_rest", self.v_rest)
        self.v_thresh = _finite_float("v_thresh", self.v_thresh)
        self.v_reset = _finite_float("v_reset", self.v_reset)
        self.r_m = _positive_float("r_m", self.r_m)
        if self.v_reset >= self.v_thresh:
            raise ValueError("v_reset must be below v_thresh")

    def evolve_to_time(self, v0: float, t: float, current: float) -> float:
        """Compute V(t) given initial voltage v0 and constant current.

        V(t) = V_rest + (v0 - V_rest) * exp(-t/tau) + R*I*(1 - exp(-t/tau))
        """
        v0 = _finite_float("v0", v0)
        t = _finite_float("t", t)
        current = _finite_float("current", current)
        if t < 0.0:
            raise ValueError("t must be non-negative")
        decay = math.exp(-t / self.tau)
        v_inf = self.v_rest + self.r_m * current
        return v_inf + (v0 - v_inf) * decay

    def next_spike_time(self, v0: float, current: float) -> float | None:
        """Compute time until next spike under constant current.

        Returns None if the neuron will never reach threshold.
        t_spike = -tau * ln((V_inf - V_thresh) / (V_inf - v0))
        where V_inf = V_rest + R*I
        """
        v0 = _finite_float("v0", v0)
        current = _finite_float("current", current)
        v_inf = self.v_rest + self.r_m * current
        if v_inf <= self.v_thresh:
            return None  # current insufficient to reach threshold
        if v0 >= self.v_thresh:
            return 0.0  # already at or above threshold

        ratio = (v_inf - self.v_thresh) / (v_inf - v0)
        if ratio <= 0:  # pragma: no cover - defensive guard after threshold/current checks.
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
        current = _finite_float("current", current)
        t_end = _finite_float("t_end", t_end)
        if t_end < 0.0:
            raise ValueError("t_end must be non-negative")
        v = _finite_float("v0", v0) if v0 is not None else self.v_rest
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
