# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Fardet-Levina 2020 eLIF author-Brian specialization

"""Source-faithful energy-based leaky integrate-and-fire neuron."""

from __future__ import annotations

import math
from dataclasses import dataclass

_VOLTAGE_MIN = -200.0
_VOLTAGE_MAX = 100.0
_ENERGY_MAX = 5.0


@dataclass
class EnergyLIFNeuron:
    """Fardet-Levina eLIF using the authors' 0.1 ms Brian RK4 profile.

    The two coupled states are membrane potential ``v`` in mV and normalized
    available energy ``epsilon``.  ``alpha`` is energetic health, ``delta`` is
    the per-spike energy cost, and ``epsilon_c`` is the energy firing gate.

    Reference
    ---------
    Fardet & Levina (2020), PLOS Computational Biology 16:e1008503,
    DOI 10.1371/journal.pcbi.1008503.
    """

    v: float = -61.0
    epsilon: float = 0.32
    capacitance: float = 100.0
    g_leak: float = 9.0
    e_0: float = -62.5
    e_u: float = -58.5
    e_d: float = -40.0
    e_f: float = -62.0
    v_threshold: float = -59.0
    v_reset: float = -62.0
    alpha: float = 1.0
    epsilon_0: float = 0.5
    epsilon_c: float = 0.18
    delta: float = 0.01
    tau_e: float = 200.0
    dt: float = 0.1

    def __post_init__(self) -> None:
        """Validate the complete eLIF state and parameter contract."""
        self._validate_state()

    def _validate_state(self) -> None:
        """Reject invalid eLIF state before any mutation."""
        for field in ("v", "e_0", "e_u", "e_d", "e_f", "v_threshold", "v_reset"):
            if not math.isfinite(getattr(self, field)):
                raise ValueError(f"{field} must be finite")
        if not _VOLTAGE_MIN <= self.v <= _VOLTAGE_MAX:
            raise ValueError("v must be inside the voltage safety envelope")
        if not _VOLTAGE_MIN <= self.v_reset <= _VOLTAGE_MAX:
            raise ValueError("v_reset must be inside the voltage safety envelope")
        for field in ("epsilon", "epsilon_0", "epsilon_c", "delta"):
            value = getattr(self, field)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field} must be finite and non-negative")
        if self.epsilon > _ENERGY_MAX:
            raise ValueError("epsilon must be inside the energy safety envelope")
        for field in ("capacitance", "g_leak", "alpha", "tau_e", "dt"):
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be finite and positive")
        if self.e_d == self.e_f:
            raise ValueError("e_d must differ from e_f")
        if self.v_threshold <= self.v_reset:
            raise ValueError("v_threshold must be greater than v_reset")
        if self.dt > 1.0 or self.dt > self.tau_e:
            raise ValueError("dt exceeds the enrolled source integration envelope")

    def _derivatives(self, v: float, epsilon: float, current: float) -> tuple[float, float]:
        """Return the Fardet-Levina eLIF right-hand side."""
        e_leak = self.e_0 + (self.e_u - self.e_0) * (1.0 - epsilon / self.epsilon_0)
        dv = (self.g_leak * (e_leak - v) + current) / self.capacitance
        production = (1.0 - epsilon / (self.alpha * self.epsilon_0)) ** 3
        voltage_cost = (v - self.e_f) / (self.e_d - self.e_f)
        depsilon = (production - voltage_cost) / self.tau_e
        return dv, depsilon

    def _rk4_candidate(self, current: float) -> tuple[float, float]:
        """Return one simultaneous author-Brian RK4 candidate."""
        dt = self.dt
        k1_v, k1_e = self._derivatives(self.v, self.epsilon, current)
        k2_v, k2_e = self._derivatives(
            self.v + 0.5 * dt * k1_v,
            self.epsilon + 0.5 * dt * k1_e,
            current,
        )
        k3_v, k3_e = self._derivatives(
            self.v + 0.5 * dt * k2_v,
            self.epsilon + 0.5 * dt * k2_e,
            current,
        )
        k4_v, k4_e = self._derivatives(
            self.v + dt * k3_v,
            self.epsilon + dt * k3_e,
            current,
        )
        scale = dt / 6.0
        return (
            self.v + scale * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v),
            self.epsilon + scale * (k1_e + 2.0 * k2_e + 2.0 * k3_e + k4_e),
        )

    def step(self, current: float) -> int:
        """Advance one source RK4 sample and return the sampled spike event."""
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_state()
        v_candidate, epsilon_candidate = self._rk4_candidate(current)
        if not (
            math.isfinite(v_candidate)
            and _VOLTAGE_MIN <= v_candidate <= _VOLTAGE_MAX
            and math.isfinite(epsilon_candidate)
            and 0.0 <= epsilon_candidate <= _ENERGY_MAX
        ):
            raise ValueError("energy-LIF RK4 candidate left the safety envelope")
        if v_candidate > self.v_threshold and epsilon_candidate > self.epsilon_c:
            epsilon_after_spike = epsilon_candidate - self.delta
            if not 0.0 <= epsilon_after_spike <= _ENERGY_MAX:
                raise ValueError("energy-LIF post-spike energy left the safety envelope")
            self.v = self.v_reset
            self.epsilon = epsilon_after_spike
            return 1
        self.v = v_candidate
        self.epsilon = epsilon_candidate
        return 0

    def reset(self) -> None:
        """Restore the source equilibrium-oriented reset state."""
        self.v = self.e_0
        self.epsilon = self.alpha * self.epsilon_0
