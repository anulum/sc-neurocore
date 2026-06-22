# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Fardet & Levina 2020 — LIF with metabolic energy constraint

from __future__ import annotations

import math
from dataclasses import dataclass

_VOLTAGE_MIN = -200.0
_VOLTAGE_MAX = 100.0
_ENERGY_GATE = 0.1


@dataclass
class EnergyLIFNeuron:
    """Fardet & Levina 2020 — LIF with metabolic energy constraint.

    Reference: Fardet, T. & Levina, A. (2020). PLoS Comput. Biol. 16(12):e1008503.
    """

    v: float = -70.0
    epsilon: float = 1.0
    v_rest: float = -70.0
    v_reset: float = -70.0
    v_threshold: float = -50.0
    tau_m: float = 10.0
    tau_e: float = 500.0
    alpha: float = 0.1
    epsilon_0: float = 1.0
    resistance: float = 1.0
    dt: float = 1.0

    def __post_init__(self) -> None:
        """Validate the energy-LIF state before first use."""
        self._validate_state()

    def _validate_state(self) -> None:
        """Reject non-physical energy-LIF state before mutation."""
        for field in ("v", "v_rest", "v_reset", "v_threshold"):
            if not math.isfinite(getattr(self, field)):
                raise ValueError(f"{field} must be finite")
        for field in ("epsilon", "epsilon_0"):
            value = getattr(self, field)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field} must be finite and non-negative")
        for field in ("tau_m", "tau_e", "resistance", "dt"):
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be finite and positive")
        if not math.isfinite(self.alpha) or self.alpha < 0.0:
            raise ValueError("alpha must be finite and non-negative")
        if self.epsilon > self.epsilon_0:
            raise ValueError("epsilon must not exceed epsilon_0")
        if not (_VOLTAGE_MIN <= self.v <= _VOLTAGE_MAX):
            raise ValueError("v must be inside the voltage safety envelope")
        if not (_VOLTAGE_MIN <= self.v_reset <= _VOLTAGE_MAX):
            raise ValueError("v_reset must be inside the voltage safety envelope")
        if self.dt > self.tau_m or self.dt > self.tau_e:
            raise ValueError("dt must not exceed tau_m or tau_e")
        if self.v_threshold <= self.v_rest:
            raise ValueError("v_threshold must be greater than v_rest")
        if self.v_threshold <= self.v_reset:
            raise ValueError("v_threshold must be greater than v_reset")

    def _exact_candidate(self, current: float) -> tuple[float, float]:
        """Return the exact constant-current `(v, epsilon)` candidate."""
        membrane_decay = math.exp(-self.dt / self.tau_m)
        energy_decay = math.exp(-self.dt / self.tau_e)
        energy_delta = self.epsilon - self.epsilon_0
        epsilon_candidate = self.epsilon_0 + energy_delta * energy_decay
        steady_energy_integral = self.epsilon_0 * self.tau_m * (1.0 - membrane_decay)
        coupled_rate = (1.0 / self.tau_m) - (1.0 / self.tau_e)
        if abs(coupled_rate) < 1.0e-12:
            transient_energy_integral = energy_delta * membrane_decay * self.dt
        else:
            transient_energy_integral = (
                energy_delta * membrane_decay * math.expm1(coupled_rate * self.dt) / coupled_rate
            )
        voltage_candidate = (
            self.v_rest
            + (self.v - self.v_rest) * membrane_decay
            + (self.resistance * current / self.tau_m)
            * (steady_energy_integral + transient_energy_integral)
        )
        return voltage_candidate, epsilon_candidate

    def step(self, current: float) -> int:
        """Advance one exact-flow step and return `1` when a spike occurs."""
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_state()

        v_candidate, epsilon_candidate = self._exact_candidate(current)
        if not (
            math.isfinite(v_candidate)
            and _VOLTAGE_MIN <= v_candidate <= _VOLTAGE_MAX
            and math.isfinite(epsilon_candidate)
            and 0.0 <= epsilon_candidate <= self.epsilon_0
        ):
            raise ValueError("energy-LIF exact-flow candidate left the safety envelope")
        if v_candidate >= self.v_threshold and epsilon_candidate > _ENERGY_GATE:
            epsilon_after_spike = max(0.0, epsilon_candidate - self.alpha)
            if not math.isfinite(epsilon_after_spike) or epsilon_after_spike > self.epsilon_0:
                raise ValueError("energy-LIF post-spike energy left the safety envelope")
            self.v = self.v_reset
            self.epsilon = epsilon_after_spike
            return 1
        self.v = v_candidate
        self.epsilon = epsilon_candidate
        return 0

    def reset(self) -> None:
        """Restore membrane voltage and energy reserve to resting state."""
        self.v = self.v_rest
        self.epsilon = self.epsilon_0
