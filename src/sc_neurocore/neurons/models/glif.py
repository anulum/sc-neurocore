# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Allen Institute GLIF5 candidate-first RK4 dynamics

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class GLIFNeuron:
    """Allen Institute GLIF5 generalized leaky integrate-and-fire neuron.

    The four dynamic states are advanced with candidate-first RK4 over the
    continuous GLIF flow. Spike reset is applied only after the candidate is
    finite and crosses the adaptive threshold.

    Reference: Teeter, C. et al. (2018). Nat. Commun. 9:709.
    """

    v: float = -70.0
    theta: float = -50.0
    theta_inf: float = -50.0
    i_asc1: float = 0.0
    i_asc2: float = 0.0
    v_rest: float = -70.0
    v_reset: float = -70.0
    tau_m: float = 10.0
    tau_theta: float = 100.0
    tau_asc1: float = 10.0
    tau_asc2: float = 200.0
    a_theta: float = 0.01
    delta_theta: float = 2.0
    r_asc1: float = 1.0
    r_asc2: float = 0.5
    resistance: float = 1.0
    dt: float = 1.0

    def __post_init__(self) -> None:
        self._raise_if_invalid_runtime()

    @staticmethod
    def _finite_values(values: tuple[float, ...]) -> bool:
        return all(math.isfinite(value) for value in values)

    def _raise_if_invalid_runtime(self) -> None:
        finite_fields = (
            "v",
            "theta",
            "theta_inf",
            "i_asc1",
            "i_asc2",
            "v_rest",
            "v_reset",
            "a_theta",
            "delta_theta",
            "r_asc1",
            "r_asc2",
            "resistance",
        )
        for field in finite_fields:
            value = getattr(self, field)
            if not math.isfinite(value):
                raise ValueError(f"{field} must be finite")
        for field in ("tau_m", "tau_theta", "tau_asc1", "tau_asc2", "dt"):
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be finite and positive")
        for field in ("delta_theta", "resistance"):
            value = getattr(self, field)
            if value < 0.0:
                raise ValueError(f"{field} must be finite and non-negative")

    def _derivatives(
        self,
        v: float,
        theta: float,
        i_asc1: float,
        i_asc2: float,
        current: float,
    ) -> tuple[float, float, float, float]:
        return (
            (-(v - self.v_rest) + self.resistance * current + i_asc1 + i_asc2) / self.tau_m,
            (self.theta_inf - theta + self.a_theta * (v - self.v_rest)) / self.tau_theta,
            -i_asc1 / self.tau_asc1,
            -i_asc2 / self.tau_asc2,
        )

    @staticmethod
    def _add_scaled(
        state: tuple[float, float, float, float],
        slope: tuple[float, float, float, float],
        scale: float,
    ) -> tuple[float, float, float, float]:
        return (
            state[0] + scale * slope[0],
            state[1] + scale * slope[1],
            state[2] + scale * slope[2],
            state[3] + scale * slope[3],
        )

    def _rk4_candidate(self, current: float) -> tuple[float, float, float, float]:
        state = (self.v, self.theta, self.i_asc1, self.i_asc2)
        half_dt = 0.5 * self.dt
        k1 = self._derivatives(*state, current)
        k2 = self._derivatives(*self._add_scaled(state, k1, half_dt), current)
        k3 = self._derivatives(*self._add_scaled(state, k2, half_dt), current)
        k4 = self._derivatives(*self._add_scaled(state, k3, self.dt), current)
        return (
            state[0] + self.dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            state[1] + self.dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
            state[2] + self.dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
            state[3] + self.dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0,
        )

    def step(self, current: float) -> int:
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._raise_if_invalid_runtime()
        candidate = self._rk4_candidate(current)
        if not self._finite_values(candidate):
            raise FloatingPointError("GLIF candidate state must be finite")

        next_v, next_theta, next_i_asc1, next_i_asc2 = candidate
        self.v = next_v
        self.theta = next_theta
        self.i_asc1 = next_i_asc1
        self.i_asc2 = next_i_asc2

        if self.v >= self.theta:
            self.v = self.v_reset
            self.theta += self.delta_theta
            self.i_asc1 += self.r_asc1
            self.i_asc2 += self.r_asc2
            return 1
        return 0

    def reset(self) -> None:
        self.v = self.v_rest
        self.theta = self.theta_inf
        self.i_asc1 = 0.0
        self.i_asc2 = 0.0
