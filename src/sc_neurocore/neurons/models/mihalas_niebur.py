# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mihalas-Niebur Generalized IF candidate-first RK4 dynamics

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass
class MihalasNieburNeuron:
    """Mihalas-Niebur generalized integrate-and-fire neuron.

    The continuous four-state flow is advanced with a candidate-first RK4
    integrator. Spike reset is applied only after the continuous candidate is
    finite and crosses the adaptive threshold.

    Reference: Mihalas, S. & Niebur, E. (2009). Neural Comput. 21:704-718.
    """

    v: float = 0.0
    theta: float = 1.0
    i1: float = 0.0
    i2: float = 0.0
    v_rest: float = 0.0
    v_reset: float = 0.0
    theta_reset: float = 1.0
    theta_inf: float = 1.0
    tau_v: float = 10.0
    tau_theta: float = 100.0
    tau_1: float = 10.0
    tau_2: float = 200.0
    a: float = 0.0
    b: float = 0.0
    r1: float = 0.0
    r2: float = 0.0
    dt: float = 1.0

    @staticmethod
    def _finite_values(values: tuple[float, ...]) -> bool:
        return all(isfinite(value) for value in values)

    def _valid_runtime(self) -> bool:
        values = (
            self.v,
            self.theta,
            self.i1,
            self.i2,
            self.v_rest,
            self.v_reset,
            self.theta_reset,
            self.theta_inf,
            self.tau_v,
            self.tau_theta,
            self.tau_1,
            self.tau_2,
            self.a,
            self.b,
            self.r1,
            self.r2,
            self.dt,
        )
        return (
            self._finite_values(values)
            and self.tau_v > 0.0
            and self.tau_theta > 0.0
            and self.tau_1 > 0.0
            and self.tau_2 > 0.0
            and self.dt > 0.0
        )

    def _derivatives(
        self,
        v: float,
        theta: float,
        i1: float,
        i2: float,
        current: float,
    ) -> tuple[float, float, float, float]:
        return (
            (-(v - self.v_rest) + i1 + i2 + current) / self.tau_v,
            (self.theta_inf - theta + self.a * (v - self.v_rest)) / self.tau_theta,
            -i1 / self.tau_1,
            -i2 / self.tau_2,
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
        state = (self.v, self.theta, self.i1, self.i2)
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
        if not isfinite(current) or not self._valid_runtime():
            return 0

        candidate = self._rk4_candidate(current)
        if not self._finite_values(candidate):
            return 0

        next_v, next_theta, next_i1, next_i2 = candidate
        self.v = next_v
        self.theta = next_theta
        self.i1 = next_i1
        self.i2 = next_i2

        if self.v >= self.theta:
            self.v = self.v_reset + self.b * (self.v - self.v_rest)
            self.theta = max(self.theta, self.theta_reset)
            self.i1 += self.r1
            self.i2 += self.r2
            return 1
        return 0

    def reset(self) -> None:
        self.v = self.v_rest
        self.theta = self.theta_reset
        self.i1 = 0.0
        self.i2 = 0.0
