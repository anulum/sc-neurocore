# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Nonlinear leaky integrate-and-fire neuron model."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class NonlinearLIFNeuron:
    """Quadratic nonlinear LIF neuron with slow adaptation.

    The membrane follows

    ``c_m dV/dt = a(V - v_rest)(V - v_crit) - w + I``

    and the adaptation current follows

    ``tau_w dw/dt = b(V - v_rest) - w``.

    The parameter validation is intentionally fail-closed: invalid geometry,
    non-finite state, or unstable integration constants are rejected before any
    state mutation can occur.
    """

    v: float = -65.0
    w: float = 0.0
    v_rest: float = -65.0
    v_crit: float = -40.0
    v_threshold: float = -20.0
    v_reset: float = -65.0
    a: float = 0.04
    b: float = 0.5
    tau_w: float = 100.0
    c_m: float = 1.0
    dt: float = 0.1

    def __post_init__(self) -> None:
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        finite_fields = {
            "v": self.v,
            "w": self.w,
            "v_rest": self.v_rest,
            "v_crit": self.v_crit,
            "v_threshold": self.v_threshold,
            "v_reset": self.v_reset,
            "a": self.a,
            "b": self.b,
            "tau_w": self.tau_w,
            "c_m": self.c_m,
            "dt": self.dt,
        }
        for name, value in finite_fields.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")

        if not self.v_rest < self.v_crit < self.v_threshold:
            raise ValueError("voltage geometry must satisfy v_rest < v_crit < v_threshold")
        if not self.v_reset < self.v_threshold:
            raise ValueError("v_reset must be below v_threshold")
        if self.a < 0.0:
            raise ValueError("a must be non-negative")
        if self.b < 0.0:
            raise ValueError("b must be non-negative")
        if self.tau_w <= 0.0:
            raise ValueError("tau_w must be positive")
        if self.c_m <= 0.0:
            raise ValueError("c_m must be positive")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.dt > self.tau_w:
            raise ValueError("dt must not exceed tau_w")

    def step(self, current: float) -> int:
        """Advance one candidate-first RK4 step and return ``1`` on spike."""
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_configuration()

        next_v, next_w = self._rk4_candidate(current)
        if not (math.isfinite(next_v) and math.isfinite(next_w)):
            raise ValueError("RK4 candidate must be finite")

        self.v = next_v
        self.w = next_w
        if next_v >= self.v_threshold:
            self.v = self.v_reset
            return 1
        return 0

    def reset(self) -> None:
        """Restore dynamic state without changing model parameters."""
        self.v = self.v_rest
        self.w = 0.0

    def _derivatives(self, v: float, w: float, current: float) -> tuple[float, float]:
        """Return NLIF right-hand-side derivatives at ``(v, w)``."""

        nonlinear = self.a * (v - self.v_rest) * (v - self.v_crit)
        dv = (nonlinear - w + current) / self.c_m
        dw = (self.b * (v - self.v_rest) - w) / self.tau_w
        return dv, dw

    def _rk4_candidate(self, current: float) -> tuple[float, float]:
        """Compute the fourth-order Runge-Kutta candidate without mutation."""

        k1_v, k1_w = self._derivatives(self.v, self.w, current)
        k2_v, k2_w = self._derivatives(
            self.v + 0.5 * self.dt * k1_v,
            self.w + 0.5 * self.dt * k1_w,
            current,
        )
        k3_v, k3_w = self._derivatives(
            self.v + 0.5 * self.dt * k2_v,
            self.w + 0.5 * self.dt * k2_w,
            current,
        )
        k4_v, k4_w = self._derivatives(
            self.v + self.dt * k3_v,
            self.w + self.dt * k3_w,
            current,
        )
        next_v = self.v + (self.dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
        next_w = self.w + (self.dt / 6.0) * (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w)
        return next_v, next_w
