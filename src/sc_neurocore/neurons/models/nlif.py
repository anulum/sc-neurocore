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
        """Advance one Euler step and return ``1`` when the neuron spikes."""
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_configuration()

        cubic = self.a * (self.v - self.v_rest) * (self.v - self.v_crit)
        dv = (cubic - self.w + current) / self.c_m * self.dt
        dw = (self.b * (self.v - self.v_rest) - self.w) / self.tau_w * self.dt
        self.v += dv
        self.w += dw
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            return 1
        return 0

    def reset(self) -> None:
        """Restore dynamic state without changing model parameters."""
        self.v = self.v_rest
        self.w = 0.0
