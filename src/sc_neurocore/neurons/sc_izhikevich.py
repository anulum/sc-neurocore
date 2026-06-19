# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Stochastic Izhikevich neuron (software-only)

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from ..constants import (
    IZH_A,
    IZH_B,
    IZH_C,
    IZH_D,
    IZH_SPIKE_THRESHOLD,
    LIF_DT,
)
from ..utils.rng import RNG
from .base import BaseNeuron


@dataclass
class SCIzhikevichNeuron(BaseNeuron):
    """
    Stochastic Izhikevich neuron (software-only).

    Standard Izhikevich model (IEEE TNN 14(6), 2003):
    v' = 0.04*v^2 + 5*v + 140 - u + I + noise
    u' = a*(b*v - u)

    When v >= 30 mV: spike, then v <- c, u <- u + d.

    Example
    -------
    >>> neuron = SCIzhikevichNeuron(noise_std=0.0)
    >>> spikes = [neuron.step(10.0) for _ in range(100)]
    >>> sum(spikes) > 0  # regular spiking with I=10
    True

    Integrator options:
    - ``baseline_half_euler`` preserves the historical two-half-step path
    - ``rk4`` is an explicit higher-order alternative path
    """

    a: float = IZH_A
    b: float = IZH_B
    c: float = IZH_C
    d: float = IZH_D
    dt: float = LIF_DT
    noise_std: float = 0.0
    seed: int | None = None
    integrator: Literal["baseline_half_euler", "rk4"] = "baseline_half_euler"

    def __post_init__(self) -> None:
        if self.integrator not in {"baseline_half_euler", "rk4"}:
            raise ValueError(f"Unsupported integrator for SCIzhikevichNeuron: {self.integrator}")
        for name in ("a", "b", "c", "d"):
            self._require_finite(name, getattr(self, name))
        self.dt = self._require_positive("dt", self.dt)
        self.noise_std = self._require_nonnegative("noise_std", self.noise_std)
        self._rng = RNG(self.seed)
        self.v: float = self.c
        self.u: float = self.b * self.c
        self.reset_state()

    @staticmethod
    def _require_finite(name: str, value: float) -> float:
        if not isinstance(value, int | float) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
        return float(value)

    @classmethod
    def _require_positive(cls, name: str, value: float) -> float:
        result = cls._require_finite(name, value)
        if result <= 0.0:
            raise ValueError(f"{name} must be positive")
        return result

    @classmethod
    def _require_nonnegative(cls, name: str, value: float) -> float:
        result = cls._require_finite(name, value)
        if result < 0.0:
            raise ValueError(f"{name} must be non-negative")
        return result

    def step(self, input_current: float) -> int:
        input_current = self._require_finite("input_current", input_current)
        if self.integrator == "baseline_half_euler":
            return self._step_baseline_half_euler(input_current)
        return self._step_rk4(input_current)

    def _rhs(self, v: float, u: float, input_current: float) -> tuple[float, float]:
        dv = 0.04 * v**2 + 5.0 * v + 140.0 - u + input_current
        du = self.a * (self.b * v - u)
        return dv, du

    def _apply_noise_and_threshold(self) -> int:
        if self.noise_std > 0.0:
            self.v += float(self._rng.normal(0.0, self.noise_std))

        if self.v >= IZH_SPIKE_THRESHOLD:
            self.v = self.c
            self.u += self.d
            return 1
        return 0

    def _step_baseline_half_euler(self, input_current: float) -> int:
        # Two half-steps for numerical stability on 0.04v² term.
        # Izhikevich (2003) recommends dt ≤ 0.5 ms; we split each dt into two.
        half_dt = self.dt * 0.5
        for _ in range(2):
            dv, du = self._rhs(self.v, self.u, input_current)
            dv *= half_dt
            du *= half_dt
            self.v += dv
            self.u += du
        return self._apply_noise_and_threshold()

    def _step_rk4(self, input_current: float) -> int:
        state = np.array([self.v, self.u], dtype=np.float64)

        def rhs(state_vec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            dv, du = self._rhs(float(state_vec[0]), float(state_vec[1]), input_current)
            return np.array([dv, du], dtype=np.float64)

        k1 = rhs(state)
        k2 = rhs(state + 0.5 * self.dt * k1)
        k3 = rhs(state + 0.5 * self.dt * k2)
        k4 = rhs(state + self.dt * k3)
        state = state + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        self.v = float(state[0])
        self.u = float(state[1])
        return self._apply_noise_and_threshold()

    def reset_state(self) -> None:
        self.v = self.c  # membrane potential
        self.u = self.b * self.v  # recovery variable

    def get_state(self) -> dict[str, Any]:
        return {"v": float(self.v), "u": float(self.u)}
