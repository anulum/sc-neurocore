# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Izhikevich 2007 biophysical spiking neuron

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from ..base import BaseNeuron


@dataclass
class Izhikevich2007Neuron(BaseNeuron):
    """Izhikevich 2007 biophysical quadratic integrate-and-fire neuron.

    Equations from Izhikevich, E. M. (2007), *Dynamical Systems in
    Neuroscience*, using the NeuroML 2 ``izhikevich2007Cell`` parameterisation:

    ``C dv/dt = k (v - vr) (v - vt) - u + I``
    ``du/dt = a (b (v - vr) - u)``

    If ``v >= vpeak`` after integration, the neuron emits one spike and applies
    ``v <- c`` and ``u <- u + d``. Units are the NeuroML base units used by the
    importer: pF, nS, mV, ms, and pA.
    """

    C: float = 100.0
    k: float = 0.7
    vr: float = -60.0
    vt: float = -40.0
    vpeak: float = 35.0
    a: float = 0.03
    b: float = -2.0
    c: float = -50.0
    d: float = 100.0
    v0: float | None = None
    dt: float = 0.1
    integrator: Literal["euler", "rk4"] = "rk4"

    def __post_init__(self) -> None:
        if self.integrator not in {"euler", "rk4"}:
            raise ValueError(f"Unsupported integrator for Izhikevich2007Neuron: {self.integrator}")
        for name in ("k", "vr", "vt", "vpeak", "a", "b", "c", "d"):
            self._require_finite(name, getattr(self, name))
        self.C = self._require_positive("C", self.C)
        self.dt = self._require_positive("dt", self.dt)
        if self.v0 is None:
            self.v0 = self.vr
        else:
            self.v0 = self._require_finite("v0", self.v0)
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

    def _rhs(self, v: float, u: float, input_current: float) -> tuple[float, float]:
        dv = (self.k * (v - self.vr) * (v - self.vt) - u + input_current) / self.C
        du = self.a * (self.b * (v - self.vr) - u)
        return dv, du

    def step(self, input_current: float) -> int:
        input_current = self._require_finite("input_current", input_current)
        if self.integrator == "euler":
            self._step_euler(input_current)
        else:
            self._step_rk4(input_current)
        return self._apply_threshold_reset()

    def _step_euler(self, input_current: float) -> None:
        dv, du = self._rhs(self.v, self.u, input_current)
        self.v += self.dt * dv
        self.u += self.dt * du

    def _step_rk4(self, input_current: float) -> None:
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

    def _apply_threshold_reset(self) -> int:
        if self.v >= self.vpeak:
            self.v = self.c
            self.u += self.d
            return 1
        return 0

    def reset_state(self) -> None:
        self.v = float(self.v0)
        self.u = self.b * (self.v - self.vr)

    def get_state(self) -> dict[str, Any]:
        return {"v": float(self.v), "u": float(self.u)}
