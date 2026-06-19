# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Breakspear, Terry & Friston 2003 — neural mass with ion

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

_STATE_NAMES = ("v", "w", "z")
_PARAM_NAMES = (
    "g_ca",
    "g_na",
    "g_k",
    "v_ca",
    "v_na",
    "v_k",
    "v_l",
    "g_l",
    "phi",
    "tau_k",
    "b",
    "a_ee",
    "v0",
    "i_ext",
    "dt",
)
_STRICTLY_POSITIVE_PARAMS = ("dt", "tau_k", "phi", "b", "g_ca", "g_na", "g_k", "g_l")


@dataclass
class LarterBreakspearNeuron:
    """Breakspear, Terry & Friston 2003 — neural mass with ion channels.

    3 ODEs per node. Combines Wilson-Cowan population dynamics with
    conductance-based ion channel kinetics for whole-brain modelling.
    Used in The Virtual Brain (TVB) simulator.

    Reference: Larter, R. et al. (1999). Chaos 9:795–804.; Breakspear, M. et al. (2003). Cereb. Cortex 13:189–202.
    """

    v: float = -0.5
    w: float = 0.0
    z: float = 0.0
    g_ca: float = 1.1
    g_na: float = 6.7
    g_k: float = 2.0
    v_ca: float = 1.0
    v_na: float = 0.53
    v_k: float = -0.7
    v_l: float = -0.5
    g_l: float = 0.5
    phi: float = 0.7
    tau_k: float = 1.0
    b: float = 0.1
    a_ee: float = 0.36
    v0: float = 0.0
    i_ext: float = 0.3
    dt: float = 0.01
    integrator: str = "rk4"

    def __post_init__(self) -> None:
        self.integrator = self.integrator.lower()
        if self.integrator not in {"rk4", "euler"}:
            raise ValueError("integrator must be 'rk4' or 'euler'")
        self._validate_configuration(coerce=True)

    def _validate_configuration(self, *, coerce: bool = False) -> None:
        for name in (*_STATE_NAMES, *_PARAM_NAMES):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if coerce:
                setattr(self, name, value)

        for name in _STRICTLY_POSITIVE_PARAMS:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.w <= 1.0:
            raise ValueError("w must remain in [0, 1]")

    def _m_ca(self, v: float) -> Any:
        return 0.5 * (1.0 + np.tanh((v - (-0.01)) / 0.15))

    def _m_na(self, v: float) -> Any:
        return 0.5 * (1.0 + np.tanh((v - 0.12) / 0.15))

    def _m_k(self, v: float) -> Any:
        return 0.5 * (1.0 + np.tanh((v - self.v0) / 0.3))

    def _derivatives(
        self, v: float, w: float, z: float, coupling: float
    ) -> tuple[float, float, float]:
        i_ca = self.g_ca * self._m_ca(v) * (v - self.v_ca)
        i_na = self.g_na * self._m_na(v) * (v - self.v_na)
        i_k = self.g_k * w * (v - self.v_k)
        i_l = self.g_l * (v - self.v_l)

        dv = -i_ca - i_na - i_k - i_l + self.i_ext + coupling + self.a_ee * v
        dw = self.phi * (self._m_k(v) - w) / self.tau_k
        dz = self.b * (v + 0.5 - z)
        return float(dv), float(dw), float(dz)

    def _set_state(self, v: float, w: float, z: float) -> None:
        if not (np.isfinite(v) and np.isfinite(w) and np.isfinite(z)):
            raise FloatingPointError("Larter-Breakspear state became non-finite")
        if not 0.0 <= w <= 1.0:
            raise FloatingPointError("Larter-Breakspear potassium gate left [0, 1]")
        self.v = float(v)
        self.w = float(w)
        self.z = float(z)

    def _step_euler(self, coupling: float) -> None:
        dv, dw, dz = self._derivatives(self.v, self.w, self.z, coupling)
        self._set_state(self.v + dv * self.dt, self.w + dw * self.dt, self.z + dz * self.dt)

    def _step_rk4(self, coupling: float) -> None:
        v0, w0, z0 = self.v, self.w, self.z
        dt = self.dt

        k1 = self._derivatives(v0, w0, z0, coupling)
        k2 = self._derivatives(
            v0 + 0.5 * dt * k1[0],
            w0 + 0.5 * dt * k1[1],
            z0 + 0.5 * dt * k1[2],
            coupling,
        )
        k3 = self._derivatives(
            v0 + 0.5 * dt * k2[0],
            w0 + 0.5 * dt * k2[1],
            z0 + 0.5 * dt * k2[2],
            coupling,
        )
        k4 = self._derivatives(v0 + dt * k3[0], w0 + dt * k3[1], z0 + dt * k3[2], coupling)

        self._set_state(
            v0 + (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
            w0 + (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
            z0 + (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]),
        )

    def step(self, coupling: float = 0.0) -> float:
        if not np.isfinite(coupling):
            raise ValueError("coupling must be finite")
        self._validate_configuration()

        if self.integrator == "rk4":
            self._step_rk4(coupling)
        else:
            self._step_euler(coupling)
        return self.v

    def reset(self) -> None:
        self.v, self.w, self.z = -0.5, 0.0, 0.0
