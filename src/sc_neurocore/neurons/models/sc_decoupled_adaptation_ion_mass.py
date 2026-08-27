# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Retained three-state project ion-mass recurrence

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SCDecoupledAdaptationIonMassNeuron:
    """Retain the former project ion-mass recurrence without paper attribution."""

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
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        values = tuple(value for name, value in vars(self).items() if name != "integrator")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("SC ion-mass state and parameters must be finite")
        if not 0.0 <= self.w <= 1.0:
            raise ValueError("w must remain in [0, 1]")
        for name in ("dt", "tau_k", "phi", "b", "g_ca", "g_na", "g_k", "g_l"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")

    @staticmethod
    def _gate(v: float, midpoint: float, width: float) -> float:
        return 0.5 * (1.0 + math.tanh((v - midpoint) / width))

    def _derivatives(
        self, v: float, w: float, z: float, coupling: float
    ) -> tuple[float, float, float]:
        m_ca = self._gate(v, -0.01, 0.15)
        m_na = self._gate(v, 0.12, 0.15)
        m_k = self._gate(v, self.v0, 0.3)
        dv = (
            -self.g_ca * m_ca * (v - self.v_ca)
            - self.g_na * m_na * (v - self.v_na)
            - self.g_k * w * (v - self.v_k)
            - self.g_l * (v - self.v_l)
            + self.i_ext
            + coupling
            + self.a_ee * v
        )
        return dv, self.phi * (m_k - w) / self.tau_k, self.b * (v + 0.5 - z)

    def _candidate(self, coupling: float) -> tuple[float, float, float]:
        v0, w0, z0 = self.v, self.w, self.z
        dt = self.dt
        k1 = self._derivatives(v0, w0, z0, coupling)
        if self.integrator == "euler":
            return v0 + dt * k1[0], w0 + dt * k1[1], z0 + dt * k1[2]
        k2 = self._derivatives(
            v0 + 0.5 * dt * k1[0], w0 + 0.5 * dt * k1[1], z0 + 0.5 * dt * k1[2], coupling
        )
        k3 = self._derivatives(
            v0 + 0.5 * dt * k2[0], w0 + 0.5 * dt * k2[1], z0 + 0.5 * dt * k2[2], coupling
        )
        k4 = self._derivatives(v0 + dt * k3[0], w0 + dt * k3[1], z0 + dt * k3[2], coupling)
        return (
            v0 + (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
            w0 + (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
            z0 + (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]),
        )

    def step(self, coupling: float = 0.0) -> float:
        """Advance the retained project recurrence atomically."""

        if not math.isfinite(coupling):
            raise ValueError("coupling must be finite")
        self._validate_configuration()
        candidate = self._candidate(coupling)
        if not all(math.isfinite(value) for value in candidate):
            raise FloatingPointError("SC ion-mass candidate state became non-finite")
        if not 0.0 <= candidate[1] <= 1.0:
            raise FloatingPointError("SC ion-mass potassium gate left [0, 1]")
        self.v, self.w, self.z = candidate
        return self.v

    def reset(self) -> None:
        """Restore dynamic state without changing configuration."""

        self.v, self.w, self.z = -0.5, 0.0, 0.0
