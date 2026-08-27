# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Breakspear-Terry-Friston cortical neural mass

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class LarterBreakspearNeuron:
    """Three-state Larter-Breakspear cortical neural mass.

    The transition follows the excitatory-voltage, potassium-channel, and
    inhibitory-voltage equations of Breakspear, Terry, and Friston (2003).
    ``coupling`` is the external excitatory firing-rate term denoted by the
    population-average :math:`Q_V` in the source model. The default parameters
    are the maintained Larter-Breakspear profile used by The Virtual Brain.

    A fixed-step classical RK4 grid is an implementation specialisation. The
    equations are continuous, produce continuous population activity, and do
    not define a spike/reset event.

    References:
        Breakspear, Terry & Friston, Network 14 (2003), 703-732,
        doi:10.1088/0954-898X/14/4/305.
    """

    v: float = 0.1
    w: float = 0.1
    z: float = 0.1
    g_ca: float = 1.1
    g_na: float = 6.7
    g_k: float = 2.0
    g_l: float = 0.5
    v_ca: float = 1.0
    v_na: float = 0.53
    v_k: float = -0.7
    v_l: float = -0.5
    t_ca: float = -0.01
    t_na: float = 0.3
    t_k: float = 0.0
    delta_ca: float = 0.15
    delta_na: float = 0.15
    delta_k: float = 0.3
    phi: float = 0.7
    tau_k: float = 1.0
    b: float = 0.1
    a_ee: float = 0.4
    a_ei: float = 2.0
    a_ie: float = 2.0
    a_ne: float = 1.0
    a_ni: float = 0.4
    r_nmda: float = 0.25
    coupling_balance: float = 0.1
    v_t: float = 0.0
    z_t: float = 0.0
    delta_v: float = 0.65
    delta_z: float = 0.7
    q_v_max: float = 1.0
    q_z_max: float = 1.0
    i_ext: float = 0.3
    t_scale: float = 1.0
    dt: float = 0.01
    integrator: str = "rk4"

    def __post_init__(self) -> None:
        self.integrator = self.integrator.lower()
        if self.integrator not in {"rk4", "euler"}:
            raise ValueError("integrator must be 'rk4' or 'euler'")
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        for name, value in vars(self).items():
            if name != "integrator" and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.w <= 1.0:
            raise ValueError("w must remain in [0, 1]")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        for name in (
            "delta_ca",
            "delta_na",
            "delta_k",
            "delta_v",
            "delta_z",
            "tau_k",
            "phi",
            "b",
            "t_scale",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in ("g_ca", "g_na", "g_k", "g_l", "q_v_max", "q_z_max", "r_nmda"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 <= self.coupling_balance <= 1.0:
            raise ValueError("coupling_balance must be within [0, 1]")

    @staticmethod
    def _sigmoid(value: float, threshold: float, width: float) -> float:
        return 0.5 * (1.0 + math.tanh((value - threshold) / width))

    def _m_ca(self, v: float) -> float:
        return self._sigmoid(v, self.t_ca, self.delta_ca)

    def _m_na(self, v: float) -> float:
        return self._sigmoid(v, self.t_na, self.delta_na)

    def _m_k(self, v: float) -> float:
        return self._sigmoid(v, self.t_k, self.delta_k)

    def _q_v(self, v: float) -> float:
        return self.q_v_max * self._sigmoid(v, self.v_t, self.delta_v)

    def _q_z(self, z: float) -> float:
        return self.q_z_max * self._sigmoid(z, self.z_t, self.delta_z)

    def _derivatives(
        self, v: float, w: float, z: float, coupling: float
    ) -> tuple[float, float, float]:
        q_v = self._q_v(v)
        q_z = self._q_z(z)
        excitation = self.a_ee * (
            (1.0 - self.coupling_balance) * q_v + self.coupling_balance * coupling
        )
        dv = (
            -(self.g_ca + self.r_nmda * excitation) * self._m_ca(v) * (v - self.v_ca)
            - self.g_k * w * (v - self.v_k)
            - self.g_l * (v - self.v_l)
            - (self.g_na * self._m_na(v) + excitation) * (v - self.v_na)
            - self.a_ie * z * q_z
            + self.a_ne * self.i_ext
        )
        dw = self.phi * (self._m_k(v) - w) / self.tau_k
        dz = self.b * (self.a_ni * self.i_ext + self.a_ei * v * q_v)
        return self.t_scale * dv, self.t_scale * dw, self.t_scale * dz

    def _candidate(self, coupling: float) -> tuple[float, float, float]:
        v0, w0, z0 = self.v, self.w, self.z
        dt = self.dt
        k1 = self._derivatives(v0, w0, z0, coupling)
        if self.integrator == "euler":
            return v0 + dt * k1[0], w0 + dt * k1[1], z0 + dt * k1[2]
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
        return (
            v0 + (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
            w0 + (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
            z0 + (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]),
        )

    def step(self, coupling: float = 0.0) -> float:
        """Advance one observation step and return excitatory voltage."""

        if not math.isfinite(coupling):
            raise ValueError("coupling must be finite")
        self._validate_configuration()
        candidate = self._candidate(coupling)
        if not all(math.isfinite(value) for value in candidate):
            raise FloatingPointError("Larter-Breakspear candidate state became non-finite")
        if not 0.0 <= candidate[1] <= 1.0:
            raise FloatingPointError("Larter-Breakspear potassium gate left [0, 1]")
        self.v, self.w, self.z = candidate
        return self.v

    def reset(self) -> None:
        """Restore the maintained source-profile initial state."""

        self.v, self.w, self.z = 0.1, 0.1, 0.1
