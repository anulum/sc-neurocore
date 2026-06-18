# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Conductance-based LIF — Destexhe et al. 2001

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class COBALIFNeuron:
    """Conductance-based LIF with coupled RK4 synaptic conductance state.

    C dV/dt = -g_L(V - E_L) - g_e(V - E_e) - g_i(V - E_i) + I
    dg_e/dt = -g_e / tau_e, dg_i/dt = -g_i / tau_i.

    Conductance injections are applied before integration. The full
    ``(v, g_e, g_i)`` candidate is advanced with RK4 and committed only after
    finite-value and envelope checks pass.
    """

    v: float = -65.0
    g_e: float = 0.0
    g_i: float = 0.0
    c_m: float = 200.0
    g_l: float = 10.0
    e_l: float = -65.0
    e_e: float = 0.0
    e_i: float = -80.0
    tau_e: float = 5.0
    tau_i: float = 10.0
    v_threshold: float = -50.0
    v_reset: float = -65.0
    dt: float = 0.1

    _V_MIN: float = -200.0
    _V_MAX: float = 100.0
    _G_MAX: float = 1.0e9

    def __post_init__(self) -> None:
        self._validated_state()

    @staticmethod
    def _finite(value: float, name: str) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    @classmethod
    def _positive(cls, value: float, name: str) -> float:
        value = cls._finite(value, name)
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
        return value

    @classmethod
    def _nonnegative(cls, value: float, name: str) -> float:
        value = cls._finite(value, name)
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")
        return value

    def _validated_state(self) -> tuple[float, float, float]:
        v = self._finite(self.v, "v")
        if not self._V_MIN <= v <= self._V_MAX:
            raise ValueError("v outside COBA LIF safety envelope")
        g_e = self._nonnegative(self.g_e, "g_e")
        g_i = self._nonnegative(self.g_i, "g_i")
        if g_e > self._G_MAX or g_i > self._G_MAX:
            raise ValueError("conductance outside COBA LIF safety envelope")

        self._positive(self.c_m, "c_m")
        self._nonnegative(self.g_l, "g_l")
        self._finite(self.e_l, "e_l")
        self._finite(self.e_e, "e_e")
        self._finite(self.e_i, "e_i")
        self._positive(self.tau_e, "tau_e")
        self._positive(self.tau_i, "tau_i")
        self._finite(self.v_threshold, "v_threshold")
        self._finite(self.v_reset, "v_reset")
        if not self._V_MIN <= self.v_reset <= self._V_MAX:
            raise ValueError("v_reset outside COBA LIF safety envelope")
        self._positive(self.dt, "dt")

        return v, g_e, g_i

    def step(self, current: float, delta_ge: float = 0.0, delta_gi: float = 0.0) -> int:
        """Advance one candidate-first RK4 timestep.

        Args:
            current: External drive current.
            delta_ge: Instantaneous excitatory conductance increment.
            delta_gi: Instantaneous inhibitory conductance increment.

        Returns:
            ``1`` when the RK4 voltage candidate crosses threshold, otherwise
            ``0``. Invalid states or candidates raise ``ValueError`` without
            mutating the stored state.
        """
        current = self._finite(current, "current")
        delta_ge = self._nonnegative(delta_ge, "delta_ge")
        delta_gi = self._nonnegative(delta_gi, "delta_gi")
        v, g_e, g_i = self._validated_state()

        g_e_pre = g_e + delta_ge
        g_i_pre = g_i + delta_gi
        if g_e_pre > self._G_MAX or g_i_pre > self._G_MAX:
            raise ValueError("conductance candidate outside COBA LIF safety envelope")

        v_candidate, g_e_candidate, g_i_candidate = self._rk4_candidate(
            v, g_e_pre, g_i_pre, current
        )
        i_syn = g_e_pre * (v - self.e_e) + g_i_pre * (v - self.e_i)

        for value, name in (
            (i_syn, "synaptic current candidate"),
            (v_candidate, "voltage candidate"),
            (g_e_candidate, "excitatory conductance candidate"),
            (g_i_candidate, "inhibitory conductance candidate"),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not self._V_MIN <= v_candidate <= self._V_MAX:
            raise ValueError("voltage candidate outside COBA LIF safety envelope")
        if g_e_candidate < 0.0 or g_i_candidate < 0.0:
            raise ValueError("conductance candidate must remain non-negative")

        if v_candidate >= self.v_threshold:
            self.v = self.v_reset
            self.g_e = g_e_candidate
            self.g_i = g_i_candidate
            return 1

        self.v = v_candidate
        self.g_e = g_e_candidate
        self.g_i = g_i_candidate
        return 0

    def reset(self) -> None:
        """Restore the membrane to leak reversal and clear conductances."""
        self.v = self.e_l
        self.g_e = 0.0
        self.g_i = 0.0

    def _derivatives(
        self, v: float, g_e: float, g_i: float, current: float
    ) -> tuple[float, float, float]:
        i_syn = g_e * (v - self.e_e) + g_i * (v - self.e_i)
        dv = (-self.g_l * (v - self.e_l) - i_syn + current) / self.c_m
        dge = -g_e / self.tau_e
        dgi = -g_i / self.tau_i
        return dv, dge, dgi

    def _rk4_candidate(
        self, v: float, g_e: float, g_i: float, current: float
    ) -> tuple[float, float, float]:
        """Return the coupled RK4 candidate for ``(v, g_e, g_i)``."""
        k1v, k1e, k1i = self._derivatives(v, g_e, g_i, current)
        k2v, k2e, k2i = self._derivatives(
            v + 0.5 * self.dt * k1v,
            g_e + 0.5 * self.dt * k1e,
            g_i + 0.5 * self.dt * k1i,
            current,
        )
        k3v, k3e, k3i = self._derivatives(
            v + 0.5 * self.dt * k2v,
            g_e + 0.5 * self.dt * k2e,
            g_i + 0.5 * self.dt * k2i,
            current,
        )
        k4v, k4e, k4i = self._derivatives(
            v + self.dt * k3v,
            g_e + self.dt * k3e,
            g_i + self.dt * k3i,
            current,
        )
        return (
            v + (self.dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v),
            g_e + (self.dt / 6.0) * (k1e + 2.0 * k2e + 2.0 * k3e + k4e),
            g_i + (self.dt / 6.0) * (k1i + 2.0 * k2i + 2.0 * k3i + k4i),
        )
