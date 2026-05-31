# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bertram et al. 2008 — phantom burster with dual slow

from __future__ import annotations

from dataclasses import dataclass
import math


_VOLTAGE_MIN = -250.0
_VOLTAGE_MAX = 250.0
_GATE_TOL = 1e-9


def _finite_float(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real value")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real value") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real value")
    return result


def _positive_float(name: str, value: float) -> float:
    result = _finite_float(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _non_negative_float(name: str, value: float) -> float:
    result = _finite_float(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _gate_value(name: str, value: float) -> float:
    result = _finite_float(name, value)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


@dataclass
class BertramPhantomBurster:
    """Bertram et al. 2008 — phantom burster with dual slow variables.

    C dV/dt  = -(I_Ca + I_K + I_s1 + I_s2 + I_L) + I_ext
    ds1/dt   = (s1_inf(V) - s1) / tau_s1
    ds2/dt   = (s2_inf(V) - s2) / tau_s2

    Two slow variables (s1, s2) with different timescales produce
    bursting via a phantom slow manifold.

    Reference: Bertram, R. et al. (1995). Biophys. J. 68:2323–2332.
    """

    v: float = -50.0
    s1: float = 0.1
    s2: float = 0.1
    g_ca: float = 3.6
    g_k: float = 10.0
    g_s1: float = 4.0
    g_s2: float = 4.0
    g_l: float = 0.2
    e_ca: float = 25.0
    e_k: float = -75.0
    e_l: float = -40.0
    c_m: float = 5.3
    v_m: float = -20.0
    s_m: float = 12.0
    v_n: float = -16.0
    s_n: float = 5.6
    v_s1: float = -40.0
    s_s1: float = 10.0
    v_s2: float = -42.0
    s_s2: float = 0.4
    tau_s1: float = 20000.0
    tau_s2: float = 100000.0
    dt: float = 0.5
    v_threshold: float = -20.0

    def __post_init__(self) -> None:
        self.v = _finite_float("v", self.v)
        self.s1 = _gate_value("s1", self.s1)
        self.s2 = _gate_value("s2", self.s2)
        self.g_ca = _non_negative_float("g_ca", self.g_ca)
        self.g_k = _non_negative_float("g_k", self.g_k)
        self.g_s1 = _non_negative_float("g_s1", self.g_s1)
        self.g_s2 = _non_negative_float("g_s2", self.g_s2)
        self.g_l = _non_negative_float("g_l", self.g_l)
        self.e_ca = _finite_float("e_ca", self.e_ca)
        self.e_k = _finite_float("e_k", self.e_k)
        self.e_l = _finite_float("e_l", self.e_l)
        self.c_m = _positive_float("c_m", self.c_m)
        self.v_m = _finite_float("v_m", self.v_m)
        self.s_m = _positive_float("s_m", self.s_m)
        self.v_n = _finite_float("v_n", self.v_n)
        self.s_n = _positive_float("s_n", self.s_n)
        self.v_s1 = _finite_float("v_s1", self.v_s1)
        self.s_s1 = _positive_float("s_s1", self.s_s1)
        self.v_s2 = _finite_float("v_s2", self.v_s2)
        self.s_s2 = _positive_float("s_s2", self.s_s2)
        self.tau_s1 = _positive_float("tau_s1", self.tau_s1)
        self.tau_s2 = _positive_float("tau_s2", self.tau_s2)
        self.dt = _positive_float("dt", self.dt)
        self.v_threshold = _finite_float("v_threshold", self.v_threshold)
        self._validate_state()

    def _validate_state(self) -> None:
        self.v = _finite_float("v", self.v)
        if self.v < _VOLTAGE_MIN or self.v > _VOLTAGE_MAX:
            raise ValueError("v outside Bertram phantom safety envelope")
        self.s1 = _gate_value("s1", self.s1)
        self.s2 = _gate_value("s2", self.s2)

    def _boltz(self, v: float, vh: float, k: float) -> float:
        v = _finite_float("v", v)
        vh = _finite_float("vh", vh)
        k = _positive_float("k", k)
        z = (vh - v) / k
        if z >= 0.0:
            exp_neg = math.exp(-z)
            return exp_neg / (1.0 + exp_neg)
        exp_pos = math.exp(z)
        return 1.0 / (1.0 + exp_pos)

    def _derivatives(
        self, v: float, s1: float, s2: float, current: float
    ) -> tuple[float, float, float]:
        m_inf = self._boltz(v, self.v_m, self.s_m)
        n_inf = self._boltz(v, self.v_n, self.s_n)
        s1_inf = self._boltz(v, self.v_s1, self.s_s1)
        s2_inf = self._boltz(v, self.v_s2, self.s_s2)

        i_ca = self.g_ca * m_inf * (v - self.e_ca)
        i_k = self.g_k * n_inf * (v - self.e_k)
        i_s1 = self.g_s1 * s1 * (v - self.e_k)
        i_s2 = self.g_s2 * s2 * (v - self.e_k)
        i_l = self.g_l * (v - self.e_l)

        dv = (-i_ca - i_k - i_s1 - i_s2 - i_l + current) / self.c_m
        ds1 = (s1_inf - s1) / self.tau_s1
        ds2 = (s2_inf - s2) / self.tau_s2
        return dv, ds1, ds2

    def _rk4_candidate(self, current: float) -> tuple[float, float, float]:
        v0, s10, s20 = self.v, self.s1, self.s2
        dt = self.dt

        k1 = self._derivatives(v0, s10, s20, current)
        k2 = self._derivatives(
            v0 + 0.5 * dt * k1[0],
            s10 + 0.5 * dt * k1[1],
            s20 + 0.5 * dt * k1[2],
            current,
        )
        k3 = self._derivatives(
            v0 + 0.5 * dt * k2[0],
            s10 + 0.5 * dt * k2[1],
            s20 + 0.5 * dt * k2[2],
            current,
        )
        k4 = self._derivatives(
            v0 + dt * k3[0],
            s10 + dt * k3[1],
            s20 + dt * k3[2],
            current,
        )
        v = v0 + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        s1 = s10 + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        s2 = s20 + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0
        return v, s1, s2

    def _validate_candidate(self, v: float, s1: float, s2: float) -> tuple[float, float, float]:
        values = {"v": v, "s1": s1, "s2": s2}
        for name, value in values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} candidate must be finite")
        if v < _VOLTAGE_MIN or v > _VOLTAGE_MAX:
            raise ValueError("v candidate outside Bertram phantom safety envelope")
        if s1 < -_GATE_TOL or s1 > 1.0 + _GATE_TOL:
            raise ValueError("s1 candidate must remain in [0, 1]")
        if s2 < -_GATE_TOL or s2 > 1.0 + _GATE_TOL:
            raise ValueError("s2 candidate must remain in [0, 1]")
        return v, min(1.0, max(0.0, s1)), min(1.0, max(0.0, s2))

    def step(self, current: float) -> int:
        current = _finite_float("current", current)
        self._validate_state()
        v_prev = self.v
        v, s1, s2 = self._validate_candidate(*self._rk4_candidate(current))
        self.v = v
        self.s1 = s1
        self.s2 = s2

        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self) -> None:
        self.v = -50.0
        self.s1 = 0.1
        self.s2 = 0.1
